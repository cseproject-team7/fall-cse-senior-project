"""
Step 4: Train Seq2Seq Pattern Prediction Model

Input: prepared_data/all_labeled_patterns.jsonl
Output: model_output/seq2seq_pattern_model.pth

Features:
- Encoder-Decoder LSTM architecture
- Predicts complete patterns (variable-length app sequences)
- Colab-friendly (auto-detects environment)
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import sys
from collections import defaultdict

# --- CONFIGURATION ---
INPUT_FILE = 'prepared_data/all_labeled_patterns.jsonl'
MODEL_OUTPUT_DIR = 'model_output'
HIDDEN_DIM = 256
EMBEDDING_DIM = 128
NUM_LAYERS = 2
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HISTORY_LENGTH = 5  # Number of past patterns to use as context

# Special tokens
PAD_IDX = 0
START_IDX = 1
END_IDX = 2

# --- DATASET ---
class PatternSeq2SeqDataset(Dataset):
    def __init__(self, encoder_inputs, decoder_inputs, decoder_targets, input_lengths, target_lengths):
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_targets = decoder_targets
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths
        
    def __len__(self):
        return len(self.encoder_inputs)
        
    def __getitem__(self, idx):
        return (self.encoder_inputs[idx], self.decoder_inputs[idx], self.decoder_targets[idx],
                self.input_lengths[idx], self.target_lengths[idx])

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    encoder_inputs, decoder_inputs, decoder_targets, input_lengths, target_lengths = zip(*batch)
    
    # Pad encoder inputs (list of lists of variable-length patterns)
    encoder_inputs_padded = []
    for enc_input in encoder_inputs:
        # enc_input is a list of patterns, each pattern is a list of app IDs
        # Flatten into a single sequence
        flattened = []
        for pattern in enc_input:
            flattened.extend(pattern)
        encoder_inputs_padded.append(torch.tensor(flattened, dtype=torch.long))
    
    encoder_inputs_padded = pad_sequence(encoder_inputs_padded, batch_first=True, padding_value=PAD_IDX)
    
    # Pad decoder inputs and targets
    decoder_inputs_padded = pad_sequence([torch.tensor(x, dtype=torch.long) for x in decoder_inputs], 
                                         batch_first=True, padding_value=PAD_IDX)
    decoder_targets_padded = pad_sequence([torch.tensor(x, dtype=torch.long) for x in decoder_targets], 
                                          batch_first=True, padding_value=PAD_IDX)
    
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    return encoder_inputs_padded, decoder_inputs_padded, decoder_targets_padded, input_lengths, target_lengths

# --- ENCODER-DECODER MODEL ---
class Seq2SeqPatternModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(Seq2SeqPatternModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Shared embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        
        # Encoder
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                               batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        
        # Decoder
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                               batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, encoder_input, decoder_input, encoder_lengths):
        # Encode
        encoder_emb = self.embedding(encoder_input)
        
        # Pack padded sequence for encoder
        packed_encoder = pack_padded_sequence(encoder_emb, encoder_lengths.cpu(), 
                                              batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.encoder(packed_encoder)
        
        # Decode
        decoder_emb = self.embedding(decoder_input)
        decoder_output, _ = self.decoder(decoder_emb, (hidden, cell))
        
        # Project to vocabulary
        output = self.fc(decoder_output)
        
        return output

# --- DATA PREPARATION ---
def prepare_data(input_file):
    """Load and prepare data for Seq2Seq training"""
    print(f"Loading data from {input_file}...")
    
    # Load all patterns
    patterns_by_user = defaultdict(list)
    
    with open(input_file, 'r') as f:
        for line in f:
            pattern = json.loads(line)
            patterns_by_user[pattern['user']].append(pattern)
    
    print(f"Loaded patterns from {len(patterns_by_user):,} users")
    
    # Sort patterns by timestamp for each user
    for user in patterns_by_user:
        patterns_by_user[user].sort(key=lambda x: x['timestamps'][0])
    
    # Build vocabulary
    all_apps = set()
    for user_patterns in patterns_by_user.values():
        for pattern in user_patterns:
            all_apps.update(pattern['apps'])
    
    # Create encoder (reserve 0=PAD, 1=START, 2=END)
    app_encoder = LabelEncoder()
    app_encoder.fit(list(all_apps))
    
    print(f"Vocabulary size: {len(app_encoder.classes_)} apps")
    
    # Create training sequences
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []
    input_lengths = []
    target_lengths = []
    
    for user, user_patterns in patterns_by_user.items():
        if len(user_patterns) < HISTORY_LENGTH + 1:
            continue
        
        # Sliding window
        for i in range(len(user_patterns) - HISTORY_LENGTH):
            # Encoder input: past HISTORY_LENGTH patterns
            history = user_patterns[i:i+HISTORY_LENGTH]
            
            # Decoder target: next pattern
            target_pattern = user_patterns[i+HISTORY_LENGTH]
            
            # Encode history (list of lists)
            encoded_history = []
            total_len = 0
            for pattern in history:
                encoded_pattern = [app_encoder.transform([app])[0] + 3 for app in pattern['apps']]  # +3 to account for PAD, START, END
                encoded_history.append(encoded_pattern)
                total_len += len(encoded_pattern)
            
            # Encode target
            encoded_target = [app_encoder.transform([app])[0] + 3 for app in target_pattern['apps']]
            
            # Decoder input: START + target[:-1]
            decoder_input = [START_IDX] + encoded_target[:-1]
            
            # Decoder target: target + END
            decoder_target = encoded_target + [END_IDX]
            
            encoder_inputs.append(encoded_history)
            decoder_inputs.append(decoder_input)
            decoder_targets.append(decoder_target)
            input_lengths.append(total_len)
            target_lengths.append(len(decoder_target))
    
    print(f"Created {len(encoder_inputs):,} training sequences")
    
    return encoder_inputs, decoder_inputs, decoder_targets, input_lengths, target_lengths, app_encoder

# --- TRAINING ---
def train(input_file, output_dir, epochs=EPOCHS, batch_size=BATCH_SIZE):
    print(f"\n=== Training Seq2Seq Pattern Model ===")
    
    # Prepare data
    encoder_inputs, decoder_inputs, decoder_targets, input_lengths, target_lengths, app_encoder = prepare_data(input_file)
    
    # Split
    (enc_train, enc_val, dec_in_train, dec_in_val, dec_tgt_train, dec_tgt_val, 
     in_len_train, in_len_val, tgt_len_train, tgt_len_val) = train_test_split(
        encoder_inputs, decoder_inputs, decoder_targets, input_lengths, target_lengths,
        test_size=0.2, random_state=42
    )
    
    train_dataset = PatternSeq2SeqDataset(enc_train, dec_in_train, dec_tgt_train, in_len_train, tgt_len_train)
    val_dataset = PatternSeq2SeqDataset(enc_val, dec_in_val, dec_tgt_val, in_len_val, tgt_len_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    vocab_size = len(app_encoder.classes_) + 3  # +3 for PAD, START, END
    print(f"Vocab size: {vocab_size}")
    
    model = Seq2SeqPatternModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for enc_in, dec_in, dec_tgt, enc_len, dec_len in train_loader:
            enc_in = enc_in.to(device)
            dec_in = dec_in.to(device)
            dec_tgt = dec_tgt.to(device)
            enc_len = enc_len.to(device)
            
            optimizer.zero_grad()
            
            output = model(enc_in, dec_in, enc_len)
            
            # Reshape for loss
            output = output.reshape(-1, output.size(-1))
            dec_tgt = dec_tgt.reshape(-1)
            
            loss = criterion(output, dec_tgt)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for enc_in, dec_in, dec_tgt, enc_len, dec_len in val_loader:
                enc_in = enc_in.to(device)
                dec_in = dec_in.to(device)
                dec_tgt = dec_tgt.to(device)
                enc_len = enc_len.to(device)
                
                output = model(enc_in, dec_in, enc_len)
                
                output = output.reshape(-1, output.size(-1))
                dec_tgt = dec_tgt.reshape(-1)
                
                loss = criterion(output, dec_tgt)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, 'seq2seq_pattern_model.pth'))
    
    # Save artifacts
    joblib.dump(app_encoder, os.path.join(output_dir, 'app_encoder.pkl'))
    
    config = {
        'vocab_size': vocab_size,
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'history_length': HISTORY_LENGTH
    }
    joblib.dump(config, os.path.join(output_dir, 'model_config.pkl'))
    
    print(f"\n✓ Training complete! Model saved to {output_dir}")

if __name__ == "__main__":
    # Colab Detection
    IN_COLAB = 'google.colab' in sys.modules
    
    if IN_COLAB:
        print("Running in Google Colab environment")
        input_file = "prepared_data/all_labeled_patterns.jsonl"
        output_dir = "model_output"
        train(input_file, output_dir, epochs=20, batch_size=32)
    else:
        import argparse
        parser = argparse.ArgumentParser(description="Step 4: Train Seq2Seq Pattern Prediction Model")
        parser.add_argument("--input", default="prepared_data/all_labeled_patterns.jsonl", help="Input labeled patterns file")
        parser.add_argument("--output_dir", default="model_output", help="Output directory for model artifacts")
        parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
        args = parser.parse_args()
        
        train(args.input, args.output_dir, args.epochs, args.batch_size)
