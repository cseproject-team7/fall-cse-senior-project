"""
Step 6: Rolling Pattern Prediction with Seq2Seq

Input: 
- model_output/ (seq2seq model, encoder, config)
- prepared_data/all_labeled_patterns.jsonl (user history)

Features:
- Selects a random user (or specific user ID)
- Replays their pattern history
- At each step, performs prediction for the next 10 patterns
- Shows complete app sequences
"""

import torch
import torch.nn as nn
import joblib
import json
import os
import sys
import random
from collections import defaultdict

# Special tokens
PAD_IDX = 0
START_IDX = 1
END_IDX = 2

class Seq2SeqPatternModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(Seq2SeqPatternModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                               batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                               batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def encode(self, encoder_input):
        encoder_emb = self.embedding(encoder_input)
        _, (hidden, cell) = self.encoder(encoder_emb)
        return hidden, cell
    
    def decode_step(self, decoder_input, hidden, cell):
        decoder_emb = self.embedding(decoder_input)
        decoder_output, (hidden, cell) = self.decoder(decoder_emb, (hidden, cell))
        output = self.fc(decoder_output)
        return output, hidden, cell

# --- FORECASTER ---
class RollingForecaster:
    def __init__(self, model_dir='model_output'):
        print(f"Loading model from {model_dir}...")
        
        # Load artifacts
        self.config = joblib.load(os.path.join(model_dir, 'model_config.pkl'))
        self.app_encoder = joblib.load(os.path.join(model_dir, 'app_encoder.pkl'))
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = Seq2SeqPatternModel(
            self.config['vocab_size'],
            self.config['embedding_dim'],
            self.config['hidden_dim'],
            self.config['num_layers']
        ).to(self.device)
        
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, 'seq2seq_pattern_model.pth'), map_location=self.device)
        )
        self.model.eval()
        
        print("✓ Model loaded successfully")
        
    def predict_next_pattern(self, pattern_history, max_len=20):
        """Predict the next pattern given a history of patterns"""
        # Encode pattern history
        encoded_history = []
        for pattern in pattern_history:
            encoded_pattern = []
            for app in pattern:
                try:
                    encoded_pattern.append(self.app_encoder.transform([app])[0] + 3)
                except:
                    encoded_pattern.append(PAD_IDX)
            encoded_history.extend(encoded_pattern)
        
        # Convert to tensor
        encoder_input = torch.tensor([encoded_history], dtype=torch.long).to(self.device)
        
        # Encode
        with torch.no_grad():
            hidden, cell = self.model.encode(encoder_input)
            
            # Decode autoregressively
            generated_apps = []
            decoder_input = torch.tensor([[START_IDX]], dtype=torch.long).to(self.device)
            
            for _ in range(max_len):
                output, hidden, cell = self.model.decode_step(decoder_input, hidden, cell)
                
                # Get next token
                next_token = torch.argmax(output[0, -1, :]).item()
                
                # Stop if END token or PAD
                if next_token == END_IDX or next_token == PAD_IDX:
                    break
                
                generated_apps.append(next_token)
                decoder_input = torch.tensor([[next_token]], dtype=torch.long).to(self.device)
        
        # Decode app IDs to names
        app_names = []
        for app_id in generated_apps:
            if app_id >= 3:
                try:
                    app_names.append(self.app_encoder.inverse_transform([app_id - 3])[0])
                except:
                    pass
        
        return app_names
    
    def predict_next_n_patterns(self, pattern_history, num_patterns=10):
        """Predict the next N patterns autoregressively"""
        predicted_patterns = []
        current_history = pattern_history.copy()
        
        for _ in range(num_patterns):
            # Predict next pattern
            next_pattern = self.predict_next_pattern(current_history[-self.config['history_length']:])
            
            if not next_pattern:
                break
            
            predicted_patterns.append(next_pattern)
            
            # Update history
            current_history.append(next_pattern)
        
        return predicted_patterns

    def interactive_mode(self):
        """Real-time interactive prediction"""
        print("\n--- Interactive Mode (10-Pattern Forecast) ---")
        print("Enter patterns as you go. Type 'q' to quit.")
        
        # Start with empty history
        pattern_history = []
        
        while True:
            if pattern_history:
                print(f"\nCurrent Pattern History (Last 3):")
                for i, p in enumerate(pattern_history[-3:]):
                    print(f"  Pattern {i+1}: {p}")
            
            pattern_str = input("\nEnter Current Pattern (comma-separated apps): ").strip()
            if pattern_str.lower() == 'q':
                break
            
            apps = [x.strip() for x in pattern_str.split(',')]
            pattern_history.append(apps)
            
            # Forecast
            predicted_patterns = self.predict_next_n_patterns(pattern_history, num_patterns=10)
            
            print(f"\n🔮 Next 10 Patterns Forecast:")
            print("=" * 80)
            
            for i, pattern in enumerate(predicted_patterns):
                apps_str = ', '.join(pattern) if pattern else '<no apps>'
                print(f"Pattern {i+1}: [{apps_str}]")
            
            print("=" * 80)

def run_simulation(data_file='prepared_data/all_labeled_patterns.jsonl'):
    print(f"Loading data from {data_file}...")
    try:
        # Load patterns by user
        patterns_by_user = defaultdict(list)
        
        with open(data_file, 'r') as f:
            for line in f:
                pattern = json.loads(line)
                patterns_by_user[pattern['user']].append(pattern)
        
        # Sort by timestamp
        for user in patterns_by_user:
            patterns_by_user[user].sort(key=lambda x: x['timestamps'][0])
        
    except FileNotFoundError:
        print("Error: all_labeled_patterns.jsonl not found. Run Step 3 first.")
        return

    forecaster = RollingForecaster()
    
    mode = input("Select Mode: (1) User Simulation, (2) Interactive: ").strip()
    
    if mode == '2':
        forecaster.interactive_mode()
        return
    
    # Get unique users
    users = list(patterns_by_user.keys())
    print(f"Found {len(users):,} users.")
    
    while True:
        print("\n--- Select User ---")
        user_input = input("Enter User ID (or press Enter for random, 'q' to quit): ").strip()
        
        if user_input == 'q':
            break
            
        if not user_input:
            user = random.choice(users)
            print(f"Selected Random User: {user}")
        else:
            user = user_input
            if user not in users:
                print("User not found!")
                continue
                
        # Get user's full timeline
        user_patterns = patterns_by_user[user]
            
        print(f"\nUser Timeline Length: {len(user_patterns)} patterns")
        
        # Rolling Simulation
        print("\n--- Rolling 10-Pattern Forecast ---")
        print("Press Enter to advance, 'q' to stop user simulation.")
        
        # Start from index 5 (after initial history)
        history_len = forecaster.config['history_length']
        
        for t in range(history_len, len(user_patterns)):
            current_history = [p['apps'] for p in user_patterns[max(0, t-history_len):t]]
            
            # Ground Truth for next 10 patterns (if available)
            ground_truth = []
            for k in range(10):
                if t + k < len(user_patterns):
                    gt_apps = ', '.join(user_patterns[t+k]['apps'])
                    ground_truth.append(f"[{gt_apps}]")
                else:
                    ground_truth.append("End of Data")
            
            # Forecast
            predicted_patterns = forecaster.predict_next_n_patterns(current_history, num_patterns=10)
            
            last_pattern_str = ', '.join(current_history[-1]) if current_history else 'None'
            print(f"\nTime Step {t} (Last Pattern: [{last_pattern_str}])")
            print("-" * 100)
            print(f"{'Step':<5} | {'Forecast':<60} | {'Ground Truth'}")
            print("-" * 100)
            
            for k, (pred_pattern, truth) in enumerate(zip(predicted_patterns, ground_truth)):
                pred_apps_str = ', '.join(pred_pattern) if pred_pattern else '<no apps>'
                pred_str = f"[{pred_apps_str}]"
                
                print(f"+{k+1:<4} | {pred_str:<60} | {truth}")
                
            cmd = input()
            if cmd.lower() == 'q':
                break

if __name__ == "__main__":
    # Colab Check
    if 'google.colab' in sys.modules:
        run_simulation("prepared_data/all_labeled_patterns.jsonl")
    else:
        run_simulation()
