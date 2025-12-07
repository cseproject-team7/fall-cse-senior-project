"""
Step 5: Test Seq2Seq Pattern Model

Input: model_output/ (seq2seq model, encoder, config)
Output: Interactive predictions

Features:
- Loads trained Seq2Seq model
- Predicts next N complete patterns
- Colab-friendly
"""

import torch
import torch.nn as nn
import joblib
import os
import sys

# Special tokens
PAD_IDX = 0
START_IDX = 1
END_IDX = 2

# --- MODEL DEFINITION (Must match Step 4 exactly) ---
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

# --- INFERENCE LOGIC ---
class ModelPredictor:
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
        """
        Predict the next pattern given a history of patterns.
        pattern_history: list of lists of app names
        Returns: list of app names
        """
        # Encode pattern history
        encoded_history = []
        for pattern in pattern_history:
            encoded_pattern = []
            for app in pattern:
                try:
                    encoded_pattern.append(self.app_encoder.transform([app])[0] + 3)  # +3 for PAD, START, END
                except:
                    encoded_pattern.append(PAD_IDX)  # Unknown app
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
            if app_id >= 3:  # Skip PAD, START, END
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

def interactive_demo():
    try:
        predictor = ModelPredictor()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have run Step 4 and have the 'model_output' directory.")
        return

    print("\n=== Seq2Seq Pattern Prediction Test ===")
    print("Enter 'q' to quit.")
    
    # Sample Data
    samples = [
        [
            ['Microsoft 365 Sign-in', 'Canvas', 'Word Online', 'Microsoft 365 Sign-out'],
            ['Microsoft 365 Sign-in', 'Canvas', 'OneDrive', 'Microsoft 365 Sign-out']
        ],
        [
            ['Microsoft 365 Sign-in', 'VS Code', 'GitHub', 'Microsoft 365 Sign-out'],
            ['Microsoft 365 Sign-in', 'GitHub', 'StackOverflow', 'Microsoft 365 Sign-out']
        ],
        [
            ['Microsoft 365 Sign-in', 'Teams', 'Outlook', 'Microsoft 365 Sign-out'],
            ['Microsoft 365 Sign-in', 'Outlook', 'Teams', 'Microsoft 365 Sign-out']
        ]
    ]
    
    print("\nTry these examples:")
    for i, patterns in enumerate(samples):
        print(f"{i+1}. Pattern History: {len(patterns)} patterns")
        for j, p in enumerate(patterns):
            print(f"   Pattern {j+1}: {p}")
        
    while True:
        print("\n--- New Prediction ---")
        choice = input("Enter example number (1-3) or 'c' for custom input: ").strip().lower()
        
        if choice == 'q':
            break
            
        pattern_history = []
        
        if choice in ['1', '2', '3']:
            idx = int(choice) - 1
            pattern_history = samples[idx]
        elif choice == 'c':
            print("Enter pattern history (one pattern per line, empty line to finish):")
            while True:
                pattern_str = input("Pattern (comma-separated apps): ").strip()
                if not pattern_str:
                    break
                apps = [x.strip() for x in pattern_str.split(',')]
                pattern_history.append(apps)
        else:
            continue
            
        if not pattern_history:
            continue
            
        print(f"\nInput Pattern History: {len(pattern_history)} patterns")
        for i, p in enumerate(pattern_history):
            print(f"  Pattern {i+1}: {p}")
        
        # Predict next N patterns
        num_patterns = int(input("\nHow many patterns to predict? (default: 10): ").strip() or "10")
        
        predicted_patterns = predictor.predict_next_n_patterns(pattern_history, num_patterns=num_patterns)
        
        print(f"\n🔮 Next {num_patterns} Patterns Forecast:")
        print("=" * 80)
        
        for i, pattern in enumerate(predicted_patterns):
            apps_str = ', '.join(pattern) if pattern else '<no apps>'
            print(f"Pattern {i+1}: [{apps_str}]")
        
        print("=" * 80)

if __name__ == "__main__":
    interactive_demo()
