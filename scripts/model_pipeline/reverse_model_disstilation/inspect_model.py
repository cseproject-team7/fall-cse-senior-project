"""
MODEL INSPECTION UTILITY

Loads and displays the architecture and configuration of the trained Dual-Head LSTM model.
Useful for debugging, verification, and understanding model capacity.

Displays:
- Model configuration (vocabularies, dimensions, layers)
- Total parameter count
- Architecture summary
- Input/output specifications

Input:  models/model_config.pkl, models/dual_head_lstm.pth
Output: Console output with model details
"""

import torch
import torch.nn as nn
import joblib

# Based on the config you have
config = joblib.load('models/model_config.pkl')
print("Model Config:")
print(config)
print()

# Define the Dual-Head LSTM architecture (from step5_test_model.py)
class DualHeadLSTM(nn.Module):
    def __init__(self, num_patterns, num_apps, embedding_dim, hidden_dim, num_layers):
        super(DualHeadLSTM, self).__init__()
        
        # Embeddings (padding_idx=0)
        self.pattern_embedding = nn.Embedding(num_patterns, embedding_dim, padding_idx=0)
        self.app_embedding = nn.Embedding(num_apps, embedding_dim, padding_idx=0)
        
        # LSTM
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # Heads
        self.pattern_head = nn.Linear(hidden_dim, num_patterns)
        self.app_head = nn.Linear(hidden_dim, num_apps)
        
    def forward(self, pattern_seq, app_seq):
        # Embed and Concat
        pat_emb = self.pattern_embedding(pattern_seq) 
        app_emb = self.app_embedding(app_seq)         
        
        x = torch.cat((pat_emb, app_emb), dim=2)      
        
        # LSTM
        lstm_out, _ = self.lstm(x)                    
        
        # Take last time step
        last_out = lstm_out[:, -1, :]                 
        
        # Heads
        pattern_logits = self.pattern_head(last_out)
        app_logits = self.app_head(last_out)
        
        return pattern_logits, app_logits

# Load the model
model = DualHeadLSTM(
    config['num_patterns'],
    config['num_apps'],
    config['embedding_dim'],
    config['hidden_dim'],
    config['num_layers']
)

state_dict = torch.load('models/dual_head_lstm.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

print("✅ Model loaded successfully!")
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
print()
print("Architecture:")
print(f"  - Pattern vocabulary: {config['num_patterns']}")
print(f"  - App vocabulary: {config['num_apps']}")
print(f"  - Embedding dim: {config['embedding_dim']}")
print(f"  - Hidden dim: {config['hidden_dim']}")
print(f"  - LSTM layers: {config['num_layers']}")
print()
print("This model predicts:")
print("  Input: Sequence of (pattern_id, app_id) pairs")
print("  Output: Next pattern + Next key app")
