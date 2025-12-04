from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import joblib
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add model_training_pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model_training_pipeline'))
from simple_gru_model import LSTMPatternPredictor
from pattern_definitions import detect_pattern_from_session, get_pattern_id_by_name, PATTERNS

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model_training_pipeline', 'model_output')
DEVICE = torch.device('cpu')

# Global variables
model = None
config = None

def load_models():
    global model, config
    try:
        print("Loading model artifacts...")
        config = joblib.load(os.path.join(MODEL_DIR, 'model_config.pkl'))
        print(f"Config loaded: {config}")
        
        # Initialize model structure (LSTMPatternPredictor)
        # Note: num_patterns in config is 8. The model class adds +1 internally.
        # Checkpoint has num_departments=2, so we default to that if not in config.
        model = LSTMPatternPredictor(
            num_patterns=config['num_patterns'],
            embedding_dim=config.get('embedding_dim', 64),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 3),
            metadata_dim=7, # Default
            num_departments=config.get('num_departments', 2)
        )
        
        model_path = os.path.join(MODEL_DIR, 'gru_pattern_model.pth')
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("✅ LSTM Pattern Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load models: {str(e)}")

load_models()

def process_logs_to_patterns(logs):
    """
    Convert a list of raw app logs into a sequence of pattern IDs.
    """
    if not logs:
        return []
        
    # Sort by time
    # Assuming logs have 'createdDateTime' or similar, or just rely on order
    # The client sends 'data' which is a list of objects.
    # Let's assume they are in order or have a timestamp.
    
    # Simple sessionization: Group by 20 min gaps
    sessions = []
    current_session_apps = []
    current_session_durations = []
    last_time = None
    
    for log in logs:
        app_name = log.get('appDisplayName')
        # Parse time if available, else assume sequential with default gap
        # The client payload in localMLService.js sends the raw log object
        # which usually has 'createdDateTime'
        timestamp_str = log.get('createdDateTime')
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                timestamp = datetime.now() # Fallback
        else:
            timestamp = datetime.now()

        if last_time and (timestamp - last_time).total_seconds() > 1200: # 20 mins
            # End current session
            if current_session_apps:
                p_name = detect_pattern_from_session(current_session_apps, current_session_durations)
                p_id = get_pattern_id_by_name(p_name)
                sessions.append(p_id)
                current_session_apps = []
                current_session_durations = []
        
        current_session_apps.append(app_name)
        current_session_durations.append(5.0) # Default 5 mins
        last_time = timestamp
        
    # Add last session
    if current_session_apps:
        p_name = detect_pattern_from_session(current_session_apps, current_session_durations)
        p_id = get_pattern_id_by_name(p_name)
        sessions.append(p_id)
        
    return sessions

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/api/predict-next-pattern', methods=['POST'])
def predict_next_pattern():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 503
        
    try:
        data = request.json.get('data', [])
        
        # Convert logs to pattern sequence
        pattern_seq = process_logs_to_patterns(data)
        
        # If no history, default to something or fail
        if not pattern_seq:
            # If we have just one session (the current one), we can't predict NEXT based on history?
            # Or we use the current partial session to predict the REST of it?
            # The model predicts NEXT session pattern given PAST session patterns.
            # If we only have current session, we might need to pad.
            pattern_seq = [7] # OTHER as padding
            
        # Prepare tensor
        pattern_tensor = torch.tensor([pattern_seq], dtype=torch.long).to(DEVICE)
        
        # Prepare metadata tensor (Dummy for now)
        metadata_tensor = torch.zeros((1, 7), dtype=torch.float32).to(DEVICE)
        
        # Multi-step forecasting (Predict next 10 steps)
        steps = 10
        predictions = []
        
        # Clone sequence for autoregressive loop
        current_seq = pattern_tensor.clone()
        
        print(f"Input pattern sequence: {pattern_seq}")

        with torch.no_grad():
            for i in range(steps):
                # Model returns 4 outputs: pred_1, pred_2, pred_3, dept_pred
                pred_1, _, _, _ = model(current_seq, metadata_tensor)
                
                # Get prediction for this step
                # Use temperature sampling to add variety to the sequence forecast
                temperature = 2.5 # Increased temperature for more variety
                
                # Apply temperature
                scaled_logits = pred_1 / temperature
                
                # Optional: Apply repetition penalty
                # If the last predicted item was the same as the one before, penalize it
                if len(predictions) > 0:
                    last_pred = predictions[-1]['pattern_id']
                    scaled_logits[0, last_pred] -= 1.0 # Penalize repetition
                
                probs = torch.softmax(scaled_logits, dim=1)
                
                # Debug: Print top probabilities
                top_probs, top_indices = torch.topk(probs, 3)
                print(f"Step {i+1} Top 3: {top_indices.tolist()} {top_probs.tolist()}")
                
                # Sample from the distribution instead of always taking top-1
                # This creates a more realistic "possible future" path
                next_pattern_id = torch.multinomial(probs, 1).item()
                confidence = probs[0][next_pattern_id].item()
                
                predictions.append({
                    "step": i + 1,
                    "pattern_id": next_pattern_id,
                    "pattern_name": PATTERNS.get(next_pattern_id, "UNKNOWN"),
                    "confidence": float(confidence)
                })
                
                # Update sequence for next step: shift left and append new prediction
                # current_seq shape is [1, seq_len]
                next_input = torch.tensor([[next_pattern_id]], dtype=torch.long).to(DEVICE)
                current_seq = torch.cat([current_seq[:, 1:], next_input], dim=1)

        return jsonify({
            "predictions": predictions,
            "top_prediction": predictions[0]['pattern_name']
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
