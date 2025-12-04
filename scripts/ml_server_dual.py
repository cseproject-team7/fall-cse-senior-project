"""
ML Server for Dual-Head LSTM Pattern and App Prediction
Uses trained model to predict next pattern + specific apps from user history
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import joblib
import numpy as np
from datetime import datetime, timedelta
import sys
import os

app = Flask(__name__)
CORS(app)

# Model Architecture
class DualHeadLSTM(nn.Module):
    def __init__(self, num_patterns, num_apps, embedding_dim=64, hidden_dim=128, num_layers=2):
        super(DualHeadLSTM, self).__init__()
        self.pattern_embedding = nn.Embedding(num_patterns, embedding_dim)
        self.app_embedding = nn.Embedding(num_apps, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, num_layers, batch_first=True)
        self.pattern_head = nn.Linear(hidden_dim, num_patterns)
        self.app_head = nn.Linear(hidden_dim, num_apps)
    
    def forward(self, pattern_ids, app_ids):
        pattern_emb = self.pattern_embedding(pattern_ids)
        app_emb = self.app_embedding(app_ids)
        combined = torch.cat([pattern_emb, app_emb], dim=-1)
        lstm_out, _ = self.lstm(combined)
        last_hidden = lstm_out[:, -1, :]
        pattern_logits = self.pattern_head(last_hidden)
        app_logits = self.app_head(last_hidden)
        return pattern_logits, app_logits

# Global model variables
model = None
pattern_encoder = None
app_encoder = None
config = None

def load_model():
    """Load the trained dual-head LSTM model and encoders"""
    global model, pattern_encoder, app_encoder, config
    
    try:
        print("Loading model components...")
        
        # Load encoders
        pattern_encoder = joblib.load('models/pattern_encoder.pkl')
        app_encoder = joblib.load('models/app_encoder.pkl')
        config = joblib.load('models/model_config_fixed.pkl')
        
        print(f"✓ Loaded pattern encoder: {len(pattern_encoder.classes_)} patterns")
        print(f"  Patterns: {list(pattern_encoder.classes_)}")
        print(f"✓ Loaded app encoder: {len(app_encoder.classes_)} app sequences")
        
        # Create model
        model = DualHeadLSTM(
            num_patterns=config['num_patterns'],
            num_apps=config['num_apps'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers']
        )
        
        # Load weights
        state_dict = torch.load('models/dual_head_lstm_fixed.pth', map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"✓ Model loaded successfully!")
        print(f"  Config: {config}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def group_apps_into_sessions(logs, gap_minutes=20):
    """
    Group logs into sessions based on time gaps.
    Returns list of sessions with apps and patterns detected.
    """
    if not logs:
        return []
    
    # Sort by timestamp
    sorted_logs = sorted(logs, key=lambda x: datetime.fromisoformat(x['createdDateTime'].replace('Z', '+00:00')))
    
    sessions = []
    current_session_apps = []
    current_session_start = None
    
    for log in sorted_logs:
        app = log['appDisplayName']
        
        # Skip sign-in/sign-out
        if app in ["Microsoft 365 Sign-in", "Microsoft 365 Sign-out"]:
            continue
        
        timestamp = datetime.fromisoformat(log['createdDateTime'].replace('Z', '+00:00'))
        
        if not current_session_apps:
            # Start new session
            current_session_apps = [app]
            current_session_start = timestamp
        else:
            # Check time gap
            gap = (timestamp - current_session_start).total_seconds() / 60.0
            
            if gap > gap_minutes:
                # Save previous session
                if current_session_apps:
                    sessions.append(current_session_apps)
                # Start new session
                current_session_apps = [app]
                current_session_start = timestamp
            else:
                # Continue current session
                current_session_apps.append(app)
    
    # Add final session
    if current_session_apps:
        sessions.append(current_session_apps)
    
    return sessions

def detect_pattern_heuristic(apps):
    """
    Heuristic pattern detection from app list.
    Maps to one of the 8 trained patterns.
    """
    apps_set = set(apps)
    apps_lower = [a.lower() for a in apps]
    
    # Define app categories
    admin_apps = ['oasis', 'degreeworks', 'myusf', 'archivum', 'navigate', 'schedule planner']
    career_apps = ['handshake', 'linkedin', 'indeed', 'glassdoor']
    coding_apps = ['github', 'visual studio', 'jupyter', 'colab', 'matlab', 'replit', 'codepen']
    research_apps = ['jstor', 'ieee', 'google scholar', 'pubmed', 'arxiv', 'library']
    social_apps = ['teams', 'yammer', 'slack', 'discord']
    exam_apps = ['canvas', 'onenote', 'word', 'pdf']
    
    # Check for matches
    has_admin = any(any(admin_app in app.lower() for admin_app in admin_apps) for app in apps)
    has_career = any(any(career_app in app.lower() for career_app in career_apps) for app in apps)
    has_coding = any(any(coding_app in app.lower() for coding_app in coding_apps) for app in apps)
    has_research = any(any(research_app in app.lower() for research_app in research_apps) for app in apps)
    has_social = any(any(social_app in app.lower() for social_app in social_apps) for app in apps)
    has_canvas = any('canvas' in app.lower() for app in apps)
    has_productivity = any(app in ['Word Online', 'Excel Online', 'PowerPoint Online', 'OneNote'] for app in apps)
    
    # Pattern detection logic
    if has_admin:
        return 'ADMIN'
    elif has_career:
        return 'CAREER'
    elif 'club' in ' '.join(apps_lower) or 'organization' in ' '.join(apps_lower):
        return 'CLUBS'
    elif has_coding:
        return 'CODING'
    elif has_research:
        return 'RESEARCH'
    elif (has_canvas or has_productivity) and len(apps) > 3:
        # Longer sessions with academic apps = exam prep
        return 'EXAM'
    elif has_social and (has_canvas or has_productivity):
        return 'COURSEWORK'
    elif has_social:
        return 'SOCIAL'
    elif has_canvas or has_productivity:
        return 'COURSEWORK'
    else:
        return 'COURSEWORK'  # Default

def encode_session_for_model(session_apps):
    """
    Convert a session (list of apps) to pattern and app IDs for the model.
    Returns pattern_id and app_sequence_id.
    """
    # Detect pattern using heuristics
    pattern_name = detect_pattern_heuristic(session_apps)
    
    try:
        pattern_id = pattern_encoder.transform([pattern_name])[0]
    except:
        pattern_id = 4  # Default to COURSEWORK if unknown
    
    # Create app sequence string (this is how the encoder was trained)
    app_sequence = ', '.join(sorted(set(session_apps)))
    
    try:
        app_id = app_encoder.transform([app_sequence])[0]
    except:
        # If this exact sequence wasn't in training, find closest match
        # For now, use the most common app from the sequence
        for app in session_apps:
            try:
                app_id = app_encoder.transform([app])[0]
                break
            except:
                continue
        else:
            app_id = 0  # Fallback
    
    return pattern_id, app_id

def predict_next_pattern_and_apps(history_sessions, num_predictions=5):
    """
    Predict next pattern and top apps based on session history.
    
    Args:
        history_sessions: List of sessions (each session is a list of apps)
        num_predictions: Number of top app predictions to return
    
    Returns:
        {
            'next_pattern': pattern_name,
            'pattern_confidence': float,
            'top_apps': [app_names],
            'app_confidences': [floats]
        }
    """
    if not history_sessions:
        return {
            'next_pattern': 'COURSEWORK',
            'pattern_confidence': 0.5,
            'top_apps': ['Canvas', 'Outlook', 'Teams'],
            'app_confidences': [0.3, 0.25, 0.2]
        }
    
    # Encode recent sessions (use last 10)
    recent_sessions = history_sessions[-10:]
    pattern_ids = []
    app_ids = []
    
    for session in recent_sessions:
        pat_id, app_id = encode_session_for_model(session)
        pattern_ids.append(pat_id)
        app_ids.append(app_id)
    
    # Convert to tensors
    pattern_tensor = torch.tensor([pattern_ids], dtype=torch.long)  # [1, seq_len]
    app_tensor = torch.tensor([app_ids], dtype=torch.long)  # [1, seq_len]
    
    # Predict
    with torch.no_grad():
        pattern_logits, app_logits = model(pattern_tensor, app_tensor)
        
        # Get pattern prediction
        pattern_probs = torch.softmax(pattern_logits, dim=-1)[0]
        pattern_id = torch.argmax(pattern_probs).item()
        pattern_confidence = pattern_probs[pattern_id].item()
        pattern_name = pattern_encoder.inverse_transform([pattern_id])[0]
        
        # Get top app predictions
        app_probs = torch.softmax(app_logits, dim=-1)[0]
        top_app_indices = torch.topk(app_probs, min(num_predictions, len(app_encoder.classes_))).indices.tolist()
        top_app_probs = [app_probs[idx].item() for idx in top_app_indices]
        
        # Decode app sequences
        top_app_sequences = app_encoder.inverse_transform(top_app_indices)
        
        # Parse app sequences into individual apps
        predicted_apps = []
        for app_seq in top_app_sequences:
            # App sequences might be comma-separated
            if ', ' in app_seq:
                apps = app_seq.split(', ')
                predicted_apps.extend(apps[:3])  # Take first 3 from sequence
            else:
                predicted_apps.append(app_seq)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_apps = []
        for app in predicted_apps:
            if app not in seen:
                seen.add(app)
                unique_apps.append(app)
        
        return {
            'next_pattern': pattern_name,
            'pattern_confidence': float(pattern_confidence),
            'top_apps': unique_apps[:num_predictions],
            'app_confidences': top_app_probs
        }

def predict_multi_step(history_sessions, steps=10):
    """
    Predict multiple steps ahead (autoregressive).
    Each prediction becomes input for the next.
    """
    predictions = []
    current_history = list(history_sessions[-10:])  # Start with last 10 sessions
    
    for step in range(steps):
        pred = predict_next_pattern_and_apps(current_history, num_predictions=3)
        predictions.append({
            'step': step + 1,
            'pattern': pred['next_pattern'],
            'confidence': pred['pattern_confidence'],
            'top_apps': pred['top_apps'][:3]
        })
        
        # Add prediction to history for next step
        # Create a synthetic session from predicted apps
        synthetic_session = pred['top_apps'][:2]  # Use top 2 apps
        current_history.append(synthetic_session)
        if len(current_history) > 10:
            current_history.pop(0)
    
    return predictions

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'patterns': list(pattern_encoder.classes_) if pattern_encoder else [],
        'num_apps': len(app_encoder.classes_) if app_encoder else 0
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict next pattern and apps from user log history.
    
    Expected input:
    {
        "logs": [
            {"appDisplayName": "Canvas", "createdDateTime": "2024-12-01T10:00:00Z"},
            ...
        ],
        "num_predictions": 10,  // optional, for multi-step
        "new_app_access": {     // optional, add a new app access to history before predicting
            "appDisplayName": "Teams",
            "createdDateTime": "2024-12-01T15:00:00Z"
        }
    }
    """
    try:
        data = request.json
        logs = data.get('logs', [])
        num_steps = data.get('num_predictions', 1)
        new_app_access = data.get('new_app_access', None)
        
        if not logs:
            return jsonify({'error': 'No logs provided'}), 400
        
        # If new app access is provided, add it to the logs
        if new_app_access and 'appDisplayName' in new_app_access:
            # Add timestamp if not provided
            if 'createdDateTime' not in new_app_access:
                from datetime import datetime
                new_app_access['createdDateTime'] = datetime.utcnow().isoformat() + 'Z'
            logs.append(new_app_access)
            print(f"📝 Added new app access: {new_app_access['appDisplayName']}")
        
        # Group logs into sessions
        sessions = group_apps_into_sessions(logs)
        
        if not sessions:
            return jsonify({'error': 'No valid sessions found'}), 400
        
        # Single-step or multi-step prediction
        if num_steps == 1:
            prediction = predict_next_pattern_and_apps(sessions, num_predictions=5)
            return jsonify({
                'success': True,
                'prediction': prediction,
                'sessions_analyzed': len(sessions),
                'logs_used': len(logs)
            })
        else:
            predictions = predict_multi_step(sessions, steps=num_steps)
            return jsonify({
                'success': True,
                'predictions': predictions,
                'sessions_analyzed': len(sessions),
                'logs_used': len(logs)
            })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/pattern-apps', methods=['POST'])
def get_pattern_specific_apps():
    """
    Get the most likely apps for a specific pattern based on user history.
    
    Expected input:
    {
        "pattern": "EXAM",
        "logs": [...],
        "top_k": 5
    }
    """
    try:
        data = request.json
        target_pattern = data.get('pattern', 'COURSEWORK')
        logs = data.get('logs', [])
        top_k = data.get('top_k', 5)
        
        # Group into sessions
        sessions = group_apps_into_sessions(logs)
        
        if not sessions:
            # Return default apps for pattern
            pattern_default_apps = {
                'ADMIN': ['OASIS', 'DegreeWorks', 'MyUSF'],
                'CAREER': ['Handshake', 'LinkedIn Learning'],
                'CLUBS': ['Teams', 'Outlook', 'OneNote'],
                'CODING': ['GitHub', 'Visual Studio Code', 'Jupyter'],
                'COURSEWORK': ['Canvas', 'Word Online', 'Outlook'],
                'EXAM': ['Canvas', 'OneNote', 'Word Online', 'PowerPoint Online'],
                'RESEARCH': ['Google Scholar', 'JSTOR', 'Library'],
                'SOCIAL': ['Teams', 'Yammer', 'Outlook']
            }
            return jsonify({
                'success': True,
                'pattern': target_pattern,
                'apps': pattern_default_apps.get(target_pattern, ['Canvas', 'Outlook']),
                'note': 'Using default apps (no history)'
            })
        
        # Encode sessions and predict with pattern constraint
        recent_sessions = sessions[-10:]
        pattern_ids = []
        app_ids = []
        
        for session in recent_sessions:
            pat_id, app_id = encode_session_for_model(session)
            pattern_ids.append(pat_id)
            app_ids.append(app_id)
        
        # Convert to tensors
        pattern_tensor = torch.tensor([pattern_ids], dtype=torch.long)
        app_tensor = torch.tensor([app_ids], dtype=torch.long)
        
        # Get app predictions
        with torch.no_grad():
            _, app_logits = model(pattern_tensor, app_tensor)
            app_probs = torch.softmax(app_logits, dim=-1)[0]
            top_indices = torch.topk(app_probs, min(top_k * 2, len(app_encoder.classes_))).indices.tolist()
            
            # Decode and flatten apps
            predicted_apps = []
            for idx in top_indices:
                app_seq = app_encoder.inverse_transform([idx])[0]
                if ', ' in app_seq:
                    predicted_apps.extend(app_seq.split(', '))
                else:
                    predicted_apps.append(app_seq)
                
                if len(predicted_apps) >= top_k:
                    break
            
            # Remove duplicates
            seen = set()
            unique_apps = []
            for app in predicted_apps:
                if app not in seen:
                    seen.add(app)
                    unique_apps.append(app)
        
        return jsonify({
            'success': True,
            'pattern': target_pattern,
            'apps': unique_apps[:top_k]
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Dual-Head LSTM Pattern & App Prediction Server")
    print("=" * 60)
    
    if not load_model():
        print("\n✗ Failed to load model. Exiting.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Server starting on http://localhost:5001")
    print("=" * 60)
    print("\nEndpoints:")
    print("  GET  /health              - Health check")
    print("  POST /predict             - Predict next pattern + apps")
    print("  POST /pattern-apps        - Get apps for specific pattern")
    print("\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
