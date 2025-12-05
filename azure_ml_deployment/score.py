"""
Azure ML Scoring Script for Pattern & App Prediction
This script provides a real-time endpoint for the Dual-Head LSTM model
"""

import os
import json
import torch
import torch.nn as nn
import joblib
from datetime import datetime
from typing import List, Dict, Any

# ============================================================================
# Model Architecture (Must match training)
# ============================================================================

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

# ============================================================================
# Global Variables
# ============================================================================

model = None
pattern_encoder = None
app_encoder = None

# ============================================================================
# Initialization
# ============================================================================

def init():
    """
    This function is called when the container is initialized/started.
    Load the model and encoders.
    """
    global model, pattern_encoder, app_encoder
    
    # Get the path to the model directory (AZUREML_MODEL_DIR)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR', './'), 'model')
    
    print(f"Loading model from: {model_path}")
    
    try:
        # Load encoders
        pattern_encoder = joblib.load(os.path.join(model_path, 'pattern_encoder.pkl'))
        app_encoder = joblib.load(os.path.join(model_path, 'app_encoder.pkl'))
        
        print(f"Loaded pattern encoder with {len(pattern_encoder.classes_)} patterns")
        print(f"Loaded app encoder with {len(app_encoder.classes_)} app sequences")
        
        # Load model
        num_patterns = len(pattern_encoder.classes_)
        num_apps = len(app_encoder.classes_)
        
        model = DualHeadLSTM(
            num_patterns=num_patterns,
            num_apps=num_apps,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2
        )
        
        model.load_state_dict(torch.load(
            os.path.join(model_path, 'dual_head_lstm.pth'),
            map_location=torch.device('cpu')
        ))
        model.eval()
        
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise

# ============================================================================
# Helper Functions
# ============================================================================

def detect_pattern_heuristic(apps: List[str]) -> str:
    """Detect pattern from app list using heuristics"""
    apps_set = set([app.lower() for app in apps])
    apps_str = ' '.join([app.lower() for app in apps])
    
    # Pattern detection rules
    admin_apps = ['oasis', 'degreeworks', 'myusf', 'archivum', 'navigate', 'schedule planner']
    career_apps = ['handshake', 'linkedin', 'indeed', 'glassdoor']
    coding_apps = ['github', 'visual studio', 'jupyter', 'colab', 'matlab', 'replit', 'codepen']
    research_apps = ['jstor', 'ieee', 'google scholar', 'pubmed', 'arxiv', 'library']
    social_apps = ['teams', 'yammer', 'slack', 'discord']
    
    has_admin = any(any(admin_app in app.lower() for admin_app in admin_apps) for app in apps)
    has_career = any(any(career_app in app.lower() for career_app in career_apps) for app in apps)
    has_coding = any(any(coding_app in app.lower() for coding_app in coding_apps) for app in apps)
    has_research = any(any(research_app in app.lower() for research_app in research_apps) for app in apps)
    has_social = any(any(social_app in app.lower() for social_app in social_apps) for app in apps)
    has_canvas = any('canvas' in app.lower() for app in apps)
    
    if has_admin:
        return 'ADMIN'
    elif has_career:
        return 'CAREER'
    elif 'club' in apps_str or 'organization' in apps_str:
        return 'CLUBS'
    elif has_coding:
        return 'CODING'
    elif has_research:
        return 'RESEARCH'
    elif has_canvas and any(exam_app in apps_str for exam_app in ['onenote', 'pdf', 'word']):
        return 'EXAM'
    elif has_social:
        return 'SOCIAL'
    else:
        return 'COURSEWORK'

def group_apps_into_sessions(logs: List[Dict], gap_minutes: int = 20) -> List[List[str]]:
    """Group logs into sessions based on time gaps"""
    if not logs:
        return []
    
    sorted_logs = sorted(logs, key=lambda x: datetime.fromisoformat(x['createdDateTime'].replace('Z', '+00:00')))
    
    sessions = []
    current_session = []
    current_time = None
    
    for log in sorted_logs:
        app = log['appDisplayName']
        
        if app in ["Microsoft 365 Sign-in", "Microsoft 365 Sign-out"]:
            continue
        
        timestamp = datetime.fromisoformat(log['createdDateTime'].replace('Z', '+00:00'))
        
        if not current_session:
            current_session = [app]
            current_time = timestamp
        else:
            gap = (timestamp - current_time).total_seconds() / 60.0
            
            if gap > gap_minutes:
                if current_session:
                    sessions.append(current_session)
                current_session = [app]
                current_time = timestamp
            else:
                current_session.append(app)
    
    if current_session:
        sessions.append(current_session)
    
    return sessions

def encode_session_for_model(session: List[str]) -> tuple:
    """Encode a session into pattern_id and app_id"""
    pattern = detect_pattern_heuristic(session)
    
    try:
        pattern_id = pattern_encoder.transform([pattern])[0]
    except:
        pattern_id = 0
    
    # Join apps as comma-separated sequence
    app_sequence = ', '.join(session[:3])  # Use first 3 apps
    
    try:
        app_id = app_encoder.transform([app_sequence])[0]
    except:
        # Try individual apps if sequence not found
        for app in session:
            try:
                app_id = app_encoder.transform([app])[0]
                break
            except:
                continue
        else:
            app_id = 0
    
    return pattern_id, app_id

def predict_next_pattern_and_apps(history_sessions: List[List[str]], num_predictions: int = 5) -> Dict[str, Any]:
    """Predict next pattern and apps from session history"""
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
    pattern_tensor = torch.tensor([pattern_ids], dtype=torch.long)
    app_tensor = torch.tensor([app_ids], dtype=torch.long)
    
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
            if ', ' in app_seq:
                apps = app_seq.split(', ')
                predicted_apps.extend(apps[:3])
            else:
                predicted_apps.append(app_seq)
        
        # Remove duplicates
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

def predict_multi_step(history_sessions: List[List[str]], steps: int = 10) -> List[Dict[str, Any]]:
    """Predict multiple steps ahead"""
    predictions = []
    current_history = list(history_sessions[-10:])
    
    for step in range(steps):
        pred = predict_next_pattern_and_apps(current_history, num_predictions=3)
        predictions.append({
            'step': step + 1,
            'pattern': pred['next_pattern'],
            'confidence': pred['pattern_confidence'],
            'top_apps': pred['top_apps'][:3]
        })
        
        # Add prediction to history
        synthetic_session = pred['top_apps'][:2]
        current_history.append(synthetic_session)
        if len(current_history) > 10:
            current_history.pop(0)
    
    return predictions

# ============================================================================
# Scoring Function
# ============================================================================

def run(raw_data: str) -> str:
    """
    This function is called for every invocation of the endpoint.
    
    Expected input format:
    {
        "logs": [
            {"appDisplayName": "Canvas", "createdDateTime": "2024-12-01T10:00:00Z"},
            ...
        ],
        "num_predictions": 10  // optional, defaults to 1
    }
    """
    try:
        # Parse input
        data = json.loads(raw_data)
        logs = data.get('logs', [])
        num_steps = data.get('num_predictions', 1)
        
        if not logs:
            return json.dumps({
                'error': 'No logs provided',
                'success': False
            })
        
        # Group logs into sessions
        sessions = group_apps_into_sessions(logs)
        
        if not sessions:
            return json.dumps({
                'error': 'No valid sessions found',
                'success': False
            })
        
        # Make predictions
        if num_steps == 1:
            prediction = predict_next_pattern_and_apps(sessions, num_predictions=5)
            return json.dumps({
                'success': True,
                'prediction': prediction,
                'sessions_analyzed': len(sessions),
                'logs_used': len(logs)
            })
        else:
            predictions = predict_multi_step(sessions, steps=num_steps)
            return json.dumps({
                'success': True,
                'predictions': predictions,
                'sessions_analyzed': len(sessions),
                'logs_used': len(logs)
            })
    
    except Exception as e:
        return json.dumps({
            'error': str(e),
            'success': False
        })
