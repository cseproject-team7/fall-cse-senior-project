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
        
        # Get app sequence predictions with boosting for multi-app sequences
        app_probs = torch.softmax(app_logits, dim=-1)[0]
        
        # BOOST LOGIC: Extreme graduated boosting to overcome model bias
        # The model heavily favors 2-app sequences, so we need very aggressive boosts
        # Target: ~20% single, ~35% 2-app, ~25% 3-app, ~18% 4-app, ~2% 5-app
        boosted_probs = app_probs.clone()
        
        for idx in range(len(app_encoder.classes_)):
            app_seq = app_encoder.classes_[idx]
            
            # Filter out ONLY sequences that are just noise apps by themselves
            if app_seq in ['Microsoft 365', 'Copilot']:
                boosted_probs[idx] = boosted_probs[idx] * 0.01
                continue
            
            # Apply extreme graduated boosting based on sequence length
            if ', ' in app_seq:  # Multi-app sequence
                apps_list = app_seq.split(', ')
                num_apps = len(apps_list)
                
                # Check if sequence has meaningful apps
                meaningful_apps = [app for app in apps_list if app not in ['Microsoft 365', 'Copilot']]
                
                # Fine-tuned boosting: Target ~20% 1-app, ~35% 2-app, ~25% 3-app, ~18% 4-app
                if num_apps == 2:
                    if len(meaningful_apps) >= 1:
                        boosted_probs[idx] = boosted_probs[idx] * 1.5  # Reduced to lower 2-app %
                elif num_apps == 3:
                    if len(meaningful_apps) >= 2:
                        boosted_probs[idx] = boosted_probs[idx] * 30.0  # Increased from 18 to boost 3-app
                    else:
                        boosted_probs[idx] = boosted_probs[idx] * 15.0
                elif num_apps == 4:
                    if len(meaningful_apps) >= 3:
                        boosted_probs[idx] = boosted_probs[idx] * 80.0
                    else:
                        boosted_probs[idx] = boosted_probs[idx] * 45.0
                elif num_apps >= 5:
                    boosted_probs[idx] = boosted_probs[idx] * 120.0  # Maximum for 5+ app
            else:
                # Single apps get slight boost to maintain ~20% presence
                boosted_probs[idx] = boosted_probs[idx] * 1.2
        
        # Re-normalize after boosting and filtering
        boosted_probs = boosted_probs / boosted_probs.sum()
        
        # Get top predictions from boosted probabilities
        top_app_indices = torch.topk(boosted_probs, min(num_predictions * 2, len(app_encoder.classes_))).indices.tolist()
        top_app_probs = [boosted_probs[idx].item() for idx in top_app_indices]
        
        # Decode app sequences - these are the actual predicted sessions
        top_app_sequences = app_encoder.inverse_transform(top_app_indices)
        
        # Parse sequences into structured format
        predicted_app_sequences = []
        
        for i, app_seq in enumerate(top_app_sequences):
            # Skip any that slipped through with zero probability
            if top_app_probs[i] == 0.0:
                continue
                
            if ', ' in app_seq:
                # This is a multi-app sequence (the actual pattern session)
                apps = app_seq.split(', ')
                predicted_app_sequences.append({
                    'apps': apps,
                    'confidence': top_app_probs[i],
                    'session_type': 'multi_app'
                })
            else:
                # Single app
                predicted_app_sequences.append({
                    'apps': [app_seq],
                    'confidence': top_app_probs[i],
                    'session_type': 'single_app'
                })
        
        # Take top num_predictions
        predicted_app_sequences = predicted_app_sequences[:num_predictions]
        
        # Return the most likely session (first prediction after reranking)
        most_likely_session = predicted_app_sequences[0]['apps'] if predicted_app_sequences else ['Canvas']
        
        return {
            'next_pattern': pattern_name,
            'pattern_confidence': float(pattern_confidence),
            'top_apps': most_likely_session,  # This is the actual sequence of apps
            'app_confidences': [seq['confidence'] for seq in predicted_app_sequences],
            'all_predicted_sequences': predicted_app_sequences  # All top sequences (boosted & reranked)
        }

def predict_multi_step(history_sessions: List[List[str]], steps: int = 10) -> List[Dict[str, Any]]:
    """Predict multiple steps ahead"""
    predictions = []
    current_history = list(history_sessions[-10:])
    
    for step in range(steps):
        pred = predict_next_pattern_and_apps(current_history, num_predictions=5)
        
        # Get the predicted app sequence (could be 2-5 apps)
        predicted_session = pred['top_apps']  # This is already a list of apps in the sequence
        
        predictions.append({
            'step': step + 1,
            'pattern': pred['next_pattern'],
            'confidence': pred['pattern_confidence'],
            'top_apps': predicted_session  # Full app sequence for this pattern
        })
        
        # Add the predicted session to history for next prediction
        current_history.append(predicted_session)
        if len(current_history) > 10:
            current_history.pop(0)
    
    return predictions

def compute_pattern_transitions() -> Dict[str, Any]:
    """
    Use the trained LSTM model to predict pattern transitions.
    For each pattern, we feed it as context and see what the model predicts next.
    """
    pattern_names = list(pattern_encoder.classes_)
    
    # We'll build a context sequence and let the model predict next patterns
    sequence_length = 5
    
    transition_probs = {}
    
    for from_pattern in pattern_names:
        from_idx = pattern_encoder.transform([from_pattern])[0]
        
        # Create a sequence where the last element is from_pattern
        pattern_sequence = [from_idx] * sequence_length
        
        # Use a common app sequence (Canvas is most common)
        try:
            canvas_idx = app_encoder.transform(['Canvas'])[0]
        except:
            canvas_idx = 0
        app_sequence = [canvas_idx] * sequence_length
        
        # Convert to tensors
        pattern_tensor = torch.tensor([pattern_sequence], dtype=torch.long)
        app_tensor = torch.tensor([app_sequence], dtype=torch.long)
        
        # Get model prediction
        with torch.no_grad():
            pattern_logits, _ = model(pattern_tensor, app_tensor)
            probabilities = torch.softmax(pattern_logits[0], dim=-1)
        
        # Extract top predictions as successors
        successors = []
        for to_idx, prob in enumerate(probabilities):
            prob_val = float(prob)
            if prob_val > 0.01:  # Only include if > 1% probability
                successors.append({
                    'pattern': pattern_names[to_idx],
                    'probability': prob_val,
                    'confidence': prob_val * 100,
                    'count': int(prob_val * 100)
                })
        
        successors.sort(key=lambda x: x['probability'], reverse=True)
        transition_probs[from_pattern] = successors[:5]
    
    # Compute predecessors (reverse transitions)
    predecessor_probs = {pattern: [] for pattern in pattern_names}
    
    for from_pattern, successors in transition_probs.items():
        for successor_info in successors:
            to_pattern = successor_info['pattern']
            predecessor_probs[to_pattern].append({
                'pattern': from_pattern,
                'probability': successor_info['probability'],
                'confidence': successor_info['confidence'],
                'count': successor_info['count']
            })
    
    # Sort predecessors by probability
    for pattern in predecessor_probs:
        predecessor_probs[pattern].sort(key=lambda x: x['probability'], reverse=True)
        predecessor_probs[pattern] = predecessor_probs[pattern][:5]
    
    return {
        'predecessors': predecessor_probs,
        'successors': transition_probs
    }

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
        "num_predictions": 10,  // optional, defaults to 1
        "request_type": "predict" // optional, can be "predict" or "pattern_transitions"
    }
    """
    try:
        # Parse input
        data = json.loads(raw_data)
        request_type = data.get('request_type', 'predict')
        
        # Handle pattern transitions request
        if request_type == 'pattern_transitions':
            transitions = compute_pattern_transitions()
            pattern_names = list(pattern_encoder.classes_)
            
            # Format for frontend: array of pattern chains
            result = []
            for pattern in pattern_names:
                result.append({
                    'pattern': pattern,
                    'predecessors': transitions['predecessors'].get(pattern, []),
                    'successors': transitions['successors'].get(pattern, [])
                })
            
            return json.dumps({
                'success': True,
                'transitions': result,
                'total_patterns': len(pattern_names)
            })
        
        # Handle prediction request
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
