from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import joblib
import numpy as np
import os
from collections import Counter

app = Flask(__name__)
CORS(app)

# Global model variables
model = None
pattern_encoder = None
config = None
device = None

# --- MODEL DEFINITION (Must match training script) ---
class LSTMPredictor(nn.Module):
    def __init__(self, num_patterns, embedding_dim, hidden_dim, num_layers):
        super(LSTMPredictor, self).__init__()
        self.embedding = nn.Embedding(num_patterns, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_patterns)
        
    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def detect_pattern_from_sequence(app_sequence):
    """
    Analyzes a sequence of apps and classifies it into a behavioral pattern.
    This uses heuristic rules to identify common student workflows.
    Returns both the pattern name and the apps used in that pattern.
    """
    apps_str = " ".join([app.lower() for app in app_sequence])
    apps_set = set([app.lower() for app in app_sequence])
    
    # Deep Research Pattern
    if any(keyword in apps_str for keyword in ["library", "jstor", "scholar", "ieee", "pubmed"]):
        if len(app_sequence) >= 3:
            return "DEEP_RESEARCH", app_sequence
    
    # Group Project Pattern
    if "teams" in apps_str and any(tool in apps_str for tool in ["word", "powerpoint", "sharepoint", "excel"]):
        return "GROUP_PROJECT", app_sequence
    
    # Exam Cramming Pattern
    if "canvas" in apps_str and any(tool in apps_str for tool in ["onenote", "word", "pdf"]):
        if len(app_sequence) >= 4:
            return "EXAM_CRAMMING", app_sequence
    
    # Coding/Development Pattern
    if any(keyword in apps_str for keyword in ["github", "vscode", "matlab", "jupyter", "colab", "stackoverflow"]):
        return "CODING_DEV", app_sequence
    
    # Creative Design Pattern
    if any(keyword in apps_str for keyword in ["adobe", "photoshop", "illustrator", "behance", "figma", "canva"]):
        return "CREATIVE_DESIGN", app_sequence
    
    # Data Analysis Pattern
    if any(keyword in apps_str for keyword in ["excel", "powerbi", "tableau", "python"]):
        if "excel" in apps_str or "powerbi" in apps_str:
            return "DATA_ANALYSIS", app_sequence
    
    # Job Hunting Pattern
    if any(keyword in apps_str for keyword in ["handshake", "linkedin", "indeed", "glassdoor"]):
        return "JOB_HUNTING", app_sequence
    
    # Administrative Pattern
    if any(keyword in apps_str for keyword in ["oasis", "degreeworks", "archivum", "navigate", "schedule"]):
        return "ADMIN_LOGISTICS", app_sequence
    
    # Casual Browsing (short sequence, random apps)
    if len(app_sequence) <= 2:
        return "CASUAL_BROWSING", app_sequence
    
    # Communication Pattern
    if apps_str.count("outlook") >= 2 or apps_str.count("teams") >= 2:
        return "COMMUNICATION", app_sequence
    
    # Default: Mixed Academic
    return "MIXED_ACADEMIC", app_sequence

def group_apps_into_sessions(data, gap_minutes=20):
    """
    Groups app access logs into sessions based on time gaps.
    Then detects the behavioral pattern for each session.
    """
    if not data:
        return []
    
    from datetime import datetime
    
    # Sort by time
    sorted_data = sorted(data, key=lambda x: x.get('createdDateTime', x.get('timestamp', '')))
    
    sessions = []
    current_session = []
    last_time = None
    
    for entry in sorted_data:
        app = entry.get('appDisplayName', '')
        time_str = entry.get('createdDateTime', entry.get('timestamp', ''))
        
        # Skip Login/Logout
        if app in ['Login', 'Logout', 'Microsoft 365 Sign-in', 'Microsoft 365 Sign-out']:
            continue
        
        try:
            current_time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        except:
            current_time = datetime.now()
        
        if last_time is None:
            current_session.append(app)
        else:
            gap = (current_time - last_time).total_seconds() / 60.0
            if gap > gap_minutes:
                # End current session
                if current_session:
                    pattern, pattern_apps = detect_pattern_from_sequence(current_session)
                    sessions.append({
                        'apps': current_session,
                        'pattern': pattern,
                        'pattern_apps': pattern_apps
                    })
                current_session = [app]
            else:
                current_session.append(app)
        
        last_time = current_time
    
    # Add last session
    if current_session:
        pattern, pattern_apps = detect_pattern_from_sequence(current_session)
        sessions.append({
            'apps': current_session,
            'pattern': pattern,
            'pattern_apps': pattern_apps
        })
    
    return sessions

def load_models():
    global model, pattern_encoder, config, device
    
    model_dir = 'model_training_pipeline/model_output'
    
    try:
        print("Loading model configuration...")
        config = joblib.load(os.path.join(model_dir, 'model_config.pkl'))
        print(f"Config: {config}")
        
        print("Loading pattern encoder...")
        pattern_encoder = joblib.load(os.path.join(model_dir, 'pattern_encoder.pkl'))
        print(f"Known patterns: {list(pattern_encoder.classes_)}")
        
        print("Loading LSTM model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = LSTMPredictor(
            num_patterns=config['num_patterns'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers']
        ).to(device)
        
        model.load_state_dict(torch.load(
            os.path.join(model_dir, 'lstm_pattern_model.pth'),
            map_location=device
        ))
        model.eval()
        
        print("✅ Pattern prediction model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'known_patterns': list(pattern_encoder.classes_) if pattern_encoder else []
    })

@app.route('/api/predict-next-pattern', methods=['POST'])
def predict_next_pattern():
    """
    Predicts upcoming behavioral patterns based on a student's semester of app usage.
    
    Request body:
    {
        "data": [
            {"appDisplayName": "Canvas", "createdDateTime": "2025-01-01T10:00:00Z"},
            {"appDisplayName": "Outlook", "createdDateTime": "2025-01-01T10:05:00Z"},
            ... (entire semester of app logs)
        ]
    }
    
    Response:
    {
        "detected_sessions": 45,
        "pattern_summary": {"EXAM_CRAMMING": 12, "CODING_DEV": 8, ...},
        "predicted_next_patterns": [
            {"pattern": "EXAM_CRAMMING", "confidence": 0.65, "rank": 1},
            {"pattern": "MIXED_ACADEMIC", "confidence": 0.20, "rank": 2},
            ...
        ]
    }
    """
    try:
        data = request.json.get('data', [])
        
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Expected "data" field with array of app access records'}), 400
        
        print(f"Received {len(data)} app access records")
        
        # Step 1: Group entire semester's apps into sessions and detect patterns
        sessions = group_apps_into_sessions(data)
        
        if len(sessions) == 0:
            return jsonify({'error': 'No valid sessions detected'}), 400
        
        pattern_history = [s['pattern'] for s in sessions]
        pattern_summary = Counter(pattern_history)
        
        # Build app usage profiles per pattern (for personalization)
        pattern_app_usage = {}
        for session in sessions:
            pattern = session['pattern']
            apps_in_pattern = session['pattern_apps']
            
            if pattern not in pattern_app_usage:
                pattern_app_usage[pattern] = []
            pattern_app_usage[pattern].extend(apps_in_pattern)
        
        # Get most common apps per pattern for this student
        pattern_app_profiles = {}
        for pattern, apps in pattern_app_usage.items():
            app_counts = Counter(apps)
            pattern_app_profiles[pattern] = [
                {'app': app, 'frequency': count}
                for app, count in app_counts.most_common(5)
            ]
        
        print(f"Detected {len(sessions)} sessions")
        print(f"Pattern distribution: {dict(pattern_summary)}")
        
        # Step 2: Use recent pattern history to predict next patterns
        sequence_length = config['sequence_length']
        encoded_patterns = []
        
        # Take the most recent patterns for prediction
        recent_patterns = pattern_history[-sequence_length:] if len(pattern_history) >= sequence_length else pattern_history
        
        for pattern in recent_patterns:
            try:
                encoded = pattern_encoder.transform([pattern])[0]
                encoded_patterns.append(encoded)
            except:
                # Unknown pattern, use 0 (padding)
                encoded_patterns.append(0)
        
        # Pad if necessary
        while len(encoded_patterns) < sequence_length:
            encoded_patterns.insert(0, 0)  # Padding at the beginning
        
        # Step 3: Predict next patterns using LSTM
        pattern_tensor = torch.tensor([encoded_patterns], dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits = model(pattern_tensor)
            probs = torch.softmax(logits, dim=1)
            
            # Get top 5 predictions
            top_k = torch.topk(probs, min(5, len(pattern_encoder.classes_)))
            
            predictions = []
            for i in range(len(top_k.indices[0])):
                pattern_idx = top_k.indices[0][i].item()
                confidence = top_k.values[0][i].item()
                pattern_name = pattern_encoder.inverse_transform([pattern_idx])[0]
                
                # Add personalized app recommendations for this pattern
                expected_apps = pattern_app_profiles.get(pattern_name, [])
                
                predictions.append({
                    'pattern': pattern_name,
                    'confidence': float(confidence),
                    'rank': i + 1,
                    'description': get_pattern_description(pattern_name),
                    'expected_apps': expected_apps  # Personalized to this student
                })
        
        return jsonify({
            'detected_sessions': len(sessions),
            'pattern_summary': dict(pattern_summary),
            'recent_patterns': pattern_history[-10:],  # Last 10 patterns for context
            'predicted_next_patterns': predictions,
            'top_prediction': predictions[0]['pattern'] if predictions else None,
            'student_app_profiles': pattern_app_profiles  # Full profile for reference
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def get_pattern_description(pattern_name):
    """Returns a human-readable description of each pattern."""
    descriptions = {
        'DEEP_RESEARCH': 'Academic research using library databases and scholarly sources',
        'GROUP_PROJECT': 'Collaborative work using Teams and Office tools',
        'EXAM_CRAMMING': 'Intensive studying with Canvas and note-taking apps',
        'CODING_DEV': 'Software development with GitHub, IDEs, and coding tools',
        'CREATIVE_DESIGN': 'Design work using Adobe Creative Suite or similar tools',
        'DATA_ANALYSIS': 'Data work with Excel, PowerBI, or Python',
        'JOB_HUNTING': 'Career preparation with LinkedIn and job boards',
        'ADMIN_LOGISTICS': 'Administrative tasks like registration and advising',
        'CASUAL_BROWSING': 'Brief, unstructured app usage',
        'COMMUNICATION': 'Email and messaging focused activity',
        'MIXED_ACADEMIC': 'General coursework without specific focus'
    }
    return descriptions.get(pattern_name, 'Unknown pattern')

@app.route('/api/analyze-patterns', methods=['POST'])
def analyze_patterns():
    """
    Analyzes a user's app history and returns detected behavioral patterns.
    This is useful for understanding what patterns the model sees in the data.
    """
    try:
        data = request.json.get('data', [])
        
        sessions = group_apps_into_sessions(data)
        
        pattern_summary = Counter([s['pattern'] for s in sessions])
        
        return jsonify({
            'total_sessions': len(sessions),
            'detected_patterns': [
                {
                    'pattern': pattern,
                    'count': count,
                    'percentage': round(count / len(sessions) * 100, 1),
                    'description': get_pattern_description(pattern)
                }
                for pattern, count in pattern_summary.most_common()
            ],
            'sessions': sessions[-10:]  # Return last 10 sessions for debugging
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🐍 Starting Pattern Prediction ML Server...")
    load_models()
    print(f"Server running on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
