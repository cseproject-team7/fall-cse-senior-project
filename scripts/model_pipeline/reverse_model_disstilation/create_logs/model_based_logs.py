import os
import json
import torch
import torch.nn as nn
import joblib
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import hashlib

# Load environment variables
load_dotenv('server/.env')

# ============================================================================
# Configuration
# ============================================================================

MODEL_DIR = 'azure_ml_deployment/model'
CONTAINER_NAME = 'json-signin-logs'
CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

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

def load_model():
    global model, pattern_encoder, app_encoder
    
    try:
        print(f"Loading model from {MODEL_DIR}...")
        pattern_encoder = joblib.load(os.path.join(MODEL_DIR, 'pattern_encoder.pkl'))
        app_encoder = joblib.load(os.path.join(MODEL_DIR, 'app_encoder.pkl'))
        config = joblib.load(os.path.join(MODEL_DIR, 'model_config.pkl'))
        
        model = DualHeadLSTM(
            num_patterns=config['num_patterns'],
            num_apps=config['num_apps'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers']
        )
        
        model.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, 'dual_head_lstm.pth'),
            map_location=torch.device('cpu')
        ))
        model.eval()
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Helper Functions
# ============================================================================

def detect_pattern_heuristic(apps: List[str]) -> str:
    """Detect pattern from app list using heuristics"""
    apps_str = ' '.join([app.lower() for app in apps])
    
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
    
    if has_admin: return 'ADMIN'
    elif has_career: return 'CAREER'
    elif 'club' in apps_str or 'organization' in apps_str: return 'CLUBS'
    elif has_coding: return 'CODING'
    elif has_research: return 'RESEARCH'
    elif has_canvas and any(exam_app in apps_str for exam_app in ['onenote', 'pdf', 'word']): return 'EXAM'
    elif has_social: return 'SOCIAL'
    else: return 'COURSEWORK'

def encode_session_for_model(session: List[str]) -> tuple:
    pattern = detect_pattern_heuristic(session)
    try:
        pattern_id = pattern_encoder.transform([pattern])[0]
    except:
        pattern_id = 0
    
    app_sequence = ', '.join(session[:3])
    try:
        app_id = app_encoder.transform([app_sequence])[0]
    except:
        for app in session:
            try:
                app_id = app_encoder.transform([app])[0]
                break
            except:
                continue
        else:
            app_id = 0
    
    return pattern_id, app_id

    return pattern_id, app_id

def anonymize_user(user_id):
    """Generate anonymized user ID and display name"""
    raw_id = f"student{user_id}@usf.edu"
    hash_obj = hashlib.sha256(raw_id.encode())
    short_hash = hash_obj.hexdigest()[:8]
    return f"User {short_hash}", short_hash

def get_random_seed_history():
    """Generate diverse initial history for users"""
    common_apps = [
        'Canvas', 'Outlook', 'Teams', 'MyUSF', 'OASIS', 'DegreeWorks', 
        'Handshake', 'GitHub', 'Jupyter', 'Visual Studio Code', 
        'Adobe Creative Cloud', 'Zoom', 'Slack', 'Discord', 
        'Google Scholar', 'PubMed', 'LinkedIn', 'Spotify', 'Netflix'
    ]
    # Create 2 initial sessions with random apps
    s1 = random.sample(common_apps, k=random.randint(1, 2))
    s2 = random.sample(common_apps, k=random.randint(1, 3))
    return [s1, s2]

def predict_next_pattern_and_apps(history_sessions: List[List[str]]) -> Dict[str, Any]:
    """Predict next pattern and apps with BOOSTING logic"""
    if not history_sessions:
        return {'top_apps': ['Canvas', 'Outlook']}
    
    recent_sessions = history_sessions[-10:]
    pattern_ids = []
    app_ids = []
    
    for session in recent_sessions:
        pat_id, app_id = encode_session_for_model(session)
        pattern_ids.append(pat_id)
        app_ids.append(app_id)
    
    pattern_tensor = torch.tensor([pattern_ids], dtype=torch.long)
    app_tensor = torch.tensor([app_ids], dtype=torch.long)
    
    with torch.no_grad():
        _, app_logits = model(pattern_tensor, app_tensor)
        
        # Temperature scaling for diversity
        temperature = 1.5
        app_probs = torch.softmax(app_logits / temperature, dim=-1)[0]
        
        # ====================================================================
        # BOOST LOGIC: Extreme graduated boosting (from score.py)
        # ====================================================================
        boosted_probs = app_probs.clone()
        
        for idx in range(len(app_encoder.classes_)):
            app_seq = app_encoder.classes_[idx]
            
            # Penalize generic/overused apps
            if app_seq in ['Microsoft 365', 'Copilot', 'MyUSF']:
                boosted_probs[idx] = boosted_probs[idx] * 0.1
                continue
            
            if ', ' in app_seq:  # Multi-app sequence
                apps_list = app_seq.split(', ')
                num_apps = len(apps_list)
                meaningful_apps = [app for app in apps_list if app not in ['Microsoft 365', 'Copilot', 'MyUSF']]
                
                if num_apps == 2:
                    if len(meaningful_apps) >= 1:
                        boosted_probs[idx] = boosted_probs[idx] * 1.5
                elif num_apps == 3:
                    if len(meaningful_apps) >= 2:
                        boosted_probs[idx] = boosted_probs[idx] * 30.0
                    else:
                        boosted_probs[idx] = boosted_probs[idx] * 15.0
                elif num_apps == 4:
                    if len(meaningful_apps) >= 3:
                        boosted_probs[idx] = boosted_probs[idx] * 80.0
                    else:
                        boosted_probs[idx] = boosted_probs[idx] * 45.0
                elif num_apps >= 5:
                    boosted_probs[idx] = boosted_probs[idx] * 120.0
            else:
                boosted_probs[idx] = boosted_probs[idx] * 1.2
        
        boosted_probs = boosted_probs / boosted_probs.sum()
        
        # Sample from distribution instead of just taking top 1 to add variety
        top_indices = torch.multinomial(boosted_probs, 1).tolist()
        top_app_seq = app_encoder.inverse_transform(top_indices)[0]
        
        if ', ' in top_app_seq:
            return {'top_apps': top_app_seq.split(', ')}
        else:
            return {'top_apps': [top_app_seq]}

# ============================================================================
# Log Generation & Upload
# ============================================================================

def generate_and_upload_logs(num_users=5, sessions_per_user=20):
    if not CONNECTION_STRING:
        print("❌ AZURE_STORAGE_CONNECTION_STRING not found in environment")
        return

    try:
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        print(f"Generating logs for {num_users} users, {sessions_per_user} sessions each...")
        
        # Clear existing blobs
        print(f"Clearing existing blobs in '{CONTAINER_NAME}'...")
        blobs = container_client.list_blobs()
        for blob in blobs:
            try:
                print(f"Deleting {blob.name}...")
                container_client.delete_blob(blob.name)
            except Exception as e:
                print(f"⚠️ Could not delete {blob.name}: {e}")
        print("✅ Container cleanup finished.")
        
        all_logs = []
        start_date = datetime.now() - timedelta(days=7)

        for user_id in range(1, num_users + 1):
            if user_id % 100 == 0:
                print(f"Generating for user {user_id}/{num_users}...")
            
            # Anonymize PII
            user_display, user_hash = anonymize_user(user_id)
            
            current_time = start_date + timedelta(hours=random.randint(0, 23))
            
            # Seed history with DIVERSITY
            history = get_random_seed_history()
            
            for _ in range(sessions_per_user):
                # Predict next session
                prediction = predict_next_pattern_and_apps(history)
                session_apps = prediction['top_apps']
                
                # Add to history
                history.append(session_apps)
                if len(history) > 10: history.pop(0)
                
                # Create logs for this session
                for app in session_apps:
                    # Add random gap between apps in session
                    current_time += timedelta(minutes=random.randint(1, 5))
                    
                    log_entry = {
                        "userPrincipalName": user_display,
                        "userId": user_hash,
                        "appDisplayName": app,
                        "createdDateTime": current_time.isoformat() + 'Z'
                    }
                    
                    # Format as Azure log (nested Body)
                    azure_log = {
                        "Body": json.dumps(log_entry)
                    }
                    all_logs.append(azure_log)
                
                # Gap between sessions
                current_time += timedelta(hours=random.randint(1, 4))
        
        # Save to file
        output_file = 'generated_azure_logs.json'
        with open(output_file, 'w') as f:
            for log in all_logs:
                f.write(json.dumps(log) + '\n')
        
        print(f"✅ Generated {len(all_logs)} logs in {output_file}")
        
        # Upload
        print(f"Uploading to Azure container '{CONTAINER_NAME}'...")
        blob_name = f"generated_logs_{int(time.time())}.json"
        
        with open(output_file, "rb") as data:
            container_client.upload_blob(name=blob_name, data=data, overwrite=True)
            
        print(f"✅ Successfully uploaded to {blob_name}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if load_model():
        generate_and_upload_logs(num_users=1000)
