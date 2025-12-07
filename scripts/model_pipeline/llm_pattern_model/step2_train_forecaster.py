"""Train Pattern Forecaster (Sequential Context Predictor)

This script trains a Random Forest model to predict the next learning context based on
a sequence of previous contexts, enabling temporal pattern forecasting.

Key Features:
- Sessionizes logs using 20-minute gaps to identify distinct learning sessions
- Creates sliding window sequences (N-grams) of context labels
- Trains on context transitions to learn temporal dependencies
- Uses label encoding for categorical context handling

Input: raw_logs/labeled_logs.json (logs with context labels)
Output: 
- shadow_simulation/pattern_forecaster.pkl (trained forecaster)
- shadow_simulation/pattern_encoder.pkl (label encoder)

Workflow:
1. Load and sessionize logs per user
2. Assign dominant context label to each session
3. Create sliding windows of size SEQUENCE_LENGTH (default: 3)
4. Train Random Forest to predict next context from previous N contexts
5. Save forecaster and encoder for inference

This enables predicting: Given [Context1, Context2, Context3] → Context4
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# --- CONFIGURATION ---
LOGS_FILE = "raw_logs/labeled_logs.json"
MODEL_FILE = "shadow_simulation/pattern_forecaster.pkl"
ENCODER_FILE = "shadow_simulation/pattern_encoder.pkl"
SEQUENCE_LENGTH = 3  # Look back at last 3 sessions

def load_logs():
    with open(LOGS_FILE, 'r') as f:
        return json.load(f)

def prepare_sequences(logs):
    """
    Groups logs into sessions, then groups sessions by user to form sequences.
    """
    df = pd.DataFrame(logs)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['user', 'timestamp'])
    
    # 1. Sessionize
    sessions = []
    
    # Group by User first
    for user_id, group in df.groupby('user'):
        group = group.sort_values('timestamp')
        
        user_sessions = []
        curr_sess = []
        
        for _, row in group.iterrows():
            if not curr_sess:
                curr_sess.append(row)
                continue
                
            last_time = curr_sess[-1]['timestamp']
            curr_time = row['timestamp']
            
            if (curr_time - last_time).total_seconds() > 20 * 60:
                # End of session
                labels = [r['context'] for r in curr_sess]
                from collections import Counter
                label = Counter(labels).most_common(1)[0][0]
                user_sessions.append(label)
                curr_sess = [row]
            else:
                curr_sess.append(row)
                
        if curr_sess:
            labels = [r['context'] for r in curr_sess]
            from collections import Counter
            label = Counter(labels).most_common(1)[0][0]
            user_sessions.append(label)
            
        sessions.append(user_sessions)
        
    print(f"Extracted session sequences for {len(sessions)} users.")
    return sessions

def main():
    print("Loading logs...")
    logs = load_logs()
    
    print("Preparing sequences...")
    user_sequences = prepare_sequences(logs)
    
    # Flatten to train encoder
    all_patterns = [p for seq in user_sequences for p in seq]
    
    print(f"Total sessions: {len(all_patterns)}")
    print(f"Unique patterns: {len(set(all_patterns))}")
    
    # Encode Labels
    le = LabelEncoder()
    le.fit(all_patterns)
    
    # Create Input Sequences (N-Grams)
    X = []
    y = []
    
    for seq in user_sequences:
        encoded_seq = le.transform(seq)
        
        # Create sliding windows
        for i in range(SEQUENCE_LENGTH, len(encoded_seq)):
            # Input: Previous N patterns
            window = encoded_seq[i-SEQUENCE_LENGTH:i]
            # Target: Current pattern
            target = encoded_seq[i]
            
            X.append(window)
            y.append(target)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Training data shape: {X.shape}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Forecaster...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save
    os.makedirs("shadow_simulation", exist_ok=True)
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(le, ENCODER_FILE)
    print(f"Model saved to {MODEL_FILE}")
    print(f"Encoder saved to {ENCODER_FILE}")

if __name__ == "__main__":
    main()
