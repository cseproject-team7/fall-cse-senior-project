"""Train Weekly Context Classifier

This script trains a Random Forest classifier to predict student context (e.g., 'MIDTERMS',
'PROJECT', 'REGULAR') based on aggregated weekly behavioral features.

Key Features:
- Aggregates events by week to capture longer-term patterns
- Extracts behavioral features (app frequencies, time-of-day, weekend activity)
- Normalizes app usage as frequency vectors
- Trains classifier to infer learning context from behavior

Input: raw_logs/usf_visible_semester.json (semester logs with context labels)
Output: shadow_simulation/weekly_classifier.pkl (trained classifier + feature names)

Workflow:
1. Load semester logs with context labels
2. Group events by week
3. Extract aggregated features per week (app frequencies, temporal patterns)
4. Train Random Forest on weekly feature vectors
5. Save model with feature schema for inference

Note: Designed for small datasets (e.g., 12 weeks/student). With more students,
split into train/test sets for proper validation.
"""

import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# --- CONFIGURATION ---
LOGS_FILE = "raw_logs/usf_visible_semester.json"
MODEL_FILE = "shadow_simulation/weekly_classifier.pkl"

def load_logs():
    with open(LOGS_FILE, 'r') as f:
        data = json.load(f)
        return data['logs'] # Extract the list of events

def extract_weekly_features(week_events, all_apps):
    """
    Extracts aggregated features for a full week of activity.
    """
    if not week_events:
        return None
        
    df = pd.DataFrame(week_events)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. App Counts (Normalized)
    app_counts = df['app'].value_counts().to_dict()
    total_events = len(df)
    
    features = {
        "total_events": total_events,
        "unique_apps": len(df['app'].unique()),
    }
    
    # 2. Time of Day Features
    df['hour'] = df['timestamp'].dt.hour
    features['late_night_activity'] = len(df[df['hour'] >= 22])  # Events after 10 PM
    features['morning_activity'] = len(df[df['hour'] < 9])       # Events before 9 AM
    
    # 3. Weekend Features
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    features['weekend_activity'] = len(df[df['day_of_week'] >= 5])
    
    # 4. App Specific Features (Normalized Frequency)
    for app in all_apps:
        count = app_counts.get(app, 0)
        features[f"freq_{app}"] = count / total_events if total_events > 0 else 0
        
    return features

def prepare_data(logs):
    """
    Groups logs by Week and prepares (X, y).
    """
    df = pd.DataFrame(logs)
    
    # Get all unique apps for consistent feature vector
    all_apps = sorted(list(set(df['app'].unique())))
    print(f"Feature space: {len(all_apps)} unique apps")
    
    X = []
    y = []
    
    # Group by Week
    # The logs have a 'week' field from the generator
    for week_num, group in df.groupby('week'):
        week_events = group.to_dict('records')
        
        # Extract Features
        feats = extract_weekly_features(week_events, all_apps)
        X.append(list(feats.values()))
        
        # Label: The context is consistent for the week
        label = week_events[0]['context']
        y.append(label)
        
    feature_names = list(feats.keys())
    return np.array(X), np.array(y), feature_names

def main():
    print("Loading logs...")
    logs = load_logs()
    
    print("Aggregating by Week...")
    X, y, feature_names = prepare_data(logs)
    
    print(f"Data shape: {X.shape} (Weeks x Features)")
    print(f"Labels: {set(y)}")
    
    # Since we only have 12 weeks for 1 student, splitting is tricky (very small data).
    # But to demonstrate the concept, we'll just train on all and see if it overfits (learns the pattern).
    # In a real scenario, we'd have many students.
    
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    print("Evaluating (Training Accuracy)...")
    y_pred = clf.predict(X)
    print(classification_report(y, y_pred, zero_division=0))
    
    # Save Model
    os.makedirs("shadow_simulation", exist_ok=True)
    joblib.dump({
        "model": clf,
        "features": feature_names
    }, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()
