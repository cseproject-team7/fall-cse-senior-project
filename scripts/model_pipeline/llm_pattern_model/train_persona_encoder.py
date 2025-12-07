"""
PERSONA ENCODER TRAINING

Trains Random Forest classifiers to infer student personas (major and work ethic)
from their app usage patterns and activity behaviors.

Features Extracted:
- App usage frequency distribution
- Time-of-day preferences (morning/day/night)
- Session intensity metrics
- Total activity counts

Outputs two classifiers:
1. Major Classifier (CS, Arts, Business, etc.)
2. Work Ethic Classifier (Consistent, Procrastinator, etc.)

Input:  raw_logs/student_histories.json
Output: shadow_simulation/persona_encoder.pkl (trained models + feature names)
"""

import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import os

# --- CONFIGURATION ---
DATA_FILE = "raw_logs/student_histories.json"
MODEL_FILE = "shadow_simulation/persona_encoder.pkl"

def load_data():
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def extract_student_features(student_data):
    """
    Aggregates a student's entire history into a single feature vector.
    """
    history = student_data['history']
    if not history:
        return None
        
    # Flatten all events
    all_events = []
    for session in history:
        all_events.extend(session['events'])
        
    if not all_events:
        return None
        
    df = pd.DataFrame(all_events)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. App Usage Counts (Normalized)
    app_counts = df['app'].value_counts(normalize=True).to_dict()
    
    # 2. Time of Day Distribution
    df['hour'] = df['timestamp'].dt.hour
    morning_pct = len(df[(df['hour'] >= 6) & (df['hour'] < 12)]) / len(df)
    day_pct = len(df[(df['hour'] >= 12) & (df['hour'] < 18)]) / len(df)
    night_pct = len(df[(df['hour'] >= 18) | (df['hour'] < 6)]) / len(df)
    
    # 3. Activity Intensity
    total_sessions = len(history)
    avg_events_per_session = len(df) / total_sessions
    
    features = {
        "total_sessions": total_sessions,
        "avg_events_per_session": avg_events_per_session,
        "morning_pct": morning_pct,
        "day_pct": day_pct,
        "night_pct": night_pct
    }
    
    # Add app features (prefix with 'app_')
    for app, freq in app_counts.items():
        features[f"app_{app}"] = freq
        
    return features

def prepare_dataset(data):
    X = []
    y_major = []
    y_work_ethic = []
    
    # First pass to get all possible apps for consistent feature vector
    all_apps = set()
    for student in data:
        history = student['history']
        for session in history:
            for event in session['events']:
                all_apps.add(event['app'])
    
    feature_names = ["total_sessions", "avg_events_per_session", "morning_pct", "day_pct", "night_pct"]
    feature_names.extend([f"app_{app}" for app in sorted(list(all_apps))])
    
    print(f"Feature vector size: {len(feature_names)}")
    
    for student in data:
        feats = extract_student_features(student)
        if not feats:
            continue
            
        # Create vector
        vector = [feats.get(f, 0) for f in feature_names]
        X.append(vector)
        
        y_major.append(student['student_profile']['major'])
        y_work_ethic.append(student['student_profile']['work_ethic'])
        
    return np.array(X), np.array(y_major), np.array(y_work_ethic), feature_names

def main():
    print("Loading data...")
    data = load_data()
    
    print("Preparing dataset...")
    X, y_major, y_work_ethic, feature_names = prepare_dataset(data)
    
    # Train Major Classifier
    print("\n--- Training Major Classifier ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y_major, test_size=0.2, random_state=42)
    clf_major = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_major.fit(X_train, y_train)
    print(classification_report(y_test, clf_major.predict(X_test), zero_division=0))
    
    # Train Work Ethic Classifier
    print("\n--- Training Work Ethic Classifier ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y_work_ethic, test_size=0.2, random_state=42)
    clf_ethic = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_ethic.fit(X_train, y_train)
    print(classification_report(y_test, clf_ethic.predict(X_test), zero_division=0))
    
    # Save Models
    joblib.dump({
        "major_model": clf_major,
        "ethic_model": clf_ethic,
        "features": feature_names
    }, MODEL_FILE)
    print(f"\nModels saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()
