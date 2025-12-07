"""
PERSONA-AWARE PATTERN FORECASTER

Trains a persona-aware pattern predictor that combines:
1. Sequential pattern history (last N patterns)
2. Inferred student persona (major + work ethic)

Compares baseline (sequence-only) vs persona-aware models to quantify
the improvement gained by incorporating student characteristics.

Key Innovation:
- Uses inferred personas (not ground truth) to simulate real-world deployment
- Demonstrates how personalization improves prediction accuracy

Input:  raw_logs/student_histories.json, shadow_simulation/persona_encoder.pkl
Output: shadow_simulation/aware_forecaster.pkl (persona-aware model + encoders)
"""

import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

# --- CONFIGURATION ---
HISTORY_FILE = "raw_logs/student_histories.json"
PERSONA_MODEL_FILE = "shadow_simulation/persona_encoder.pkl"
FORECASTER_FILE = "shadow_simulation/aware_forecaster.pkl"
SEQUENCE_LENGTH = 3

def load_resources():
    with open(HISTORY_FILE, 'r') as f:
        data = json.load(f)
        
    persona_models = joblib.load(PERSONA_MODEL_FILE)
    
    return data, persona_models

def extract_student_features(student_data, feature_names):
    # Re-implement feature extraction (should be shared code, but duplicating for simplicity)
    history = student_data['history']
    if not history:
        return None
        
    all_events = []
    for session in history:
        all_events.extend(session['events'])
        
    if not all_events:
        return None
        
    df = pd.DataFrame(all_events)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    app_counts = df['app'].value_counts(normalize=True).to_dict()
    
    df['hour'] = df['timestamp'].dt.hour
    morning_pct = len(df[(df['hour'] >= 6) & (df['hour'] < 12)]) / len(df)
    day_pct = len(df[(df['hour'] >= 12) & (df['hour'] < 18)]) / len(df)
    night_pct = len(df[(df['hour'] >= 18) | (df['hour'] < 6)]) / len(df)
    
    total_sessions = len(history)
    avg_events_per_session = len(df) / total_sessions
    
    features = {
        "total_sessions": total_sessions,
        "avg_events_per_session": avg_events_per_session,
        "morning_pct": morning_pct,
        "day_pct": day_pct,
        "night_pct": night_pct
    }
    
    for app, freq in app_counts.items():
        features[f"app_{app}"] = freq
        
    return [features.get(f, 0) for f in feature_names]

def prepare_sequences(data, persona_models):
    X_seq = []
    X_persona = []
    y = []
    
    # Encoders
    pattern_le = LabelEncoder()
    major_le = LabelEncoder()
    ethic_le = LabelEncoder()
    
    # Collect all patterns for fitting LE
    all_patterns = []
    for student in data:
        for session in student['history']:
            all_patterns.append(session['context'])
    pattern_le.fit(all_patterns)
    
    # Fit Persona LEs (using ground truth for fitting, but we will use inferred for training)
    majors = [s['student_profile']['major'] for s in data]
    ethics = [s['student_profile']['work_ethic'] for s in data]
    major_le.fit(majors)
    ethic_le.fit(ethics)
    
    print("Generating sequences...")
    
    for student in data:
        # 1. Infer Persona
        feats = extract_student_features(student, persona_models['features'])
        if not feats:
            continue
            
        inferred_major = persona_models['major_model'].predict([feats])[0]
        inferred_ethic = persona_models['ethic_model'].predict([feats])[0]
        
        # Encode Persona
        try:
            p_major = major_le.transform([inferred_major])[0]
            p_ethic = ethic_le.transform([inferred_ethic])[0]
        except ValueError:
            continue # Skip if unknown
            
        # 2. Create Sequences
        # Sort history by timestamp
        history = sorted(student['history'], key=lambda x: x['timestamp'])
        patterns = [s['context'] for s in history]
        encoded_patterns = pattern_le.transform(patterns)
        
        for i in range(SEQUENCE_LENGTH, len(encoded_patterns)):
            seq = encoded_patterns[i-SEQUENCE_LENGTH:i]
            target = encoded_patterns[i]
            
            X_seq.append(seq)
            X_persona.append([p_major, p_ethic])
            y.append(target)
            
    return np.array(X_seq), np.array(X_persona), np.array(y), pattern_le, major_le, ethic_le

def main():
    data, persona_models = load_resources()
    
    X_seq, X_persona, y, pattern_le, major_le, ethic_le = prepare_sequences(data, persona_models)
    
    print(f"Total sequences: {len(y)}")
    
    # Baseline: Sequence Only
    print("\n--- Baseline Model (Sequence Only) ---")
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)
    clf_base = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_base.fit(X_train, y_train)
    acc_base = accuracy_score(y_test, clf_base.predict(X_test))
    print(f"Baseline Accuracy: {acc_base:.4f}")
    
    # Aware: Sequence + Persona
    print("\n--- Persona-Aware Model ---")
    # Concatenate Sequence + Persona
    X_combined = np.hstack((X_seq, X_persona))
    
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    clf_aware = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_aware.fit(X_train, y_train)
    acc_aware = accuracy_score(y_test, clf_aware.predict(X_test))
    print(f"Aware Accuracy:    {acc_aware:.4f}")
    
    improvement = (acc_aware - acc_base) * 100
    print(f"\nImprovement: +{improvement:.2f}%")
    
    # Save
    joblib.dump({
        "model": clf_aware,
        "pattern_le": pattern_le,
        "major_le": major_le,
        "ethic_le": ethic_le
    }, FORECASTER_FILE)
    print(f"Model saved to {FORECASTER_FILE}")

if __name__ == "__main__":
    main()
