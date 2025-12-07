"""Predict Next Learning Context and Recommended Apps

This script demonstrates end-to-end prediction: given a user's session history (raw apps),
it infers their context sequence and forecasts their next likely context with app recommendations.

Key Features:
- Infers context from raw app sequences using trained classifier
- Forecasts next context using sequential forecaster
- Retrieves matching blueprints to recommend specific apps
- Works with unlabeled user history (no context labels needed)

Inputs:
- shadow_simulation/pattern_classifier.pkl (session classifier)
- shadow_simulation/pattern_forecaster.pkl (sequential forecaster)
- shadow_simulation/pattern_encoder.pkl (label encoder)
- shadow_simulation/blueprints.json (context templates with app sequences)

Workflow:
1. Analyze user's past sessions (3+ sessions with timestamps and apps)
2. Extract features and classify each session's context
3. Use last N contexts to forecast next context
4. Look up blueprint for predicted context
5. Return recommended apps and activities

Example: [Session1: Canvas+Word] → 'STUDY' → Predicts → 'PROJECT' → Recommends: [GitHub, Overleaf]
"""

import json
import joblib
import pandas as pd
import numpy as np
import argparse
import random

# --- CONFIGURATION ---
CLASSIFIER_FILE = "shadow_simulation/pattern_classifier.pkl"
FORECASTER_FILE = "shadow_simulation/pattern_forecaster.pkl"
ENCODER_FILE = "shadow_simulation/pattern_encoder.pkl"
BLUEPRINTS_FILE = "shadow_simulation/blueprints.json"
SEQUENCE_LENGTH = 3

def load_models():
    print("Loading models...")
    classifier_data = joblib.load(CLASSIFIER_FILE)
    classifier = classifier_data["model"]
    feature_names = classifier_data["features"]
    
    forecaster = joblib.load(FORECASTER_FILE)
    encoder = joblib.load(ENCODER_FILE)
    
    with open(BLUEPRINTS_FILE, 'r') as f:
        blueprints = json.load(f)
        
    return classifier, feature_names, forecaster, encoder, blueprints

def extract_features(session_events, feature_names):
    """
    Extracts features from a session to feed into the classifier.
    """
    if not session_events:
        return None
        
    start_time = pd.to_datetime(session_events[0]['timestamp'])
    end_time = pd.to_datetime(session_events[-1]['timestamp'])
    duration_minutes = (end_time - start_time).total_seconds() / 60
    
    apps = [e['app'] for e in session_events]
    unique_apps = set(apps)
    
    # Base features
    features = {
        "duration_minutes": duration_minutes,
        "num_events": len(session_events),
        "num_unique_apps": len(unique_apps),
        "hour_of_day": start_time.hour,
        "is_weekend": 1 if start_time.dayofweek >= 5 else 0,
    }
    
    # App presence features
    for feat in feature_names:
        if feat.startswith("has_"):
            app_name = feat.replace("has_", "")
            features[feat] = 1 if app_name in unique_apps else 0
            
    # Ensure order matches training
    vector = [features.get(f, 0) for f in feature_names]
    return [vector]

def predict_next_step(user_history, classifier, feature_names, forecaster, encoder, blueprints):
    """
    Predicts the next likely pattern and apps based on user history.
    """
    # 1. Classify Past Sessions
    past_patterns = []
    
    print("\n--- Analyzing History ---")
    for i, session in enumerate(user_history):
        X = extract_features(session, feature_names)
        predicted_label = classifier.predict(X)[0]
        past_patterns.append(predicted_label)
        print(f"Session {i+1}: {len(session)} events -> Classified as: {predicted_label}")
        
    # 2. Forecast Next Pattern
    # We need at least SEQUENCE_LENGTH patterns
    if len(past_patterns) < SEQUENCE_LENGTH:
        print(f"Not enough history to forecast (Need {SEQUENCE_LENGTH}, got {len(past_patterns)})")
        return
        
    # Get last N patterns
    recent_sequence = past_patterns[-SEQUENCE_LENGTH:]
    
    # Encode
    try:
        encoded_seq = encoder.transform(recent_sequence)
    except ValueError as e:
        print(f"Error encoding sequence (unknown label): {e}")
        return

    # Predict
    prediction_idx = forecaster.predict([encoded_seq])[0]
    predicted_pattern = encoder.inverse_transform([prediction_idx])[0]
    
    print(f"\n>>> FORECAST: The student will likely enter context: '{predicted_pattern}'")
    
    # 3. Retrieve Blueprint for this Pattern
    # Find blueprints that match this context
    matching_blueprints = [b for b in blueprints if b.get('context_type') == predicted_pattern]
    
    if matching_blueprints:
        # Pick one (or the best match)
        bp = random.choice(matching_blueprints)
        print(f"\n>>> RECOMMENDED APPS (from Blueprint):")
        print(f"Persona: {bp['persona']}")
        print("Likely Apps:")
        for event in bp['usf_log']:
            print(f"  - {event['app']} ({event['original_activity']})")
    else:
        print(f"No specific blueprint found for '{predicted_pattern}'.")

def main():
    # Load Models
    clf, feats, forecaster, encoder, blueprints = load_models()
    
    # Simulate a User History (3 sessions)
    # Let's pretend we have a user who just finished 3 sessions
    
    # Session 1: Normal study
    s1 = [
        {"timestamp": "2023-11-01T10:00:00", "app": "Canvas"},
        {"timestamp": "2023-11-01T10:05:00", "app": "Word Online"},
        {"timestamp": "2023-11-01T10:20:00", "app": "Canvas"}
    ]
    
    # Session 2: Research
    s2 = [
        {"timestamp": "2023-11-02T14:00:00", "app": "Library Database"},
        {"timestamp": "2023-11-02T14:10:00", "app": "Google Scholar"},
        {"timestamp": "2023-11-02T14:30:00", "app": "OneNote"}
    ]
    
    # Session 3: Late night panic (should trigger "FINALS" or similar)
    s3 = [
        {"timestamp": "2023-11-03T23:00:00", "app": "Canvas"},
        {"timestamp": "2023-11-03T23:05:00", "app": "Turnitin"},
        {"timestamp": "2023-11-03T23:10:00", "app": "Canvas"},
        {"timestamp": "2023-11-03T23:15:00", "app": "Word Online"}
    ]
    
    history = [s1, s2, s3]
    
    predict_next_step(history, clf, feats, forecaster, encoder, blueprints)

if __name__ == "__main__":
    main()
