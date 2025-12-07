"""Rolling Pattern Prediction (Multi-Step Forecasting)

This script performs iterative multi-step forecasting to predict a sequence of future
learning contexts and app patterns, demonstrating the model's ability to simulate
student trajectories over time.

Key Features:
- Iterative forecasting: predicts next 10 contexts in sequence
- Uses sliding window approach (predict → append → shift → repeat)
- Maps each predicted context to blueprint app sequences
- Demonstrates shadow simulation capability (future behavior modeling)

Inputs:
- shadow_simulation/pattern_classifier.pkl (session classifier)
- shadow_simulation/pattern_forecaster.pkl (sequential forecaster)  
- shadow_simulation/pattern_encoder.pkl (label encoder)
- shadow_simulation/blueprints.json (context templates)

Workflow:
1. Infer context history from user's raw app sessions
2. Initialize sliding window with last N contexts
3. For each prediction step:
   a. Forecast next context from current window
   b. Look up blueprint app sequence
   c. Update window (append prediction, drop oldest)
4. Display 10-step trajectory with contexts and app sequences

Example Output:
[History: STUDY→PROJECT→STUDY] → Predicts:
Step 1: PROJECT → [GitHub, Overleaf, Canvas]
Step 2: STUDY → [Canvas, Word, Library]
Step 3: MIDTERMS → [Canvas, Turnitin, Quizlet]
...
"""

import json
import joblib
import pandas as pd
import numpy as np
import random
import argparse

# --- CONFIGURATION ---
CLASSIFIER_FILE = "shadow_simulation/pattern_classifier.pkl"
FORECASTER_FILE = "shadow_simulation/pattern_forecaster.pkl"
ENCODER_FILE = "shadow_simulation/pattern_encoder.pkl"
BLUEPRINTS_FILE = "shadow_simulation/blueprints.json"
SEQUENCE_LENGTH = 3
PREDICTION_STEPS = 10

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

def get_blueprint_for_context(context, blueprints):
    """
    Finds a blueprint that matches the predicted context.
    """
    matches = [b for b in blueprints if b.get('context_type') == context]
    if matches:
        return random.choice(matches)
    return None

def rolling_prediction(user_history, classifier, feature_names, forecaster, encoder, blueprints):
    """
    Performs rolling prediction for the next 10 steps.
    """
    # 1. Infer Context History from Raw Apps
    context_history = []
    
    print("\n--- Step 1: Inferring Context from History ---")
    for i, session in enumerate(user_history):
        X = extract_features(session, feature_names)
        predicted_label = classifier.predict(X)[0]
        context_history.append(predicted_label)
        print(f"Session {i+1}: {len(session)} events -> Inferred Context: '{predicted_label}'")
        
    if len(context_history) < SEQUENCE_LENGTH:
        print(f"Not enough history. Padding with last known context.")
        while len(context_history) < SEQUENCE_LENGTH:
            context_history.insert(0, context_history[0])

    # 2. Rolling Forecast
    print(f"\n--- Step 2: Forecasting Next {PREDICTION_STEPS} Patterns ---")
    
    current_sequence = context_history[-SEQUENCE_LENGTH:]
    
    for step in range(1, PREDICTION_STEPS + 1):
        # Encode current sequence
        try:
            encoded_seq = encoder.transform(current_sequence)
        except ValueError:
            # Fallback if unknown label appears (shouldn't happen with closed set)
            print("Unknown label in sequence, stopping.")
            break
            
        # Predict Next Context
        prediction_idx = forecaster.predict([encoded_seq])[0]
        predicted_context = encoder.inverse_transform([prediction_idx])[0]
        
        # Look up Blueprint (The "Sequence of Apps")
        bp = get_blueprint_for_context(predicted_context, blueprints)
        
        print(f"\n[Prediction {step}/{PREDICTION_STEPS}]")
        print(f"  Context: '{predicted_context}'")
        
        if bp:
            app_sequence = [e['app'] for e in bp['usf_log']]
            # Truncate for display
            display_seq = app_sequence[:5] + ["..."] if len(app_sequence) > 5 else app_sequence
            print(f"  App Sequence: {display_seq}")
        else:
            print("  App Sequence: [No Blueprint Found]")
            
        # Update Sequence for Next Step (Sliding Window)
        current_sequence.append(predicted_context)
        current_sequence.pop(0)

def main():
    clf, feats, forecaster, encoder, blueprints = load_models()
    
    # Simulate a User History (Raw Apps Only - No Labels!)
    # This proves we can start from just a list of apps.
    
    # Session 1: Coding
    s1 = [
        {"timestamp": "2023-11-01T10:00:00", "app": "GitHub"},
        {"timestamp": "2023-11-01T10:05:00", "app": "StackOverflow"},
        {"timestamp": "2023-11-01T10:20:00", "app": "Copilot"}
    ]
    
    # Session 2: More Coding
    s2 = [
        {"timestamp": "2023-11-02T14:00:00", "app": "GitHub"},
        {"timestamp": "2023-11-02T14:10:00", "app": "MATLAB Online"},
        {"timestamp": "2023-11-02T14:30:00", "app": "Canvas"}
    ]
    
    # Session 3: Submission (Trigger for next state?)
    s3 = [
        {"timestamp": "2023-11-03T23:00:00", "app": "Canvas"},
        {"timestamp": "2023-11-03T23:05:00", "app": "Turnitin"},
        {"timestamp": "2023-11-03T23:10:00", "app": "Canvas"}
    ]
    
    history = [s1, s2, s3]
    
    rolling_prediction(history, clf, feats, forecaster, encoder, blueprints)

if __name__ == "__main__":
    main()
