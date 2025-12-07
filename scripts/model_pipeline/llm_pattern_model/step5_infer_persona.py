"""
PERSONA INFERENCE FROM BEHAVIORAL PATTERNS

Simulates a student's year-long activity and infers their persona by analyzing
the distribution of detected behavioral patterns.

Workflow:
1. Select a target persona (e.g., CS Major)
2. Generate 1 year of synthetic logs (80% target persona, 20% noise)
3. Classify each session using the trained pattern classifier
4. Build pattern signature (frequency distribution)
5. Match signature to most likely persona from blueprints

Demonstrates:
- How behavioral patterns reveal student characteristics
- Robustness to noise in real-world data
- Practical application of pattern classification

Input:  shadow_simulation/pattern_classifier.pkl, shadow_simulation/blueprints.json
Output: Console report with inferred persona and confidence score
"""

import json
import joblib
import pandas as pd
import numpy as np
import random
from collections import Counter
from datetime import datetime, timedelta

# --- CONFIGURATION ---
CLASSIFIER_FILE = "shadow_simulation/pattern_classifier.pkl"
BLUEPRINTS_FILE = "shadow_simulation/blueprints.json"

def load_resources():
    print("Loading resources...")
    classifier_data = joblib.load(CLASSIFIER_FILE)
    classifier = classifier_data["model"]
    feature_names = classifier_data["features"]
    
    with open(BLUEPRINTS_FILE, 'r') as f:
        blueprints = json.load(f)
        
    return classifier, feature_names, blueprints

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

def generate_year_of_logs(target_persona_blueprints, other_blueprints):
    """
    Generates 1 year of logs. 
    80% of the time, it uses the 'Target Persona' blueprints.
    20% of the time, it uses random 'General' blueprints (noise).
    """
    logs = []
    start_date = datetime(2024, 1, 1)
    current_date = start_date
    
    print(f"Simulating 1 year of activity...")
    
    while current_date < datetime(2025, 1, 1):
        # 1-3 sessions per day
        num_sessions = random.randint(1, 3)
        
        for _ in range(num_sessions):
            # Pick blueprint
            if random.random() < 0.8:
                bp = random.choice(target_persona_blueprints)
            else:
                bp = random.choice(other_blueprints)
                
            # Generate events (simplified version of generate_logs logic)
            session_events = []
            hour = random.randint(8, 22)
            session_start = current_date.replace(hour=hour, minute=random.randint(0, 59))
            
            for event in bp['usf_log']:
                event_time = session_start + timedelta(minutes=event['time_offset'])
                session_events.append({
                    "timestamp": event_time.isoformat(),
                    "app": event['app']
                })
            
            logs.append(session_events)
            
        current_date += timedelta(days=1)
        
    return logs

def infer_persona(logs, classifier, feature_names, blueprints):
    """
    Infers the persona by analyzing the distribution of patterns.
    """
    print(f"Analyzing {len(logs)} sessions...")
    
    # 1. Classify all sessions
    pattern_counts = Counter()
    
    for session in logs:
        X = extract_features(session, feature_names)
        predicted_label = classifier.predict(X)[0]
        pattern_counts[predicted_label] += 1
        
    print("\n--- Detected Pattern Signature (Top 5) ---")
    total_sessions = len(logs)
    for pattern, count in pattern_counts.most_common(5):
        percentage = (count / total_sessions) * 100
        print(f"  {pattern}: {percentage:.1f}%")
        
    # 2. Find Best Matching Persona in Blueprints
    # We look for the persona whose 'context_type' appears most frequently in our detected patterns.
    # Since blueprints have unique personas but shared context types (sometimes), this is an approximation.
    
    print("\n--- Inferring Persona ---")
    
    # Score each blueprint based on how often its context_type appears in the user's history
    best_score = -1
    best_bp = None
    
    for bp in blueprints:
        context = bp.get('context_type')
        # Score = Count of this context in user history
        score = pattern_counts.get(context, 0)
        
        if score > best_score:
            best_score = score
            best_bp = bp
            
    if best_bp:
        print(f">>> MATCH FOUND!")
        print(f"Most likely Persona: {best_bp['persona']}")
        print(f"Major: {best_bp['major']}")
        print(f"Dominant Context: {best_bp['context_type']}")
        print(f"Confidence Score: {best_score} matching sessions")
    else:
        print("Could not determine persona.")

def main():
    clf, feats, blueprints = load_resources()
    
    # 1. Pick a Target Persona to Simulate
    # Let's pick a distinct one, e.g., a CS Major or Arts Major
    # We filter blueprints by major to create a "Persona Set"
    
    target_major = "Computer Science"
    target_blueprints = [b for b in blueprints if b['major'] == target_major]
    other_blueprints = [b for b in blueprints if b['major'] != target_major]
    
    if not target_blueprints:
        print(f"No blueprints found for {target_major}. Picking random.")
        target_blueprints = [blueprints[0]]
        other_blueprints = blueprints[1:]
        
    print(f"\nTARGET PERSONA: Student majoring in {target_major}")
    print(f"(Using {len(target_blueprints)} specific blueprints)")
    
    # 2. Generate 1 Year of Logs for this student
    user_logs = generate_year_of_logs(target_blueprints, other_blueprints)
    
    # 3. Infer Persona from the logs
    infer_persona(user_logs, clf, feats, blueprints)

if __name__ == "__main__":
    main()
