"""
Step 3: Train Random Forest and Apply to All Sessions

Input: 
- prepared_data/labeled_sessions.jsonl (10k labeled)
- prepared_data/sessions.jsonl (all 20M)
Output: prepared_data/distilled_sequences.csv

Memory: O(batch_size) - processes sessions in batches
"""

import json
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def extract_features_batch(sessions):
    """Extract features from a batch of sessions."""
    data = []
    for s in sessions:
        timestamps = [datetime.fromisoformat(t) for t in s['timestamps']]
        duration = sum([(timestamps[i+1] - timestamps[i]).total_seconds()/60 for i in range(len(timestamps)-1)]) if len(timestamps) > 1 else 5.0
        
        app_str = " ".join([app.replace(" ", "_") for app in s['apps']])
        
        hour = timestamps[0].hour
        weekday = timestamps[0].weekday()
        is_weekend = 1 if weekday >= 5 else 0
        
        devices = s.get('devices', [])
        is_mobile = 1 if any("iOS" in d or "Android" in d for d in devices) else 0
        device_switches = len(set(devices)) - 1
        
        ips = s.get('ips', [])
        is_on_campus = 1 if any(ip.startswith("131.247") for ip in ips) else 0
        ip_switches = len(set(ips)) - 1
        
        auth_events = s.get('auth_events', [])
        has_auth_issues = 1 if len(auth_events) > 0 else 0
        
        data.append({
            'app_str': app_str,
            'duration': duration,
            'hour': hour,
            'weekday': weekday,
            'is_weekend': is_weekend,
            'n_apps': len(s['apps']),
            'is_mobile': is_mobile,
            'device_switches': device_switches,
            'is_on_campus': is_on_campus,
            'ip_switches': ip_switches,
            'has_auth_issues': has_auth_issues
        })
    return pd.DataFrame(data)

def train_model(labeled_sessions_file):
    """Train Random Forest on labeled sessions (Pattern and Key Apps)."""
    print("\n=== Step 3: Train and Apply ===")
    print(f"Loading labeled sessions from {labeled_sessions_file}...")
    
    labeled_sessions = []
    with open(labeled_sessions_file, 'r') as f:
        for line in f:
            labeled_sessions.append(json.loads(line))
    
    print(f"Loaded {len(labeled_sessions):,} labeled sessions")
    
    # Extract features
    print("Extracting features...")
    df = extract_features_batch(labeled_sessions)
    
    # Parse labels (PATTERN | KEY_APPS)
    raw_labels = [s.get('pattern', 'UNKNOWN | UNKNOWN') for s in labeled_sessions]
    patterns = []
    apps = []
    
    for label in raw_labels:
        if '|' in label:
            p, a = label.split('|', 1)
            patterns.append(p.strip())
            apps.append(a.strip())
        else:
            patterns.append(label.strip())
            apps.append("Unknown")
            
    # Train/test split
    vectorizer = CountVectorizer()
    X_apps = vectorizer.fit_transform(df['app_str']).toarray()
    X_numeric = df[['duration', 'hour', 'weekday', 'is_weekend', 'n_apps', 'is_mobile', 'device_switches', 'is_on_campus', 'ip_switches', 'has_auth_issues']].values
    
    X = np.hstack((X_apps, X_numeric))
    
    # Train Pattern Model
    print("Training Pattern Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, patterns, test_size=0.2, random_state=42)
    clf_pattern = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_pattern.fit(X_train, y_train)
    print(f"Pattern Accuracy: {clf_pattern.score(X_test, y_test):.4f}")
    
    # Train Key Apps Model
    print("Training Key Apps Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, apps, test_size=0.2, random_state=42)
    clf_app = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_app.fit(X_train, y_train)
    print(f"Key Apps Accuracy: {clf_app.score(X_test, y_test):.4f}")
    
    # Retrain on full dataset
    clf_pattern.fit(X, patterns)
    clf_app.fit(X, apps)
    
    return clf_pattern, clf_app, vectorizer

def apply_model_streaming(clf_pattern, clf_app, vectorizer, sessions_file, output_file, batch_size=10000):
    """Apply model to all sessions in batches."""
    print(f"\nApplying model to all sessions (batch size: {batch_size:,})...")
    
    # First pass: Apply classifier
    print("Pass 1: Classifying sessions...")
    predictions_file = output_file.replace('.csv', '_predictions.jsonl')
    
    batch = []
    total_processed = 0
    
    with open(sessions_file, 'r') as infile, open(predictions_file, 'w') as outfile:
        for line in infile:
            session = json.loads(line)
            batch.append(session)
            
            if len(batch) >= batch_size:
                # Process batch
                df = extract_features_batch(batch)
                X_apps = vectorizer.transform(df['app_str']).toarray()
                X_numeric = df[['duration', 'hour', 'weekday', 'is_weekend', 'n_apps', 'is_mobile', 'device_switches', 'is_on_campus', 'ip_switches', 'has_auth_issues']].values
                X = np.hstack((X_apps, X_numeric))
                
                preds_pattern = clf_pattern.predict(X)
                preds_app = clf_app.predict(X)
                
                # Write predictions
                for s, p, a in zip(batch, preds_pattern, preds_app):
                    s['pattern'] = p
                    s['key_apps'] = a
                    outfile.write(json.dumps(s) + '\n')
                
                total_processed += len(batch)
                print(f"  Processed {total_processed:,} sessions...")
                batch = []
        
        # Process remaining
        if batch:
            df = extract_features_batch(batch)
            X_apps = vectorizer.transform(df['app_str']).toarray()
            X_numeric = df[['duration', 'hour', 'weekday', 'is_weekend', 'n_apps', 'is_mobile', 'device_switches', 'is_on_campus', 'ip_switches', 'has_auth_issues']].values
            X = np.hstack((X_apps, X_numeric))
            
            preds_pattern = clf_pattern.predict(X)
            preds_app = clf_app.predict(X)
            
            for s, p, a in zip(batch, preds_pattern, preds_app):
                s['pattern'] = p
                s['key_apps'] = a
                outfile.write(json.dumps(s) + '\n')
            
            total_processed += len(batch)
    
    print(f"✓ Classified {total_processed:,} sessions")
    
    # Second pass: Write pattern-level sequences (NOT flattened)
    print("\nPass 2: Writing pattern-level sequences...")
    
    patterns_written = 0
    
    with open(predictions_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            pat = json.loads(line)
            
            # Keep pattern as complete sequence
            output_pattern = {
                'user': pat['user'],
                'pattern_id': pat.get('pattern_id', f"{pat['user']}_{pat['timestamps'][0]}"),
                'apps': pat['apps'],
                'timestamps': pat['timestamps'],
                'pattern_label': pat['pattern']
            }
            
            outfile.write(json.dumps(output_pattern) + '\n')
            patterns_written += 1
    
    # Clean up intermediate file
    os.remove(predictions_file)
    
    print(f"\n✓ Complete!")
    print(f"  Output: {output_file}")
    print(f"  Patterns: {patterns_written:,}")

if __name__ == "__main__":
    import sys
    
    # Check if running in Colab
    IN_COLAB = 'google.colab' in sys.modules
    
    if IN_COLAB:
        print("Running in Google Colab environment")
        
        # Default settings for Colab
        # Default settings for Colab
        # Default settings for Colab
        labeled_file = "prepared_data/labeled_patterns.jsonl"
        sessions_file = "prepared_data/patterns.jsonl"
        output_file = "prepared_data/all_labeled_patterns.jsonl"
        batch_size = 5000  # Smaller batch for Colab RAM
        
        os.makedirs("prepared_data", exist_ok=True)
        
        # Train model
        clf_pattern, clf_app, vectorizer = train_model(labeled_file)
        
        # Apply to all sessions
        apply_model_streaming(clf_pattern, clf_app, vectorizer, sessions_file, output_file, batch_size)
    else:
        import argparse
        parser = argparse.ArgumentParser(description="Step 3: Train RF and apply to all sessions")
        parser.add_argument("--labeled", default="prepared_data/labeled_patterns.jsonl", help="Labeled patterns file")
        parser.add_argument("--patterns", default="prepared_data/patterns.jsonl", help="All patterns file")
        parser.add_argument("--output", default="prepared_data/all_labeled_patterns.jsonl", help="Output labeled patterns file")
        parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for inference")
        args = parser.parse_args()
        
        os.makedirs("prepared_data", exist_ok=True)
        
        # Train model
        clf_pattern, clf_app, vectorizer = train_model(args.labeled)
        
        # Apply to all sessions
        apply_model_streaming(clf_pattern, clf_app, vectorizer, args.patterns, args.output, args.batch_size)
