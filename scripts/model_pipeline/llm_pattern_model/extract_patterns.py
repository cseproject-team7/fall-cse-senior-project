"""Extract Patterns from Raw Logs

This script processes raw cohort logs and extracts behavioral patterns by sessionizing
student activity into discrete learning sessions.

Key Features:
- Sessionizes logs using 20-minute inactivity gaps
- Creates pattern objects with app sequences, context, and duration
- Generates pattern signatures for each session
- Preserves temporal and contextual information (week, context)

Input: raw_logs/cohort_visible_semester.json (cohort data with logs)
Output: shadow_simulation/student_patterns.json (sessionized patterns per student)

Workflow:
1. Load cohort data with raw logs
2. For each student, sort logs by timestamp
3. Split into sessions using 20-min gap threshold
4. Create pattern objects with metadata (apps, duration, context)
5. Save student patterns with persona information
"""

import json
import pandas as pd
import os

# --- CONFIGURATION ---
INPUT_FILE = "raw_logs/cohort_visible_semester.json"
OUTPUT_FILE = "shadow_simulation/student_patterns.json"
SESSION_GAP_MINUTES = 20

def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please run generate_cohort.py first.")
        return None
        
    with open(INPUT_FILE, 'r') as f:
        return json.load(f)

def extract_patterns(student_data):
    """
    Converts a student's raw logs into a sequence of Patterns (Sessions).
    """
    logs = student_data['logs']
    if not logs:
        return []
        
    df = pd.DataFrame(logs)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    patterns = []
    current_session = []
    
    for _, row in df.iterrows():
        if not current_session:
            current_session.append(row)
            continue
            
        last_time = current_session[-1]['timestamp']
        curr_time = row['timestamp']
        
        if (curr_time - last_time).total_seconds() > SESSION_GAP_MINUTES * 60:
            # End of Session -> Create Pattern
            patterns.append(create_pattern_object(current_session))
            current_session = [row]
        else:
            current_session.append(row)
            
    if current_session:
        patterns.append(create_pattern_object(current_session))
        
    return patterns

def create_pattern_object(session_events):
    """
    Summarizes a session into a Pattern object.
    """
    apps = [e['app'] for e in session_events]
    unique_apps = sorted(list(set(apps)))
    
    # Context is consistent within a session (from generator)
    context = session_events[0].get('context', 'Unknown')
    week = session_events[0].get('week', 0)
    
    start_time = session_events[0]['timestamp']
    end_time = session_events[-1]['timestamp']
    duration = (end_time - start_time).total_seconds() / 60
    
    return {
        "timestamp": start_time.isoformat(),
        "week": week,
        "context": context,
        "duration_mins": duration,
        "apps": unique_apps,
        "sequence": apps, # The literal sequence of apps in this session
        "pattern_signature": "|".join(unique_apps) # Simple signature
    }

def main():
    print("Loading cohort data...")
    cohort = load_data()
    if not cohort:
        return

    all_student_patterns = []
    
    print(f"Extracting patterns for {len(cohort)} students...")
    
    for student in cohort:
        patterns = extract_patterns(student)
        
        all_student_patterns.append({
            "student_id": student['student_id'],
            "persona": student['persona'],
            "patterns": patterns
        })
        
        print(f"   > {student['student_id']}: {len(patterns)} patterns extracted.")
        
    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_student_patterns, f, indent=2)
        
    print(f"\nSuccess! Saved pattern sequences to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
