import os
import json
import time
from datetime import datetime, timedelta
from generate_semester_logs import SemesterGenerator, APPS, USF_VISIBLE_APPS

import random

# --- CONFIGURATION ---
COHORT_SIZE = 10
OUTPUT_FILE = "raw_logs/cohort_semester.json"
VISIBLE_OUTPUT_FILE = "raw_logs/cohort_visible_semester.json"

def main():
    generator = SemesterGenerator()
    cohort_data = []
    
    print(f"Starting generation for cohort of {COHORT_SIZE} students...")
    
    semester_start = datetime(2024, 8, 26) # Fall 2024 start
    
    # USF-like Major Distribution (Approximate for 10 students)
    # Biology/Pre-Med: 30%
    # Business: 20%
    # Engineering: 20%
    # Computer Science: 20%
    # Arts: 10%
    major_pool = (
        ["Biology"] * 3 +
        ["Business"] * 2 +
        ["Engineering"] * 2 +
        ["Computer Science"] * 2 +
        ["Arts"] * 1
    )
    random.shuffle(major_pool)
    
    for i in range(COHORT_SIZE):
        major = major_pool[i % len(major_pool)]
        print(f"\n--- Generating Student {i+1}/{COHORT_SIZE} ({major}) ---")
        
        # 1. Generate Persona
        print("   Generating Persona...")
        persona = generator.generate_persona(force_major=major)
        print(f"   > {persona['major']} | {persona['work_ethic']} | {persona['description'][:50]}...")
        
        # 2. Generate Plan
        print("   Generating Semester Plan...")
        plan = generator.generate_semester_plan(persona)
        
        # 3. Generate Logs
        print("   Generating Weekly Logs...")
        student_logs = []
        for week in plan['weeks']:
            # print(f"      Week {week['week_num']}...", end="\r")
            week_start_date = semester_start + timedelta(weeks=week['week_num']-1)
            logs = generator.generate_weekly_logs(persona, week, week_start_date.isoformat())
            student_logs.extend(logs)
            
        print(f"   > Generated {len(student_logs)} events.")
        
        # Add Student ID to logs
        student_id = f"student_{i+1:03d}"
        for log in student_logs:
            log['user'] = student_id
            
        cohort_data.append({
            "student_id": student_id,
            "persona": persona,
            "semester_plan": plan,
            "logs": student_logs
        })
        
        # Sleep briefly to avoid rate limits if necessary
        time.sleep(1)

    # Save Full Cohort
    print(f"\nSaving full cohort data to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(cohort_data, f, indent=2)
        
    # Save Visible Cohort (Filtered)
    print(f"Saving visible cohort data to {VISIBLE_OUTPUT_FILE}...")
    visible_cohort = []
    for student in cohort_data:
        visible_logs = [
            event for event in student['logs']
            if event['app'] in USF_VISIBLE_APPS or event['app'] in APPS["SPECIALIZED"].get(student['persona']['major'], [])
        ]
        visible_cohort.append({
            "student_id": student['student_id'],
            "persona": student['persona'],
            "semester_plan": student['semester_plan'],
            "logs": visible_logs
        })
        
    with open(VISIBLE_OUTPUT_FILE, 'w') as f:
        json.dump(visible_cohort, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
