import os
import json
import random
import sys
from datetime import datetime, timedelta
try:
    import ollama
except ImportError:
    print("Please install ollama: pip install ollama")
    sys.exit(1)

FULL_OUTPUT_FILE = "raw_logs/full_student_semester.json"
VISIBLE_OUTPUT_FILE = "raw_logs/usf_visible_semester.json"
MODEL = "llama3.1:latest"  # Use Ollama model

# --- APPS & CATEGORIES (Reused for context) ---
APPS = {
    "COMMUNICATION": ["Outlook", "Teams"],
    "PRODUCTIVITY": ["Word Online", "Excel Online", "PowerPoint Online", "OneDrive", "OneNote", "SharePoint"],
    "LMS": ["Canvas", "MyUSF"],
    "ADMIN": ["OASIS", "DegreeWorks", "Archivum", "Advisor Appointments", "Schedule Planner"],
    "CAREER": ["Handshake", "LinkedIn", "LinkedIn Learning"],
    "RESEARCH": ["Library Database", "Google Scholar"],
    "EXAM_PROCTORING": ["Respondus LockDown Browser", "Honorlock", "Turnitin"],
    "CLASSROOM": ["TopHat"],
    "CLUBS": ["BullsConnect"],
    "PERSONAL": ["Instagram", "TikTok", "Snapchat", "YouTube", "Netflix", "Spotify", "Steam", "Discord", "Twitch", "iMessage", "WhatsApp", "Uber Eats", "Amazon"],
    "SPECIALIZED": {
        "Computer Science": ["GitHub", "MATLAB Online", "Copilot", "StackOverflow"],
        "Engineering": ["MATLAB Online", "AutoCAD Web", "Copilot"],
        "Arts": ["Adobe Creative Cloud", "Adobe Photoshop", "Adobe Illustrator", "Behance", "Copilot"],
        "Business": ["Excel Online", "PowerBI", "Copilot"],
        "Pre-Med": ["Khan Academy", "Kaplan", "Anki", "Copilot"],
        "Psychology": ["Qualtrics", "Copilot"],
        "Biology": ["LabArchives", "PubMed", "Copilot"],
        "General": ["Copilot"]
    }
}

# Flatten visible apps
USF_VISIBLE_APPS = {"Microsoft 365 Sign-in", "Microsoft 365 Sign-out"}
for category, apps in APPS.items():
    if isinstance(apps, list):
        USF_VISIBLE_APPS.update(apps)
    elif isinstance(apps, dict):
        for sub_apps in apps.values():
            USF_VISIBLE_APPS.update(sub_apps)

class SemesterGenerator:
    def __init__(self):
        # Ollama doesn't require API keys - uses local models
        pass
            
    def generate_persona(self, force_major=None):
        """Generates a detailed student persona."""
        major_instruction = f"Major: {force_major}" if force_major else "Major (Pick one: Computer Science, Arts, Business, Biology, Engineering)"
        
        prompt = f"""
        Create a UNIQUE and detailed persona for a University of South Florida (USF) student.
        
        Requirements:
        - {major_instruction}
        - Year: Choose from Freshman, Sophomore, Junior, Senior
        - Work Ethic: Choose from Procrastinator, High Achiever, Balanced, Perfectionist, Laid-back
        - Personality Traits: Choose 2-3 unique traits (e.g., Night Owl, Gamer, Athlete, Club Officer, Social Butterfly, Introvert, Overachiever, Artist, Musician, etc.)
        - Courses: List 4 specific courses relevant to their major
        - Description: Write a UNIQUE 1-2 sentence description that reflects their major, personality, and work style
        
        IMPORTANT: Make each student different! Vary their personality, work ethic, and description based on their major.
        - Biology students might be lab-focused, pre-med, or research-oriented
        - Engineering students might be hands-on, design-focused, or analytical
        - Business students might be entrepreneurial, networking-focused, or finance-oriented
        - Arts students might be creative, portfolio-driven, or exhibition-focused
        - Computer Science students might be coders, theorists, or startup-minded
        
        Output strictly valid JSON (DO NOT copy this example, create your own):
        {{
            "major": "<the major>",
            "year": "<year level>",
            "work_ethic": "<work ethic type>",
            "traits": ["<trait1>", "<trait2>"],
            "courses": ["<course1>", "<course2>", "<course3>", "<course4>"],
            "description": "<unique description here>"
        }}
        """
        
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a creative writer. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            format="json"
        )
        return json.loads(response['message']['content'])

    def generate_semester_plan(self, persona):
        """Generates a 12-week academic calendar."""
        prompt = f"""
        Create a 12-week academic calendar for this student:
        {json.dumps(persona, indent=2)}
        
        For each week, list the "Key Events" (Assignments, Midterms, Projects, Social Events, Breaks).
        Ensure the schedule reflects their major (e.g., CS has coding projects, Arts has portfolios).
        Include a mix of busy weeks and lighter weeks.
        
        Output strictly valid JSON:
        {{
            "weeks": [
                {{
                    "week_num": 1,
                    "theme": "Syllabus Week",
                    "key_events": ["First day of classes", "Club Fair"]
                }},
                ...
            ]
        }}
        """
        
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an academic planner. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            format="json"
        )
        return json.loads(response['message']['content'])

    def generate_weekly_logs(self, persona, week_plan, start_date):
        """Generates logs for a specific week."""
        
        # Filter relevant apps for this major
        relevant_apps = APPS["SPECIALIZED"].get(persona['major'], APPS["SPECIALIZED"]["General"])
        relevant_apps.extend(APPS["LMS"] + APPS["PRODUCTIVITY"] + APPS["COMMUNICATION"] + APPS["PERSONAL"])
        
        prompt = f"""
        Generate a realistic digital activity log for **Week {week_plan['week_num']}** ({week_plan['theme']}).
        
        **Student Persona:**
        {json.dumps(persona, indent=2)}
        
        **Week Context:**
        {json.dumps(week_plan['key_events'], indent=2)}
        
        **Instructions:**
        1. **CRITICAL**: Include "Personal" apps (Social Media, Gaming, Streaming) to reflect their lifestyle. 
           - If they are a "Gamer", show Steam/Discord.
           - If they are a "Procrastinator", show Netflix/TikTok binging before deadlines.
        2. Use specific apps from this list: {', '.join(relevant_apps)}.
        3. Include "Microsoft 365 Sign-in" and "Microsoft 365 Sign-out" events ONLY when accessing USF resources (Canvas, Outlook, Teams). Personal apps do NOT need USF login.
        4. Create a list of events for the entire week (Mon-Sun).
        5. The `timestamp` should be relative to the start of the week (Day 0 = Monday).
        
        **Output strictly valid JSON:**
        {{
            "events": [
                {{
                    "day_offset": 0,
                    "time": "09:00",
                    "app": "Instagram",
                    "activity": "Scrolling in bed"
                }},
                {{
                    "day_offset": 0,
                    "time": "09:30",
                    "app": "Microsoft 365 Sign-in",
                    "activity": "Logging in to check email"
                }},
                ...
            ]
        }}
        """
        
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a log generator. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            format="json"
        )
        
        result = json.loads(response['message']['content'])
        
        # Post-process timestamps
        final_events = []
        week_start = datetime.fromisoformat(start_date)
        
        for event in result.get('events', []):
            try:
                day_offset = event['day_offset']
                time_str = event['time']
                hour, minute = map(int, time_str.split(':'))
                
                event_dt = week_start + timedelta(days=day_offset)
                event_dt = event_dt.replace(hour=hour, minute=minute, second=0)
                
                final_events.append({
                    "timestamp": event_dt.isoformat(),
                    "app": event['app'],
                    "details": event['activity'],
                    "week": week_plan['week_num'],
                    "context": week_plan['theme'] # Ground truth context
                })
            except Exception as e:
                print(f"Skipping malformed event: {event} ({e})")
                
        return final_events

def main():
    generator = SemesterGenerator()
    
    print("1. Generating Persona...")
    persona = generator.generate_persona()
    print(f"   > {persona['description']}")
    
    print("2. Generating Semester Plan...")
    plan = generator.generate_semester_plan(persona)
    print(f"   > Generated {len(plan['weeks'])} weeks.")
    
    all_logs = []
    semester_start = datetime(2024, 8, 26) # Fall 2024 start
    
    print("3. Generating Weekly Logs...")
    for week in plan['weeks']:
        week_num = week['week_num']
        print(f"   > Processing Week {week_num}: {week['theme']}...")
        
        week_start_date = semester_start + timedelta(weeks=week_num-1)
        logs = generator.generate_weekly_logs(persona, week, week_start_date.isoformat())
        all_logs.extend(logs)
        
    # Save Full Logs
    full_output = {
        "persona": persona,
        "semester_plan": plan,
        "logs": all_logs
    }
    
    os.makedirs(os.path.dirname(FULL_OUTPUT_FILE), exist_ok=True)
    with open(FULL_OUTPUT_FILE, 'w') as f:
        json.dump(full_output, f, indent=2)
        
    # Save USF-Visible Logs (Filter out PERSONAL apps)
    visible_logs = [
        event for event in all_logs 
        if event['app'] in USF_VISIBLE_APPS or event['app'] in APPS["SPECIALIZED"].get(persona['major'], [])
    ]
    
    visible_output = {
        "persona": persona,
        "semester_plan": plan,
        "logs": visible_logs
    }
    
    with open(VISIBLE_OUTPUT_FILE, 'w') as f:
        json.dump(visible_output, f, indent=2)
        
    print(f"\nSuccess! Generated {len(all_logs)} total events.")
    print(f"  -> Full History: {FULL_OUTPUT_FILE}")
    print(f"  -> USF Visible:  {VISIBLE_OUTPUT_FILE} ({len(visible_logs)} events)")

if __name__ == "__main__":
    main()
