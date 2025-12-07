"""
Step 2: LLM Labeling of Sampled Sessions

Input: prepared_data/sessions.jsonl (all sessions)
Output: prepared_data/labeled_sessions.jsonl (10k labeled sessions)

Memory: O(sample_size) - only loads sampled sessions
"""

import json
import os
import random
import argparse
from datetime import datetime

# --- LLM BACKENDS ---

class LLMBackend:
    def generate_label(self, session):
        raise NotImplementedError

class OpenAIBackend(LLMBackend):
    def __init__(self, api_key, model="gpt-4o-mini"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
            print(f"✓ Connected to OpenAI ({model})")
        except ImportError:
            print("Error: 'openai' library not found. Install: pip install openai")
            self.client = None

    def generate_label(self, session):
        if not self.client: return "ERROR"
        
        user_id = session.get('user', 'Unknown')
        apps = session['apps']
        timestamps = [datetime.fromisoformat(t) for t in session['timestamps']]
        devices = session.get('devices', [])
        ips = session.get('ips', [])
        auth_events = session.get('auth_events', [])
        
        duration = sum([(timestamps[i+1] - timestamps[i]).total_seconds()/60 for i in range(len(timestamps)-1)]) if len(timestamps) > 1 else 5.0
        hour = timestamps[0].hour
        weekday = timestamps[0].strftime("%A")
        
        device_type = "Mobile" if any("iOS" in d or "Android" in d for d in devices) else "Desktop"
        ip_pattern = "On-Campus" if any(ip.startswith("131.247") for ip in ips) else "Off-Campus"
        has_auth_issues = len(auth_events) > 0
        
        prompt = f"""You are an expert student behavior analyst at USF.
Analyze this sequence of app usage (a 'Pattern') to determine the student's intent and the tools used.

Pattern Context:
- User ID: {user_id}
- App Sequence: {' -> '.join(apps)}
- Duration: {duration:.1f} mins
- Time: {weekday} at {hour}:00
- Device: {device_type} ({ip_pattern})

Categories:
- COURSEWORK: Canvas, assignments, studying, LMS
- RESEARCH: Library, Google Scholar, research papers
- CODING: GitHub, VS Code, technical development
- ADMIN: Registration, OASIS, email, scheduling
- CAREER: Handshake, LinkedIn, job applications
- SOCIAL: Teams, Outlook, communication
- CLUBS: BullsConnect, student organizations
- EXAM: Honorlock, Respondus, Kaplan, quizzes

Instructions:
1. Infer the student's persona/major strictly from the **App Sequence** (e.g., VS Code -> CS, Kaplan -> Pre-Med).
2. Determine the specific activity.
3. Select the best Category.
4. Identify the "Key Apps".

Output VALID JSON only:
{{
  "reasoning": "Brief explanation of why this category fits the app usage patterns.",
  "category": "CATEGORY_NAME",
  "key_apps": ["App1", "App2"]
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            return f"{data['category']} | {', '.join(data['key_apps'])}"
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return "ERROR"



# --- MAIN LOGIC ---

def sample_and_label(sessions_file, output_file, backend, sample_size):
    """
    Sample sessions and label them with LLM.
    """
    print(f"\n=== Step 2: LLM Labeling ===")
    print(f"Input: {sessions_file}")
    print(f"Sample size: {sample_size:,}")
    
    # Count total sessions
    print("Counting sessions...")
    with open(sessions_file, 'r') as f:
        total_sessions = sum(1 for _ in f)
    print(f"Total sessions: {total_sessions:,}")
    
    # Sample session indices
    if sample_size >= total_sessions:
        sample_indices = set(range(total_sessions))
    else:
        random.seed(42)
        sample_indices = set(random.sample(range(total_sessions), sample_size))
    
    print(f"Sampling {len(sample_indices):,} sessions...")
    
    # Load sampled sessions
    sampled_sessions = []
    with open(sessions_file, 'r') as f:
        for idx, line in enumerate(f):
            if idx in sample_indices:
                sampled_sessions.append(json.loads(line))
    
    print(f"Loaded {len(sampled_sessions):,} sessions")
    
    # Label with LLM
    print(f"Labeling with LLM...")
    labeled_count = 0
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w') as outfile:
        for i, session in enumerate(sampled_sessions):
            if (i + 1) % 100 == 0:
                print(f"  Labeled {i + 1:,}/{len(sampled_sessions):,} sessions...")
            
            label = backend.generate_label(session)
            session['pattern'] = label
            
            outfile.write(json.dumps(session) + '\n')
            labeled_count += 1
    
    print(f"\n✓ Labeling complete!")
    print(f"  Labeled: {labeled_count:,} sessions")
    print(f"  Output: {output_file}")

if __name__ == "__main__":
    import sys
    
    # Check if running in Colab
    IN_COLAB = 'google.colab' in sys.modules
    
    if IN_COLAB:
        print("Running in Google Colab environment")
        print("!pip install openai  # Ensure openai is installed")
        
        # Default settings for Colab
        input_file = "prepared_data/patterns.jsonl"
        output_file = "prepared_data/labeled_patterns.jsonl"
        os.makedirs("prepared_data", exist_ok=True)
        
        try:
            from google.colab import userdata
            api_key = userdata.get('OPENAI_API_KEY')
            backend = OpenAIBackend(api_key)
        except:
            print("Error: OPENAI_API_KEY not found in Colab secrets.")
            sys.exit(1)
                
        sample_size = 5000
        sample_and_label(input_file, output_file, backend, sample_size)
    else:
        parser = argparse.ArgumentParser(description="Step 2: LLM labeling of sampled sessions")
        parser.add_argument("--input", default="prepared_data/patterns.jsonl", help="Input patterns file")
        parser.add_argument("--output", default="prepared_data/labeled_patterns.jsonl", help="Output labeled patterns file")
        parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI Model name")
        parser.add_argument("--sample_size", type=int, default=10000, help="Number of sessions to label")
        args = parser.parse_args()
        
        # Initialize backend
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            exit(1)
            
        backend = OpenAIBackend(api_key, model=args.model)
        sample_and_label(args.input, args.output, backend, args.sample_size)
