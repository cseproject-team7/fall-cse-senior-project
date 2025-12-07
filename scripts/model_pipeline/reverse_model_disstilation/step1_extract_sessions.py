import json
import os
import sys
"""
STEP 1: SESSION EXTRACTION FROM RAW LOGS

This script processes raw authentication logs and extracts user sessions based on
explicit sign-in/sign-out events. Each session is further split into behavioral
patterns using a 20-minute idle time threshold.

Key Features:
- Streaming log processing for memory efficiency
- Authentication-driven session boundaries (Sign-in → Activities → Sign-out)
- Automatic pattern segmentation based on activity gaps
- Metadata capture (devices, IPs, timestamps)

Input:  raw_logs/logs.json (line-delimited JSON logs)
Output: prepared_data/patterns.jsonl (extracted session patterns)
"""

from datetime import datetime
from collections import defaultdict

# --- CONFIGURATION ---
IN_COLAB = 'google.colab' in sys.modules


def stream_sessions(log_file, output_file):
    print(f"Streaming logs from {log_file}...")

    if not os.path.exists(log_file):
        print(f"❌ ERROR: {log_file} not found.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Dictionary to hold the ACTIVE session for each user
    user_sessions = defaultdict(lambda: {
        'apps': [],
        'timestamps': [],
        'devices': [],
        'ips': [],
        'auth_events': []
    })

    sessions_written = 0
    logs_processed = 0

    with open(log_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if not line.strip(): continue

            logs_processed += 1
            if logs_processed % 100000 == 0:
                print(f"Processed {logs_processed:,} logs, written {sessions_written:,} sessions...")

            try:
                envelope = json.loads(line)
                body = json.loads(envelope['Body'])

                app = body['appDisplayName']
                user = body['userPrincipalName']
                timestamp = datetime.fromisoformat(body['createdDateTime'].replace('Z', '+00:00'))
                
                # Metadata
                device_detail = body.get('deviceDetail', {})
                device_str = f"{device_detail.get('operatingSystem', 'Unk')}/{device_detail.get('browser', 'Unk')}"
                ip_address = body.get('ipAddress', 'Unknown')
                
                # Auth Status
                error_code = body.get('status', {}).get('errorCode', 0)
                auth_status = "Success" if error_code == 0 else ("MFA" if error_code == 50058 else "Failure")

                # --- NEW LOGIC: AUTHENTICATION DRIVEN SESSIONS ---

                # 1. Handle Explicit SIGN-IN (Start of new session)
                if app == "Microsoft 365 Sign-in":
                    if auth_status == "Success":
                        # If there was an open session, close it (it was left hanging)
                        if user_sessions[user]['timestamps']:
                            write_session(outfile, user, user_sessions[user])
                            sessions_written += 1
                        
                        # Start fresh session
                        user_sessions[user] = {
                            'apps': ["Microsoft 365 Sign-in"], 
                            'timestamps': [timestamp], # Track login time
                            'devices': [device_str], 
                            'ips': [ip_address], 
                            'auth_events': ["Sign-in"]
                        }
                    else:
                        # Track failed logins in the "void" or attach to previous if closely timed
                        # For simplicity, we just note it if a session exists, or ignore if not
                        if user_sessions[user]['timestamps']:
                             user_sessions[user]['auth_events'].append(f"Failed Login: {auth_status}")
                    continue

                # 2. Handle Explicit SIGN-OUT (End of session)
                if app == "Microsoft 365 Sign-out":
                    if user_sessions[user]['timestamps']:
                        # Add Sign-out as an app event
                        user_sessions[user]['apps'].append("Microsoft 365 Sign-out")
                        user_sessions[user]['timestamps'].append(timestamp)
                        user_sessions[user]['devices'].append(device_str)
                        user_sessions[user]['ips'].append(ip_address)
                        
                        # Add sign-out marker
                        user_sessions[user]['auth_events'].append("Sign-out")
                        
                        # Close and write the session
                        write_session(outfile, user, user_sessions[user])
                        sessions_written += 1
                        
                        # Clear memory for this user
                        del user_sessions[user]
                    continue

                # 3. Handle Normal App Activity
                current_session = user_sessions[user]
                
                # Edge Case: Activity without a Sign-in event (e.g. long-lived token or missed log)
                if not current_session['timestamps']:
                    current_session['timestamps'].append(timestamp)
                    current_session['auth_events'].append("Implicit Start")



                # Normal Append
                current_session['apps'].append(app)
                current_session['timestamps'].append(timestamp)
                current_session['devices'].append(device_str)
                current_session['ips'].append(ip_address)

            except Exception as e:
                continue

        # Flush remaining open sessions
        print("Writing remaining sessions...")
        for user, session in user_sessions.items():
            if session['apps'] or len(session['timestamps']) > 1:
                write_session(outfile, user, session)
                sessions_written += 1

    print(f"\n✓ Extraction complete!")
    print(f"  Processed: {logs_processed:,} logs")
    print(f"  Sessions:  {sessions_written:,}")

def write_session(outfile, user, session):
    """Helper to write JSON line"""
    # Filter out empty sessions that might occur from just a login/logout with no apps
    if not session['apps'] and len(session['auth_events']) < 2:
        return

    # Split session into patterns based on 20-minute gaps
    patterns = split_into_patterns(session)
    
    session_id = f"{user}_{session['timestamps'][0].isoformat()}"
    
    for pat in patterns:
        pat_obj = {
            'user': user,
            'session_id': session_id,
            'pattern_id': f"{session_id}_{pat['start_idx']}",
            'apps': pat['apps'],
            'timestamps': [t.isoformat() for t in pat['timestamps']],
            'devices': pat['devices'],
            'ips': pat['ips'],
            'auth_events': session['auth_events'] if pat['start_idx'] == 0 else [] # Only attach auth events to first pattern
        }
        outfile.write(json.dumps(pat_obj) + '\n')

def split_into_patterns(session):
    """Splits a session into sub-sessions (patterns) based on 20 min idle time."""
    patterns = []
    
    apps = session['apps']
    timestamps = session['timestamps']
    devices = session['devices']
    ips = session['ips']
    
    if not apps:
        return []
        
    current_pattern = {
        'apps': [apps[0]],
        'timestamps': [timestamps[0]],
        'devices': [devices[0]],
        'ips': [ips[0]],
        'start_idx': 0
    }
    
    for i in range(1, len(apps)):
        time_diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 60.0
        
        if time_diff > 20:
            # Gap found -> Save current and start new
            patterns.append(current_pattern)
            current_pattern = {
                'apps': [],
                'timestamps': [],
                'devices': [],
                'ips': [],
                'start_idx': i
            }
            
        current_pattern['apps'].append(apps[i])
        current_pattern['timestamps'].append(timestamps[i])
        current_pattern['devices'].append(devices[i])
        current_pattern['ips'].append(ips[i])
        
    patterns.append(current_pattern)
    return patterns

# --- ENTRY POINT ---
if __name__ == "__main__":
    if IN_COLAB:
        stream_sessions("raw_logs/logs.json", "prepared_data/patterns.jsonl")
    else:
        stream_sessions("raw_logs/logs.json", "prepared_data/patterns.jsonl")
