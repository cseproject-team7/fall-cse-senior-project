import os
import json
import time
import hashlib
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# Load environment variables
load_dotenv('server/.env')

CONTAINER_NAME = 'json-signin-logs'
CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

def anonymize_user(user_id):
    """Generate anonymized user ID and display name (Consistent with generate_azure_logs.py)"""
    raw_id = f"student{user_id}@usf.edu"
    hash_obj = hashlib.sha256(raw_id.encode())
    short_hash = hash_obj.hexdigest()[:8]
    return f"User {short_hash}", short_hash

def inject_logs(user_id, apps):
    if not CONNECTION_STRING:
        print("❌ AZURE_STORAGE_CONNECTION_STRING not found")
        return

    user_display, user_hash = anonymize_user(user_id)
    print(f"Injecting logs for User ID: {user_id} ({user_display})")
    print(f"Apps: {apps}")

    current_time = datetime.now()
    logs = []

    for app in apps:
        # Add random gap
        current_time += timedelta(minutes=random.randint(1, 10))
        
        log_entry = {
            "userPrincipalName": user_display,
            "userId": user_hash,
            "appDisplayName": app,
            "createdDateTime": current_time.isoformat() + 'Z'
        }
        
        azure_log = {
            "Body": json.dumps(log_entry)
        }
        logs.append(azure_log)

    # Save to temporary file
    output_file = f"manual_logs_{user_id}_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        for log in logs:
            f.write(json.dumps(log) + '\n')

    # Upload
    try:
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        blob_name = output_file
        print(f"Uploading to {blob_name}...")
        with open(output_file, "rb") as data:
            container_client.upload_blob(name=blob_name, data=data)
            
        print(f"✅ Successfully uploaded {len(logs)} logs to {blob_name}")
        
        # Cleanup
        os.remove(output_file)
        
    except Exception as e:
        print(f"❌ Error uploading: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Inject logs for a specific user.')
    parser.add_argument('--user-id', type=int, help='User ID (integer)')
    parser.add_argument('--apps', nargs='+', help='List of apps to inject')
    
    args = parser.parse_args()
    
    user_id = args.user_id
    apps = args.apps
    
    # Interactive mode if args not provided
    if user_id is None:
        try:
            user_input = input("Enter User ID (default 1): ")
            user_id = int(user_input) if user_input.strip() else 1
        except ValueError:
            print("Invalid User ID. Using default: 1")
            user_id = 1
            
    if apps is None:
        apps_input = input("Enter apps separated by comma (default 'Canvas, Outlook'): ")
        if apps_input.strip():
            apps = [app.strip() for app in apps_input.split(',')]
        else:
            apps = ['Canvas', 'Outlook']
            
    inject_logs(user_id, apps)
