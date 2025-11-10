import json
import random
import uuid
import os
import time
import hashlib
from datetime import datetime, timedelta
from faker import Faker
from azure.eventhub import EventHubProducerClient, EventData

# --- CONFIGURATION ---
# IMPORTANT: Set these as environment variables in a real application
EVENT_HUB_CONNECTION_STRING = ""
EVENT_HUB_NAME = ""
PII_SALT = os.getenv("PII_SALT")

NUM_USERS = 50
SIMULATION_DURATION_MINUTES = 1 # Run for a shorter time for a live demo
EVENTS_PER_MINUTE = 10 # Control the rate of event generation

# --- (Persona and Location definitions remain the same as before) ---
LOCATIONS = { "ENG_Building": "131.247.34.", "Admin_Building_ADM": "131.247.1.", "Library_LIB": "131.247.12.", "Fine_Arts_FAH": "131.247.40.", "USF_Health_MDC": "131.247.60.", "Juniper_Poplar_Hall": "131.247.112.", "Off_Campus_Tampa": "72.229.28."}
PERSONAS = {"engineering_junior": {"apps": {"Canvas": 0.3, "MyUSF (OASIS)": 0.2, "MATLAB": 0.15, "SolidWorks": 0.1, "IEEE Xplore": 0.1, "GitHub": 0.1, "Outlook": 0.05},"locations": {"ENG_Building": 0.5, "Library_LIB": 0.2, "Juniper_Poplar_Hall": 0.2, "Off_Campus_Tampa": 0.1},"activity_schedule": lambda dt: random.uniform(0.1, 0.9)},"admin_employee": {"apps": {"Outlook": 0.4, "Microsoft Teams": 0.3, "MyUSF (GEMS/HR)": 0.2, "Finance Portal": 0.1},"locations": {"Admin_Building_ADM": 0.8, "Off_Campus_Tampa": 0.2},"activity_schedule": lambda dt: 1.0 if dt.weekday() < 5 and 8 <= dt.hour < 17 else 0.0},"arts_freshman": {"apps": {"Canvas": 0.5, "MyUSF (OASIS)": 0.2, "JSTOR": 0.15, "Library Catalog": 0.1, "Outlook": 0.05},"locations": {"Fine_Arts_FAH": 0.4, "Library_LIB": 0.3, "Juniper_Poplar_Hall": 0.3},"activity_schedule": lambda dt: 0.7 if 9 <= dt.hour < 22 else 0.1},"medical_resident": {"apps": {"USF Health Portal": 0.4, "Microsoft Teams": 0.2, "PubMed": 0.2, "UpToDate": 0.1, "Canvas": 0.1},"locations": {"USF_Health_MDC": 0.7, "Off_Campus_Tampa": 0.3},"activity_schedule": lambda dt: random.uniform(0.2, 0.6)}}


def deterministic_hash(value: str, salt: str) -> str:
    """Create a deterministic SHA-256 hash using the provided salt."""
    if value is None:
        return ""
    hasher = hashlib.sha256()
    hasher.update(f"{salt}:{value}".encode("utf-8"))
    return hasher.hexdigest()


def mask_email(email: str, salt: str) -> str:
    """Hash the email local-part while keeping the domain for debugging."""
    if not email:
        return email
    if "@" not in email:
        return deterministic_hash(email, salt)
    local_part, domain = email.split("@", 1)
    masked_local = deterministic_hash(local_part.lower(), salt)[:12]
    return f"{masked_local}@{domain}"


def mask_ip(ip_address: str, salt: str) -> str:
    """Keep IPv4 subnet visible while masking the host with a hash prefix."""
    if not ip_address:
        return ip_address
    octets = ip_address.split(".")
    if len(octets) == 4:
        subnet = ".".join(octets[:3])
        masked_host = deterministic_hash(octets[3], salt)[:4]
        return f"{subnet}.{masked_host}"
    # Fallback for non IPv4 addresses
    return deterministic_hash(ip_address, salt)[:12]


def mask_user_id(user_id: str, salt: str) -> str:
    """Fully hash user identifiers to prevent reverse engineering."""
    if not user_id:
        return user_id
    return deterministic_hash(user_id, salt)


def mask_log_entry(log_entry: dict, salt: str) -> dict:
    """Return a new log entry with all PII fields masked."""
    masked_entry = log_entry.copy()
    masked_entry["userPrincipalName"] = mask_email(log_entry.get("userPrincipalName"), salt)
    masked_entry["userId"] = mask_user_id(log_entry.get("userId"), salt)
    masked_entry["ipAddress"] = mask_ip(log_entry.get("ipAddress"), salt)
    return masked_entry

def generate_users(num_users):
    fake = Faker()
    users = []
    persona_names = list(PERSONAS.keys())
    for _ in range(num_users):
        persona = random.choice(persona_names)
        users.append({"userId": str(uuid.uuid4()), "userPrincipalName": fake.unique.email(domain="usf.edu"), "displayName": fake.name(), "persona": persona})
    return users

def create_signin_log(user, timestamp):
    persona_profile = PERSONAS[user["persona"]]
    app = random.choices(list(persona_profile["apps"].keys()), weights=persona_profile["apps"].values(), k=1)[0]
    location_name = random.choices(list(persona_profile["locations"].keys()), weights=persona_profile["locations"].values(), k=1)[0]
    ip_base = LOCATIONS[location_name]
    ip_address = ip_base + str(random.randint(1, 254))
    status = {"errorCode": 0, "failureReason": "Success"}
    if random.random() < 0.05:
        status = {"errorCode": 50126, "failureReason": "InvalidUserNameOrPassword"}
    return {"id": str(uuid.uuid4()), "createdDateTime": timestamp.isoformat() + "Z", "userPrincipalName": user["userPrincipalName"], "userId": user["userId"], "appDisplayName": app, "ipAddress": ip_address, "location": {"city": "Tampa", "state": "Florida", "countryOrRegion": "US"}, "status": status}

def main():
    """Main function to generate logs and send them to Event Hub in real-time."""
    if "YOUR_EVENT_HUB" in EVENT_HUB_CONNECTION_STRING:
        print("ERROR: Please set your Event Hub connection string and name in the script or as environment variables.")
        return
    if not PII_SALT:
        print("ERROR: Please set the PII_SALT environment variable for deterministic hashing.")
        return

    # Create a producer client to send messages to the event hub.
    producer = EventHubProducerClient.from_connection_string(
        conn_str=EVENT_HUB_CONNECTION_STRING, eventhub_name=EVENT_HUB_NAME
    )
    
    users = generate_users(NUM_USERS)
    start_time = time.time()
    end_time = start_time + (SIMULATION_DURATION_MINUTES * 60)
    
    print(f"Streaming synthetic sign-in logs to '{EVENT_HUB_NAME}' for {SIMULATION_DURATION_MINUTES} minutes...")
    
    try:
        event_data_batch = producer.create_batch()
        event_count = 0
        
        while time.time() < end_time:
            for _ in range(EVENTS_PER_MINUTE):
                # Pick a random user and generate a log
                user = random.choice(users)
                log_entry = create_signin_log(user, datetime.now())
                masked_log_entry = mask_log_entry(log_entry, PII_SALT)
                
                # Convert the log entry to a sendable EventData object
                event_data = EventData(json.dumps(masked_log_entry).encode("UTF-8"))
                
                try:
                    event_data_batch.add(event_data)
                except ValueError:
                    # Batch is full, send it and create a new one
                    producer.send_batch(event_data_batch)
                    print(f"Sent batch of {len(event_data_batch)} events.")
                    event_count += len(event_data_batch)
                    event_data_batch = producer.create_batch()
                    event_data_batch.add(event_data)

            time.sleep(60) # Wait a minute before generating the next burst of events
            
        # Send any remaining events in the final batch
        if len(event_data_batch) > 0:
            producer.send_batch(event_data_batch)
            print(f"Sent final batch of {len(event_data_batch)} events.")
            event_count += len(event_data_batch)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the producer client.
        producer.close()
        print(f"Finished streaming. Total events sent: {event_count}.")


if __name__ == "__main__":
    main()
