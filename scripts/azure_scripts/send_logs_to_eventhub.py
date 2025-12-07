"""
Send generated logs to Azure Event Hub
Reads logs from logs.json and sends them to Event Hub in batches
"""

import json
import time
import os
from azure.eventhub import EventHubProducerClient, EventData
from datetime import datetime

# Configuration - Set these as environment variables or replace with your values
EVENT_HUB_CONNECTION_STRING = os.getenv("EVENT_HUB_CONNECTION_STRING", "YOUR_EVENT_HUB_CONNECTION_STRING")
EVENT_HUB_NAME = os.getenv("EVENT_HUB_NAME", "signin-logs-stream")

# Batch settings
BATCH_SIZE = 100  # Send 100 logs per batch
DELAY_BETWEEN_BATCHES = 1  # Seconds between batches
MAX_LOGS_TO_SEND = 10  # Limit number of logs to send (set to None for all logs)

def load_logs(filename="logs.json"):
    """Load logs from JSON file."""
    logs = []
    print(f"Loading logs from {filename}...")
    
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    
    print(f"Loaded {len(logs)} log entries")
    
    # Limit logs if MAX_LOGS_TO_SEND is set
    if MAX_LOGS_TO_SEND is not None and len(logs) > MAX_LOGS_TO_SEND:
        print(f"Limiting to first {MAX_LOGS_TO_SEND} logs for testing")
        logs = logs[:MAX_LOGS_TO_SEND]
    
    return logs

def create_event_data(log_entry):
    """Create EventData from log entry.
    
    Sends only the Body content (the actual log data) to Event Hub,
    as the Event Hub will add its own sequence numbers and metadata.
    """
    # Extract the actual log data from the Body field
    body_data = json.loads(log_entry["Body"])
    
    # Create event with the log data
    event = EventData(json.dumps(body_data))
    
    # Add custom properties if needed
    event.properties = {
        "source": "training_pipeline",
        "persona": body_data["userPrincipalName"].split("_")[0]
    }
    
    return event

def send_logs_to_eventhub(logs, connection_string, event_hub_name):
    """Send logs to Event Hub in batches."""
    
    if not connection_string:
        print("\n❌ ERROR: Event Hub connection string not provided!")
        print("Set the EVENT_HUB_CONNECTION_STRING environment variable or update the script.")
        print("\nTo get your connection string:")
        print("  az eventhubs namespace authorization-rule keys list \\")
        print("    --resource-group <resource-group> \\")
        print("    --namespace-name <namespace-name> \\")
        print("    --name SendListenRule \\")
        print("    --query primaryConnectionString -o tsv")
        return False
    
    try:
        # Create Event Hub producer client
        print(f"\nConnecting to Event Hub: {event_hub_name}...")
        producer = EventHubProducerClient.from_connection_string(
            conn_str=connection_string,
            eventhub_name=event_hub_name
        )
        
        print(f"✓ Connected successfully!")
        
        # Send logs in batches
        total_logs = len(logs)
        batches = (total_logs + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"\nSending {total_logs} logs in {batches} batches of {BATCH_SIZE}...")
        print("=" * 60)
        
        sent_count = 0
        failed_count = 0
        
        for batch_idx in range(batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, total_logs)
            batch_logs = logs[start_idx:end_idx]
            
            try:
                # Create event batch
                event_data_batch = producer.create_batch()
                
                # Add events to batch
                for log in batch_logs:
                    event = create_event_data(log)
                    try:
                        event_data_batch.add(event)
                    except ValueError:
                        # Batch is full, send it and create a new one
                        producer.send_batch(event_data_batch)
                        event_data_batch = producer.create_batch()
                        event_data_batch.add(event)
                
                # Send the batch
                producer.send_batch(event_data_batch)
                sent_count += len(batch_logs)
                
                # Progress indicator
                progress = (batch_idx + 1) / batches * 100
                print(f"Batch {batch_idx + 1}/{batches} sent ({sent_count}/{total_logs} logs, {progress:.1f}%)")
                
                # Delay between batches to avoid throttling
                if batch_idx < batches - 1:
                    time.sleep(DELAY_BETWEEN_BATCHES)
                
            except Exception as e:
                print(f"❌ Error sending batch {batch_idx + 1}: {e}")
                failed_count += len(batch_logs)
        
        # Close producer
        producer.close()
        
        print("=" * 60)
        print(f"\n✓ Sending complete!")
        print(f"  Successfully sent: {sent_count} logs")
        if failed_count > 0:
            print(f"  Failed: {failed_count} logs")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error connecting to Event Hub: {e}")
        print("\nTroubleshooting:")
        print("  1. Verify connection string is correct")
        print("  2. Ensure Event Hub name matches deployment")
        print("  3. Check network connectivity to Azure")
        print("  4. Verify Event Hub namespace is running")
        return False

def send_test_message(connection_string, event_hub_name):
    """Send a single test message to verify connectivity."""
    try:
        producer = EventHubProducerClient.from_connection_string(
            conn_str=connection_string,
            eventhub_name=event_hub_name
        )
        
        # Create test event
        test_log = {
            "id": "test-001",
            "createdDateTime": datetime.utcnow().isoformat() + "Z",
            "userPrincipalName": "test_user@usf.edu",
            "userId": "test-user-001",
            "appDisplayName": "Canvas",
            "ipAddress": "131.247.1.1",
            "location": {"city": "Tampa", "state": "Florida", "countryOrRegion": "US"},
            "status": {"errorCode": 0, "failureReason": "Success"}
        }
        
        event_data_batch = producer.create_batch()
        event_data_batch.add(EventData(json.dumps(test_log)))
        producer.send_batch(event_data_batch)
        producer.close()
        
        print("✓ Test message sent successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test message failed: {e}")
        return False

def display_summary(logs):
    """Display summary of logs to be sent."""
    print("\n=== Log Summary ===")
    print(f"Total logs: {len(logs)}")
    
    # Count by persona
    persona_counts = {}
    for log in logs:
        body = json.loads(log["Body"])
        email = body["userPrincipalName"]
        persona = email.split('_')[0]
        persona_counts[persona] = persona_counts.get(persona, 0) + 1
    
    print("\nLogs by persona:")
    for persona, count in sorted(persona_counts.items()):
        print(f"  {persona}: {count}")
    
    # Count by app
    app_counts = {}
    for log in logs:
        body = json.loads(log["Body"])
        app = body["appDisplayName"]
        app_counts[app] = app_counts.get(app, 0) + 1
    
    print("\nLogs by application:")
    for app, count in sorted(app_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {app}: {count}")

def main():
    print("=" * 60)
    print("Azure Event Hub Log Sender")
    print("=" * 60)
    
    # Check if logs file exists
    if not os.path.exists("logs.json"):
        print("\n❌ Error: logs.json not found!")
        print("Run generate_logs.py first to create the logs.")
        return
    
    # Load logs
    logs = load_logs("logs.json")
    
    if len(logs) == 0:
        print("❌ No logs found in logs.json")
        return
    
    # Display summary
    display_summary(logs)
    
    # Check connection string
    connection_string = EVENT_HUB_CONNECTION_STRING
    event_hub_name = EVENT_HUB_NAME
    
    if not connection_string:
        print("\n" + "=" * 60)
        print("CONNECTION STRING REQUIRED")
        print("=" * 60)
        print("\nOption 1: Set environment variable")
        print("  export EVENT_HUB_CONNECTION_STRING='<your-connection-string>'")
        print("  export EVENT_HUB_NAME='<your-event-hub-name>'")
        print("  python send_logs_to_eventhub.py")
        print("\nOption 2: Edit this script")
        print("  Update EVENT_HUB_CONNECTION_STRING and EVENT_HUB_NAME at the top")
        print("\nTo get your connection string:")
        print("  az eventhubs namespace authorization-rule keys list \\")
        print("    --resource-group rg-mlappanalytics-dev \\")
        print("    --namespace-name evhns-mlappanalytics-dev \\")
        print("    --name SendListenRule \\")
        print("    --query primaryConnectionString -o tsv")
        return
    
    print(f"\nEvent Hub Name: {event_hub_name}")
    print(f"Connection String: {connection_string[:50]}..." if len(connection_string) > 50 else connection_string)
    
    # Ask for confirmation
    print("\n" + "=" * 60)
    response = input(f"Send {len(logs)} logs to Event Hub? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("Cancelled by user.")
        return
    
    # Option to send test message first
    print("\n" + "=" * 60)
    test_response = input("Send test message first? (yes/no): ").strip().lower()
    
    if test_response in ['yes', 'y']:
        print("\nSending test message...")
        if not send_test_message(connection_string, event_hub_name):
            print("\n❌ Test failed. Fix the issue before sending all logs.")
            return
        print("\n✓ Test successful! Proceeding with full send...")
        time.sleep(2)
    
    # Send all logs
    success = send_logs_to_eventhub(logs, connection_string, event_hub_name)
    
    if success:
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("\n1. Verify logs in Event Hub:")
        print("   - Check Azure Portal > Event Hub > Metrics")
        print("   - Look for 'Incoming Messages' spike")
        print("\n2. Check AVRO capture:")
        print("   az storage blob list --account-name <storage-name> \\")
        print("     --container-name eventhub-capture --output table")
        print("\n3. Trigger ADF pipeline to convert AVRO to JSON:")
        print("   az datafactory pipeline create-run \\")
        print("     --resource-group rg-mlappanalytics-dev \\")
        print("     --factory-name adf-mlappanalytics-dev \\")
        print("     --name AvroToJsonPipeline")
        print("\n4. Train models with new data:")
        print("   cd ..")
        print("   python run_dual_pipeline.py")

if __name__ == "__main__":
    main()
