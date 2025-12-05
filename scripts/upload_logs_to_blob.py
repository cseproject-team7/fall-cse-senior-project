"""
Upload Synthetic Logs to Azure Blob Storage

This script:
1. Deletes all existing logs in the Azure Blob Storage container
2. Uploads the newly generated synthetic logs
"""

import os
import sys
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables from server/.env
env_path = os.path.join(os.path.dirname(__file__), '..', 'server', '.env')
load_dotenv(env_path)

# Get connection string
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
if not connection_string:
    print("❌ Error: AZURE_STORAGE_CONNECTION_STRING not found in server/.env")
    sys.exit(1)

# Configuration
CONTAINER_NAME = "json-signin-logs"
LOG_FILE = "../logs.json"  # Generated JSONL file from model_training_pipeline
BLOB_PREFIX = "json-output/"  # Folder prefix in blob storage

def delete_all_blobs():
    """Delete all blobs in the json-output folder"""
    print(f"🗑️  Connecting to Azure Blob Storage...")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    
    try:
        print(f"📋 Listing all blobs in '{CONTAINER_NAME}/{BLOB_PREFIX}'...")
        blob_count = 0
        for blob in container_client.list_blobs(name_starts_with=BLOB_PREFIX):
            blob_count += 1
            print(f"   Deleting: {blob.name}")
            container_client.delete_blob(blob.name)
        
        print(f"✅ Deleted {blob_count} blobs")
        return blob_count
    except Exception as e:
        print(f"❌ Error deleting blobs: {e}")
        return 0

def upload_logs():
    """Upload the generated logs to blob storage as part files"""
    if not os.path.exists(LOG_FILE):
        print(f"❌ Error: Log file not found at {LOG_FILE}")
        return False
    
    print(f"\n📤 Uploading logs from {LOG_FILE}...")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    
    try:
        # Get file size
        file_size = os.path.getsize(LOG_FILE)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"   File size: {file_size_mb:.2f} MB")
        
        # Upload as a single part file (mimicking Spark output structure)
        blob_name = f"{BLOB_PREFIX}part-00000-generated.json"
        blob_client = container_client.get_blob_client(blob_name)
        
        print(f"   Uploading to blob: {blob_name}")
        with open(LOG_FILE, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)
        
        # Create _SUCCESS marker file (Spark convention)
        success_blob = container_client.get_blob_client(f"{BLOB_PREFIX}_SUCCESS")
        success_blob.upload_blob(b"", overwrite=True)
        print(f"   Created success marker: {BLOB_PREFIX}_SUCCESS")
        
        print(f"✅ Successfully uploaded logs to {blob_name}")
        
        # Count lines
        with open(LOG_FILE, 'r') as f:
            line_count = sum(1 for line in f if line.strip())
        
        print(f"   Total log entries: {line_count:,}")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading logs: {e}")
        return False

def main():
    print("=" * 60)
    print("Azure Blob Storage Log Management")
    print("=" * 60)
    
    # Step 1: Delete existing logs
    deleted_count = delete_all_blobs()
    
    if deleted_count == 0:
        print("\n⚠️  No blobs were deleted. Container might be empty or there was an error.")
    
    # Step 2: Upload new logs
    success = upload_logs()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ SUCCESS! Logs have been replaced in Azure Blob Storage")
        print("=" * 60)
        print("\n💡 Next steps:")
        print("   1. Restart your Node.js server")
        print("   2. Refresh the dashboard to see the new logs")
    else:
        print("\n" + "=" * 60)
        print("❌ FAILED to upload logs")
        print("=" * 60)
        sys.exit(1)

if __name__ == '__main__':
    main()
