import requests
import json
import os

# -----------------------------
# Configuration
# -----------------------------
# App Registration with permissions to create users (User.ReadWrite.All)
CREATOR_APP = {
    "client_id": "CLIENT_ID",
    "client_secret": "CLIENT_SECRET",
    "tenant_id": "TENANT_ID"
}

# Toggle between talking to the real tenant ("REAL") and ("LOCAL")
# CREATION_MODE = os.getenv("USER_CREATION_MODE", "REAL").strip().upper()  # REAL
CREATION_MODE = os.getenv("USER_CREATION_MODE", "LOCAL").strip().upper() # LOCAL
VALID_CREATION_MODES = {"REAL", "LOCAL"}

# Define all the users you want to create here
USERS_TO_CREATE = [
    {
        "displayName": "USER_NAME",
        "userPrincipalName": "USER_NAME@fallcseseniorproject.onmicrosoft.com",
        "password": "USER_PASSWORD"
    },
]

OUTPUT_DIR = os.path.join("data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "users.json")

# -----------------------------
# Helper Functions
# -----------------------------
def get_access_token(client_id, client_secret, tenant_id):
    """Acquires an access token from Microsoft Entra ID."""
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default"
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("Access token acquired successfully!")
        return response.json()["access_token"]
    else:
        print(f"Failed to get access token. Status: {response.status_code}, Response: {response.text}")
        return None

# -----------------------------
# Main User Creation Logic
# -----------------------------
def create_users(access_token):
    """Creates multiple virtual users and saves their details to a JSON file."""
    print("\n--- Starting User Creation Process ---")
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    
    created_users_data = []

    for user_details in USERS_TO_CREATE:
        print(f"\nProcessing user: {user_details['userPrincipalName']}...")
        user_data = {
            "accountEnabled": True,
            "displayName": user_details["displayName"],
            "mailNickname": user_details["userPrincipalName"].split("@")[0],
            "userPrincipalName": user_details["userPrincipalName"],
            "passwordProfile": {
                "forceChangePasswordNextSignIn": False,
                "password": user_details["password"]
            }
        }
        
        create_response = requests.post("https://graph.microsoft.com/v1.0/users", headers=headers, json=user_data)
        
        user_id = None
        if create_response.status_code == 201:
            created_user = create_response.json()
            user_id = created_user['id']
            print(f"Successfully created user with ID: {user_id}")
        elif "userPrincipalName already exists" in create_response.text:
             print(f"User {user_details['userPrincipalName']} already exists. Retrieving their ID...")
             get_user_url = f"https://graph.microsoft.com/v1.0/users/{user_details['userPrincipalName']}"
             get_user_response = requests.get(get_user_url, headers=headers)
             if get_user_response.status_code == 200:
                 user_id = get_user_response.json()['id']
                 print(f"Found existing user with ID: {user_id}")
             else:
                 print(f"[ERROR] Could not retrieve existing user details: {get_user_response.text}")
                 continue # Skip to the next user
        else:
            print(f"[ERROR] Failed to create user: {create_response.text}")
            continue # Skip to the next user

        if user_id:
            created_users_data.append({
                "id": user_id,
                "displayName": user_details["displayName"],
                "userPrincipalName": user_details["userPrincipalName"],
                "password": user_details["password"],
                "source": "REAL"
            })

    if created_users_data:
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(created_users_data, f, indent=4)
            print(f"\nSuccessfully wrote {len(created_users_data)} user(s) to {OUTPUT_FILE}")
        except IOError as e:
            print(f"[ERROR] Could not write to file {OUTPUT_FILE}: {e}")

    print("\n--- User Creation Process Complete ---")

def main():
    """Main function to drive the script."""
    if CREATION_MODE not in VALID_CREATION_MODES:
        print(f"[ERROR] Invalid CREATION_MODE '{CREATION_MODE}'. Choose from {sorted(VALID_CREATION_MODES)}.")
        return

    if CREATION_MODE == "LOCAL":
        print("Running in LOCAL mode: skipping Microsoft Graph calls and generating stub users.")
        from helper_create_vusers_local import generate_local_user_stubs

        existing_users_data = []
        if os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, 'r') as f:
                    existing_users_data = json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                print(f"[WARNING] Could not read existing user file {OUTPUT_FILE}: {e}. Proceeding with empty list.")
                existing_users_data = []

        result = generate_local_user_stubs(existing_users_data)
        if result is None:
            return

        new_local_users = result["new_users"]
        reset_requested = result["reset_requested"]

        if reset_requested:
            retained_users = [
                user for user in existing_users_data if user.get("source") != "LOCAL"
            ]
        else:
            retained_users = list(existing_users_data)

        final_users = retained_users + new_local_users

        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(final_users, f, indent=4)
            print(
                f"\nSuccessfully wrote {len(new_local_users)} new local user stub(s) to {OUTPUT_FILE}. Total records: {len(final_users)}"
            )
        except IOError as e:
            print(f"[ERROR] Could not write to file {OUTPUT_FILE}: {e}")
        return

    token = get_access_token(
        CREATOR_APP["client_id"],
        CREATOR_APP["client_secret"],
        CREATOR_APP["tenant_id"]
    )

    if not token:
        print("Exiting script due to authentication failure.")
        return

    create_users(token)

if __name__ == "__main__":
    main()
