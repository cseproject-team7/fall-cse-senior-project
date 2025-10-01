import requests
import json
import os

# -----------------------------
# Configuration
# -----------------------------
# App Registration with permissions to assign apps and read service principals
# (AppRoleAssignment.ReadWrite.All, Application.Read.All)
CREATOR_APP = {
    "client_id": "CLIENT_ID",
    "client_secret": "CLIENT_SECRET",
    "tenant_id": "TENANT_ID"
}

# Add every application you want to grant access to in this list. Make sure to give Application.Read permission to it.
TARGET_APPS = [
    {
        "name": "Canvas App",
        "client_id": "CLIENT_ID",
        "client_secret": "CLIENT_SECRET"
    },
    {
        "name": "Outlook App",
        "client_id": "CLIENT_ID",
        "client_secret": "CLIENT_SECRET"
    }
]

INPUT_FILE = os.path.join("data", "users.json")
OUTPUT_FILE = "assignments.json"

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
# Main Assignment Logic
# -----------------------------
def assign_apps(access_token):
    """Reads users from a JSON file and assigns them to multiple apps."""
    print("\n--- Starting User to App Assignment Process ---")
    
    try:
        with open(INPUT_FILE, 'r') as f:
            users_to_process = json.load(f)
        print(f"Successfully read {len(users_to_process)} user(s) from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {INPUT_FILE}. Please run create_vusers.py first.")
        return
    except json.JSONDecodeError:
        print(f"[ERROR] Could not decode JSON from {INPUT_FILE}. The file may be empty or corrupted.")
        return

    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    assignments_for_simulation = []

    for user in users_to_process:
        principal_id = user["id"]
        upn = user["userPrincipalName"]
        print(f"\n{'='*20}\n--- Processing User: {upn} ---\n{'='*20}")

        for app in TARGET_APPS:
            app_name = app["name"]
            app_client_id = app["client_id"]
            print(f"\n--- Assigning to application: '{app_name}' ---")
            
            sp_url = f"https://graph.microsoft.com/v1.0/servicePrincipals?$filter=appId eq '{app_client_id}'"
            sp_response = requests.get(sp_url, headers=headers)
            
            if sp_response.status_code != 200 or not sp_response.json().get("value"):
                print(f"[ERROR] Could not find SP for {app_name}. Skipping.")
                continue
            
            service_principal = sp_response.json()["value"][0]
            sp_id = service_principal["id"]
            app_role_id = "00000000-0000-0000-0000-000000000000"
            print(f"Found Service Principal ID: {sp_id}")

            assign_url = f"https://graph.microsoft.com/v1.0/users/{principal_id}/appRoleAssignments"
            payload = {"principalId": principal_id, "resourceId": sp_id, "appRoleId": app_role_id}
            assign_response = requests.post(assign_url, headers=headers, json=payload)

            assignment_exists = False
            if assign_response.status_code == 201:
                print(f"Successfully assigned user to '{app_name}'.")
                assignment_exists = True
            elif "Permission being assigned already exists" in assign_response.text:
                print(f"User already has access to '{app_name}'.")
                assignment_exists = True
            else:
                print(f"[ERROR] Failed to assign user to '{app_name}': {assign_response.text}")

            if assignment_exists:
                # Add the details needed for the simulation script to our list
                assignments_for_simulation.append({
                    "userPrincipalName": user["userPrincipalName"],
                    "password": user["password"],
                    "appName": app["name"],
                    "appClientId": app["client_id"],
                    "appClientSecret": app["client_secret"]
                })

    if assignments_for_simulation:
        try:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(assignments_for_simulation, f, indent=4)
            print(f"\nSuccessfully wrote {len(assignments_for_simulation)} assignment(s) to {OUTPUT_FILE}")
        except IOError as e:
            print(f"[ERROR] Could not write to file {OUTPUT_FILE}: {e}")

    print("\n--- Assignment Process Complete ---")

# -----------------------------
# Main Execution
# -----------------------------
def main():
    """Main function to drive the script."""
    token = get_access_token(
        CREATOR_APP["client_id"],
        CREATOR_APP["client_secret"],
        CREATOR_APP["tenant_id"]
    )
    if not token:
        print("Exiting script due to authentication failure.")
        return
    assign_apps(token)

if __name__ == "__main__":
    main()
