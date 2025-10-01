"""Generate simulated Microsoft Entra sign-in logs for local users."""

import json
import os
import random
import uuid
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

USERS_FILE = Path("data/users.json")
ASSIGNMENTS_FILE = Path("assignments.json")
DEFAULT_OUTPUT = Path("data/local_signin_logs.json")
DEFAULT_TEMPLATE_FILE = Path("data/ApplicationSignIns_2025-09-24_2025-10-01.json")
LOGS_PER_ASSIGNMENT_ENV = "LOCAL_LOG_COUNT"
RESET_ENV = "LOCAL_LOG_RESET"
TENANT_ID_ENV = "LOCAL_TENANT_ID"
TENANT_NAME_ENV = "LOCAL_TENANT_NAME"
APP_FALLBACK_NAME = "Local Simulation App"
APP_FALLBACK_ID = "00000000-0000-0000-0000-000000000000"
USER_AGENT = "CodexLocalSimulator/1.0"
DEVICE_OS = "Windows 11"
DEVICE_BROWSER = "Edge 120.0"
LOCATION_PRESETS = [
    ("Tampa", "Florida", "US", 28.01999, -82.36786),
    ("Seattle", "Washington", "US", 47.6062, -122.3321),
    ("Austin", "Texas", "US", 30.2672, -97.7431),
    ("Toronto", "Ontario", "CA", 43.65107, -79.347015),
    ("Madrid", "Madrid", "ES", 40.4168, -3.7038),
]

BASE_TEMPLATE: Dict[str, object] = {
    "id": "",
    "createdDateTime": "",
    "userDisplayName": None,
    "userPrincipalName": None,
    "userId": "",
    "appId": "",
    "appDisplayName": "",
    "ipAddress": "",
    "ipAddressFromResourceProvider": None,
    "clientAppUsed": "Browser",
    "userAgent": USER_AGENT,
    "correlationId": "",
    "conditionalAccessStatus": "notApplied",
    "originalRequestId": "",
    "isInteractive": True,
    "tokenIssuerName": None,
    "tokenIssuerType": "AzureAd",
    "clientCredentialType": "none",
    "processingTimeInMilliseconds": 0,
    "riskDetail": "none",
    "riskLevelAggregated": "none",
    "riskLevelDuringSignIn": "none",
    "riskState": "none",
    "riskEventTypes_v2": [],
    "resourceDisplayName": "",
    "resourceId": "",
    "resourceTenantId": None,
    "homeTenantId": "",
    "homeTenantName": "",
    "authenticationMethodsUsed": ["Password"],
    "authenticationRequirement": "singleFactorAuthentication",
    "signInIdentifier": "",
    "signInIdentifierType": "userPrincipalName",
    "servicePrincipalName": "",
    "signInEventTypes": ["interactiveUser"],
    "servicePrincipalId": "",
    "federatedCredentialId": "",
    "userType": "member",
    "flaggedForReview": False,
    "isTenantRestricted": False,
    "autonomousSystemNumber": 0,
    "crossTenantAccessType": "none",
    "servicePrincipalCredentialKeyId": "",
    "servicePrincipalCredentialThumbprint": "",
    "uniqueTokenIdentifier": "",
    "incomingTokenType": "none",
    "authenticationProtocol": "oAuth2",
    "resourceServicePrincipalId": "",
    "signInTokenProtectionStatus": "none",
    "originalTransferMethod": "none",
    "isThroughGlobalSecureAccess": False,
    "globalSecureAccessIpAddress": None,
    "conditionalAccessAudiences": ["00000002-0000-0000-c000-000000000000"],
    "sessionId": "",
    "appOwnerTenantId": "",
    "resourceOwnerTenantId": "",
    "mfaDetail": None,
    "authenticationAppDeviceDetails": None,
    "agent": {"agentType": "notAgentic", "parentAppId": ""},
    "status": {"errorCode": 0, "failureReason": "Other.", "additionalDetails": None},
    "deviceDetail": {
        "deviceId": "",
        "displayName": "",
        "operatingSystem": DEVICE_OS,
        "browser": DEVICE_BROWSER,
        "isCompliant": True,
        "isManaged": False,
        "trustType": "AzureAD",
    },
    "location": {
        "city": "",
        "state": "",
        "countryOrRegion": "",
        "geoCoordinates": {"altitude": None, "latitude": 0.0, "longitude": 0.0},
    },
    "appliedConditionalAccessPolicies": [],
    "authenticationContextClassReferences": [],
    "authenticationProcessingDetails": [],
    "networkLocationDetails": [],
    "authenticationDetails": [],
    "authenticationRequirementPolicies": [],
    "sessionLifetimePolicies": [],
    "privateLinkDetails": {
        "policyId": "",
        "policyName": "",
        "resourceId": "",
        "policyTenantId": "",
    },
    "appliedEventListeners": [],
    "authenticationAppPolicyEvaluationDetails": [],
    "managedServiceIdentity": {
        "msiType": "none",
        "associatedResourceId": "",
        "federatedTokenId": "",
        "federatedTokenIssuer": "",
    },
    "tokenProtectionStatusDetails": {
        "signInSessionStatus": "none",
        "signInSessionStatusCode": None,
    },
}


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


def _load_json(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    try:
        with path.open("r") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as exc:
        print(f"[WARNING] Could not parse {path}: {exc}. Using empty list.")
        return []


def _load_template() -> Dict[str, object]:
    if DEFAULT_TEMPLATE_FILE.exists():
        entries = _load_json(DEFAULT_TEMPLATE_FILE)
        if entries:
            return entries[0]
    return BASE_TEMPLATE


def _random_ip() -> str:
    block = random.choice([(10, 0), (172, random.randint(16, 31)), (192, 168)])
    if block[0] == 10:
        return f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    if block[0] == 172:
        return f"172.{block[1]}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    return f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"


def _random_timestamp(index: int) -> str:
    base = datetime.now(timezone.utc) - timedelta(
        minutes=random.randint(5, 240), seconds=index
    )
    return base.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_authentication_details(timestamp: str) -> List[Dict[str, object]]:
    return [
        {
            "authenticationMethod": "Password",
            "authenticationMethodDetail": "UsernamePassword",
            "authenticationStepDateTime": timestamp,
            "authenticationStepRequirement": "primary",
            "succeeded": True,
        }
    ]


def _choose_location() -> Dict[str, object]:
    city, state, country, lat, lon = random.choice(LOCATION_PRESETS)
    return {
        "city": city,
        "state": state,
        "countryOrRegion": country,
        "geoCoordinates": {
            "altitude": None,
            "latitude": lat,
            "longitude": lon,
        },
    }


def _build_log_entry(
    template: Dict[str, object],
    user: Dict[str, object],
    assignment: Dict[str, object],
    tenant_id: str,
    tenant_name: str,
    index: int,
) -> Dict[str, object]:
    entry = deepcopy(template)

    timestamp = _random_timestamp(index)
    app_name = assignment.get("appName", APP_FALLBACK_NAME)
    app_id = assignment.get("appClientId", APP_FALLBACK_ID)

    entry.update(
        {
            "id": str(uuid.uuid4()),
            "createdDateTime": timestamp,
            "userDisplayName": user.get("displayName") or user.get("userPrincipalName"),
            "userPrincipalName": user.get("userPrincipalName"),
            "userId": user.get("id", ""),
            "appId": app_id,
            "appDisplayName": app_name,
            "ipAddress": _random_ip(),
            "correlationId": str(uuid.uuid4()),
            "processingTimeInMilliseconds": random.randint(120, 850),
            "resourceDisplayName": app_name,
            "resourceId": app_id,
            "resourceTenantId": tenant_id,
            "homeTenantId": tenant_id,
            "homeTenantName": tenant_name,
            "signInIdentifier": user.get("userPrincipalName", ""),
            "servicePrincipalId": assignment.get("resourceServicePrincipalId", app_id),
            "resourceServicePrincipalId": assignment.get("resourceServicePrincipalId", app_id),
            "sessionId": str(uuid.uuid4()),
            "appOwnerTenantId": tenant_id,
            "resourceOwnerTenantId": tenant_id,
            "uniqueTokenIdentifier": uuid.uuid4().hex,
        }
    )

    entry["clientAppUsed"] = "Browser"
    entry["userAgent"] = USER_AGENT
    entry["isInteractive"] = True
    entry["tokenIssuerType"] = "AzureAd"
    entry["tokenIssuerName"] = tenant_name
    entry["clientCredentialType"] = "none"
    entry["authenticationRequirement"] = "singleFactorAuthentication"
    entry["signInIdentifierType"] = "userPrincipalName"
    entry["servicePrincipalName"] = ""
    entry["signInEventTypes"] = ["interactiveUser"]
    entry["authenticationMethodsUsed"] = ["Password"]
    entry["incomingTokenType"] = "none"
    entry["authenticationProtocol"] = "oAuth2"
    entry["riskDetail"] = "none"
    entry["riskLevelAggregated"] = "none"
    entry["riskLevelDuringSignIn"] = "none"
    entry["riskEventTypes_v2"] = []
    entry["servicePrincipalCredentialKeyId"] = ""
    entry["servicePrincipalCredentialThumbprint"] = ""
    entry["signInTokenProtectionStatus"] = "none"
    entry["originalTransferMethod"] = "none"
    entry["conditionalAccessStatus"] = "notApplied"
    entry["status"] = {"errorCode": 0, "failureReason": "Other.", "additionalDetails": None}
    entry["authenticationRequirementPolicies"] = []
    entry["appliedConditionalAccessPolicies"] = []
    entry["appliedEventListeners"] = []
    entry["authenticationAppPolicyEvaluationDetails"] = []
    entry["authenticationContextClassReferences"] = []
    entry["authenticationProcessingDetails"] = [
        {"key": "Is CAE Token", "value": random.choice(["True", "False"])}
    ]
    entry["authenticationDetails"] = _build_authentication_details(timestamp)
    entry["networkLocationDetails"] = []
    entry["sessionLifetimePolicies"] = []
    entry["privateLinkDetails"] = {
        "policyId": "",
        "policyName": "",
        "resourceId": "",
        "policyTenantId": "",
    }
    entry["managedServiceIdentity"] = {
        "msiType": "none",
        "associatedResourceId": "",
        "federatedTokenId": "",
        "federatedTokenIssuer": "",
    }
    entry["tokenProtectionStatusDetails"] = {
        "signInSessionStatus": "none",
        "signInSessionStatusCode": None,
    }
    entry["flaggedForReview"] = False
    entry["isTenantRestricted"] = False
    entry["autonomousSystemNumber"] = random.randint(64512, 65534)
    entry["crossTenantAccessType"] = "none"
    entry["userType"] = "member"

    entry["deviceDetail"] = deepcopy(entry.get("deviceDetail", {})) or {}
    entry["deviceDetail"].update(
        {
            "deviceId": str(uuid.uuid4()),
            "displayName": f"{user.get('displayName', 'device')}'s Device",
            "operatingSystem": DEVICE_OS,
            "browser": DEVICE_BROWSER,
            "isCompliant": True,
            "isManaged": False,
            "trustType": "AzureAD",
        }
    )
    entry["location"] = _choose_location()
    entry["location"]["geoCoordinates"] = {
        "altitude": None,
        "latitude": entry["location"].get("geoCoordinates", {}).get("latitude", 0.0),
        "longitude": entry["location"].get("geoCoordinates", {}).get("longitude", 0.0),
    }

    return entry


def main() -> None:
    users = _load_json(USERS_FILE)
    if not users:
        print(f"[ERROR] No users found at {USERS_FILE}. Run create_vusers.py first.")
        return

    assignments = _load_json(ASSIGNMENTS_FILE)
    assignments_by_user: Dict[str, List[Dict[str, object]]] = {}
    for assignment in assignments:
        upn = assignment.get("userPrincipalName")
        if upn:
            assignments_by_user.setdefault(upn, []).append(assignment)

    output_path = Path(os.getenv("LOCAL_LOG_OUTPUT", DEFAULT_OUTPUT))
    reset_requested = _parse_bool(os.getenv(RESET_ENV))
    logs_per_assignment = _parse_int(os.getenv(LOGS_PER_ASSIGNMENT_ENV), 1)
    tenant_id = os.getenv(TENANT_ID_ENV, "00000000-0000-0000-0000-000000000000")
    tenant_name = os.getenv(TENANT_NAME_ENV, "Local Tenant")

    existing_logs: List[Dict[str, object]] = []
    if output_path.exists() and not reset_requested:
        existing_logs = _load_json(output_path)

    template = _load_template()
    new_logs: List[Dict[str, object]] = []
    entry_counter = 0

    for user in users:
        user_assignments = assignments_by_user.get(
            user.get("userPrincipalName"),
            [{"appName": APP_FALLBACK_NAME, "appClientId": APP_FALLBACK_ID}],
        )
        for assignment in user_assignments:
            for _ in range(logs_per_assignment):
                entry_counter += 1
                new_logs.append(
                    _build_log_entry(template, user, assignment, tenant_id, tenant_name, entry_counter)
                )

    final_logs = existing_logs + new_logs if not reset_requested else new_logs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(final_logs, f, indent=4)

    print(
        f"Generated {len(new_logs)} log(s) for {len(users)} user(s). Total stored records: {len(final_logs)}."
    )


if __name__ == "__main__":
    main()
