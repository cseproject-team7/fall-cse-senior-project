"""Helpers for creating local-only virtual user stubs."""

import os
import uuid
import secrets
from typing import Dict, List, Optional

LOCAL_USER_COUNT_ENV = "LOCAL_USER_COUNT"
LOCAL_USER_DOMAIN_ENV = "LOCAL_USER_DOMAIN"
LOCAL_USER_RESET_ENV = "LOCAL_USER_RESET"
DEFAULT_LOCAL_USER_COUNT = 7
DEFAULT_LOCAL_USER_DOMAIN = "axgile.com"


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _find_next_index(existing_local_users: List[Dict[str, str]]) -> int:
    max_index = 0
    for user in existing_local_users:
        name = user.get("displayName", "")
        if name.startswith("vuser"):
            suffix = name[5:]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
    return max_index + 1


def generate_local_user_stubs(
    existing_users: Optional[List[Dict[str, str]]] = None,
) -> Optional[Dict[str, object]]:
    """Generate sequential local users with random passwords.

    Returns a dict with keys:
    - ``new_users``: list of newly generated local user records
    - ``reset_requested``: bool indicating whether existing locals should be replaced

    Returns ``None`` if configuration is invalid.
    """

    print("\n--- Starting Local User Stub Generation ---")

    count_str = os.getenv(LOCAL_USER_COUNT_ENV, str(DEFAULT_LOCAL_USER_COUNT))
    domain = os.getenv(LOCAL_USER_DOMAIN_ENV, DEFAULT_LOCAL_USER_DOMAIN)
    reset_requested = _parse_bool(os.getenv(LOCAL_USER_RESET_ENV))

    try:
        desired_count = int(count_str)
        if desired_count < 1:
            raise ValueError
    except ValueError:
        print(
            f"[ERROR] LOCAL_USER_COUNT must be a positive integer (current value: '{count_str}')"
        )
        return None

    existing_users = existing_users or []
    existing_local_users = [
        user for user in existing_users if user.get("source") == "LOCAL"
    ]

    start_index = 1 if reset_requested else _find_next_index(existing_local_users)

    local_users_data: List[Dict[str, str]] = []
    for offset in range(desired_count):
        index = start_index + offset
        display_name = f"vuser{index}"
        upn = f"{display_name}@{domain}"
        local_id = str(uuid.uuid4())
        password = secrets.token_urlsafe(12)
        print(f"Simulating user '{upn}' with local ID {local_id}")
        local_users_data.append(
            {
                "id": local_id,
                "displayName": display_name,
                "userPrincipalName": upn,
                "password": password,
                "source": "LOCAL",
            }
        )

    print("\n--- Local User Stub Generation Complete ---")
    return {"new_users": local_users_data, "reset_requested": reset_requested}
