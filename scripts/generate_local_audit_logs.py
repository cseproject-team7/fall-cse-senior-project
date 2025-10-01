"""Utilities to simulate Microsoft Entra audit logs for local users."""

from __future__ import annotations

import json
import os
import random
import uuid
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

USERS_FILE = Path("data/users.json")
DEFAULT_TEMPLATE_FILE = Path("data/AuditLogs_2025-10-01.json")
DEFAULT_OUTPUT_ROOT = Path("data/audit_logs")
DEFAULT_ACTIONS_FILE = Path("data/audit_actions.json")
DEFAULT_LOG_NAME = "audit_logs.json"
LOG_COUNT_ENV = "LOCAL_AUDIT_COUNT"
RESET_ENV = "LOCAL_AUDIT_RESET"
TARGET_USERS_ENV = "LOCAL_AUDIT_TARGET"
ACTIONS_FILE_ENV = "LOCAL_AUDIT_ACTIONS"
TENANT_ID_ENV = "LOCAL_TENANT_ID"
TENANT_NAME_ENV = "LOCAL_TENANT_NAME"

BASE_TEMPLATE: Dict[str, object] = {
    "id": "",
    "category": "UserManagement",
    "correlationId": "",
    "result": "success",
    "resultReason": "",
    "activityDisplayName": "Update user",
    "activityDateTime": "",
    "loggedByService": "Local Directory Simulator",
    "initiatedBy": {
        "app": {
            "appId": None,
            "displayName": "Local Simulator",
            "servicePrincipalId": str(uuid.uuid4()),
            "servicePrincipalName": None,
        }
    },
    "userAgent": None,
    "targetResources": [],
    "additionalDetails": [],
}

TARGET_RESOURCE_TEMPLATE = {
    "id": "",
    "displayName": None,
    "type": "User",
    "userPrincipalName": "",
    "groupType": None,
    "modifiedProperties": [],
}

MODIFIED_PROPERTY_TEMPLATE = {
    "displayName": "Included Updated Properties",
    "oldValue": None,
    "newValue": None,
}

DEFAULT_ACTIONS: Sequence[Dict[str, object]] = (
    {
        "activityDisplayName": "Update user",
        "category": "UserManagement",
        "loggedByService": "Core Directory",
        "result": "success",
        "resultReason": "",
        "initiatedBy": {
            "app": {
                "appId": None,
                "displayName": "Codex Persona Simulator",
                "servicePrincipalId": str(uuid.uuid4()),
                "servicePrincipalName": None,
            }
        },
        "modifiedProperties": [
            {
                "displayName": "Department",
                "oldValue": '"Sales"',
                "newValue": '"Security"',
            },
            {
                "displayName": "Included Updated Properties",
                "oldValue": None,
                "newValue": '"Department"',
            },
        ],
        "additionalDetails": [
            {"key": "Persona", "value": "Security Analyst"},
        ],
    },
    {
        "activityDisplayName": "Reset user password",
        "category": "UserManagement",
        "loggedByService": "Core Directory",
        "result": "success",
        "resultReason": "",
        "initiatedBy": {
            "user": {
                "displayName": "Local Helpdesk",
                "userPrincipalName": "helpdesk@local",
                "id": str(uuid.uuid4()),
            }
        },
        "modifiedProperties": [
            {
                "displayName": "PasswordProfile",
                "oldValue": None,
                "newValue": '"********"',
            }
        ],
        "additionalDetails": [
            {"key": "ResetReason", "value": "User request"},
        ],
    },
)


def _parse_bool(value: Optional[str]) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "y"}


def _parse_int(value: Optional[str], default: int) -> int:
    if not value:
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
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARNING] Could not parse {path}: {exc}. Falling back to empty list.")
        return []


def _load_template() -> Dict[str, object]:
    entries = _load_json(DEFAULT_TEMPLATE_FILE)
    if entries:
        return entries[0]
    return deepcopy(BASE_TEMPLATE)


def _normalise_modified_properties(props: Optional[Iterable[Dict[str, object]]]) -> List[Dict[str, object]]:
    normalised: List[Dict[str, object]] = []
    if not props:
        return normalised
    for item in props:
        template = deepcopy(MODIFIED_PROPERTY_TEMPLATE)
        template.update(item)
        normalised.append(template)
    return normalised


def _build_target_resources(user: Dict[str, object], action: Dict[str, object]) -> List[Dict[str, object]]:
    resource = deepcopy(TARGET_RESOURCE_TEMPLATE)
    resource.update(
        {
            "id": user.get("id", ""),
            "displayName": user.get("displayName"),
            "userPrincipalName": user.get("userPrincipalName"),
            "modifiedProperties": _normalise_modified_properties(action.get("modifiedProperties")),
        }
    )
    return [resource]


def _timestamp(offset_seconds: int) -> str:
    ts = datetime.now(timezone.utc) - timedelta(seconds=offset_seconds)
    return ts.isoformat()


def _choose_action(actions: Sequence[Dict[str, object]]) -> Dict[str, object]:
    weights = [action.get("weight", 1) for action in actions]
    return random.choices(actions, weights=weights, k=1)[0]


def _build_entry(
    template: Dict[str, object],
    user: Dict[str, object],
    action: Dict[str, object],
    tenant_id: str,
    tenant_name: str,
    sequence: int,
) -> Dict[str, object]:
    base = deepcopy(template)
    correlation = str(uuid.uuid4())
    suffix = uuid.uuid4().hex[:12].upper()
    base.update(
        {
            "id": f"Directory_{correlation}_{suffix}",
            "correlationId": correlation,
            "activityDateTime": _timestamp(sequence * 5),
            "homeTenantId": tenant_id,
            "homeTenantName": tenant_name,
            "initiatedBy": deepcopy(action.get("initiatedBy", base.get("initiatedBy", {}))),
            "targetResources": _build_target_resources(user, action),
            "additionalDetails": deepcopy(action.get("additionalDetails", [])),
        }
    )

    for key in ("activityDisplayName", "category", "loggedByService", "result", "resultReason"):
        if key in action:
            base[key] = action[key]

    return base


def generate_audit_logs_for_user(
    user: Dict[str, object],
    actions: Sequence[Dict[str, object]],
    count: int,
    tenant_id: str,
    tenant_name: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    reset_existing: bool = False,
    template: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Create simulated audit logs for a single user.

    Returns metadata containing number of new and total records and the path used.
    """

    if not user.get("id"):
        raise ValueError("User is missing an 'id' field required for folder naming.")

    template = template or _load_template()
    user_dir = output_root / user["id"]
    output_path = user_dir / DEFAULT_LOG_NAME
    user_dir.mkdir(parents=True, exist_ok=True)

    existing: List[Dict[str, object]] = []
    if output_path.exists() and not reset_existing:
        existing = _load_json(output_path)

    new_entries: List[Dict[str, object]] = []
    for idx in range(1, count + 1):
        action = deepcopy(_choose_action(actions))
        entry = _build_entry(template, user, action, tenant_id, tenant_name, sequence=len(existing) + idx)
        new_entries.append(entry)

    combined = new_entries if reset_existing else existing + new_entries

    with output_path.open("w") as f:
        json.dump(combined, f, indent=4)

    return {
        "user": user.get("userPrincipalName", user.get("id")),
        "new_records": len(new_entries),
        "total_records": len(combined),
        "path": str(output_path),
    }


def _load_actions(path: Path) -> Sequence[Dict[str, object]]:
    if path.exists():
        data = _load_json(path)
        if data:
            return data
    return DEFAULT_ACTIONS


def _select_users(users: List[Dict[str, object]], selector: Optional[str]) -> List[Dict[str, object]]:
    if not selector:
        return users
    wanted = {item.strip().lower() for item in selector.split(",") if item.strip()}
    if not wanted:
        return users
    result = []
    for user in users:
        upn = str(user.get("userPrincipalName", "")).lower()
        uid = str(user.get("id", "")).lower()
        if upn in wanted or uid in wanted:
            result.append(user)
    return result


def main() -> None:
    users = _load_json(USERS_FILE)
    if not users:
        print(f"[ERROR] No users found in {USERS_FILE}. Run create_vusers.py first.")
        return

    tenant_id = os.getenv(TENANT_ID_ENV, "00000000-0000-0000-0000-000000000000")
    tenant_name = os.getenv(TENANT_NAME_ENV, "Local Tenant")
    count = _parse_int(os.getenv(LOG_COUNT_ENV), 1)
    reset = _parse_bool(os.getenv(RESET_ENV))
    target_raw = os.getenv(TARGET_USERS_ENV)
    actions_file = Path(os.getenv(ACTIONS_FILE_ENV, str(DEFAULT_ACTIONS_FILE)))

    actions = _load_actions(actions_file)
    selected_users = _select_users(users, target_raw)

    template = _load_template()
    summaries = []
    for user in selected_users:
        summary = generate_audit_logs_for_user(
            user=user,
            actions=actions,
            count=count,
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            output_root=DEFAULT_OUTPUT_ROOT,
            reset_existing=reset,
            template=template,
        )
        summaries.append(summary)

    for summary in summaries:
        print(
            f"User {summary['user']}: added {summary['new_records']} record(s). Total now {summary['total_records']} (stored at {summary['path']})."
        )


if __name__ == "__main__":
    main()
