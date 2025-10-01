"""High-level persona audit log simulator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from generate_local_audit_logs import (
    DEFAULT_ACTIONS,
    DEFAULT_TEMPLATE_FILE,
    DEFAULT_OUTPUT_ROOT,
    generate_audit_logs_for_user,
)

USERS_FILE = Path("data/users.json")
PERSONA_ACTIONS_FILE = Path("data/persona_actions.json")
LOG_COUNT_ENV = "PERSONA_LOG_COUNT"
RESET_ENV = "PERSONA_RESET"
TARGET_ENV = "PERSONA_TARGET_USERS"
ACTIONS_FILE_ENV = "PERSONA_ACTIONS_FILE"
TENANT_ID_ENV = "PERSONA_TENANT_ID"
TENANT_NAME_ENV = "PERSONA_TENANT_NAME"


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
        print(f"[WARNING] Could not parse {path}: {exc}. Using empty list.")
        return []


def _load_actions(path: Path) -> Sequence[Dict[str, object]]:
    data = _load_json(path)
    return data if data else DEFAULT_ACTIONS


def _filter_users(users: List[Dict[str, object]], selector: Optional[str]) -> List[Dict[str, object]]:
    if not selector:
        return users
    wanted = {token.strip().lower() for token in selector.split(",") if token.strip()}
    if not wanted:
        return users
    filtered: List[Dict[str, object]] = []
    for user in users:
        upn = str(user.get("userPrincipalName", "")).lower()
        uid = str(user.get("id", "")).lower()
        if upn in wanted or uid in wanted:
            filtered.append(user)
    return filtered


def main() -> None:
    users = _load_json(USERS_FILE)
    if not users:
        print(f"[ERROR] No users available in {USERS_FILE}. Run create_vusers.py first.")
        return

    tenant_id = os.getenv(TENANT_ID_ENV) or os.getenv("LOCAL_TENANT_ID") or "00000000-0000-0000-0000-000000000000"
    tenant_name = os.getenv(TENANT_NAME_ENV) or os.getenv("LOCAL_TENANT_NAME") or "Local Tenant"
    count = _parse_int(os.getenv(LOG_COUNT_ENV), 3)
    reset = _parse_bool(os.getenv(RESET_ENV))
    target_selector = os.getenv(TARGET_ENV)

    actions_file = Path(os.getenv(ACTIONS_FILE_ENV, str(PERSONA_ACTIONS_FILE)))
    actions = _load_actions(actions_file)
    template_entries = _load_json(DEFAULT_TEMPLATE_FILE)
    template = template_entries[0] if template_entries else None

    selected_users = _filter_users(users, target_selector)
    if not selected_users:
        print("[INFO] No users matched the supplied selector. Nothing to do.")
        return

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
        print(
            f"Persona logs for {summary['user']}: added {summary['new_records']} record(s). Total now {summary['total_records']} (see {summary['path']})."
        )


if __name__ == "__main__":
    main()
