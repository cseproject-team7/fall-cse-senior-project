# fall-cse-senior-project

## User creation modes
- The `scripts/create_vusers.py` script now supports two modes controlled by the `USER_CREATION_MODE` environment variable.
- `REAL` (default) talks to Microsoft Entra and provisions accounts in the tenant configured in `CREATOR_APP`.
- `LOCAL` skips Microsoft Graph calls and instead writes stub users to `users.json` so you can test downstream flows without touching the tenant.

### Switch modes
- macOS/Linux: `export USER_CREATION_MODE=LOCAL` (use `REAL` to switch back).
- Windows PowerShell: `setx USER_CREATION_MODE LOCAL` and restart the shell, or `$Env:USER_CREATION_MODE = "LOCAL"` for the current session.
- After setting the variable, run `python scripts/create_vusers.py` as usual; the script prints which mode is active.

### Local mode options
- `LOCAL_USER_COUNT` controls how many additional stub accounts are generated on each run (defaults to 1). Names are sequential (`vuser1`, `vuser2`, …) and continue from the highest existing local index.
- Passwords for local users are random per run; check `data/users.json` for the generated value if you need to reuse it downstream.
- `LOCAL_USER_DOMAIN` overrides the email domain used for those UPNs when needed.
- `LOCAL_USER_RESET=1` (or `true`) discards existing local entries before generating the next batch, letting you rebuild the list from scratch.

### Output location
- Whenever `scripts/create_vusers.py` runs, it now writes `users.json` into the `data/` folder (`data/users.json`). Update any manual edits accordingly.
- Downstream scripts (e.g., `assign_users.py`) already point to the new location; remove any old root-level `users.json` copies to avoid confusion.

## Local sign-in log simulation
- Run `python scripts/generate_local_signin_logs.py` to create mock Microsoft Entra sign-in entries for every user in `data/users.json`.
- Logs are stored in `data/local_signin_logs.json` and follow the structure of Azure's Application Sign-ins export (`data/ApplicationSignIns_*.json`).

Example session:

```bash
export LOCAL_LOG_COUNT=2
export LOCAL_LOG_RESET=1
python scripts/generate_local_signin_logs.py
```

### Log options
- `LOCAL_LOG_COUNT` controls how many log entries are generated per user (per assignment) on each run (defaults to 1).
- `LOCAL_LOG_RESET=1` starts a new log file instead of appending to existing entries.
- `LOCAL_TENANT_ID` / `LOCAL_TENANT_NAME` let you brand the simulated tenant metadata.
- `LOCAL_LOG_OUTPUT` overrides the destination file path when you want multiple scenarios.

## Local audit log simulation
- Run `python scripts/generate_local_audit_logs.py` to create audit events that mirror the structure of Azure’s Audit Logs export (`data/AuditLogs_*.json`).
- Each user gets their own folder under `data/audit_logs/<userId>/audit_logs.json`, making it easy to separate personas or wipe an individual user’s history.

Example session:

```bash
export LOCAL_AUDIT_COUNT=2
export LOCAL_AUDIT_TARGET=vuser1@axgile.com
python scripts/generate_local_audit_logs.py
```

### Audit log options
- `LOCAL_AUDIT_COUNT` controls how many entries are added per user on each run (defaults to 1).
- `LOCAL_AUDIT_RESET=1` recreates the log file for targeted users instead of appending to the existing data.
- `LOCAL_AUDIT_TARGET` accepts a comma-separated list of user IDs or UPNs to limit generation (leave unset to process every user in `data/users.json`).
- `LOCAL_AUDIT_ACTIONS` points to a JSON file describing the available actions. If omitted, sensible defaults are used. Pair with `LOCAL_TENANT_ID` / `LOCAL_TENANT_NAME` to brand tenant metadata.

## Persona log simulator
- Run `python scripts/simulate_persona_logs.py` when you want a higher-level controller that randomly assigns user-defined actions (from `data/persona_actions.json` or the default set) and streams them into the per-user audit logs.
- Use `PERSONA_LOG_COUNT` to configure how many entries each targeted user receives in the current run.
- Supply `PERSONA_TARGET_USERS` (comma-separated IDs or UPNs) to scope the update to specific accounts.
- Set `PERSONA_RESET=1` to rebuild logs for the targeted users from scratch before generating new entries.
- `PERSONA_ACTIONS_FILE`, `PERSONA_TENANT_ID`, and `PERSONA_TENANT_NAME` mirror the lower-level script’s options for custom action libraries or tenant branding.
