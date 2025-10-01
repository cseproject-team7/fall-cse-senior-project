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
