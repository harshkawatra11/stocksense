"""
Claude Code CLI access gate (Phase F2). StockSense never touches
`~/.claude/.credentials.json` or any auth token -- it only ever calls
`claude auth status --json` (read-only) to learn WHO is currently logged
in, and gates its own Claude-invoking code paths behind a LOCAL
authorize/decline flag the user controls from the desktop app.

This is what makes "my Claude Pro account may change in future" safe by
construction: detection, not storage. Every check compares the
currently-logged-in email against the one the user last explicitly
approved; a mismatch auto-revokes access rather than silently carrying
authorization over to a different account.

Enforced at a single choke point -- `agent/claude_cli.invoke()` itself,
not duplicated at every caller (foreman/planner.py, foreman/codegen.py,
foreman/assess.py, statements/report.py, rag/agent.py) -- so a caller
that forgets to check cannot accidentally bypass this. On ambiguity
(the auth check itself fails, e.g. the database is momentarily busy
because another job holds DuckDB's write lock), this fails CLOSED --
the same discipline `foreman/verifier.py`'s research-gate fix already
established: a control the user relies on to mean "off" must actually
mean off, not silently pass open on an unrelated error.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass

_KEY_ACCESS_GRANTED = "claude_access_granted"
_KEY_APPROVED_EMAIL = "claude_access_approved_email"


class ClaudeAccessNotGranted(Exception):
    """Raised when Claude CLI access has not been authorized, or the
    detected account no longer matches the one that was approved."""


@dataclass(frozen=True)
class ClaudeAuthStatus:
    logged_in: bool
    email: str | None
    plan: str | None
    raw_error: str | None = None


def check_claude_auth(timeout_s: int = 15) -> ClaudeAuthStatus:
    """Read-only: `claude auth status --json`. Never touches
    `~/.claude/.credentials.json` or any token directly -- this is the
    ONLY interaction this module has with Claude's own auth state."""
    binary = shutil.which("claude")
    if binary is None:
        return ClaudeAuthStatus(logged_in=False, email=None, plan=None, raw_error="claude CLI not found on PATH")
    try:
        proc = subprocess.run([binary, "auth", "status", "--json"], capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return ClaudeAuthStatus(logged_in=False, email=None, plan=None, raw_error="claude auth status timed out")
    except Exception as e:  # noqa: BLE001 -- any failure here means "can't confirm logged in," never a crash
        return ClaudeAuthStatus(logged_in=False, email=None, plan=None, raw_error=str(e))

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return ClaudeAuthStatus(logged_in=False, email=None, plan=None, raw_error=f"could not parse auth status: {proc.stdout[:200]}")

    return ClaudeAuthStatus(
        logged_in=bool(data.get("loggedIn")), email=data.get("email"), plan=data.get("subscriptionType"),
    )


def is_access_granted(store) -> bool:
    """True only if the local flag is set AND the currently-logged-in
    account still matches the one that was approved. Any DB read
    failure (e.g. another job holds the write lock) is treated as NOT
    granted -- fail closed, per the module docstring."""
    try:
        granted = store.get_app_setting(_KEY_ACCESS_GRANTED)
    except Exception:  # noqa: BLE001
        return False
    if granted != "true":
        return False

    approved_email = store.get_app_setting(_KEY_APPROVED_EMAIL)
    current = check_claude_auth()
    if not current.logged_in or current.email != approved_email:
        revoke_access(store)
        return False
    return True


def grant_access(store) -> ClaudeAuthStatus:
    """Explicit user action from the UI's Authorize control. Records
    the CURRENTLY logged-in email as the approved one -- future checks
    compare against this, never against a name the user typed by hand."""
    status = check_claude_auth()
    if not status.logged_in:
        raise ClaudeAccessNotGranted(
            f"cannot grant access: no Claude CLI session is logged in ({status.raw_error or 'not logged in'})"
        )
    store.set_app_setting(_KEY_ACCESS_GRANTED, "true")
    store.set_app_setting(_KEY_APPROVED_EMAIL, status.email)
    return status


def revoke_access(store) -> None:
    store.set_app_setting(_KEY_ACCESS_GRANTED, "false")
    store.set_app_setting(_KEY_APPROVED_EMAIL, None)


def require_claude_access(store) -> None:
    """The single enforcement point, called from
    `agent.claude_cli.invoke()` before it ever shells out to `claude`."""
    if not is_access_granted(store):
        raise ClaudeAccessNotGranted(
            "Claude CLI access has not been authorized (or the logged-in account changed) -- "
            "grant access from the desktop app's Claude Connection panel first"
        )
