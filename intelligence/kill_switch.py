"""
Application-level trading kill switch — the Upstox Agent Skill's guardrail
#6, implemented as our own DB-backed flag rather than calling Upstox's
account-level UserApi.update_kill_switch (that toggles a setting on the
user's real Upstox account; flipping it untested from this app is a risk
of its own kind, and it isn't clearly scoped to sandbox vs. live). This
flag lives entirely in our own DB and is checked before every order
placement (sandbox today, the future live path too) — halting here is
just as effective and carries none of that risk.

Stored in app_config under key 'kill_switch' as {"enabled": bool, "reason": str}.
Default: not tripped (bool absent -> False).
"""
from __future__ import annotations

import json
import logging

import asyncpg

from config import settings

log = logging.getLogger(__name__)

_KEY = "kill_switch"


async def is_tripped(conn=None) -> tuple[bool, str]:
    """(tripped, reason). Never raises — a DB hiccup fails CLOSED (tripped=True)
    so a broken kill-switch check can't silently let orders through."""
    own_conn = conn is None
    if own_conn:
        conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        row = await conn.fetchrow("SELECT value FROM app_config WHERE key = $1", _KEY)
        if row is None:
            return False, ""
        value = row["value"]
        if isinstance(value, str):
            value = json.loads(value)
        return bool(value.get("enabled")), str(value.get("reason") or "")
    except Exception as e:  # noqa: BLE001
        log.error("kill_switch.is_tripped: check failed, failing CLOSED: %s", e)
        return True, f"kill-switch check itself failed: {e}"
    finally:
        if own_conn:
            await conn.close()


async def trip(reason: str, conn=None) -> None:
    own_conn = conn is None
    if own_conn:
        conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        await conn.execute(
            """
            INSERT INTO app_config (key, value, updated_at)
            VALUES ($1, $2::jsonb, NOW())
            ON CONFLICT (key) DO UPDATE SET value = $2::jsonb, updated_at = NOW()
            """,
            _KEY, json.dumps({"enabled": True, "reason": reason}),
        )
        log.warning("Kill switch TRIPPED: %s", reason)
    finally:
        if own_conn:
            await conn.close()


async def reset(conn=None) -> None:
    own_conn = conn is None
    if own_conn:
        conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        await conn.execute(
            """
            INSERT INTO app_config (key, value, updated_at)
            VALUES ($1, $2::jsonb, NOW())
            ON CONFLICT (key) DO UPDATE SET value = $2::jsonb, updated_at = NOW()
            """,
            _KEY, json.dumps({"enabled": False, "reason": ""}),
        )
        log.info("Kill switch reset")
    finally:
        if own_conn:
            await conn.close()
