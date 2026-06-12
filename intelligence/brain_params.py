"""
Brain parameters — the knobs the system tunes on itself.

Every adaptive parameter lives in the brain_params table with hard min/max
bounds. The pipeline and auto-trader read them at runtime; the nightly Claude
calibration step (intelligence/calibration.py) may adjust them — always clamped
to bounds, always audited in brain_param_history + activity_log. There is no
UI to change these: the brain runs itself.

Tables: brain_params, brain_param_history (see data/db/schema_v4_brain.sql).
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

# Fallback defaults if the table is empty (mirror the schema seed).
DEFAULTS: dict[str, float] = {
    "confidence_threshold": 0.55,
    "max_position_pct": 0.25,
    "max_open_positions": 5,
    "macro_nudge_cap": 0.10,
    "agreement_boost": 0.05,
}

_cache: dict[str, float] = {}
_cache_at: float = 0.0
_CACHE_TTL = 120  # seconds — refreshed at least once per pipeline run


async def get_params(conn, force_refresh: bool = False) -> dict[str, float]:
    """All brain params as {name: value}, cached briefly. Falls back to DEFAULTS."""
    global _cache, _cache_at
    if not force_refresh and _cache and (time.time() - _cache_at) < _CACHE_TTL:
        return dict(_cache)
    try:
        rows = await conn.fetch("SELECT param_name, value FROM brain_params")
        params = {r["param_name"]: float(r["value"]) for r in rows}
    except Exception as e:
        log.warning("brain_params read failed (%s) — using defaults", e)
        params = {}
    merged = {**DEFAULTS, **params}
    _cache, _cache_at = merged, time.time()
    return dict(merged)


async def get_param(conn, name: str, default: float | None = None) -> float:
    params = await get_params(conn)
    if name in params:
        return params[name]
    if default is not None:
        return default
    raise KeyError(f"Unknown brain param {name}")


async def set_param(conn, name: str, value: float, changed_by: str, reason: str = "") -> dict:
    """
    Change a parameter, clamped to its bounds. Writes the audit row and a
    PARAM_CHANGE activity event. Returns {name, old, new, clamped}.
    Raises KeyError for unknown params — calibration may not invent knobs.
    """
    row = await conn.fetchrow(
        "SELECT value, min_value, max_value FROM brain_params WHERE param_name = $1", name
    )
    if not row:
        raise KeyError(f"Unknown brain param {name}")

    old = float(row["value"])
    clamped = max(float(row["min_value"]), min(float(row["max_value"]), float(value)))
    was_clamped = clamped != float(value)

    await conn.execute(
        """
        UPDATE brain_params
        SET value = $2, updated_at = NOW(), updated_by = $3, reason = $4
        WHERE param_name = $1
        """,
        name, clamped, changed_by, reason,
    )
    await conn.execute(
        """
        INSERT INTO brain_param_history (param_name, old_value, new_value, changed_by, reason)
        VALUES ($1,$2,$3,$4,$5)
        """,
        name, old, clamped, changed_by, reason,
    )
    from intelligence.activity import log_activity
    await log_activity(
        conn, event_type="PARAM_CHANGE", note=f"{name}: {old} → {clamped}",
        payload={"param": name, "old": old, "new": clamped, "by": changed_by,
                 "reason": reason, "clamped": was_clamped},
    )

    global _cache, _cache_at
    _cache, _cache_at = {}, 0.0  # invalidate

    log.info("Brain param %s: %s → %s (by %s%s) — %s",
             name, old, clamped, changed_by, ", clamped" if was_clamped else "", reason)
    return {"name": name, "old": old, "new": clamped, "clamped": was_clamped}


async def get_param_history(conn, limit: int = 100) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT param_name, old_value, new_value, changed_by, reason, changed_at
        FROM brain_param_history ORDER BY changed_at DESC LIMIT $1
        """,
        limit,
    )
    return [dict(r) for r in rows]
