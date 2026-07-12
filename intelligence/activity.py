"""
Activity log — the lifecycle feed the user sees in the app.

Captures every meaningful event on a stock/signal:
    SUGGESTED   — engine produced a signal for you
    RATED       — you liked/disliked it (with a reason, e.g. "confidence too low")
    BOUGHT      — you bought it (officially in the portfolio)
    SOLD        — you exited
    REANALYZED  — engine re-checked a held position vs its prediction
    NOTE        — free-form note

This is the audit trail that makes the platform feel like it remembers what you
did and why — and feeds the learning loop (what was disliked / skipped, not just
what was acted on).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

EVENT_TYPES = {
    "SUGGESTED", "RATED", "BOUGHT", "SOLD", "REANALYZED", "NOTE",
    # Autonomous-brain events — every self-directed action is auditable.
    "AUTO_BUY", "AUTO_SELL", "AUTO_PASS", "PARAM_CHANGE", "RETRAIN", "JOB_RUN",
}


async def log_activity(
    conn,
    *,
    event_type: str,
    ticker: str | None = None,
    signal_id: int | None = None,
    rating: str | None = None,        # LIKE | DISLIKE
    note: str = "",
    payload: dict | None = None,
) -> int:
    """Insert an activity event. Returns its id."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event_type {event_type}; expected one of {EVENT_TYPES}")
    created_at = datetime.now(timezone.utc)
    event_id = await conn.fetchval(
        """
        INSERT INTO activity_log (event_type, ticker, signal_id, rating, note, payload, created_at)
        VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id
        """,
        event_type, ticker, signal_id, rating, note,
        json.dumps(payload) if payload else None,
        created_at,
    )

    # Best-effort push to the in-process SSE bus (backend/services/activity_bus)
    # so connected frontends see the event immediately. Never allowed to fail
    # the DB write — this module is also used from offline scripts where the
    # backend package may not be importable.
    try:
        from backend.services import activity_bus
        activity_bus.publish({
            "id": event_id,
            "event_type": event_type,
            "ticker": ticker,
            "signal_id": signal_id,
            "rating": rating,
            "note": note,
            "payload": payload,
            "created_at": created_at.isoformat(),
        })
    except Exception:
        log.debug("activity_bus publish skipped/failed", exc_info=True)

    return event_id


async def rate_signal(conn, signal_id: int, ticker: str, like: bool, reason: str = "") -> int:
    """Record that the user liked/disliked a suggested signal, with an optional reason."""
    return await log_activity(
        conn, event_type="RATED", ticker=ticker, signal_id=signal_id,
        rating="LIKE" if like else "DISLIKE", note=reason,
    )


async def get_activity_feed(conn, limit: int = 100) -> list[dict]:
    """Reverse-chronological activity feed for the UI."""
    rows = await conn.fetch(
        """
        SELECT a.id, a.event_type, a.ticker, a.signal_id, a.rating, a.note,
               a.payload, a.created_at, st.name
        FROM activity_log a
        LEFT JOIN stocks st ON st.ticker = a.ticker
        ORDER BY a.created_at DESC
        LIMIT $1
        """,
        limit,
    )
    out = []
    for r in rows:
        d = dict(r)
        if d.get("payload"):
            try:
                d["payload"] = json.loads(d["payload"])
            except (TypeError, json.JSONDecodeError):
                pass
        out.append(d)
    return out
