"""
In-process pub/sub for activity_log events.

intelligence.activity.log_activity publishes each new event here (best-effort,
never blocks or fails the DB write); the SSE endpoint /api/stream/activity
fans events out to connected frontend clients so the Live tab's activity feed
updates incrementally instead of full-page refetching.

Deliberately simple: one bounded asyncio.Queue per connected client, in the
backend process only. Events logged by other processes (standalone scripts,
cron jobs run outside uvicorn) are NOT seen here — the frontend keeps a slow
poll as fallback for those.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)

_QUEUE_MAX = 200
_subscribers: set[asyncio.Queue] = set()


def subscribe() -> asyncio.Queue:
    """Register a new subscriber queue. Caller must unsubscribe() when done."""
    q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX)
    _subscribers.add(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    _subscribers.discard(q)


def publish(event: dict[str, Any]) -> None:
    """Fan an event out to all subscribers. Non-blocking; drops on full queue."""
    for q in list(_subscribers):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            # Slow/stuck client — drop the event for it rather than block.
            log.debug("activity_bus: subscriber queue full, dropping event")
        except Exception:
            log.exception("activity_bus: publish to subscriber failed")


def subscriber_count() -> int:
    return len(_subscribers)
