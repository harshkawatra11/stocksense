"""Thread-safe in-memory state shared by the feed threads and the API (one LabStore per app)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from stocksense.core.clock import Clock


@dataclass
class FeedStat:
    """Health of one background feed."""

    interval_s: float = 0.0
    last_ok: datetime | None = None
    last_err: str | None = None  # "<ExceptionName>: <first 120 chars>"; never carries secrets
    errors: int = 0


@dataclass
class LabStore:
    """All mutable dashboard state; read and write under `lock` and copy out before returning."""

    lock: threading.RLock = field(default_factory=threading.RLock)
    tape: dict[str, Any] = field(default_factory=dict)
    sx: dict[str, Any] = field(default_factory=dict)
    univ: dict[str, Any] = field(default_factory=dict)
    flow: dict[str, Any] = field(default_factory=dict)
    feeds: dict[str, FeedStat] = field(default_factory=dict)
    catchup: dict[str, Any] = field(default_factory=lambda: {"state": "idle", "steps": []})
    slow_until: dict[str, float] = field(default_factory=dict)  # source -> epoch seconds (429 back-off)

    def ok(self, feed: str, now: datetime) -> None:
        """Record a successful iteration of `feed` at `now`."""
        with self.lock:
            self.feeds.setdefault(feed, FeedStat()).last_ok = now

    def fail(self, feed: str, now: datetime, err: BaseException) -> None:
        """Record a failed iteration: count it and keep a short, secret-free description."""
        with self.lock:
            stat = self.feeds.setdefault(feed, FeedStat())
            stat.errors += 1
            stat.last_err = f"{type(err).__name__}: {str(err)[:120]}"

    def set_interval(self, feed: str, seconds: float) -> None:
        """Remember the nominal interval of `feed` (shown in /api/status)."""
        with self.lock:
            self.feeds.setdefault(feed, FeedStat()).interval_s = float(seconds)

    def snapshot_status(self) -> dict[str, Any]:
        """JSON-ready copy of feed health and catch-up progress; `ready` is true once the tape has data."""
        with self.lock:
            feeds = {
                name: {
                    "interval_s": s.interval_s,
                    "last_ok": s.last_ok.isoformat() if s.last_ok else None,
                    "errors": s.errors,
                    "last_err": s.last_err,
                }
                for name, s in self.feeds.items()
            }
            catchup = {
                "state": self.catchup.get("state"),
                "steps": [dict(x) for x in self.catchup.get("steps", [])],
            }
            return {
                "ready": bool(self.tape) or bool(self.sx.get("indices")),
                "feeds": feeds,
                "catchup": catchup,
                "progress": self.univ.get("progress", ""),
                "classified": len(self.univ.get("cls", {})),
                "history": len(self.univ.get("hist", {})),
            }


class TTLCache:
    """Thread-safe get-or-compute cache with per-key expiry (replaces market-lab app.py `cached`)."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._items: dict[Any, tuple[float, Any]] = {}

    def get(self, key: Any, ttl_s: float, fn: Callable[[], Any]) -> Any:
        """Cached value for `key` if younger than `ttl_s` seconds, else compute it with `fn()` and keep it."""
        now = self._clock.now().timestamp()
        with self._lock:
            hit = self._items.get(key)
            if hit is not None and now - hit[0] < ttl_s:
                return hit[1]
        value = fn()
        with self._lock:
            self._items[key] = (now, value)
        return value
