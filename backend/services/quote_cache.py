"""
In-memory live-quote cache — the consumer-side counterpart to
data/pipeline/upstox_feed.py (producer side, built by a parallel agent).

Holds the latest tick per symbol so routers (e.g. backend/routers/ws_prices.py)
can serve snapshots/deltas without hitting the DB or the upstream feed directly.

Tick contract this cache expects (per WHAT_TO_DO_NEXT.txt Phase 1 / Section 4
and docs/UPSTOX_API_NOTES.md):
    {"symbol": <ticker string>, "ltp": float, "close": float | None, "ts": <unix timestamp float>}

This is a cache, not a ledger — reads are eventually consistent. We still guard
writes with an asyncio.Lock to avoid interleaved dict mutation under concurrency.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

log = logging.getLogger(__name__)

_store: dict[str, dict[str, Any]] = {}
_lock = asyncio.Lock()


async def update(tick: dict[str, Any]) -> None:
    """Ingest one tick and update the cache. Expects the contract above."""
    symbol = tick.get("symbol")
    if not symbol:
        log.warning("quote_cache.update: dropping tick with no symbol: %r", tick)
        return

    entry = {
        "ltp": tick.get("ltp"),
        "close": tick.get("close"),
        "ts": tick.get("ts", time.time()),
    }
    async with _lock:
        _store[symbol] = entry


def get(symbol: str) -> dict[str, Any] | None:
    """Latest known quote for a symbol, or None if untracked."""
    entry = _store.get(symbol)
    return dict(entry) if entry is not None else None


def get_all() -> dict[str, dict[str, Any]]:
    """Snapshot of the full cache: {symbol: {"ltp", "close", "ts"}}."""
    return {sym: dict(entry) for sym, entry in _store.items()}


def age_seconds(symbol: str) -> float | None:
    """Seconds since the last tick for `symbol`, or None if untracked."""
    entry = _store.get(symbol)
    if entry is None:
        return None
    return time.time() - entry["ts"]


async def start_quote_cache_feed(instrument_keys: list[str], symbol_map: dict[str, str] | None = None) -> None:
    """
    FastAPI lifespan-compatible startup hook.

    Lazily/defensively imports data.pipeline.upstox_feed, which is being built by
    a parallel agent and may not exist yet. If it's missing, log a clear message
    and return without blocking app startup — it'll be picked up automatically
    the next time the app restarts once that module lands.

    symbol_map: {instrument_key: ticker/index-symbol} — forwarded to
    upstox_feed.start_feed so ticks land in this cache keyed by the same
    symbol strings the frontend expects (e.g. "Nifty 50", not the raw
    instrument key).
    """
    try:
        from data.pipeline.upstox_feed import start_feed
    except ImportError as e:
        log.warning(
            "quote_cache: data.pipeline.upstox_feed not available yet (%s) — "
            "live price feed disabled until that module lands; will retry on next app restart.",
            e,
        )
        return

    try:
        await start_feed(instrument_keys, on_tick=update, symbol_map=symbol_map)
    except Exception:
        log.exception("quote_cache: upstox_feed.start_feed raised — live price feed not started")
