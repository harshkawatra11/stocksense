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
from collections import deque
from typing import Any

log = logging.getLogger(__name__)

_store: dict[str, dict[str, Any]] = {}
# Tiny per-symbol ring buffer of recent LTPs (for frontend sparklines).
# Bounded (HISTORY_LEN) and in-process only — not a time series store.
HISTORY_LEN = 30
_history: dict[str, deque[float]] = {}
_lock = asyncio.Lock()

# ---- Feed health / lifecycle state -----------------------------------
# The running MarketDataStreamerV3 instance (or None before first connect /
# while a reconnect backoff is in progress). Needed so add_to_watchlist()
# can call .subscribe() on an already-open connection instead of forcing a
# full reconnect for every new position.
_streamer: Any = None
_connected: bool = False
_last_connect_error: str | None = None
_last_tick_ts: float | None = None
_subscribed_keys: set[str] = set()
_symbol_map: dict[str, str] = {}
_feed_lock = asyncio.Lock()

RECONNECT_BACKOFFS = [10, 30, 60]  # seconds; caps at the last value thereafter


def feed_status() -> dict[str, Any]:
    """Snapshot for backend/routers/system_health.py's quote_feed component."""
    now = time.time()
    age = (now - _last_tick_ts) if _last_tick_ts is not None else None
    return {
        "connected": _connected,
        "subscribed_count": len(_subscribed_keys),
        "last_tick_age_seconds": age,
        "last_connect_error": _last_connect_error,
    }


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
        ltp = entry.get("ltp")
        if ltp is not None:
            buf = _history.get(symbol)
            if buf is None:
                buf = _history[symbol] = deque(maxlen=HISTORY_LEN)
            # Only append actual movement — a flat resend adds no sparkline info.
            if not buf or buf[-1] != ltp:
                buf.append(float(ltp))

    global _last_tick_ts
    _last_tick_ts = time.time()


def get(symbol: str) -> dict[str, Any] | None:
    """Latest known quote for a symbol, or None if untracked."""
    entry = _store.get(symbol)
    return dict(entry) if entry is not None else None


def get_all() -> dict[str, dict[str, Any]]:
    """Snapshot of the full cache: {symbol: {"ltp", "close", "ts"}}."""
    return {sym: dict(entry) for sym, entry in _store.items()}


def get_history(symbol: str) -> list[float]:
    """Recent LTPs (up to HISTORY_LEN) for a symbol, oldest first."""
    buf = _history.get(symbol)
    return list(buf) if buf else []


def get_all_history() -> dict[str, list[float]]:
    """{symbol: [recent LTPs, oldest first]} for every tracked symbol."""
    return {sym: list(buf) for sym, buf in _history.items() if buf}


def age_seconds(symbol: str) -> float | None:
    """Seconds since the last tick for `symbol`, or None if untracked."""
    entry = _store.get(symbol)
    if entry is None:
        return None
    return time.time() - entry["ts"]


async def add_to_watchlist(instrument_key: str, symbol: str) -> bool:
    """
    Add one ticker to the already-open live feed without a full reconnect.

    MarketDataStreamerV3 (upstox-python-sdk) exposes .subscribe(keys, mode)
    on a connected streamer, so a same-day BUY can start getting live ticks
    within one call instead of waiting for the next app restart (which is
    what silently starved intraday positions of fast-stop coverage before).

    Returns True if the subscribe call was issued (streamer connected),
    False otherwise (feed not up yet / instrument_key already tracked with
    no streamer — caller should treat False as "will fall back to REST/slow
    path", not as an error to surface loudly every time).
    """
    global _subscribed_keys, _symbol_map
    async with _feed_lock:
        if instrument_key in _subscribed_keys:
            return True
        if _streamer is None or not _connected:
            log.warning(
                "quote_cache.add_to_watchlist: feed not connected — %s (%s) will rely on "
                "REST/slow-path fallback until the feed reconnects",
                symbol, instrument_key,
            )
            return False
        try:
            _symbol_map[instrument_key] = symbol
            # upstox_feed keeps its own key->symbol map for decoding ticks;
            # update that too so incoming messages resolve to the right symbol.
            from data.pipeline import upstox_feed
            upstox_feed._key_to_symbol[instrument_key] = symbol
            _streamer.subscribe([instrument_key], "ltpc")
            _subscribed_keys.add(instrument_key)
            log.info("quote_cache: added %s (%s) to live feed subscription", symbol, instrument_key)
            return True
        except Exception:
            log.exception("quote_cache.add_to_watchlist: subscribe() failed for %s (%s)", symbol, instrument_key)
            return False


async def start_quote_cache_feed(instrument_keys: list[str], symbol_map: dict[str, str] | None = None) -> None:
    """
    FastAPI lifespan-compatible startup hook. Runs forever: connects, and on
    any disconnect/error reconnects with backoff (10s, 30s, 60s, then holds
    at 60s) instead of giving up permanently. A dead feed used to silently
    disable ALL fast intraday stop enforcement until the next app restart —
    this loop exists so that never happens again without at least a
    surfaced `quote_feed` health component (see backend/routers/system_health.py)
    going degraded/unavailable while it's down.

    Lazily/defensively imports data.pipeline.upstox_feed. If it's genuinely
    missing (not just erroring), we still retry — the module may land via a
    hot deploy without a full app restart in dev.

    symbol_map: {instrument_key: ticker/index-symbol} — forwarded to
    upstox_feed.start_feed so ticks land in this cache keyed by the same
    symbol strings the frontend expects (e.g. "Nifty 50", not the raw
    instrument key).
    """
    global _streamer, _connected, _last_connect_error, _subscribed_keys, _symbol_map

    _symbol_map = dict(symbol_map or {})
    _subscribed_keys = set(instrument_keys)

    attempt = 0
    while True:
        try:
            from data.pipeline.upstox_feed import start_feed
        except ImportError as e:
            _connected = False
            _last_connect_error = f"upstox_feed module not importable: {e}"
            log.warning(
                "quote_cache: data.pipeline.upstox_feed not available (%s) — "
                "live price feed disabled; retrying in %ss.",
                e, RECONNECT_BACKOFFS[-1],
            )
            await asyncio.sleep(RECONNECT_BACKOFFS[-1])
            continue

        try:
            streamer = await start_feed(instrument_keys, on_tick=update, symbol_map=_symbol_map)
        except Exception:
            streamer = None
            log.exception("quote_cache: upstox_feed.start_feed raised — live price feed not started this attempt")

        if streamer is None:
            _connected = False
            _last_connect_error = _last_connect_error or "start_feed returned None (no token or SDK missing)"
            backoff = RECONNECT_BACKOFFS[min(attempt, len(RECONNECT_BACKOFFS) - 1)]
            attempt += 1
            log.warning("quote_cache: feed connect failed — retrying in %ss (attempt %d)", backoff, attempt)
            await asyncio.sleep(backoff)
            continue

        # Connected. Reset backoff state and watch the connection; the SDK's
        # own auto_reconnect handles transient WS drops internally, but if it
        # ever gives up (streamer object becomes unusable / on("close") fires
        # and never recovers), we detect that via tick staleness and force a
        # fresh start_feed() call ourselves rather than trusting it forever.
        _streamer = streamer
        _connected = True
        _last_connect_error = None
        attempt = 0
        log.info("quote_cache: feed connected (%d instrument keys)", len(instrument_keys))

        stall_threshold = 120  # seconds with zero ticks across the whole feed = assume dead
        try:
            while True:
                await asyncio.sleep(15)
                if _last_tick_ts is not None and (time.time() - _last_tick_ts) > stall_threshold:
                    log.warning(
                        "quote_cache: no ticks for any symbol in %ss — assuming feed is dead, forcing reconnect",
                        stall_threshold,
                    )
                    break
        finally:
            _connected = False
            _last_connect_error = _last_connect_error or "tick stream stalled — forced reconnect"
            try:
                _streamer.disconnect()
            except Exception:
                pass
            _streamer = None

        backoff = RECONNECT_BACKOFFS[0]
        log.info("quote_cache: reconnecting in %ss", backoff)
        await asyncio.sleep(backoff)
