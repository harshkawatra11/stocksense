"""
Upstox live tick feed — thin wrapper around the official SDK's
MarketDataStreamerV3 (WS V3, ltpc mode). See docs/UPSTOX_API_NOTES.md §4/§6.

Do NOT hand-roll protobuf decoding here — the SDK already does it. This
module's only job is: authenticate, subscribe to the given instrument keys,
translate each SDK tick into the exact dict shape the consumer-side
quote_cache.py expects, and not swallow errors (this project just did a
"kill silent fallbacks" pass — reconnects are logged, not silently retried
into a black hole).

Tick contract handed to on_tick — DO NOT change these key names, other
agents' consumer code depends on this exact shape:
    {"symbol": <ticker>, "ltp": <float>, "close": <float | None>, "ts": <unix epoch float>}

Usage:
    from data.pipeline.upstox_feed import start_feed

    def handle_tick(tick: dict) -> None:
        print(tick)  # {"symbol": "RELIANCE", "ltp": 1234.5, "close": 1230.0, "ts": 1720700000.123}

    start_feed(["NSE_EQ|INE002A01018"], handle_tick)
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from config import settings
from data.pipeline.upstox_client import get_valid_token

log = logging.getLogger(__name__)

# instrument_key ("NSE_EQ|<ISIN>") -> ticker symbol, populated by start_feed()
# so incoming ticks (keyed by instrument_key) can be translated to the
# ticker-symbol-keyed contract the consumer side expects.
_key_to_symbol: dict[str, str] = {}


def _extract_ltpc(tick_data: dict) -> tuple[float | None, float | None]:
    """
    Pull (ltp, close) out of one instrument's feed payload. The SDK's decoded
    ltpc-mode message nests the fields under ltpc; be defensive about exact
    key casing/nesting since the SDK's dict shape isn't pinned by the docs.
    """
    ltpc = tick_data.get("ltpc") or tick_data
    ltp = ltpc.get("ltp")
    close = ltpc.get("cp", ltpc.get("close"))
    try:
        ltp = float(ltp) if ltp is not None else None
    except (TypeError, ValueError):
        ltp = None
    try:
        close = float(close) if close is not None else None
    except (TypeError, ValueError):
        close = None
    return ltp, close


def _make_message_handler(on_tick: Callable[[dict], None]) -> Callable[[dict], None]:
    def handle_message(message: dict) -> None:
        try:
            feeds = message.get("feeds") or {}
            for instrument_key, tick_data in feeds.items():
                symbol = _key_to_symbol.get(instrument_key, instrument_key)
                ltp, close = _extract_ltpc(tick_data)
                if ltp is None:
                    # No usable price in this delta (e.g. a market-status-only
                    # message) — skip rather than emit a fabricated tick.
                    continue
                on_tick({
                    "symbol": symbol,
                    "ltp": ltp,
                    "close": close,
                    "ts": time.time(),
                })
        except Exception as e:
            # Explicitly logged — do not swallow. A bad tick shouldn't kill
            # the stream, but it must be visible.
            log.error("Upstox feed: failed to process message: %s | raw=%s", e, message)

    return handle_message


async def start_feed(instrument_keys: list[str], on_tick: Callable[[dict], None], symbol_map: dict[str, str] | None = None) -> "object":
    """
    Start the Upstox MarketDataStreamerV3 in ltpc mode for the given
    instrument keys and call on_tick(dict) for every price update, with the
    dict shape documented at the top of this module.

    symbol_map: optional {instrument_key: ticker} override — if omitted, the
    raw instrument_key is used as "symbol" (callers that already resolved
    tickers via upstox_client.resolve_instrument_keys should pass the
    inverse of that mapping here).

    Returns the running MarketDataStreamerV3 instance (caller can call
    .disconnect() on it). Returns None if no valid token is available —
    caller must re-auth via upstox_client's OAuth flow first.

    Async because it's called from within an already-running event loop
    (backend/services/quote_cache.py's FastAPI lifespan task) — must NOT
    call asyncio.run() internally, that raises "cannot run event loop while
    another loop is running".
    """
    if not instrument_keys:
        log.warning("Upstox feed: no instrument keys given — nothing to subscribe to")
        return None

    token = await get_valid_token()
    if token is None:
        log.warning("Upstox feed: no valid access token — cannot start feed (re-auth required via /api/upstox/login)")
        return None

    if symbol_map:
        _key_to_symbol.update(symbol_map)

    try:
        import upstox_client
        from upstox_client.feeder.market_data_streamer_v3 import MarketDataStreamerV3
    except ImportError as e:
        log.error("Upstox feed: upstox-python-sdk not installed (`pip install upstox-python-sdk`): %s", e)
        return None

    try:
        configuration = upstox_client.Configuration()
        configuration.access_token = token
        api_client = upstox_client.ApiClient(configuration)

        streamer = MarketDataStreamerV3(
            api_client,
            instrumentKeys=instrument_keys,
            mode="ltpc",
        )
        streamer.auto_reconnect(True, interval=10, retry_count=3)
        streamer.on("message", _make_message_handler(on_tick))
        streamer.on("error", lambda err: log.error("Upstox feed error: %s", err))
        streamer.on("close", lambda *a: log.warning("Upstox feed closed: %s", a))
        streamer.connect()
        log.info("Upstox feed connected: %d instrument(s), mode=ltpc", len(instrument_keys))
        return streamer
    except Exception as e:
        log.error("Upstox feed: failed to start streamer: %s", e)
        return None


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    def _print_tick(tick: dict) -> None:
        print(tick)

    asyncio.run(start_feed(["NSE_INDEX|Nifty 50"], _print_tick, symbol_map={"NSE_INDEX|Nifty 50": "Nifty 50"}))
