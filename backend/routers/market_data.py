"""
Live market data endpoint — fetches index quotes from Angel One SmartAPI.
Returns Nifty 50 and Sensex LTP + change. Falls back gracefully if
Angel One is unavailable (network timeout, missing credentials, etc.).

Provider role: Angel One is the FALLBACK-ONLY market-data provider in this app's
provider chain (see intelligence/provider_config.py: market_data_provider_chain()).
Upstox is the primary live feed once wired (data/pipeline/upstox_client.py /
upstox_feed.py). Angel One is IP-bound and frequently blocked on home networks —
this module (and its circuit breaker below) exists purely as a safety net for when
Upstox is unavailable or not yet configured, not as the default source. Do not
promote this to primary without updating provider_config.py's chain.
"""
import asyncio
import logging
import time
from functools import lru_cache

from fastapi import APIRouter

from config import settings

router = APIRouter()
log = logging.getLogger(__name__)

# Suppress SmartAPI library's own noisy ERROR-level connection logs
logging.getLogger("smartConnect").setLevel(logging.CRITICAL)

# Cache quote for 15 seconds to avoid hammering the API on rapid frontend polls
_cache: dict = {"data": None, "ts": 0.0}
CACHE_TTL = 15  # seconds

# Circuit breaker: after a network failure back off for 5 minutes before retrying
_BACKOFF_SECS = 300
_fail_ts: float = 0.0   # time of last failure
_last_fetch_error: str | None = None

# Angel One instrument tokens
# exchange, tradingsymbol, token, display_name
INDICES = [
    ("NSE", "Nifty 50",  "99926000", "Nifty 50"),
    ("BSE", "Sensex",    "99919000", "Sensex"),
]


def _fetch_quotes_sync() -> list[dict]:
    """Synchronous Angel One LTP fetch — runs in a thread pool.

    Fallback-only path (see module docstring): called when there's no primary
    (Upstox) index-quote source wired into this endpoint yet.
    """
    global _last_fetch_error, _fail_ts
    from data.pipeline.fetch_live import get_session, get_ltp

    # Circuit breaker: skip network call if we failed recently
    if _fail_ts and (time.monotonic() - _fail_ts) < _BACKOFF_SECS:
        return [{"symbol": d, "ltp": None, "available": False} for _, _, _, d in INDICES]

    try:
        obj = get_session()
        tokens = [(ex, sym, tok) for ex, sym, tok, _ in INDICES]
        prices = get_ltp(obj, tokens)

        result = []
        for ex, sym, tok, display in INDICES:
            ltp = prices.get(sym)
            result.append({
                "symbol": display,
                "ltp": ltp,
                "available": ltp is not None,
            })
        _last_fetch_error = None
        _fail_ts = 0.0  # recovered — reset circuit breaker
        return result
    except Exception as e:
        msg = str(e)
        _fail_ts = time.monotonic()  # arm circuit breaker
        if msg != _last_fetch_error:
            log.warning("Angel One LTP unavailable, backing off %ds: %s", _BACKOFF_SECS, msg[:120])
            _last_fetch_error = msg
        return [{"symbol": d, "ltp": None, "available": False} for _, _, _, d in INDICES]


@router.get("/indices")
async def get_indices():
    """
    Returns live LTP for Nifty 50 and Sensex.
    Cached for 15s. Returns available=false per index if Angel One is unreachable.
    """
    now = time.monotonic()
    if _cache["data"] is not None and (now - _cache["ts"]) < CACHE_TTL:
        return {"indices": _cache["data"], "cached": True}

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _fetch_quotes_sync)

    _cache["data"] = data
    _cache["ts"] = now
    return {"indices": data, "cached": False}
