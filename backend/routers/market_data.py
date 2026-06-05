"""
Live market data endpoint — fetches index quotes from Angel One SmartAPI.
Returns Nifty 50 and Sensex LTP + change. Falls back gracefully if
Angel One is unavailable (network timeout, missing credentials, etc.).
"""
import asyncio
import logging
import time
from functools import lru_cache

from fastapi import APIRouter

from config import settings

router = APIRouter()
log = logging.getLogger(__name__)

# Cache quote for 15 seconds to avoid hammering the API on rapid frontend polls
_cache: dict = {"data": None, "ts": 0.0}
CACHE_TTL = 15  # seconds

# Angel One instrument tokens
# exchange, tradingsymbol, token, display_name
INDICES = [
    ("NSE", "Nifty 50",  "99926000", "Nifty 50"),
    ("BSE", "Sensex",    "99919000", "Sensex"),
]


def _fetch_quotes_sync() -> list[dict]:
    """Synchronous Angel One LTP fetch — runs in a thread pool."""
    from data.pipeline.fetch_live import get_session, get_ltp

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
        return result
    except Exception as e:
        log.warning("Angel One quote fetch failed: %s", e)
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
