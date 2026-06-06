"""
Angel One SmartAPI — live data feed.

Provides:
  - get_session()          : authenticate and return SmartConnect session
  - get_ltp(tokens)        : fetch last traded price for a list of tokens
  - get_quote(tokens)      : full OHLCV + OI quote for a list of tokens
  - start_websocket(...)   : stream live ticks via SmartWebSocket
  - search_token(ticker)   : resolve NSE ticker symbol → SmartAPI token

Usage:
    from data.pipeline.fetch_live import get_session, get_ltp
    obj = get_session()
    prices = get_ltp(obj, [("NSE", "RELIANCE", "2885")])
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

import pyotp
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

from config import settings

log = logging.getLogger(__name__)

# SmartAPI exchange + segment constants
NSE_CM = "NSE"       # Cash/equity
NSE_FO = "NFO"       # F&O
FEED_MODE_LTP = 1    # Last Traded Price only
FEED_MODE_QUOTE = 2  # Full OHLCV quote
FEED_MODE_SNAP = 3   # Snap quote (bid/ask depth)

_session: SmartConnect | None = None


def _iter_fetched(fetched: Any) -> list[dict]:
    """
    Normalize the ``data.fetched`` field of a getMarketData response into a flat
    list of instrument dicts. SmartAPI returns it as a list, but harden against a
    dict-of-lists shape so a future API change can't silently break the poller.
    """
    if isinstance(fetched, list):
        return fetched
    if isinstance(fetched, dict):
        items: list[dict] = []
        for v in fetched.values():
            if isinstance(v, list):
                items.extend(v)
            elif isinstance(v, dict):
                items.append(v)
        return items
    return []


def _generate_totp() -> str:
    return pyotp.TOTP(settings.ANGEL_ONE_TOTP_KEY).now()


def get_session(force_new: bool = False) -> SmartConnect:
    """Authenticate with Angel One and return a live SmartConnect session."""
    global _session
    if _session is not None and not force_new:
        return _session

    obj = SmartConnect(api_key=settings.ANGEL_ONE_API_KEY)
    totp = _generate_totp()
    data = obj.generateSession(
        settings.ANGEL_ONE_CLIENT_ID,
        settings.ANGEL_ONE_PIN,
        totp,
    )
    if data["status"] is False:
        raise RuntimeError(f"Angel One login failed: {data['message']}")

    _session = obj
    log.info("Angel One session established for client %s", settings.ANGEL_ONE_CLIENT_ID)
    return obj


def refresh_session() -> SmartConnect:
    """Force a new session (call when token expires, typically after 24h)."""
    return get_session(force_new=True)


def search_token(obj: SmartConnect, ticker: str, exchange: str = "NSE") -> str | None:
    """
    Resolve NSE ticker symbol (e.g. 'RELIANCE') to SmartAPI instrument token.
    Returns the token string or None if not found.
    """
    try:
        data = obj.searchScrip(exchange, ticker)
        if data["status"] and data["data"]:
            return str(data["data"][0]["symboltoken"])
    except Exception as e:
        log.warning("Token lookup failed for %s: %s", ticker, e)
    return None


def get_ltp(obj: SmartConnect, tokens: list[tuple[str, str, str]]) -> dict[str, float]:
    """
    Fetch last traded price for multiple instruments.

    tokens: list of (exchange, tradingsymbol, token)
            e.g. [("NSE", "RELIANCE", "2885"), ("NSE", "INFY", "1594")]

    Returns: {tradingsymbol: ltp}
    """
    exchange_tokens: dict[str, list[str]] = {}
    symbol_map: dict[str, str] = {}  # token → symbol

    for exchange, symbol, token in tokens:
        exchange_tokens.setdefault(exchange, []).append(token)
        symbol_map[token] = symbol

    payload = {
        "mode": "LTP",
        "exchangeTokens": exchange_tokens,
    }

    try:
        resp = obj.getMarketData(**payload)
        if not resp["status"]:
            log.warning("getMarketData failed: %s", resp.get("message"))
            return {}

        # SmartAPI returns data.fetched as a LIST of instrument dicts.
        result: dict[str, float] = {}
        for item in _iter_fetched(resp["data"]["fetched"]):
            sym = symbol_map.get(str(item.get("symbolToken")), item.get("tradingSymbol"))
            result[sym] = float(item.get("ltp", 0))
        return result
    except Exception as e:
        log.error("get_ltp error: %s", e)
        return {}


def get_quote(obj: SmartConnect, tokens: list[tuple[str, str, str]]) -> dict[str, dict]:
    """
    Fetch full OHLCV quote for multiple instruments.

    Returns: {tradingsymbol: {open, high, low, close, ltp, volume, ...}}
    """
    exchange_tokens: dict[str, list[str]] = {}
    symbol_map: dict[str, str] = {}

    for exchange, symbol, token in tokens:
        exchange_tokens.setdefault(exchange, []).append(token)
        symbol_map[token] = symbol

    payload = {
        "mode": "FULL",
        "exchangeTokens": exchange_tokens,
    }

    try:
        resp = obj.getMarketData(**payload)
        if not resp["status"]:
            log.warning("getMarketData FULL failed: %s", resp.get("message"))
            return {}

        result: dict[str, dict] = {}
        for item in _iter_fetched(resp["data"]["fetched"]):
            sym = symbol_map.get(str(item.get("symbolToken")), item.get("tradingSymbol"))
            result[sym] = {
                "ltp": float(item.get("ltp", 0)),
                "open": float(item.get("open", 0)),
                "high": float(item.get("high", 0)),
                "low": float(item.get("low", 0)),
                "close": float(item.get("close", 0)),
                "volume": int(item.get("tradeVolume", 0)),
                "token": item.get("symbolToken"),
            }
        return result
    except Exception as e:
        log.error("get_quote error: %s", e)
        return {}


def start_websocket(
    obj: SmartConnect,
    tokens: list[tuple[str, str]],   # list of (exchange, token)
    on_tick: Callable[[dict], None],
    mode: int = FEED_MODE_LTP,
) -> SmartWebSocketV2:
    """
    Start a SmartWebSocketV2 stream.

    tokens: list of (exchange_type, token)
            exchange_type: "1" = NSE CM, "2" = NSE FO
    on_tick: callback called with each tick dict

    Returns the ws object (call ws.close_connection() to stop).
    """
    feed_token = obj.getfeedToken()
    client_code = settings.ANGEL_ONE_CLIENT_ID

    ws = SmartWebSocketV2(
        feed_token,
        settings.ANGEL_ONE_API_KEY,
        client_code,
        obj,
    )

    token_list = [{"exchangeType": int(ex), "tokens": [tok]} for ex, tok in tokens]

    def _on_open(wsapp):
        log.info("SmartWebSocket connected — subscribing %d tokens", len(tokens))
        ws.subscribe(client_code, mode, token_list)

    def _on_data(wsapp, message):
        on_tick(message)

    def _on_error(wsapp, error):
        log.error("SmartWebSocket error: %s", error)

    def _on_close(wsapp):
        log.info("SmartWebSocket closed")

    ws.on_open = _on_open
    ws.on_data = _on_data
    ws.on_error = _on_error
    ws.on_close = _on_close

    ws.connect()
    return ws


def test_connection() -> bool:
    """Quick smoke test — authenticate and fetch Nifty 50 LTP."""
    try:
        obj = get_session(force_new=True)
        # Nifty 50 index token = 99926000 on NSE
        result = get_ltp(obj, [("NSE", "Nifty 50", "99926000")])
        if result:
            log.info("Angel One connection OK. Nifty 50 LTP: %s", result)
            return True
        log.warning("Angel One connected but LTP fetch returned empty")
        return False
    except Exception as e:
        log.error("Angel One connection test failed: %s", e)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ok = test_connection()
    print("Connection:", "OK" if ok else "FAILED")
