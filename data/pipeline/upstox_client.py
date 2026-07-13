"""
Upstox data-layer foundation — OAuth (daily token) + REST historical candles
+ instrument-key resolution.

Why Upstox: primary live/intraday data source (Stage 1 of WHAT_TO_DO_NEXT.txt),
replacing the fake 5-minute Groww snapshot. See docs/UPSTOX_API_NOTES.md for
the full researched reference this module implements against.

Auth shape mirrors data/pipeline/fetch_groww.py's daily-login pattern: Upstox
access tokens expire once per (IST) day, so there is no long-lived session to
cache in memory across process restarts — instead we cache the token + its
acquisition timestamp in the `upstox_token` DB table (schema_v7_upstox.sql)
and treat "acquired today, IST" as valid.

Analytics Token (correction, 2026-07-11): Upstox also offers a special
"Analytics Token" — generated ONCE from the developer portal, no daily
re-authorization, and it specifically powers the Market Data and
Realtime & Streaming APIs (the only APIs this module's data layer needs).
The daily OAuth access token below still exists and is still required for
order-capable APIs (not in scope — StockSense stays paper-only), but the
live feed and historical candles should prefer the Analytics Token when
one is configured, since it removes the daily-re-auth requirement entirely
for the data path. See docs/UPSTOX_API_NOTES.md and the "Learn how to
generate an analytics token" page under Upstox's Getting Started docs.

Provides:
  - get_authorization_url(state=None)      : build the OAuth dialog URL (order-API path)
  - exchange_code_for_token(code)          : POST code -> access_token, stores it
  - store_token(token)                     : persist a token (used by the callback route)
  - get_valid_token()                      : cached daily OAuth token if acquired today (IST), else None
  - get_data_token()                       : Analytics Token if configured, else falls back to get_valid_token()
  - resolve_instrument_key(ticker)         : "NSE_EQ|<ISIN>" lookup from stocks.isin
  - get_historical_candles(...)            : REST V3 historical/intraday candles (uses get_data_token)

Everything degrades explicitly: missing creds or a failed/expired token makes
get_valid_token() return None and callers must re-auth via the OAuth flow —
no silent fallback to stale/fake data (per the project's "kill silent
fallbacks" pass).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import asyncpg
import httpx

from config import settings

log = logging.getLogger(__name__)

AUTH_DIALOG_URL = "https://api.upstox.com/v2/login/authorization/dialog"
TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"
HISTORICAL_CANDLE_URL = "https://api.upstox.com/v3/historical-candle"

IST = timezone(timedelta(hours=5, minutes=30))


def creds_present() -> bool:
    return bool(settings.UPSTOX_CLIENT_ID and settings.UPSTOX_CLIENT_SECRET)


# --------------------------------------------------------------------------- #
# OAuth — authorization dialog + code exchange                                #
# --------------------------------------------------------------------------- #

def get_authorization_url(state: str | None = None) -> str:
    """Build the Upstox OAuth authorization dialog URL to redirect the user to."""
    params = {
        "response_type": "code",
        "client_id": settings.UPSTOX_CLIENT_ID,
        "redirect_uri": settings.UPSTOX_REDIRECT_URI,
    }
    if state:
        params["state"] = state
    return f"{AUTH_DIALOG_URL}?{urlencode(params)}"


async def exchange_code_for_token(code: str) -> str | None:
    """
    Exchange a single-use OAuth auth code for an access token, and persist it.
    Returns the access token, or None if creds are missing or the exchange fails.
    """
    if not creds_present():
        log.warning("Upstox credentials not configured — cannot exchange code (set UPSTOX_CLIENT_ID + UPSTOX_CLIENT_SECRET in .env)")
        return None

    payload = {
        "code": code,
        "client_id": settings.UPSTOX_CLIENT_ID,
        "client_secret": settings.UPSTOX_CLIENT_SECRET,
        "redirect_uri": settings.UPSTOX_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(TOKEN_URL, data=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.warning("Upstox token exchange failed: %s", e)
        return None

    token = data.get("access_token")
    if not token:
        log.warning("Upstox token exchange returned no access_token: %s", data)
        return None

    await store_token(token)
    log.info("Upstox access token acquired and stored.")
    return token


# --------------------------------------------------------------------------- #
# Token cache (DB-backed, daily expiry)                                       #
# --------------------------------------------------------------------------- #

async def store_token(token: str) -> None:
    """Persist a newly acquired access token with the current timestamp."""
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        await conn.execute(
            "INSERT INTO upstox_token (token, acquired_at) VALUES ($1, NOW())",
            token,
        )
    finally:
        await conn.close()


async def get_valid_token() -> str | None:
    """
    Returns the most recently cached access token if it was acquired today
    (IST) — Upstox tokens expire daily. Returns None if no token is cached
    or the cached one is stale, in which case the caller must re-auth via
    get_authorization_url() / exchange_code_for_token().
    """
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        row = await conn.fetchrow(
            "SELECT token, acquired_at FROM upstox_token ORDER BY acquired_at DESC LIMIT 1"
        )
    finally:
        await conn.close()

    if row is None:
        return None

    acquired_at: datetime = row["acquired_at"]
    if acquired_at.tzinfo is None:
        acquired_at = acquired_at.replace(tzinfo=timezone.utc)

    acquired_ist_date = acquired_at.astimezone(IST).date()
    today_ist_date = datetime.now(timezone.utc).astimezone(IST).date()

    if acquired_ist_date != today_ist_date:
        log.info("Cached Upstox token is from %s (not today, IST) — re-auth required", acquired_ist_date)
        return None

    return row["token"]


async def get_data_token() -> str | None:
    """
    Token to use for Market Data / Realtime & Streaming APIs (live feed,
    historical candles) — the calls this module's data layer actually needs.

    Prefers the Analytics Token (settings.UPSTOX_ANALYTICS_TOKEN): generated
    once from the developer portal, no daily re-authorization required. Falls
    back to the daily OAuth access token (get_valid_token()) if no analytics
    token is configured, so the data layer still works via the OAuth flow
    while you're setting up the analytics token for the first time.

    Order-capable APIs must NOT use this function — they require the daily
    OAuth access token specifically; call get_valid_token() directly for those
    (not currently used anywhere, since StockSense stays paper-only today).
    """
    if settings.UPSTOX_ANALYTICS_TOKEN:
        return settings.UPSTOX_ANALYTICS_TOKEN
    return await get_valid_token()


# --------------------------------------------------------------------------- #
# Instrument key resolution                                                   #
# --------------------------------------------------------------------------- #

async def resolve_instrument_key(ticker: str) -> str | None:
    """
    Resolve an NSE equity ticker to its Upstox instrument_key: "NSE_EQ|<ISIN>".
    Uses the isin column on the stocks table (schema_v7_upstox.sql). Returns
    None if the ticker is unknown or has no ISIN on file.
    """
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        row = await conn.fetchrow(
            "SELECT isin FROM stocks WHERE ticker = $1", ticker
        )
    finally:
        await conn.close()

    if row is None or not row["isin"]:
        log.debug("No ISIN on file for %s — cannot resolve Upstox instrument_key", ticker)
        return None

    return f"NSE_EQ|{row['isin']}"


async def resolve_instrument_keys(tickers: list[str]) -> dict[str, str]:
    """Batch version of resolve_instrument_key. Returns {ticker: instrument_key} for resolvable tickers."""
    if not tickers:
        return {}
    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        rows = await conn.fetch(
            "SELECT ticker, isin FROM stocks WHERE ticker = ANY($1::text[]) AND isin IS NOT NULL",
            tickers,
        )
    finally:
        await conn.close()
    return {r["ticker"]: f"NSE_EQ|{r['isin']}" for r in rows if r["isin"]}


# --------------------------------------------------------------------------- #
# REST — historical / intraday candles (V3)                                   #
# --------------------------------------------------------------------------- #

async def get_historical_candles(
    instrument_key: str,
    unit: str,
    interval: str,
    to_date: str,
    from_date: str | None = None,
) -> list[dict[str, Any]]:
    """
    GET /v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}

    unit: "minutes" | "hours" | "days" | "weeks" | "months"
    interval: e.g. "1", "5", "15" for minutes; "1"-"5" for hours; "1" for days/weeks/months
    to_date / from_date: "YYYY-MM-DD" strings. from_date is optional per the API.

    Returns a list of {timestamp, open, high, low, close, volume} dicts,
    oldest-to-newest is NOT guaranteed by Upstox — caller should sort if order matters.
    Returns [] on missing token or request failure (no silent fallback data).
    """
    token = await get_data_token()
    if token is None:
        log.warning("No valid Upstox token — cannot fetch historical candles for %s (re-auth required, or set UPSTOX_ANALYTICS_TOKEN)", instrument_key)
        return []

    path_parts = [instrument_key, unit, interval, to_date]
    if from_date:
        path_parts.append(from_date)
    url = f"{HISTORICAL_CANDLE_URL}/{'/'.join(path_parts)}"

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.warning("Upstox historical-candle request failed for %s: %s", instrument_key, e)
        return []

    candles = (data.get("data") or {}).get("candles") or []
    out: list[dict[str, Any]] = []
    for c in candles:
        # Response row shape: [timestamp, open, high, low, close, volume, open_interest]
        if not isinstance(c, (list, tuple)) or len(c) < 6:
            continue
        try:
            out.append({
                "timestamp": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": int(c[5]) if c[5] is not None else 0,
            })
        except (TypeError, ValueError):
            continue

    return out


# --------------------------------------------------------------------------- #
# REST — LTP snapshot (fallback for when the WS feed has no tick yet)         #
# --------------------------------------------------------------------------- #

LTP_URL = "https://api.upstox.com/v2/market-quote/ltp"


async def get_ltp_rest(instrument_key: str) -> float | None:
    """
    One-shot REST LTP lookup for a single instrument_key, e.g. "NSE_EQ|INE...".
    Used as a fallback when the live WS feed has no tick for a position yet
    (new same-day buy not subscribed, or the feed itself is down) — slower
    than the WS path (a real HTTP round trip) but still far better than no
    price at all for stop/target enforcement. Returns None on any failure
    (missing token, bad response, unresolvable key) — never fabricates a price.
    """
    token = await get_data_token()
    if token is None:
        log.warning("get_ltp_rest: no valid Upstox token — cannot fetch LTP for %s", instrument_key)
        return None

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    params = {"instrument_key": instrument_key}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(LTP_URL, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.warning("get_ltp_rest: request failed for %s: %s", instrument_key, e)
        return None

    quotes = (data.get("data") or {})
    if not quotes:
        return None
    # Response is keyed by a normalized symbol string (e.g. "NSE_EQ:RELIANCE"),
    # not the instrument_key we sent — there's exactly one entry per request
    # since we only asked for one key, so just take the first value.
    row = next(iter(quotes.values()), None)
    if not row:
        return None
    try:
        return float(row["last_price"])
    except (KeyError, TypeError, ValueError):
        return None


if __name__ == "__main__":
    import asyncio

    async def main():
        logging.basicConfig(level=logging.INFO)
        print("Authorization URL:", get_authorization_url(state="test"))
        token = await get_valid_token()
        print("Cached daily OAuth token present:", bool(token))
        data_token = await get_data_token()
        print("Data token (analytics-first, falls back to daily) present:", bool(data_token))
        print("Analytics token configured:", bool(settings.UPSTOX_ANALYTICS_TOKEN))

    asyncio.run(main())
