"""
Upstox intraday data spine (Phase E1). 1-minute NSE equity bars, the raw
truth behind the 5-minute research grain (features/intraday.py derives
5-min by resampling -- never fetched separately, so the two can't
disagree).

Three facts verified live before writing this module, not assumed:
- The v3 historical-candle endpoint needs NO auth token (unlike Upstox's
  live/quote endpoints) and returns 200 with real OHLCV even with no
  Authorization header. One less thing to break nightly -- Upstox access
  tokens expire daily per the .env comment, and this fetcher never needs
  one.
- 1-minute history is available back to 2022-01 only; 2021 and earlier
  return an empty candle list (still 200 OK, not an error).
- Max range per request is 31 calendar days for the 1-minute grain
  (verified: 31 days -> 200 OK/9000 candles, 32 days -> 400 UDAPI1148
  "Invalid date range"). fetch_range chunks by calendar month to stay
  under this.

Reuses nse_archive.py's proven shape: content-hash disk cache, a
resumable GENERATOR for the range fetch (not a list-returning function --
see nse_archive.fetch_range's docstring for why eager collection lost an
entire backfill to one interruption, commit 731c262), and a politeness
delay.
"""

from __future__ import annotations

import gzip
import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
import structlog

from stocksense.core.config import DATA_STORE

log = structlog.get_logger(__name__)

CACHE_DIR = DATA_STORE / "upstox_intraday"
INSTRUMENT_MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
_HEADERS = {"Accept": "application/json"}
_POLITE_DELAY_S = 0.4
EARLIEST_1MIN_DATE = date(2022, 1, 1)  # verified: 2021 and earlier return empty, not an error
MAX_DAYS_PER_REQUEST = 31  # verified: 32 days -> 400 UDAPI1148 "Invalid date range"


class FetchError(Exception):
    """Genuine error (network, unexpected shape) -- not a routine empty-
    candles response, which is expected for weekends/holidays/pre-2022
    dates and is returned as [] with a 200, not raised."""


def _instrument_master_cache_path() -> Path:
    return CACHE_DIR / "instrument_master.json"


def fetch_instrument_master(force: bool = False) -> list[dict]:
    """The ~2,600-row NSE_EQ instrument master, cached to disk (static
    enough day-to-day that re-fetching every run is wasteful; force=True
    refreshes after a listing change)."""
    path = _instrument_master_cache_path()
    if path.exists() and not force:
        return json.loads(path.read_text())

    try:
        resp = requests.get(INSTRUMENT_MASTER_URL, timeout=30)
    except requests.exceptions.RequestException as e:
        raise FetchError(f"network error fetching instrument master: {e}") from e
    if resp.status_code != 200:
        raise FetchError(f"unexpected status {resp.status_code} fetching instrument master")

    data = json.loads(gzip.decompress(resp.content))
    eq = [d for d in data if d.get("segment") == "NSE_EQ" and d.get("instrument_type") == "EQ"]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(eq))
    return eq


def resolve_symbol_map(symbols: list[str]) -> pd.DataFrame:
    """Maps bhavcopy symbols -> Upstox instrument_key by exact
    trading_symbol match (Upstox's trading_symbol IS the NSE ticker, e.g.
    'RELIANCE' -- verified directly against the master). Unmapped symbols
    get resolved=False rather than being silently dropped: a caller that
    wants to know why 12 of 250 symbols never got bars needs this table,
    not a log line that scrolled past overnight."""
    master = fetch_instrument_master()
    by_symbol = {d["trading_symbol"]: d for d in master}

    rows = []
    for sym in symbols:
        m = by_symbol.get(sym)
        if m is None:
            rows.append({"symbol": sym, "isin": None, "instrument_key": None, "resolved": False})
        else:
            rows.append({
                "symbol": sym, "isin": m.get("isin"),
                "instrument_key": m["instrument_key"], "resolved": True,
            })
    return pd.DataFrame(rows, columns=["symbol", "isin", "instrument_key", "resolved"])


def _cache_path(instrument_key: str, window_start: date) -> Path:
    safe_key = instrument_key.replace("|", "_")
    return CACHE_DIR / "bars_1min" / safe_key / f"{window_start.isoformat()}.json"


def _cached_or_fetch_window(instrument_key: str, window_start: date, window_end: date) -> list:
    """Content-cached fetch for one <=31-day window. An empty candle
    list (pre-2022, or a window with no trading days) is cached as `[]`
    and returned as such -- distinct from a raised FetchError, which
    means the fetch itself failed, not that there was nothing to fetch."""
    path = _cache_path(instrument_key, window_start)
    if path.exists():
        return json.loads(path.read_text())

    url = (
        f"https://api.upstox.com/v3/historical-candle/{instrument_key}/minutes/1/"
        f"{window_end.isoformat()}/{window_start.isoformat()}"
    )
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
    except requests.exceptions.RequestException as e:
        raise FetchError(f"network error fetching {url}: {e}") from e

    if resp.status_code == 429:
        time.sleep(5.0)
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30)
        except requests.exceptions.RequestException as e:
            raise FetchError(f"network error on retry fetching {url}: {e}") from e

    if resp.status_code != 200:
        raise FetchError(f"unexpected status {resp.status_code} fetching {url}: {resp.text[:200]}")

    candles = resp.json().get("data", {}).get("candles", [])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candles))
    time.sleep(_POLITE_DELAY_S)
    return candles


def _month_windows(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Splits [start, end] into <=31-day chunks, one per calendar month,
    so every request stays under Upstox's verified 31-day cap."""
    cur = start
    while cur <= end:
        next_month_start = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)
        window_end = min(end, next_month_start - timedelta(days=1))
        yield (cur, window_end)
        cur = next_month_start


def _candles_to_frame(symbol: str, candles: list) -> pd.DataFrame:
    cols = ["symbol", "ts", "interval", "open", "high", "low", "close", "volume"]
    if not candles:
        return pd.DataFrame(columns=cols)
    # Upstox candle shape: [ts, open, high, low, close, volume, oi]
    df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume", "oi"])
    df["symbol"] = symbol
    df["interval"] = "1minute"
    # IST wall-clock only -- this project trades one exchange/timezone, so the
    # +05:30 offset is dropped rather than converted, matching how a trader
    # reads it and how bhavcopy_eq's `date` column is already stored.
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    return df[cols]


def fetch_range(
    instrument_map: pd.DataFrame, start: date, end: date,
) -> Iterator[tuple[str, date, date, pd.DataFrame | None]]:
    """Fetches every resolved symbol's 1-minute bars across [start, end],
    month-window by month-window, yielded as they arrive -- a GENERATOR,
    not a list, for the same reason nse_archive.fetch_range is one: a
    caller that writes each result to the database as it arrives can be
    killed at any point without losing progress already fetched to disk
    AND already written to the DB (see nse_archive.py's docstring on this
    property, and the resumability bug fixed in 731c262).

    `start` is clamped to EARLIEST_1MIN_DATE -- requesting earlier
    dates returns an empty (not erroring) result every time, so there's
    no point asking. A None dataframe means the window's fetch itself
    failed (network/API error); an empty-but-present dataframe means the
    fetch succeeded and there was genuinely nothing there.
    """
    start = max(start, EARLIEST_1MIN_DATE)
    resolved = instrument_map[instrument_map["resolved"]]

    for _, row in resolved.iterrows():
        symbol, instrument_key = row["symbol"], row["instrument_key"]
        for window_start, window_end in _month_windows(start, end):
            try:
                candles = _cached_or_fetch_window(instrument_key, window_start, window_end)
            except FetchError as e:
                log.warning(
                    "upstox_intraday_fetch_failed", symbol=symbol,
                    window_start=str(window_start), window_end=str(window_end), error=str(e),
                )
                yield (symbol, window_start, window_end, None)
                continue
            yield (symbol, window_start, window_end, _candles_to_frame(symbol, candles))
