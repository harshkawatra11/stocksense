"""
Angel One daily OHLCV backfill / updater.

Angel One's getCandleData works on the user's network where NSE Bhavcopy (503)
and yfinance are blocked — so this is the primary daily-data path.

Resolves NSE-EQ instrument tokens from Angel One's scrip master (one download,
cached), then pulls ONE_DAY candles per ticker and upserts into ohlcv_daily.

Run:
    python -m data.pipeline.fetch_angel_daily            # update last 7 days
    python -m data.pipeline.fetch_angel_daily 30         # last 30 days
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import asyncpg
import requests

from config import settings
from data.pipeline.fetch_live import get_session

log = logging.getLogger(__name__)

SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)
_SCRIP_CACHE = Path(settings.NSE_DATA_DIR) / "angel_scrip_master.json"
# Angel One historical API: ~3 requests/sec. Stay under it.
_REQ_DELAY = 0.35


def _load_scrip_master(max_age_hours: int = 24) -> list[dict]:
    """Download (and cache) the Angel One scrip master JSON."""
    if _SCRIP_CACHE.exists():
        age = time.time() - _SCRIP_CACHE.stat().st_mtime
        if age < max_age_hours * 3600:
            return json.loads(_SCRIP_CACHE.read_text(encoding="utf-8"))
    log.info("Downloading Angel One scrip master…")
    resp = requests.get(SCRIP_MASTER_URL, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    _SCRIP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _SCRIP_CACHE.write_text(json.dumps(data), encoding="utf-8")
    log.info("Scrip master cached: %d instruments", len(data))
    return data


def build_nse_eq_token_map(scrip: list[dict]) -> dict[str, str]:
    """
    Map base ticker -> token for NSE cash-equity instruments.
    Angel One EQ rows: exch_seg == 'NSE', symbol like 'RELIANCE-EQ'.
    """
    out: dict[str, str] = {}
    for row in scrip:
        if row.get("exch_seg") != "NSE":
            continue
        sym = row.get("symbol", "")
        if not sym.endswith("-EQ"):
            continue
        base = sym[:-3]  # strip '-EQ'
        out[base] = str(row.get("token"))
    return out


def fetch_daily_candles(obj, token: str, days_back: int) -> list[list]:
    """getCandleData ONE_DAY for the last `days_back` days. Returns raw candle rows."""
    params = {
        "exchange": "NSE",
        "symboltoken": token,
        "interval": "ONE_DAY",
        "fromdate": (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d 09:15"),
        "todate": datetime.now().strftime("%Y-%m-%d 15:30"),
    }
    try:
        data = obj.getCandleData(params)
        if data.get("status"):
            return data.get("data") or []
        log.debug("getCandleData failed for %s: %s", token, data.get("message"))
    except Exception as e:
        log.debug("getCandleData error for %s: %s", token, e)
    return []


async def _upsert_candles(conn, ticker: str, candles: list[list]) -> int:
    """Upsert raw Angel candles [ts,o,h,l,c,v] into ohlcv_daily."""
    rows = []
    for c in candles:
        # c = ["2026-06-05T00:00:00+05:30", open, high, low, close, volume]
        # The IST calendar date IS the trading day. Do NOT convert to UTC first —
        # IST-midnight → UTC rolls back to the previous day. Take the date as-is and
        # store as UTC-midnight to match the Bhavcopy convention.
        dt = datetime.fromisoformat(c[0])
        ts = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
        rows.append((ts, ticker, float(c[1]), float(c[2]), float(c[3]),
                     float(c[4]), int(c[5]), float(c[4])))
    if not rows:
        return 0
    await conn.execute(
        "INSERT INTO stocks (ticker, name, exchange) VALUES ($1,$1,'NSE') ON CONFLICT (ticker) DO NOTHING",
        ticker,
    )
    await conn.executemany(
        """
        INSERT INTO ohlcv_daily (time, ticker, open, high, low, close, volume, adj_close)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
        ON CONFLICT (time, ticker) DO UPDATE SET
            open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
            close=EXCLUDED.close, volume=EXCLUDED.volume, adj_close=EXCLUDED.adj_close
        """,
        rows,
    )
    return len(rows)


async def run_angel_backfill(days_back: int = 7, tickers: list[str] | None = None) -> dict:
    """
    Update ohlcv_daily for the active universe (or a given ticker list) via Angel One.
    Returns a summary {tickers, updated, rows, skipped}.
    """
    obj = get_session()
    token_map = build_nse_eq_token_map(_load_scrip_master())
    log.info("Resolved %d NSE-EQ tokens from scrip master", len(token_map))

    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        if tickers is None:
            rows = await conn.fetch("SELECT ticker FROM stocks WHERE active = TRUE ORDER BY ticker")
            tickers = [r["ticker"] for r in rows]

        total_rows = updated = skipped = 0
        for t in tickers:
            token = token_map.get(t)
            if not token:
                skipped += 1
                continue
            candles = fetch_daily_candles(obj, token, days_back)
            time.sleep(_REQ_DELAY)
            if candles:
                n = await _upsert_candles(conn, t, candles)
                total_rows += n
                updated += 1
        summary = {"tickers": len(tickers), "updated": updated, "rows": total_rows, "skipped": skipped}
        log.info("Angel backfill done: %s", summary)
        return summary
    finally:
        await conn.close()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    s = asyncio.run(run_angel_backfill(days_back=days))
    print(f"\nAngel One backfill: {s['updated']}/{s['tickers']} tickers updated, "
          f"{s['rows']} rows, {s['skipped']} skipped (no token).")
