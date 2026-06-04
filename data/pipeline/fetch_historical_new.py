"""
NSE CM Bhavcopy downloader for the NEW format (2024-07-08 onwards).

New URL:  https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip
Old format (pre-2024-07-08) is handled by fetch_historical.py.

Run:
    python -m data.pipeline.fetch_historical_new
    python -m data.pipeline.fetch_historical_new --start 2024-07-08 --end 2025-12-31
"""
from __future__ import annotations

import argparse
import io
import logging
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import asyncpg
import pandas as pd
import requests

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NEW_BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{YYYYMMDD}_F_0000.csv.zip"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nseindia.com/",
}

# NSE holidays from mid-2024 onwards (Bhavcopy won't exist for these)
NSE_HOLIDAYS: set[date] = {
    date(2024, 7, 17), date(2024, 8, 15), date(2024, 10, 2),
    date(2024, 10, 24), date(2024, 11, 1), date(2024, 11, 15),
    date(2024, 12, 25),
    date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31),
    date(2025, 4, 10), date(2025, 4, 14), date(2025, 4, 18),
    date(2025, 5, 1), date(2025, 8, 15), date(2025, 8, 27),
    date(2025, 10, 2), date(2025, 10, 21), date(2025, 10, 22),
    date(2025, 11, 5), date(2025, 12, 25),
}

PROGRESS_FILE = Path(settings.NSE_DATA_DIR) / ".progress_new"


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in NSE_HOLIDAYS


def build_url(d: date) -> str:
    return NEW_BHAVCOPY_URL.format(YYYYMMDD=d.strftime("%Y%m%d"))


def download_and_parse(d: date) -> pd.DataFrame | None:
    url = build_url(d)
    try:
        time.sleep(0.5)
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 404:
            log.debug("Not found for %s (404)", d)
            return None
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("Download failed %s: %s", d, e)
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, low_memory=False)
    except Exception as e:
        log.warning("Parse failed %s: %s", d, e)
        return None

    # Filter EQ series only
    df = df[df["SctySrs"] == "EQ"].copy()
    if df.empty:
        return None

    df = df.rename(columns={
        "TckrSymb": "ticker",
        "OpnPric":  "open",
        "HghPric":  "high",
        "LwPric":   "low",
        "ClsPric":  "close",
        "TtlTradgVol": "volume",
    })

    df["time"] = pd.to_datetime(d)
    df["adj_close"] = df["close"]
    df["ticker"] = df["ticker"].str.strip()

    return df[["time", "ticker", "open", "high", "low", "close", "volume", "adj_close"]]


async def upsert(conn, d: date, df: pd.DataFrame) -> int:
    rows = [
        (
            r["time"].to_pydatetime().replace(tzinfo=None),
            r["ticker"],
            float(r["open"]),
            float(r["high"]),
            float(r["low"]),
            float(r["close"]),
            int(r["volume"]) if pd.notna(r["volume"]) else 0,
            float(r["adj_close"]),
        )
        for _, r in df.iterrows()
        if pd.notna(r["close"]) and pd.notna(r["ticker"])
    ]
    if not rows:
        return 0

    # Only insert tickers that exist in the stocks table
    known = {r["ticker"] for r in await conn.fetch("SELECT ticker FROM stocks")}
    rows = [r for r in rows if r[1] in known]

    await conn.executemany(
        """
        INSERT INTO ohlcv_daily (time, ticker, open, high, low, close, volume, adj_close)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (time, ticker) DO UPDATE SET
            open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
            close=EXCLUDED.close, volume=EXCLUDED.volume, adj_close=EXCLUDED.adj_close
        """,
        rows,
    )
    return len(rows)


def _load_progress() -> date | None:
    if PROGRESS_FILE.exists():
        try:
            return date.fromisoformat(PROGRESS_FILE.read_text().strip())
        except Exception:
            return None
    return None


def _save_progress(d: date) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(d.isoformat())


async def run(start: date, end: date, resume: bool = True) -> None:
    import asyncio  # noqa: F401 — already imported at call site

    conn = await asyncpg.connect(settings.DATABASE_DSN)
    try:
        if resume:
            last = _load_progress()
            if last and last >= start:
                start = last + timedelta(days=1)
                log.info("Resuming new-format fetch from %s", start)

        current = start
        total = 0
        errors = 0

        while current <= end:
            if not is_trading_day(current):
                current += timedelta(days=1)
                continue

            df = download_and_parse(current)
            if df is not None and not df.empty:
                try:
                    n = await upsert(conn, current, df)
                    total += n
                    log.info("New-format Bhavcopy %s: %d EQ records stored", current, n)
                except Exception as e:
                    log.error("DB error for %s: %s", current, e)
                    errors += 1

            _save_progress(current)
            current += timedelta(days=1)

        log.info("New-format pull complete. total=%d errors=%d", total, errors)
    finally:
        await conn.close()


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser(description="Fetch NSE Bhavcopy (new format, 2024-07-08+)")
    parser.add_argument("--start", default="2024-07-08", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=date.today().isoformat(), help="End date YYYY-MM-DD")
    parser.add_argument("--no-resume", action="store_true", help="Ignore progress file, start fresh")
    args = parser.parse_args()

    asyncio.run(run(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        resume=not args.no_resume,
    ))
