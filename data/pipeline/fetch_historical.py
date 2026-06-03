"""
NSE daily OHLCV data fetcher.
PRIMARY source: NSE CM Bhavcopy (authoritative, free)
SUPPLEMENT: yfinance (used only to fill gaps)

Run once to seed 2000-2026, then use incremental for daily updates.
Progress is checkpointed so downloads can be resumed after interruption.
"""
from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import asyncpg
import pandas as pd
import requests
import yfinance as yf

from config import settings

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #
CM_BHAVCOPY_URL = (
    "https://archives.nseindia.com/content/historical/EQUITIES"
    "/{YYYY}/{MMM}/cm{DD}{MMM}{YYYY}bhav.csv.zip"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nseindia.com/",
}

PROGRESS_FILE = Path(settings.NSE_DATA_DIR) / ".progress"

# NSE holidays (major holidays; Bhavcopy simply won't exist for these dates)
NSE_HOLIDAYS: set[date] = {
    date(2024, 1, 22), date(2024, 3, 25), date(2024, 3, 29), date(2024, 4, 14),
    date(2024, 4, 17), date(2024, 4, 21), date(2024, 5, 23), date(2024, 6, 17),
    date(2024, 7, 17), date(2024, 8, 15), date(2024, 10, 2), date(2024, 10, 24),
    date(2024, 11, 1), date(2024, 11, 15), date(2024, 12, 25),
    date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31), date(2025, 4, 10),
    date(2025, 4, 14), date(2025, 4, 18), date(2025, 5, 1), date(2025, 8, 15),
    date(2025, 8, 27), date(2025, 10, 2), date(2025, 10, 21),
    date(2025, 10, 22), date(2025, 11, 5), date(2025, 12, 25),
}


def is_trading_day(d: date) -> bool:
    """Returns True if the date is a likely NSE trading day."""
    return d.weekday() < 5 and d not in NSE_HOLIDAYS


def build_bhavcopy_url(d: date) -> str:
    """Build the NSE CM Bhavcopy URL for a given date."""
    return CM_BHAVCOPY_URL.format(
        YYYY=d.strftime("%Y"),
        MMM=d.strftime("%b").upper(),
        DD=d.strftime("%d"),
    )


# ------------------------------------------------------------------ #
# Primary: NSE Bhavcopy                                                #
# ------------------------------------------------------------------ #

def download_bhavcopy(d: date) -> Path | None:
    """
    Downloads the NSE CM Bhavcopy zip for the given date.
    Returns local path to zip file, or None if unavailable.
    Adds 0.5s delay before the request to avoid rate-limiting.
    """
    url = build_bhavcopy_url(d)
    out_dir = Path(settings.NSE_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"cm_{d.strftime('%Y%m%d')}.csv.zip"

    if zip_path.exists():
        return zip_path

    try:
        time.sleep(0.5)
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 404:
            log.debug(f"Bhavcopy not found for {d} (404 — holiday or future date?)")
            return None
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        log.debug(f"Downloaded Bhavcopy {d}: {len(resp.content):,} bytes")
        return zip_path
    except requests.HTTPError as e:
        log.debug(f"HTTP error downloading Bhavcopy {d}: {e}")
        return None
    except Exception as e:
        log.warning(f"Failed to download Bhavcopy {d}: {e}")
        return None


def parse_bhavcopy(zip_path: Path) -> pd.DataFrame | None:
    """
    Parses NSE CM Bhavcopy zip file.
    Extracts SYMBOL, OPEN, HIGH, LOW, CLOSE, TOTTRDQTY, filters SERIES == 'EQ'.
    Returns DataFrame with columns: ticker, open, high, low, close, volume
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            csv_names = [n for n in z.namelist() if n.endswith(".csv")]
            if not csv_names:
                log.warning(f"No CSV in zip: {zip_path}")
                return None
            with z.open(csv_names[0]) as f:
                df = pd.read_csv(f)

        df.columns = [c.strip().upper() for c in df.columns]

        required = {"SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE", "SERIES"}
        if not required.issubset(set(df.columns)):
            log.warning(f"Missing columns in Bhavcopy. Found: {df.columns.tolist()}")
            return None

        # Filter EQ series
        df = df[df["SERIES"].str.strip() == "EQ"].copy()
        if df.empty:
            return None

        # Pick volume column — different Bhavcopy versions use different names
        vol_col = None
        for candidate in ["TOTTRDQTY", "TOTAL TRADED QUANTITY", "QUANTITY", "VOLUME"]:
            if candidate in df.columns:
                vol_col = candidate
                break

        result = pd.DataFrame()
        result["ticker"] = df["SYMBOL"].str.strip()
        result["open"] = pd.to_numeric(df["OPEN"], errors="coerce")
        result["high"] = pd.to_numeric(df["HIGH"], errors="coerce")
        result["low"] = pd.to_numeric(df["LOW"], errors="coerce")
        result["close"] = pd.to_numeric(df["CLOSE"], errors="coerce")
        result["volume"] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0).astype(int) if vol_col else 0

        result = result.dropna(subset=["ticker", "close"])
        return result

    except Exception as e:
        log.error(f"Error parsing Bhavcopy {zip_path}: {e}")
        return None


async def upsert_ohlcv_bhavcopy(conn, d: date, df: pd.DataFrame) -> int:
    """Upserts OHLCV rows from a parsed Bhavcopy DataFrame."""
    ts = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    rows = []
    for _, row in df.iterrows():
        rows.append((
            ts,
            str(row["ticker"]),
            float(row["open"]) if pd.notna(row["open"]) else None,
            float(row["high"]) if pd.notna(row["high"]) else None,
            float(row["low"]) if pd.notna(row["low"]) else None,
            float(row["close"]) if pd.notna(row["close"]) else None,
            int(row["volume"]) if pd.notna(row["volume"]) else None,
            float(row["close"]) if pd.notna(row["close"]) else None,  # adj_close = close for now
        ))

    if not rows:
        return 0

    # Ensure all tickers exist in stocks table
    tickers = list({r[1] for r in rows})
    await conn.executemany(
        "INSERT INTO stocks (ticker, name, exchange) VALUES ($1, $2, 'NSE') ON CONFLICT (ticker) DO NOTHING",
        [(t, t) for t in tickers],
    )

    await conn.executemany(
        """
        INSERT INTO ohlcv_daily (time, ticker, open, high, low, close, volume, adj_close)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (time, ticker) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            adj_close = EXCLUDED.adj_close
        """,
        rows,
    )
    return len(rows)


# ------------------------------------------------------------------ #
# Progress checkpoint                                                  #
# ------------------------------------------------------------------ #

def _load_progress() -> date | None:
    """Load last successfully processed date from progress file."""
    if PROGRESS_FILE.exists():
        try:
            return date.fromisoformat(PROGRESS_FILE.read_text().strip())
        except Exception:
            pass
    return None


def _save_progress(d: date):
    """Save last successfully processed date."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(d.isoformat())


# ------------------------------------------------------------------ #
# Main historical runner                                               #
# ------------------------------------------------------------------ #

async def run_bhavcopy_historical(
    start_date: date = date(2000, 1, 1),
    end_date: date = date(2026, 6, 3),
    resume: bool = True,
    delay: float = 0.5,
):
    """
    Iterates all trading days from start_date to end_date.
    Downloads NSE CM Bhavcopy, parses and stores each day.
    Deletes zip after parsing. Supports resumption via progress file.
    """
    conn = await asyncpg.connect(settings.DATABASE_DSN)

    if resume:
        last = _load_progress()
        if last and last >= start_date:
            start_date = last + timedelta(days=1)
            log.info(f"Resuming Bhavcopy history from {start_date}")

    current = start_date
    total_rows = 0
    errors = 0

    while current <= end_date:
        if not is_trading_day(current):
            current += timedelta(days=1)
            continue

        zip_path = download_bhavcopy(current)
        if zip_path is None:
            # Not available — skip silently (holiday/future date)
            current += timedelta(days=1)
            continue

        df = parse_bhavcopy(zip_path)
        if df is not None and not df.empty:
            try:
                rows = await upsert_ohlcv_bhavcopy(conn, current, df)
                total_rows += rows
                if rows > 0:
                    log.info(f"Bhavcopy {current}: {rows} EQ records stored")
            except Exception as e:
                log.error(f"DB error for {current}: {e}")
                errors += 1

        # Always delete zip to save space
        try:
            zip_path.unlink()
        except Exception:
            pass

        _save_progress(current)
        current += timedelta(days=1)
        # delay already applied inside download_bhavcopy

    log.info(f"Bhavcopy historical pull complete. Total rows: {total_rows:,}, Errors: {errors}")
    await conn.close()


# ------------------------------------------------------------------ #
# Supplement: yfinance gap-filler                                      #
# ------------------------------------------------------------------ #

def ticker_to_yf(ticker: str) -> str:
    return f"{ticker.replace(' ', '')}.NS"


async def fetch_yfinance_gaps(conn, ticker: str, start: str = "2000-01-01", end: str = "2026-06-04"):
    """
    For any ticker with gaps > 5 days, backfill via yfinance.
    Only used as a gap-filler supplement to Bhavcopy data.
    """
    yf_ticker = ticker_to_yf(ticker)
    try:
        df = yf.download(yf_ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            log.warning(f"yfinance: No data for {ticker}")
            return 0

        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]

        # Ensure stock exists
        await conn.execute(
            "INSERT INTO stocks (ticker, name, exchange) VALUES ($1, $2, 'NSE') ON CONFLICT (ticker) DO NOTHING",
            ticker, ticker,
        )

        rows = [
            (
                row.Index.to_pydatetime(),
                ticker,
                float(row.open) if pd.notna(row.open) else None,
                float(row.high) if pd.notna(row.high) else None,
                float(row.low) if pd.notna(row.low) else None,
                float(row.close) if pd.notna(row.close) else None,
                int(row.volume) if pd.notna(row.volume) else None,
                float(row.close) if pd.notna(row.close) else None,
            )
            for row in df.itertuples()
        ]

        await conn.executemany(
            """
            INSERT INTO ohlcv_daily (time, ticker, open, high, low, close, volume, adj_close)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (time, ticker) DO NOTHING
            """,
            rows,
        )
        log.info(f"yfinance gap-fill {ticker}: {len(rows)} rows")
        return len(rows)

    except Exception as e:
        log.error(f"yfinance error for {ticker}: {e}")
        return 0


# ------------------------------------------------------------------ #
# Incremental (daily scheduler use)                                    #
# ------------------------------------------------------------------ #

async def run_incremental(days_back: int = 5):
    """Pull recent Bhavcopy data (last N calendar days)."""
    end = date.today()
    start = end - timedelta(days=days_back)
    conn = await asyncpg.connect(settings.DATABASE_DSN)

    current = start
    while current <= end:
        if not is_trading_day(current):
            current += timedelta(days=1)
            continue

        zip_path = download_bhavcopy(current)
        if zip_path is None:
            current += timedelta(days=1)
            continue

        df = parse_bhavcopy(zip_path)
        if df is not None and not df.empty:
            try:
                rows = await upsert_ohlcv_bhavcopy(conn, current, df)
                log.info(f"Incremental {current}: {rows} records stored")
            except Exception as e:
                log.error(f"DB error for {current}: {e}")

        try:
            zip_path.unlink()
        except Exception:
            pass

        current += timedelta(days=1)

    await conn.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_bhavcopy_historical())
