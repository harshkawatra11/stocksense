"""
NSE F&O Bhavcopy downloader and parser.
Downloads daily derivatives bhavcopy, aggregates OI data per stock per day,
stores into fo_daily table.
"""
from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import asyncpg
import pandas as pd
import requests

from config import settings

log = logging.getLogger(__name__)

FO_BHAVCOPY_URL = (
    "https://archives.nseindia.com/content/historical/DERIVATIVES"
    "/{YYYY}/{MMM}/fo{DD}{MMM}{YYYY}bhav.csv.zip"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nseindia.com/",
}

# NSE holidays (partial list — major holidays 2000-2026)
NSE_HOLIDAYS = {
    date(2024, 1, 22), date(2024, 3, 25), date(2024, 3, 29), date(2024, 4, 14),
    date(2024, 4, 17), date(2024, 4, 21), date(2024, 5, 23), date(2024, 6, 17),
    date(2024, 7, 17), date(2024, 8, 15), date(2024, 10, 2), date(2024, 10, 24),
    date(2024, 11, 1), date(2024, 11, 15), date(2024, 12, 25),
    date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31), date(2025, 4, 10),
    date(2025, 4, 14), date(2025, 4, 18), date(2025, 5, 1), date(2025, 8, 15),
    date(2025, 8, 27), date(2025, 10, 2), date(2025, 10, 2), date(2025, 10, 21),
    date(2025, 10, 22), date(2025, 11, 5), date(2025, 12, 25),
}

PROGRESS_FILE = Path(settings.NSE_DATA_DIR) / ".fo_progress"


def is_trading_day(d: date) -> bool:
    """Returns True if the date is a likely NSE trading day."""
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    if d in NSE_HOLIDAYS:
        return False
    return True


def build_fo_url(d: date) -> str:
    """Build the NSE F&O Bhavcopy URL for a given date."""
    return FO_BHAVCOPY_URL.format(
        YYYY=d.strftime("%Y"),
        MMM=d.strftime("%b").upper(),
        DD=d.strftime("%d"),
    )


def download_fo_bhavcopy(d: date) -> Path | None:
    """
    Downloads NSE F&O Bhavcopy zip for a given date.
    Returns the local path to the zip file, or None if download fails.
    """
    url = build_fo_url(d)
    out_dir = Path(settings.NSE_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"fo_{d.strftime('%Y%m%d')}.csv.zip"

    if zip_path.exists():
        return zip_path

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 404:
            log.debug(f"F&O Bhavcopy not found for {d} (holiday or weekend?)")
            return None
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        log.debug(f"Downloaded F&O Bhavcopy for {d}: {len(resp.content):,} bytes")
        return zip_path
    except requests.HTTPError as e:
        log.debug(f"HTTP error for F&O {d}: {e}")
        return None
    except Exception as e:
        log.warning(f"Failed to download F&O Bhavcopy for {d}: {e}")
        return None


def parse_fo_bhavcopy(zip_path: Path) -> pd.DataFrame | None:
    """
    Parses the F&O Bhavcopy zip file.
    Returns aggregated DataFrame with columns:
    ticker, total_oi, oi_change, put_oi, call_oi, pcr
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

        # Required columns: SYMBOL, EXPIRY_DT, OPTION_TYP, STRIKE_PR, OPEN_INT, CHG_IN_OI, CONTRACTS
        required = {"SYMBOL", "OPEN_INT", "CHG_IN_OI"}
        if not required.issubset(set(df.columns)):
            log.warning(f"Missing columns in F&O CSV. Found: {df.columns.tolist()}")
            return None

        df["SYMBOL"] = df["SYMBOL"].str.strip()
        df["OPEN_INT"] = pd.to_numeric(df["OPEN_INT"], errors="coerce").fillna(0)
        df["CHG_IN_OI"] = pd.to_numeric(df["CHG_IN_OI"], errors="coerce").fillna(0)

        # Determine option type
        if "OPTION_TYP" in df.columns:
            df["OPTION_TYP"] = df["OPTION_TYP"].str.strip().str.upper()
            put_mask = df["OPTION_TYP"] == "PE"
            call_mask = df["OPTION_TYP"] == "CE"
        else:
            put_mask = pd.Series(False, index=df.index)
            call_mask = pd.Series(False, index=df.index)

        # Aggregate per symbol
        total_oi = df.groupby("SYMBOL")["OPEN_INT"].sum()
        oi_change = df.groupby("SYMBOL")["CHG_IN_OI"].sum()
        put_oi = df[put_mask].groupby("SYMBOL")["OPEN_INT"].sum()
        call_oi = df[call_mask].groupby("SYMBOL")["OPEN_INT"].sum()

        result = pd.DataFrame({
            "total_oi": total_oi,
            "oi_change": oi_change,
        })
        result["put_oi"] = put_oi.reindex(result.index).fillna(0).astype(int)
        result["call_oi"] = call_oi.reindex(result.index).fillna(0).astype(int)
        result["pcr"] = result.apply(
            lambda r: round(r["put_oi"] / r["call_oi"], 4) if r["call_oi"] > 0 else None,
            axis=1,
        )
        result.index.name = "ticker"
        result = result.reset_index()
        return result

    except Exception as e:
        log.error(f"Error parsing F&O Bhavcopy {zip_path}: {e}")
        return None


async def upsert_fo_daily(conn, d: date, df: pd.DataFrame) -> int:
    """Upserts F&O daily aggregates into fo_daily table."""
    from datetime import timezone
    ts = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    rows = []
    for _, row in df.iterrows():
        rows.append((
            ts,
            str(row["ticker"]),
            int(row["total_oi"]),
            int(row["oi_change"]),
            int(row["put_oi"]),
            int(row["call_oi"]),
            float(row["pcr"]) if row["pcr"] is not None else None,
        ))

    if not rows:
        return 0

    await conn.executemany(
        """
        INSERT INTO fo_daily (time, ticker, total_oi, oi_change, put_oi, call_oi, pcr)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (time, ticker) DO UPDATE SET
            total_oi = EXCLUDED.total_oi,
            oi_change = EXCLUDED.oi_change,
            put_oi = EXCLUDED.put_oi,
            call_oi = EXCLUDED.call_oi,
            pcr = EXCLUDED.pcr
        """,
        rows,
    )
    return len(rows)


def _load_fo_progress() -> date | None:
    """Load last successfully processed F&O date from progress file."""
    if PROGRESS_FILE.exists():
        try:
            txt = PROGRESS_FILE.read_text().strip()
            return date.fromisoformat(txt)
        except Exception:
            pass
    return None


def _save_fo_progress(d: date):
    """Save last successfully processed F&O date to progress file."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(d.isoformat())


async def run_fo_historical(
    start_date: date = date(2010, 1, 1),
    end_date: date = date(2026, 6, 3),
    resume: bool = True,
    delay: float = 0.5,
):
    """
    Bulk historical F&O data pull.
    Iterates all trading days, downloads + parses + stores each day.
    Skips weekends and known NSE holidays.
    Supports resume via progress checkpoint.
    """
    conn = await asyncpg.connect(settings.DATABASE_DSN)

    if resume:
        last = _load_fo_progress()
        if last and last >= start_date:
            start_date = last + timedelta(days=1)
            log.info(f"Resuming F&O history from {start_date}")

    current = start_date
    total_rows = 0

    while current <= end_date:
        if not is_trading_day(current):
            current += timedelta(days=1)
            continue

        zip_path = download_fo_bhavcopy(current)
        if zip_path is None:
            current += timedelta(days=1)
            time.sleep(delay)
            continue

        df = parse_fo_bhavcopy(zip_path)
        if df is not None and not df.empty:
            try:
                rows = await upsert_fo_daily(conn, current, df)
                total_rows += rows
                log.info(f"F&O {current}: {rows} records stored")
            except Exception as e:
                log.error(f"DB error for F&O {current}: {e}")

        # Delete zip to save space
        try:
            zip_path.unlink()
        except Exception:
            pass

        _save_fo_progress(current)
        current += timedelta(days=1)
        time.sleep(delay)

    log.info(f"F&O historical pull complete. Total records: {total_rows:,}")
    await conn.close()


async def run_fo_incremental(days_back: int = 5, delay: float = 0.5):
    """Pull recent F&O data (last N calendar days)."""
    end = date.today()
    start = end - timedelta(days=days_back)
    conn = await asyncpg.connect(settings.DATABASE_DSN)

    current = start
    while current <= end:
        if not is_trading_day(current):
            current += timedelta(days=1)
            continue

        zip_path = download_fo_bhavcopy(current)
        if zip_path is None:
            current += timedelta(days=1)
            time.sleep(delay)
            continue

        df = parse_fo_bhavcopy(zip_path)
        if df is not None and not df.empty:
            try:
                rows = await upsert_fo_daily(conn, current, df)
                log.info(f"F&O {current}: {rows} records stored")
            except Exception as e:
                log.error(f"DB error for F&O {current}: {e}")

        try:
            zip_path.unlink()
        except Exception:
            pass

        current += timedelta(days=1)
        time.sleep(delay)

    await conn.close()


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run_fo_historical())
