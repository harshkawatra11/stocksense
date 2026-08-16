"""
NSE archive ingesters (docs/17-data-spine.md). Four fetchers, both
format eras — endpoints verified reachable and their schemas observed
directly during planning, not assumed.

NSE retired the legacy bhavcopy format on 2024-07-08 (Circular 62424) in
favour of UDiFF. Every fetcher here switches on that date and normalizes
both eras to one canonical schema, so nothing downstream needs to know
which era a given row came from. This is the classic silent-breakage
source named in the plan — the era boundary is unit-tested explicitly,
not just handled and hoped for.

Politeness, required (not optional): NSE 403s requests without a
browser-like User-Agent and Referer. A 404 on any endpoint is EXPECTED on
market holidays and must never be treated as an error — this module
follows core.calendar's philosophy of deriving the trading calendar
empirically rather than hardcoding a holiday list that goes stale.
"""

from __future__ import annotations

import hashlib
import io
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
import structlog

from stocksense.core.config import DATA_STORE

log = structlog.get_logger(__name__)

UDIFF_CUTOVER = date(2024, 7, 8)
CACHE_DIR = DATA_STORE / "nse_archive"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/"}
_POLITE_DELAY_S = 1.0


class FetchError(Exception):
    """A genuine error (network failure, unexpected response format) --
    NOT raised for a 404, which is treated as an expected holiday."""


def _cache_path(kind: str, d: date, ext: str) -> Path:
    return CACHE_DIR / kind / f"{d.isoformat()}.{ext}"


def _cached_or_fetch(kind: str, d: date, url: str, ext: str = "bin") -> bytes | None:
    """Content-hash-cached fetch: a URL is downloaded at most once, ever
    — ~6,500 files should not be re-fetched on every run. Returns None
    on a 404 (holiday), raises FetchError on anything else unexpected."""
    path = _cache_path(kind, d, ext)
    if path.exists():
        return path.read_bytes()

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
    except requests.exceptions.RequestException as e:
        raise FetchError(f"network error fetching {url}: {e}") from e

    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise FetchError(f"unexpected status {resp.status_code} fetching {url}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    time.sleep(_POLITE_DELAY_S)
    return resp.content


def _unzip_single_csv(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    return df


# ---- CM (Capital Market / equity) bhavcopy ----

def fetch_cm_bhavcopy(d: date) -> pd.DataFrame | None:
    """Canonical columns: symbol, series, date, open, high, low, close,
    prev_close, volume, turnover_inr, era ('legacy'|'udiff')."""
    if d < UDIFF_CUTOVER:
        return _fetch_cm_legacy(d)
    return _fetch_cm_udiff(d)


def _fetch_cm_legacy(d: date) -> pd.DataFrame | None:
    mon = d.strftime("%b").upper()
    ddmonyyyy = d.strftime("%d%b%Y").upper()
    url = f"https://nsearchives.nseindia.com/content/historical/EQUITIES/{d.year}/{mon}/cm{ddmonyyyy}bhav.csv.zip"
    content = _cached_or_fetch("cm_legacy", d, url, ext="zip")
    if content is None:
        return None
    raw = _unzip_single_csv(content)
    return pd.DataFrame({
        "symbol": raw["SYMBOL"].astype(str).str.strip(),
        "series": raw["SERIES"].astype(str).str.strip(),
        "date": d,
        "open": raw["OPEN"].astype(float),
        "high": raw["HIGH"].astype(float),
        "low": raw["LOW"].astype(float),
        "close": raw["CLOSE"].astype(float),
        "prev_close": raw["PREVCLOSE"].astype(float),
        "volume": raw["TOTTRDQTY"].astype(float),
        "turnover_inr": raw["TOTTRDVAL"].astype(float),
        "era": "legacy",
    })


def _fetch_cm_udiff(d: date) -> pd.DataFrame | None:
    yyyymmdd = d.strftime("%Y%m%d")
    url = f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"
    content = _cached_or_fetch("cm_udiff", d, url, ext="zip")
    if content is None:
        return None
    raw = _unzip_single_csv(content)
    return pd.DataFrame({
        "symbol": raw["TckrSymb"].astype(str).str.strip(),
        "series": raw["SctySrs"].astype(str).str.strip(),
        "date": d,
        "open": raw["OpnPric"].astype(float),
        "high": raw["HghPric"].astype(float),
        "low": raw["LwPric"].astype(float),
        "close": raw["ClsPric"].astype(float),
        "prev_close": raw["PrvsClsgPric"].astype(float),
        "volume": raw["TtlTradgVol"].astype(float),
        "turnover_inr": raw["TtlTrfVal"].astype(float),
        "era": "udiff",
    })


# ---- Delivery percentage (~2021+) ----

def fetch_delivery(d: date) -> pd.DataFrame | None:
    """Canonical columns: symbol, series, date, delivery_qty, delivery_pct."""
    ddmmyyyy = d.strftime("%d%m%Y")
    url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
    content = _cached_or_fetch("delivery", d, url, ext="csv")
    if content is None:
        return None
    raw = pd.read_csv(io.BytesIO(content))
    raw.columns = [c.strip() for c in raw.columns]
    deliv_col = next((c for c in raw.columns if "DELIV_QTY" in c.upper()), None)
    pct_col = next((c for c in raw.columns if "DELIV_PER" in c.upper()), None)
    if deliv_col is None or pct_col is None:
        raise FetchError(f"delivery columns not found in response for {d}: {list(raw.columns)}")

    out = pd.DataFrame({
        "symbol": raw["SYMBOL"].astype(str).str.strip(),
        "series": raw["SERIES"].astype(str).str.strip(),
        "date": d,
        "delivery_qty": pd.to_numeric(raw[deliv_col], errors="coerce"),
        "delivery_pct": pd.to_numeric(raw[pct_col], errors="coerce"),
    })
    return out.dropna(subset=["delivery_pct"])


# ---- F&O bhavcopy (2005+) ----

def fetch_fo_bhavcopy(d: date) -> pd.DataFrame | None:
    """Canonical columns: symbol, instrument, expiry_date, strike, option_type,
    open, high, low, close, open_interest, chg_in_oi, date, era."""
    if d < UDIFF_CUTOVER:
        return _fetch_fo_legacy(d)
    return _fetch_fo_udiff(d)


def _fetch_fo_legacy(d: date) -> pd.DataFrame | None:
    mon = d.strftime("%b").upper()
    ddmonyyyy = d.strftime("%d%b%Y").upper()
    url = f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{d.year}/{mon}/fo{ddmonyyyy}bhav.csv.zip"
    content = _cached_or_fetch("fo_legacy", d, url, ext="zip")
    if content is None:
        return None
    raw = _unzip_single_csv(content)
    return pd.DataFrame({
        "symbol": raw["SYMBOL"].astype(str).str.strip(),
        "instrument": raw["INSTRUMENT"].astype(str).str.strip(),
        "expiry_date": raw["EXPIRY_DT"].astype(str).str.strip(),
        "strike": pd.to_numeric(raw["STRIKE_PR"], errors="coerce"),
        "option_type": raw["OPTION_TYP"].astype(str).str.strip(),
        "open": raw["OPEN"].astype(float),
        "high": raw["HIGH"].astype(float),
        "low": raw["LOW"].astype(float),
        "close": raw["CLOSE"].astype(float),
        "open_interest": pd.to_numeric(raw["OPEN_INT"], errors="coerce"),
        "chg_in_oi": pd.to_numeric(raw["CHG_IN_OI"], errors="coerce"),
        "date": d,
        "era": "legacy",
    })


def _fetch_fo_udiff(d: date) -> pd.DataFrame | None:
    yyyymmdd = d.strftime("%Y%m%d")
    url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
    content = _cached_or_fetch("fo_udiff", d, url, ext="zip")
    if content is None:
        return None
    raw = _unzip_single_csv(content)
    return pd.DataFrame({
        "symbol": raw["TckrSymb"].astype(str).str.strip(),
        "instrument": raw["FinInstrmTp"].astype(str).str.strip(),
        "expiry_date": raw["XpryDt"].astype(str).str.strip(),
        "strike": pd.to_numeric(raw["StrkPric"], errors="coerce"),
        "option_type": raw["OptnTp"].astype(str).str.strip(),
        "open": raw["OpnPric"].astype(float),
        "high": raw["HghPric"].astype(float),
        "low": raw["LwPric"].astype(float),
        "close": raw["ClsPric"].astype(float),
        "open_interest": pd.to_numeric(raw["OpnIntrst"], errors="coerce"),
        "chg_in_oi": pd.to_numeric(raw["ChngInOpnIntrst"], errors="coerce"),
        "date": d,
        "era": "udiff",
    })


# ---- range fetch, resumable ----

def fetch_range(start: date, end: date, kind: str) -> Iterator[tuple[date, pd.DataFrame | None]]:
    """Fetches every calendar day from start to end (inclusive) for the
    given `kind` ('cm', 'delivery', 'fo'). A None entry means that date
    was a holiday (404) -- expected, not an error.

    A GENERATOR, not a list-returning function -- this is what makes a
    caller's progress genuinely resumable, not just re-download-avoiding.
    A caller that writes each (date, df) to the database AS IT ARRIVES
    (rather than collecting the full return value first) can be killed
    at any point and keep every day already yielded: _cached_or_fetch's
    on-disk cache means a re-run never re-downloads those days, and
    nothing is lost from the database either, because it was already
    written before the process died. A version of this function that
    only returned after the entire range completed would defeat that --
    an interrupted run would have fetched everything to disk but written
    NOTHING to the database, which is exactly the gap this generator
    form closes.
    """
    fetcher = {"cm": fetch_cm_bhavcopy, "delivery": fetch_delivery, "fo": fetch_fo_bhavcopy}[kind]
    d = start
    while d <= end:
        if d.weekday() < 5:  # skip weekends without even attempting a request
            try:
                df = fetcher(d)
            except FetchError as e:
                log.warning("nse_archive_fetch_failed", kind=kind, date=str(d), error=str(e))
                df = None
            yield (d, df)
        d += timedelta(days=1)
