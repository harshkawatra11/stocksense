"""NSE daily bhavcopy ingestion -- the point-in-time spine.

Three traps live here, all of them found the hard way in earlier builds. Each
one silently corrupts data rather than raising, which is why they are called out
in the code and covered by tests:

1. **"NA" is a real SERIES code.** NSE uses it for certain bond instruments.
   pandas' default NA list contains the string "NA", so a plain `read_csv`
   turns those rows' series into NaN. In a previous build that violated a NOT
   NULL constraint and killed an entire 11-year backfill on its first day.
   Every CSV read here passes `keep_default_na=False, na_values=[""]`.

2. **Two formats, one dataset.** NSE switched from the legacy layout to UDiFF
   on 2024-07-08 and the column names share nothing. Both are normalised to one
   schema, and `era` records which file a row came from so a format-specific bug
   is traceable after the fact.

3. **Resumability is a property, not a nicety.** Fetch is a generator and each
   day is written the moment it arrives. A previous build accumulated everything
   in memory and wrote at the end -- one interruption lost 1,652 days of work.

Verified live against NSE on 2026-09-02: UDiFF 2026-08-28 -> 3,612 rows
(2,629 EQ); legacy 2020-01-01 -> 1,910 rows, and `SERIES == "NA"` is present in
that file, confirming trap #1 is real and not folklore.
"""

from __future__ import annotations

import hashlib
import io
import time
import uuid
import zipfile
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from stocksense.data.store import Store

# NSE served the legacy layout up to and including 2024-07-05, and UDiFF from
# 2024-07-08. The changeover weekend is the boundary.
UDIFF_FROM = date(2024, 7, 8)

_UDIFF_URL = (
    "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
)
_LEGACY_URL = (
    "https://nsearchives.nseindia.com/content/historical/EQUITIES/"
    "{year}/{mon}/cm{dd}{mon}{year}bhav.csv.zip"
)
_DELIVERY_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# NSE tolerates a steady trickle. This is not a rate limit we are told about; it
# is politeness, and it is why a 16-year backfill takes hours rather than being
# blocked after ten minutes.
POLITE_DELAY_S = 0.35

# Bounded in-run retry for transient network failures. 3 attempts with
# 2s/4s backoff absorbs a read timeout without turning one flaky second into
# a permanent hole that only a whole re-run repairs.
_MAX_FETCH_ATTEMPTS = 3
_BACKOFF_BASE_S = 2.0

# Normalised output schema. Everything downstream depends on exactly these names.
CANON_COLS = [
    "symbol", "series", "date", "open", "high", "low", "close",
    "prev_close", "last_price", "volume", "turnover_inr", "n_trades", "era",
]

_UDIFF_MAP = {
    "TckrSymb": "symbol",
    "SctySrs": "series",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "PrvsClsgPric": "prev_close",
    "LastPric": "last_price",
    "TtlTradgVol": "volume",
    "TtlTrfVal": "turnover_inr",
    "TtlNbOfTxsExctd": "n_trades",
}

_LEGACY_MAP = {
    "SYMBOL": "symbol",
    "SERIES": "series",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "PREVCLOSE": "prev_close",
    "LAST": "last_price",
    "TOTTRDQTY": "volume",
    "TOTTRDVAL": "turnover_inr",
    "TOTALTRADES": "n_trades",
}

# The "legacy" era is not ONE format. Measured against live NSE files:
#   2010-2011 : SYMBOL..TIMESTAMP            -- no TOTALTRADES, no ISIN
#   2012-2024 : ...TIMESTAMP,TOTALTRADES,ISIN
# So `n_trades` is genuinely absent for the first two years and demanding it
# fails EVERY day in 2010-2011 -- which is exactly what happened on the first
# real backfill run. Columns are therefore split into required (a missing one is
# a real format break worth raising on) and optional (filled with NaN, and the
# absence is a fact about NSE, not a bug).
_OPTIONAL_COLS = {"n_trades"}


# --------------------------------------------------------------------- fetching
def _cache_path(cache_root: Path, kind: str, d: date, ext: str) -> Path:
    return cache_root / kind / f"{d.year}" / f"{d.isoformat()}.{ext}"


def _cached_or_fetch(cache_root: Path, kind: str, d: date, url: str, ext: str) -> bytes | None:
    """Fetch with an on-disk cache keyed by (kind, date).

    The cache is authoritative once written: a re-run never re-downloads, which
    is what makes replaying a long backfill fast instead of another 8 hours of
    network. Returns None for a 404, which is how NSE says "market holiday".
    """
    path = _cache_path(cache_root, kind, d, ext)
    if path.exists():
        return path.read_bytes()

    # Self-recovery: a read timeout or a 5xx is transient and must not cost the
    # whole day. Retry with exponential backoff, but ONLY on transient classes --
    # a 404 is a definitive "market holiday" and retrying it just wastes the
    # politeness budget. Observed in a real backfill: exactly one day in ~330
    # hit `ReadTimeout ... read timeout=45`, which this now absorbs.
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_FETCH_ATTEMPTS + 1):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=45)
            time.sleep(POLITE_DELAY_S)
            if resp.status_code == 404:
                return None
            if resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code} from {url}")
            resp.raise_for_status()
            break
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            last_exc = exc
            if attempt == _MAX_FETCH_ATTEMPTS:
                raise
            backoff = _BACKOFF_BASE_S * (2 ** (attempt - 1))
            log_line = (
                f"    retry {attempt}/{_MAX_FETCH_ATTEMPTS - 1} for {kind} {d} "
                f"after {type(exc).__name__} -- backing off {backoff:.1f}s"
            )
            print(log_line, flush=True)
            time.sleep(backoff)
    else:  # pragma: no cover - the loop always breaks or raises
        raise last_exc  # type: ignore[misc]

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    tmp.write_bytes(resp.content)
    tmp.replace(path)
    return resp.content


def _read_zipped_csv(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        with z.open(z.namelist()[0]) as f:
            # keep_default_na=False: "NA" is a REAL NSE series code (trap #1).
            df = pd.read_csv(f, keep_default_na=False, na_values=[""])
    df.columns = [c.strip() for c in df.columns]
    return df


# -------------------------------------------------------------------- parsing
def _normalise(raw: pd.DataFrame, d: date, era: str) -> pd.DataFrame:
    mapping = _UDIFF_MAP if era == "udiff" else _LEGACY_MAP
    present = {src: dst for src, dst in mapping.items() if src in raw.columns}
    missing_required = [
        src for src, dst in mapping.items()
        if src not in raw.columns and dst not in _OPTIONAL_COLS
    ]
    if missing_required:
        raise ValueError(f"{era} bhavcopy for {d} is missing required columns {missing_required}")

    out = raw[list(present)].rename(columns=present).copy()
    for dst in mapping.values():
        if dst not in out.columns:
            out[dst] = pd.NA
    # Drop null keys BEFORE the str cast. Order matters: `na_values=[""]` turns a
    # blank field into NaN, and `astype(str)` would then render it as the literal
    # string "nan" -- which is not "" and so survives every emptiness check, and
    # lands in the spine as a tradeable symbol named `nan`. Found by
    # test_blank_symbol_footer_rows_are_dropped.
    out = out[out["symbol"].notna() & out["series"].notna()]
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out["series"] = out["series"].astype(str).str.strip()
    out["date"] = pd.Timestamp(d)
    out["era"] = era

    for col in ("open", "high", "low", "close", "prev_close", "last_price",
                "volume", "turnover_inr", "n_trades"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if era == "legacy":
        # Legacy TOTTRDVAL is already rupees; UDiFF TtlTrfVal likewise. No scaling.
        pass

    # A footer/summary row with a blank symbol appears in some legacy files and
    # crashed a previous backfill on a NOT NULL constraint. Belt and braces after
    # the cast: catch whitespace-only and the stringified-null forms too.
    bad = {"", "nan", "none", "<na>"}
    out = out[~out["symbol"].str.lower().isin(bad) & ~out["series"].str.lower().isin(bad)]
    return out[CANON_COLS].reset_index(drop=True)


def parse_delivery(content: bytes, d: date) -> pd.DataFrame:
    """sec_bhavdata_full is a plain CSV (not zipped) with padded column names."""
    df = pd.read_csv(io.BytesIO(content), keep_default_na=False, na_values=[""])
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame(
        {
            "symbol": df["SYMBOL"].astype(str).str.strip(),
            "series": df["SERIES"].astype(str).str.strip(),
            "date": pd.Timestamp(d),
            "deliv_qty": pd.to_numeric(df["DELIV_QTY"], errors="coerce"),
            "deliv_pct": pd.to_numeric(df["DELIV_PER"], errors="coerce"),
        }
    )
    return out[(out["symbol"] != "") & (out["series"] != "")].reset_index(drop=True)


def fetch_day(cache_root: Path, d: date) -> pd.DataFrame | None:
    """One trading day, normalised. None means NSE had no file (holiday/weekend)."""
    if d >= UDIFF_FROM:
        url = _UDIFF_URL.format(ymd=d.strftime("%Y%m%d"))
        content = _cached_or_fetch(cache_root, "cm_udiff", d, url, "zip")
        if content is None:
            return None
        return _normalise(_read_zipped_csv(content), d, "udiff")

    url = _LEGACY_URL.format(
        year=d.strftime("%Y"), mon=d.strftime("%b").upper(), dd=d.strftime("%d")
    )
    content = _cached_or_fetch(cache_root, "cm_legacy", d, url, "zip")
    if content is None:
        return None
    return _normalise(_read_zipped_csv(content), d, "legacy")


def fetch_delivery_day(cache_root: Path, d: date) -> pd.DataFrame | None:
    url = _DELIVERY_URL.format(ddmmyyyy=d.strftime("%d%m%Y"))
    content = _cached_or_fetch(cache_root, "delivery", d, url, "csv")
    if content is None:
        return None
    return parse_delivery(content, d)


# ------------------------------------------------------------------- ingestion
def _weekdays(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        if d.weekday() < 5:  # NSE never trades a weekend; skip without a request
            yield d
        d += timedelta(days=1)


def backfill(
    store: Store,
    cache_root: Path,
    start: date,
    end: date,
    with_delivery: bool = True,
    resume: bool = True,
    progress: bool = True,
    publish_every: int = 25,
) -> dict[str, int]:
    """Ingest bhavcopy (and delivery) for every trading day in a range.

    Resumable by construction: each day is fetched, written and RECORDED
    individually, so interrupting this loses at most the day in flight. On a
    re-run, days already recorded as ok/empty are skipped without a request.

    Failures are recorded, never raised -- a single bad day must not abort a
    16-year backfill, and an unrecorded failure is a silent hole in the spine.
    """
    done = store.completed_units("nse_bhavcopy") if resume else set()
    stats = {"days_ok": 0, "days_empty": 0, "days_failed": 0, "rows": 0, "delivery_rows": 0, "skipped": 0}
    processed = 0

    for d in _weekdays(start, end):
        unit = d.isoformat()
        if unit in done:
            stats["skipped"] += 1
            continue

        started = datetime.now()
        run = {
            "run_id": hashlib.sha1(f"nse_bhavcopy:{unit}:{started}".encode()).hexdigest()[:16],
            "source": "nse_bhavcopy",
            "unit": unit,
            "started_at": started,
            "attempt": 1,
        }
        try:
            df = fetch_day(cache_root, d)
            if df is None or df.empty:
                run |= {"status": "empty", "rows_written": 0, "error": None}
                stats["days_empty"] += 1
            else:
                store.write_bhavcopy_eq(df)
                stats["rows"] += len(df)
                stats["days_ok"] += 1
                run |= {"status": "ok", "rows_written": len(df), "error": None}

                if with_delivery:
                    try:
                        dl = fetch_delivery_day(cache_root, d)
                        if dl is not None and not dl.empty:
                            store.write_delivery(dl)
                            stats["delivery_rows"] += len(dl)
                    except Exception as exc:
                        # Delivery is a bonus feed; never let it fail the day.
                        run["error"] = f"delivery: {type(exc).__name__}: {exc}"
        except Exception as exc:
            run |= {"status": "failed", "rows_written": 0, "error": f"{type(exc).__name__}: {exc}"}
            stats["days_failed"] += 1

        run["finished_at"] = datetime.now()
        store.record_ingest_run(run)
        processed += 1

        # Publish telemetry periodically, not just at the end. `ingest_runs`
        # lives in DuckDB, which only readers-of-Parquet can see AFTER a publish
        # -- so without this the pipeline monitor is blind for the entire hour a
        # backfill runs, which is precisely the window you most want to watch.
        # Cheap: the table is small and the write is an atomic rename.
        if processed % publish_every == 0:
            store.publish()

        if progress and (stats["days_ok"] + stats["days_empty"]) % 50 == 0:
            print(
                f"  {unit}  ok={stats['days_ok']} empty={stats['days_empty']} "
                f"failed={stats['days_failed']} rows={stats['rows']:,}",
                flush=True,
            )

    return stats
