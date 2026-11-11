from __future__ import annotations

import zipfile
from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.data.ingest_common import (
    NSE_ARCHIVES,
    IngestContext,
    IngestResult,
    absent,
    csv_from_zip,
    daterange,
    fetch_cached,
    marker,
    num,
    read_strings,
)
from stocksense.data.io import read, upsert, write_partition

log = get_logger(__name__)

FO_START = date(2024, 7, 8)  # UDiFF F&O history is never extended earlier than this
MIN_FO_ROWS = 1000  # a real F&O day has ~36,000 rows; fewer means the file is not a full session
_KIND = "fo"
DATASET = "fo_bhavcopy"


def fo_url(d: date) -> str:
    """NSE archive URL of the UDiFF F&O bhavcopy zip for `d`."""
    return f"{NSE_ARCHIVES}/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"


def _cache(ctx: IngestContext, d: date):
    return ctx.settings.paths.data_root / "cache" / "fo_bhavcopy" / f"{d.year}" / f"{d:%Y-%m-%d}.zip"


def parse_fo(raw: bytes) -> pl.DataFrame:
    """fo_bhavcopy rows (prices in rupees, oi in contracts) from a UDiFF F&O zip.

    Futures have null strike_price and option_type.
    """
    df = read_strings(csv_from_zip(raw))
    return df.select(
        pl.col("TckrSymb").alias("symbol"),
        pl.col("FinInstrmTp").alias("instrument_type"),
        pl.col("XpryDt").str.strptime(pl.Date, "%Y-%m-%d").alias("expiry_date"),
        num("StrkPric").alias("strike_price"),
        pl.col("OptnTp").alias("option_type"),
        pl.col("TradDt").str.strptime(pl.Date, "%Y-%m-%d").alias("date"),
        num("OpnPric").alias("open"),
        num("HghPric").alias("high"),
        num("LwPric").alias("low"),
        num("ClsPric").alias("close"),
        num("PrvsClsgPric").alias("prev_close"),
        num("SttlmPric").alias("settle_price"),
        num("OpnIntrst").cast(pl.Int64).alias("oi"),
        num("ChngInOpnIntrst").cast(pl.Int64).alias("change_in_oi"),
        num("UndrlygPric").alias("underlying_price"),
    )


def _stored_counts(ctx: IngestContext, a: date, b: date) -> dict[date, int]:
    g = (
        read(ctx.settings, DATASET, start=a, end=b, columns=["date"])
        .group_by("date")
        .agg(pl.len().alias("n"))
        .collect()
    )
    return dict(zip(g["date"].to_list(), g["n"].to_list(), strict=True))


def ingest_fo_day(ctx: IngestContext, d: date, counts: dict[date, int] | None = None) -> IngestResult:
    """Ingest the F&O bhavcopy for one calendar day (404 rules as for the equity ingester)."""
    if d < FO_START:
        return IngestResult(DATASET, d, 0, "skipped")
    if counts is None:
        counts = _stored_counts(ctx, d, d)
    have = counts.get(d, 0)
    if have >= MIN_FO_ROWS:
        return IngestResult(DATASET, d, have, "skipped")
    if ctx.state.kv_get(marker(_KIND, d)) is not None:
        return IngestResult(DATASET, d, 0, "skipped")
    try:
        raw = fetch_cached(ctx, fo_url(d), _cache(ctx, d))
    except (DataNotAvailable, httpx.HTTPError) as e:
        log.warning("fo_fetch_failed", date=str(d), error=str(e))
        return IngestResult(DATASET, d, 0, "failed")
    if raw is None:
        return absent(ctx, DATASET, _KIND, d)
    try:
        df = parse_fo(raw)
    except (zipfile.BadZipFile, pl.exceptions.PolarsError, DataNotAvailable, ValueError) as e:
        log.warning("fo_parse_failed", date=str(d), error=str(e))
        return IngestResult(DATASET, d, 0, "failed")
    if df.height == 0 or (df["date"] != d).any():
        log.warning("fo_wrong_date", date=str(d))
        return IngestResult(DATASET, d, 0, "no_session")
    if df.height < MIN_FO_ROWS:
        log.warning("fo_too_few_rows", date=str(d), rows=df.height)
        res = absent(ctx, DATASET, _KIND, d)
        return IngestResult(DATASET, d, df.height, res.status)
    upsert(ctx.settings, DATASET, df, clock=ctx.clock)
    counts[d] = df.height
    log.info("ingest_done", dataset=DATASET, date=str(d), rows=df.height)
    return IngestResult(DATASET, d, df.height, "ok")


def ingest_fo_range(
    ctx: IngestContext,
    a: date,
    b: date,
    on_result: Callable[[IngestResult], None] | None = None,
) -> list[IngestResult]:
    """Ingest every calendar day in [a, b] (weekends included); never earlier than 2024-07-08."""
    days = daterange(max(a, FO_START), b)
    if not days:
        return []
    counts = _stored_counts(ctx, days[0], days[-1])
    out: list[IngestResult] = []
    for d in days:
        r = ingest_fo_day(ctx, d, counts)
        out.append(r)
        if on_result is not None:
            on_result(r)
    return out


def rebuild_fo_from_cache(ctx: IngestContext, a: date, b: date) -> dict[date, int]:
    """Re-parse every cached F&O file in [a, b] and replace those days' stored rows (rows per day).

    The adopted store has expiry_date entirely null and about 40% of the rows (keys collapsed without
    the expiry), so upsert by the full key cannot repair it. Raises DataNotAvailable, writing nothing,
    if a stored day in the range has no cached file.
    """
    root = ctx.settings.paths.data_root / "cache" / "fo_bhavcopy"
    cached: dict[date, Path] = {}
    for f in sorted(root.glob("*/*.zip")):
        try:
            d = date.fromisoformat(f.stem)
        except ValueError:
            continue
        if max(a, FO_START) <= d <= b:
            cached[d] = f
    stored = _stored_counts(ctx, max(a, FO_START), b)
    lacking = sorted(d for d in stored if d not in cached)
    if lacking:
        raise DataNotAvailable(f"stored F&O days without a cached file: {lacking[:5]} ({len(lacking)})")
    parsed: dict[date, pl.DataFrame] = {}
    for d, f in cached.items():
        df = parse_fo(f.read_bytes())
        if df.height < MIN_FO_ROWS or (df["date"] != d).any():
            log.warning("fo_rebuild_skipped", date=str(d), rows=df.height)
            continue
        parsed[d] = df
    months = sorted({(d.year, d.month) for d in parsed})
    for y, m in months:
        days = [d for d in parsed if (d.year, d.month) == (y, m)]
        new = pl.concat([parsed[d] for d in days])
        first, last = date(y, m, 1), date(y + (m == 12), m % 12 + 1, 1)
        old = read(ctx.settings, DATASET, start=first, end=last).collect()
        keep = old.filter(~pl.col("date").is_in(days) & (pl.col("date") < last))
        merged = pl.concat([keep, new], how="diagonal_relaxed") if keep.height else new
        write_partition(ctx.settings, DATASET, f"{y}-{m:02d}", merged, clock=ctx.clock)
        log.info("fo_rebuilt_month", month=f"{y}-{m:02d}", days=len(days), rows=merged.height)
    return {d: df.height for d, df in parsed.items()}
