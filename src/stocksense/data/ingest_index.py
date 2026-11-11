from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from datetime import date, timedelta

import httpx
import polars as pl

from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.data.ingest_common import (
    NSE_ARCHIVES,
    IngestContext,
    IngestResult,
    absent,
    daterange,
    marker,
    num,
    read_strings,
)
from stocksense.data.io import list_partitions, read, upsert, write_partition

log = get_logger(__name__)

DATASET = "index_daily"
_KIND = "index"
MAX_MISSES = 60  # backfill stops after this many consecutive days with no file
FLOOR = date(2005, 1, 1)  # hard stop for the backward walk (NSE has nothing this old)

# Old (2012-2016 "CNX ...") and new names of the same index map to one canonical, upper-case name.
# Keys are upper-case with single spaces. The first 20 are the plan's; the rest were read from real
# files of 2012-2026 (see docs/DECISIONS.md T2.5).
INDEX_NAME_MAP: dict[str, str] = {
    "CNX NIFTY": "NIFTY 50",
    "CNX NIFTY JUNIOR": "NIFTY NEXT 50",
    "CNX 100": "NIFTY 100",
    "CNX 200": "NIFTY 200",
    "CNX 500": "NIFTY 500",
    "CNX MIDCAP": "NIFTY MIDCAP 100",
    "CNX SMALLCAP": "NIFTY SMALLCAP 100",
    "BANK NIFTY": "NIFTY BANK",
    "CNX IT": "NIFTY IT",
    "CNX FMCG": "NIFTY FMCG",
    "CNX PHARMA": "NIFTY PHARMA",
    "CNX AUTO": "NIFTY AUTO",
    "CNX METAL": "NIFTY METAL",
    "CNX REALTY": "NIFTY REALTY",
    "CNX ENERGY": "NIFTY ENERGY",
    "CNX INFRA": "NIFTY INFRASTRUCTURE",
    "CNX PSU BANK": "NIFTY PSU BANK",
    "CNX MEDIA": "NIFTY MEDIA",
    "CNX FINANCE": "NIFTY FINANCIAL SERVICES",
    "INDIA VIX": "INDIA VIX",
    # extensions from real files
    "CNX BANK": "NIFTY BANK",
    "CNX INFRASTRUCTURE": "NIFTY INFRASTRUCTURE",
    "CNX PSE": "NIFTY PSE",
    "CNX COMMODITIES": "NIFTY COMMODITIES",
    "CNX CONSUMPTION": "NIFTY INDIA CONSUMPTION",
    "CNX MNC": "NIFTY MNC",
    "CNX SERVICE SECTOR": "NIFTY SERVICES SECTOR",
    "CNX DIVIDEND OPPORTUNITIES": "NIFTY DIVIDEND OPPORTUNITIES 50",
    "CNX HIGH BETA": "NIFTY HIGH BETA 50",
    "CNX LOW VOLATILITY": "NIFTY LOW VOLATILITY 50",
    "CNX ALPHA INDEX": "NIFTY ALPHA 50",
    "CNX SHARIAH25": "NIFTY SHARIAH 25",
    "CNX NIFTY SHARIAH": "NIFTY50 SHARIAH",
    "CNX 500 SHARIAH": "NIFTY500 SHARIAH",
    "CNX NIFTY DIVIDEND": "NIFTY50 DIVIDEND POINTS",
    "CNX 100 EQUAL WEIGHT": "NIFTY100 EQUAL WEIGHT",
    "NIFTY PR 1X INVERSE": "NIFTY50 PR 1X INVERSE",
    "NIFTY PR 2X LEVERAGE": "NIFTY50 PR 2X LEVERAGE",
    "NIFTY TR 1X INVERSE": "NIFTY50 TR 1X INVERSE",
    "NIFTY TR 2X LEVERAGE": "NIFTY50 TR 2X LEVERAGE",
    "CPSE": "NIFTY CPSE",
    # 2012-02 .. 2013-02-07 files use "S&P CNX ..." names
    "S&P CNX NIFTY": "NIFTY 50",
    "S&P CNX 500": "NIFTY 500",
    "S&P CNX 500 SHARIAH": "NIFTY500 SHARIAH",
    "S&P CNX NIFTY SHARIAH": "NIFTY50 SHARIAH",
    "S&P CNX NIFTY DIVIDEND": "NIFTY50 DIVIDEND POINTS",
}


def canonical_index_name(raw: str) -> str:
    """Upper-case, single-spaced index name with old names mapped to the current one."""
    name = re.sub(r"\s+", " ", raw.strip()).upper()
    return INDEX_NAME_MAP.get(name, name)


def index_url(d: date) -> str:
    """NSE archive URL of ind_close_all (every index's daily values) for `d`."""
    return f"{NSE_ARCHIVES}/content/indices/ind_close_all_{d:%d%m%Y}.csv"


def _cache(ctx: IngestContext, d: date):
    return ctx.settings.paths.data_root / "cache" / "index" / f"{d.year}" / f"{d:%Y-%m-%d}.csv"


def parse_index(text: str, d: date) -> pl.DataFrame | None:
    """index_daily rows (values in index points; volume in shares; turnover in Rs crore) for day `d`.

    Returns None if the file is not for `d`: the most common `Index Date` differs from `d` (NSE serves
    the previous session's file for a holiday). Stray rows dated otherwise are dropped. `-` becomes null.
    """
    df = read_strings(text.encode("utf-8"))
    dates = df.select(pl.col("Index Date").str.strptime(pl.Date, "%d-%m-%Y", strict=False))["Index Date"]
    if df.height == 0 or dates.is_null().all():
        return None
    top = Counter(x for x in dates.to_list() if x is not None).most_common(1)[0][0]
    if top != d:
        return None
    out = df.with_columns(dates.alias("_d")).filter(pl.col("_d") == d)
    if out.height < df.height:
        log.warning("index_stray_rows", date=str(d), dropped=df.height - out.height)
    mapped = out.select(
        pl.col("Index Name").map_elements(canonical_index_name, return_dtype=pl.Utf8).alias("index_name"),
        pl.lit(d).alias("date"),
        num("Open Index Value").alias("open"),
        num("High Index Value").alias("high"),
        num("Low Index Value").alias("low"),
        num("Closing Index Value").alias("close"),
        num("P/E").alias("pe"),
        num("P/B").alias("pb"),
        num("Div Yield").alias("div_yield"),
        num("Volume").alias("volume"),
        num("Turnover (Rs. Cr.)").alias("turnover_cr"),
    )
    return mapped.unique(subset=["index_name", "date"], keep="last", maintain_order=True)


def _stored_days(ctx: IngestContext) -> dict[date, int]:
    if not list_partitions(ctx.settings, DATASET):
        return {}
    g = read(ctx.settings, DATASET, columns=["date"]).group_by("date").agg(pl.len().alias("n")).collect()
    return dict(zip(g["date"].to_list(), g["n"].to_list(), strict=True))


def _last_vix_date(ctx: IngestContext) -> date | None:
    if not list_partitions(ctx.settings, "india_vix"):
        return None
    return read(ctx.settings, "india_vix", columns=["date"]).select(pl.col("date").max()).collect().item()


def _fetch(ctx: IngestContext, d: date) -> tuple[str | None, bool]:
    """(text, is_valid). A downloaded file is cached only after its Index Date validated for `d`."""
    cache = _cache(ctx, d)
    if cache.is_file():
        return cache.read_text(encoding="utf-8", errors="replace"), True
    raw = ctx.nse.get_archive(index_url(d))
    if raw is None:
        return None, False
    text = raw.decode("utf-8", errors="replace")
    try:
        ok = parse_index(text, d) is not None
    except (pl.exceptions.PolarsError, ValueError):
        ok = False
    if ok:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
    return text, ok


def ingest_index_day(
    ctx: IngestContext, d: date, counts: dict[date, int] | None = None
) -> list[IngestResult]:
    """Ingest all index values of one calendar day; also extends india_vix when `d` is newer than it."""
    if counts is None:
        counts = _stored_days(ctx)
    have = counts.get(d, 0)
    if have > 0:
        return [IngestResult(DATASET, d, have, "skipped")]
    if ctx.state.kv_get(marker(_KIND, d)) is not None:
        return [IngestResult(DATASET, d, 0, "skipped")]
    try:
        text, ok = _fetch(ctx, d)
    except (DataNotAvailable, httpx.HTTPError) as e:
        log.warning("index_fetch_failed", date=str(d), error=str(e))
        return [IngestResult(DATASET, d, 0, "failed")]
    if text is None:
        return [absent(ctx, DATASET, _KIND, d)]
    if not ok:  # another day's file (holiday trap) or unparseable
        age = (ctx.clock.today() - d).days
        if age > 3:
            ctx.state.kv_set(marker(_KIND, d), "1")
        log.info("index_not_for_day", date=str(d))
        return [IngestResult(DATASET, d, 0, "no_session")]
    df = parse_index(text, d)
    assert df is not None
    upsert(ctx.settings, DATASET, df, clock=ctx.clock)
    counts[d] = df.height
    log.info("ingest_done", dataset=DATASET, date=str(d), rows=df.height)
    out = [IngestResult(DATASET, d, df.height, "ok")]
    vix = df.filter(pl.col("index_name") == "INDIA VIX").drop_nulls("close")
    last = _last_vix_date(ctx)
    if vix.height and (last is None or d > last):
        upsert(ctx.settings, "india_vix", vix.select("date", "close"), clock=ctx.clock)
        out.append(IngestResult("india_vix", d, vix.height, "ok"))
    return out


def ingest_index_range(
    ctx: IngestContext,
    a: date,
    b: date,
    on_result: Callable[[IngestResult], None] | None = None,
) -> list[IngestResult]:
    """Ingest every calendar day in [a, b] (weekends included)."""
    counts = _stored_days(ctx)
    out: list[IngestResult] = []
    for d in daterange(a, b):
        for r in ingest_index_day(ctx, d, counts):
            out.append(r)
            if on_result is not None:
                on_result(r)
    return out


def backfill_index(
    ctx: IngestContext,
    end: date,
    max_misses: int = MAX_MISSES,
    on_result: Callable[[IngestResult], None] | None = None,
) -> list[IngestResult]:
    """Walk backward from `end` over every calendar day until `max_misses` consecutive days have no file.

    A stored day is a hit; a 404, a wrong-date file or a remembered miss is a miss; a failed download is
    neutral (neither resets nor counts), so a flaky network cannot end the walk early.
    """
    counts = _stored_days(ctx)
    out: list[IngestResult] = []
    misses = 0
    d = end
    while misses < max_misses and d >= FLOOR:
        for r in ingest_index_day(ctx, d, counts):
            out.append(r)
            if on_result is not None:
                on_result(r)
            if r.dataset != DATASET:
                continue
            if r.status == "ok" or (r.status == "skipped" and r.rows > 0):
                misses = 0
            elif r.status in ("no_session", "skipped", "not_published"):
                misses += 1
        d -= timedelta(days=1)
    return out


def rebuild_index_from_cache(ctx: IngestContext) -> dict[date, int]:
    """Re-parse every cached index file and replace those days' stored rows (rows per day).

    Used after INDEX_NAME_MAP changes; the cache holds only validated files, so no HTTP is made.
    """
    root = ctx.settings.paths.data_root / "cache" / "index"
    parsed: dict[date, pl.DataFrame] = {}
    for f in sorted(root.glob("*/*.csv")):
        try:
            d = date.fromisoformat(f.stem)
        except ValueError:
            continue
        df = parse_index(f.read_text(encoding="utf-8", errors="replace"), d)
        if df is not None and df.height:
            parsed[d] = df
    for year in sorted({d.year for d in parsed}):
        days = [d for d in parsed if d.year == year]
        new = pl.concat([parsed[d] for d in days])
        keep = pl.DataFrame(schema=new.schema)
        if list_partitions(ctx.settings, DATASET):
            old = read(ctx.settings, DATASET, start=date(year, 1, 1), end=date(year, 12, 31)).collect()
            keep = old.filter(~pl.col("date").is_in(days))
        merged = pl.concat([keep.select(new.columns), new], how="vertical_relaxed") if keep.height else new
        write_partition(ctx.settings, DATASET, f"{year}", merged, clock=ctx.clock)
    return {d: df.height for d, df in parsed.items()}
