from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

import httpx
import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.data.ingest_common import (
    NSE_ARCHIVES,
    IngestContext,
    IngestResult,
    fetch_cached,
    num,
    read_strings,
)
from stocksense.data.io import list_partitions, read, upsert, write_partition

log = get_logger(__name__)

HOLIDAY_FIRST_YEAR = 2011  # the holiday API answers nothing for earlier years (probed 2026-11-11)
_INDEX_SYMBOLS = frozenset({"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"})
_FNO_STOCK_MARKER = "derivatives on individual securities"

SYMBOLCHANGE_URL = f"{NSE_ARCHIVES}/content/equities/symbolchange.csv"
LISTED_URL = f"{NSE_ARCHIVES}/content/equities/EQUITY_L.csv"
ETF_URL = f"{NSE_ARCHIVES}/content/equities/eq_etfseclist.csv"
PRICE_BANDS_URL = f"{NSE_ARCHIVES}/content/equities/sec_list.csv"
FNO_LOTS_URL = f"{NSE_ARCHIVES}/content/fo/fo_mktlots.csv"
SECTOR_FILES = ("ind_nifty500list.csv", "ind_niftytotalmarket_list.csv")  # first wins on overlap


def _cache(ctx: IngestContext, name: str, ext: str = "csv"):
    d = ctx.clock.today()
    return ctx.settings.paths.data_root / "cache" / "reference" / name / f"{d:%Y-%m-%d}.{ext}"


def _date_expr(col: str) -> pl.Expr:
    """`%d-%b-%Y` strings (any month case) to Date; unparseable becomes null."""
    return pl.col(col).str.to_titlecase().str.strptime(pl.Date, "%d-%b-%Y", strict=False)


# ---- parsers ------------------------------------------------------------------------------------


def parse_symbol_changes(text: str) -> pl.DataFrame:
    """symbol_changes rows (company, old_symbol, new_symbol, change_date) from symbolchange.csv."""
    df = read_strings(text.encode("utf-8"))
    out = df.select(
        pl.col("NAME OF THE COMPANY").alias("company"),
        pl.col("OLD SYMBOL").alias("old_symbol"),
        pl.col("NEW SYMBOL").alias("new_symbol"),
        _date_expr("EFFECTIVE DATE").alias("change_date"),
    )
    return out.drop_nulls(["old_symbol", "new_symbol", "change_date"]).unique(
        subset=["old_symbol", "new_symbol", "change_date"], keep="last", maintain_order=True
    )


def parse_listed(text: str, as_of: date) -> pl.DataFrame:
    """listed_equities snapshot rows from EQUITY_L.csv (header names stripped; face_value in rupees)."""
    df = read_strings(text.encode("utf-8"))
    out = df.select(
        pl.col("SYMBOL").alias("symbol"),
        pl.col("NAME OF COMPANY").alias("name"),
        pl.col("SERIES").alias("series"),
        _date_expr("DATE OF LISTING").alias("listing_date"),
        pl.col("ISIN NUMBER").alias("isin"),
        num("FACE VALUE").alias("face_value"),
        pl.lit(as_of).alias("as_of"),
    )
    return out.unique(subset=["symbol"], keep="last", maintain_order=True)


def parse_etf(text: str, as_of: date) -> pl.DataFrame:
    """etf_list snapshot rows (symbol, underlying, name) from eq_etfseclist.csv."""
    df = read_strings(text.encode("utf-8"))
    out = df.select(
        pl.col("Symbol").alias("symbol"),
        pl.col("Underlying Asset").alias("underlying"),
        pl.col("SecurityName").alias("name"),
        pl.lit(as_of).alias("as_of"),
    )
    return out.unique(subset=["symbol"], keep="first", maintain_order=True)


def parse_sectors(text: str, source: str, as_of: date) -> pl.DataFrame:
    """sectors snapshot rows from one Nifty index list file; `source` is the file name."""
    df = read_strings(text.encode("utf-8"))
    out = df.select(
        pl.col("Symbol").alias("symbol"),
        pl.col("Industry").alias("industry"),
        pl.col("Company Name").alias("company"),
        pl.col("ISIN Code").alias("isin"),
        pl.lit(source).alias("source"),
        pl.lit(as_of).alias("as_of"),
    )
    return out.unique(subset=["symbol"], keep="first", maintain_order=True)


def parse_price_bands(text: str, d: date) -> pl.DataFrame:
    """price_bands rows (symbol, series, band, date=d) from sec_list.csv; band is the file's text."""
    df = read_strings(text.encode("utf-8"))
    out = df.select(
        pl.col("Symbol").alias("symbol"),
        pl.col("Series").alias("series"),
        pl.col("Band").alias("band"),
        pl.lit(d).alias("date"),
    )
    return out.unique(subset=["symbol", "series"], keep="last", maintain_order=True)


def parse_holidays(payload: dict[str, Any], fetched_at: datetime) -> pl.DataFrame:
    """holidays rows (date, description, fetched_at) from the CM list of /api/holiday-master."""
    rows = payload.get("CM") or []
    return pl.DataFrame(
        {
            "date": [datetime.strptime(r["tradingDate"], "%d-%b-%Y").date() for r in rows],
            "description": [str(r.get("description") or "") for r in rows],
        },
        schema={"date": pl.Date, "description": pl.Utf8},
    ).with_columns(pl.lit(fetched_at).alias("fetched_at"))


def parse_fno_symbols(text: str) -> list[str]:
    """F&O stock symbols from fo_mktlots.csv: rows after the 'Derivatives on Individual Securities' header.

    Without that header row, every row whose symbol is not one of the five index symbols is taken.
    """
    df = read_strings(text.encode("utf-8"))
    under = [str(u).lower() for u in df["UNDERLYING"].to_list()]
    syms = df["SYMBOL"].to_list()
    start = next((i + 1 for i, u in enumerate(under) if u.startswith(_FNO_STOCK_MARKER)), None)
    if start is not None:
        picked = syms[start:]
    else:
        picked = [s for s in syms if s not in _INDEX_SYMBOLS and s.lower() != "symbol"]
    return [s for s in picked if s and s.upper() not in _INDEX_SYMBOLS]


# ---- ingesters ----------------------------------------------------------------------------------


def _get_text(ctx: IngestContext, url: str, name: str) -> str | None:
    raw = fetch_cached(ctx, url, _cache(ctx, name))
    return None if raw is None else raw.decode("utf-8", errors="replace")


def _failed(dataset: str, d: date, why: str) -> IngestResult:
    log.warning("reference_failed", dataset=dataset, why=why)
    return IngestResult(dataset, d, 0, "failed")


def _snapshot(
    ctx: IngestContext,
    dataset: str,
    texts: dict[str, str | None],
    build: Callable[[dict[str, str], date], pl.DataFrame],
) -> IngestResult:
    """Append a dated snapshot of `dataset` only if the downloaded content changed since the last one."""
    d = ctx.clock.today()
    if any(t is None for t in texts.values()):
        return _failed(dataset, d, "file not found")
    have = {k: t for k, t in texts.items() if t is not None}
    digest = hashlib.sha1("\n".join(have[k] for k in sorted(have)).encode("utf-8")).hexdigest()
    key = f"ref_hash:{dataset}"
    if ctx.state.kv_get(key) == digest and list_partitions(ctx.settings, dataset):
        return IngestResult(dataset, d, 0, "skipped")
    try:
        df = build(have, d)
    except (pl.exceptions.PolarsError, ValueError, KeyError) as e:
        return _failed(dataset, d, f"parse error {type(e).__name__}: {e}")
    upsert(ctx.settings, dataset, df, clock=ctx.clock)
    ctx.state.kv_set(key, digest)
    log.info("ingest_done", dataset=dataset, date=str(d), rows=df.height)
    return IngestResult(dataset, d, df.height, "ok")


def ingest_symbol_changes(ctx: IngestContext) -> IngestResult:
    """Replace symbol_changes with the current symbolchange.csv."""
    d = ctx.clock.today()
    try:
        text = _get_text(ctx, SYMBOLCHANGE_URL, "symbolchange")
        if text is None:
            return _failed("symbol_changes", d, "file not found")
        df = parse_symbol_changes(text)
    except (DataNotAvailable, httpx.HTTPError, pl.exceptions.PolarsError, ValueError) as e:
        return _failed("symbol_changes", d, f"{type(e).__name__}: {e}")
    write_partition(ctx.settings, "symbol_changes", "all", df, clock=ctx.clock)
    log.info("ingest_done", dataset="symbol_changes", date=str(d), rows=df.height)
    return IngestResult("symbol_changes", d, df.height, "ok")


def _download(ctx: IngestContext, url: str, name: str) -> str | None:
    try:
        return _get_text(ctx, url, name)
    except (DataNotAvailable, httpx.HTTPError) as e:
        log.warning("reference_download_failed", name=name, error=str(e))
        return None


def ingest_listed_equities(ctx: IngestContext) -> IngestResult:
    """Append a listed_equities snapshot (as_of = today) if EQUITY_L.csv changed."""
    texts = {"listed": _download(ctx, LISTED_URL, "listed")}
    return _snapshot(ctx, "listed_equities", texts, lambda t, d: parse_listed(t["listed"], d))


def ingest_etf_list(ctx: IngestContext) -> IngestResult:
    """Append an etf_list snapshot if eq_etfseclist.csv changed."""
    texts = {"etf": _download(ctx, ETF_URL, "etf")}
    return _snapshot(ctx, "etf_list", texts, lambda t, d: parse_etf(t["etf"], d))


def ingest_sectors(ctx: IngestContext) -> IngestResult:
    """Append a sectors snapshot if either index list changed; the Nifty 500 row wins on overlap."""
    texts = {
        f: _download(ctx, f"{NSE_ARCHIVES}/content/indices/{f}", f.removesuffix(".csv")) for f in SECTOR_FILES
    }

    def build(t: dict[str, str], d: date) -> pl.DataFrame:
        frames = [parse_sectors(t[f], f, d) for f in SECTOR_FILES]
        return pl.concat(frames).unique(subset=["symbol"], keep="first", maintain_order=True)

    return _snapshot(ctx, "sectors", texts, build)


def ingest_price_bands(ctx: IngestContext) -> IngestResult:
    """Today's circuit-limit bands from sec_list.csv, appended daily (upsert by symbol, series, date)."""
    d = ctx.clock.today()
    if not ctx.calendar.is_session(d):
        return IngestResult("price_bands", d, 0, "no_session")
    text = _download(ctx, PRICE_BANDS_URL, "price_bands")
    if text is None:
        return _failed("price_bands", d, "file not found")
    try:
        df = parse_price_bands(text, d)
    except (pl.exceptions.PolarsError, ValueError, KeyError) as e:
        return _failed("price_bands", d, f"parse error {type(e).__name__}: {e}")
    upsert(ctx.settings, "price_bands", df, clock=ctx.clock)
    log.info("ingest_done", dataset="price_bands", date=str(d), rows=df.height)
    return IngestResult("price_bands", d, df.height, "ok")


def _holiday_years(ctx: IngestContext) -> list[int]:
    """This year and next, plus every earlier year (from 2011) not yet in the store and not known empty."""
    today = ctx.clock.today()
    stored: set[int] = set()
    if list_partitions(ctx.settings, "holidays"):
        col = read(ctx.settings, "holidays", columns=["date"]).collect()["date"]
        stored = {x.year for x in col.to_list()}
    history = [
        y
        for y in range(HOLIDAY_FIRST_YEAR, today.year)
        if y not in stored and ctx.state.kv_get(f"holidays_empty:{y}") is None
    ]
    return [*history, today.year, today.year + 1]


def ingest_holidays(ctx: IngestContext, years: list[int] | None = None) -> IngestResult:
    """Replace the CM trading-holiday rows of each fetched year (other years untouched)."""
    d = ctx.clock.today()
    fetched_at = ctx.clock.now()
    frames: list[pl.DataFrame] = []
    got_years: list[int] = []
    for y in years if years is not None else _holiday_years(ctx):
        try:
            payload = ctx.nse.get_api(
                "/api/holiday-master",
                {"type": "trading", "year": str(y)},
                "resources/exchange-communication-holidays",
            )
        except (DataNotAvailable, httpx.HTTPError) as e:
            return _failed("holidays", d, f"year {y}: {type(e).__name__}: {e}")
        df = parse_holidays(payload if isinstance(payload, dict) else {}, fetched_at)
        if df.height == 0:
            if y < d.year:
                ctx.state.kv_set(f"holidays_empty:{y}", "1")
            continue
        frames.append(df)
        got_years.append(y)
    if not frames:
        return IngestResult("holidays", d, 0, "skipped")
    new = pl.concat(frames)
    if list_partitions(ctx.settings, "holidays"):
        old = read(ctx.settings, "holidays").collect()
        new = pl.concat([old.filter(~pl.col("date").dt.year().is_in(got_years)), new], how="diagonal_relaxed")
    write_partition(
        ctx.settings, "holidays", "all", new.unique(subset=["date"], keep="last"), clock=ctx.clock
    )
    log.info("ingest_done", dataset="holidays", date=str(d), rows=new.height, years=got_years)
    return IngestResult("holidays", d, new.height, "ok")


def ingest_fno_list(ctx: IngestContext) -> IngestResult:
    """Store today's F&O stock list (from fo_mktlots.csv) in kv as `fno_symbols:<date>` (JSON list)."""
    d = ctx.clock.today()
    text = _download(ctx, FNO_LOTS_URL, "fo_mktlots")
    if text is None:
        return _failed("fno_symbols", d, "file not found")
    try:
        syms = parse_fno_symbols(text)
    except (pl.exceptions.PolarsError, ValueError, KeyError) as e:
        return _failed("fno_symbols", d, f"parse error {type(e).__name__}: {e}")
    ctx.state.kv_set(f"fno_symbols:{d.isoformat()}", json.dumps(syms))
    log.info("ingest_done", dataset="fno_symbols", date=str(d), rows=len(syms))
    return IngestResult("fno_symbols", d, len(syms), "ok")


def ingest_reference(ctx: IngestContext) -> list[IngestResult]:
    """Run every reference ingester (holidays first, then rebuild ctx.calendar); returns all results."""
    out = [ingest_holidays(ctx)]
    ctx.calendar = TradingCalendar.from_store(ctx.settings, ctx.clock)
    for fn in (
        ingest_symbol_changes,
        ingest_listed_equities,
        ingest_etf_list,
        ingest_sectors,
        ingest_price_bands,
        ingest_fno_list,
    ):
        out.append(fn(ctx))
    return out
