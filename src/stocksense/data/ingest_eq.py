from __future__ import annotations

import asyncio
import json
import random
import zipfile
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx
import polars as pl

from stocksense.core.config import Settings
from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.data.ingest_common import (
    MARKER_AGE_DAYS,
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
from stocksense.data.io import read, upsert
from stocksense.data.nse_http import NseClient

__all__ = ["IngestContext", "IngestResult", "NseClient"]  # re-exported for callers and tests

log = get_logger(__name__)

UDIFF_START = date(2024, 7, 8)
DELIVERY_START = date(2019, 8, 23)  # stored delivery history starts here; never extended earlier
MIN_SESSION_ROWS = 500
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

# PLAN B.3 Finding 2: special sessions found missing from the adopted data.
MISSING_SPECIAL_SESSIONS: tuple[date, ...] = tuple(
    date.fromisoformat(s)
    for s in [
        "2010-02-06",
        "2012-01-07",
        "2012-03-03",
        "2012-04-28",
        "2012-09-08",
        "2013-05-11",
        "2013-11-03",
        "2014-03-22",
        "2015-02-28",
        "2016-10-30",
        "2019-10-27",
        "2020-02-01",
        "2020-11-14",
        "2023-11-12",
        "2024-01-20",
        "2024-03-02",
        "2024-05-18",
        "2025-02-01",
        "2026-02-01",
    ]
)

_EQ_COLUMNS = (
    "symbol", "series", "date", "open", "high", "low", "close", "prev_close", "last_price",
    "volume", "turnover_inr", "n_trades", "era", "isin", "name", "avg_price",
)  # fmt: skip
_FLOATS = ("open", "high", "low", "close", "prev_close", "last_price", "turnover_inr", "n_trades")


# ---- URLs and cache paths -----------------------------------------------------------------------


def eq_url(d: date) -> str:
    """NSE archive URL of the cash-market bhavcopy zip for `d` (UDiFF from 2024-07-08, else legacy)."""
    if d >= UDIFF_START:
        return f"{NSE_ARCHIVES}/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    mon = _MONTHS[d.month - 1]
    return f"{NSE_ARCHIVES}/content/historical/EQUITIES/{d.year}/{mon}/cm{d.day:02d}{mon}{d.year}bhav.csv.zip"


def delivery_url(d: date) -> str:
    """NSE archive URL of sec_bhavdata_full (prices + delivery) for `d`."""
    return f"{NSE_ARCHIVES}/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv"


def _eq_cache(settings: Settings, d: date):
    sub = "cm_udiff" if d >= UDIFF_START else "cm_legacy"
    return settings.paths.data_root / "cache" / sub / f"{d.year}" / f"{d:%Y-%m-%d}.zip"


def _delivery_cache(settings: Settings, d: date):
    return settings.paths.data_root / "cache" / "delivery" / f"{d.year}" / f"{d:%Y-%m-%d}.csv"


# ---- parsers ------------------------------------------------------------------------------------


def _finish_eq(df: pl.DataFrame, era: str) -> pl.DataFrame:
    for c in ("isin", "name", "avg_price", "n_trades"):
        if c not in df.columns:
            df = df.with_columns(
                pl.lit(None, dtype=pl.Utf8 if c in ("isin", "name") else pl.Float64).alias(c)
            )
    df = df.with_columns(
        *(pl.col(c).cast(pl.Float64) for c in _FLOATS),
        pl.col("volume").cast(pl.Float64).cast(pl.Int64),
        pl.col("avg_price").cast(pl.Float64),
        pl.lit(era).alias("era"),
    )
    return df.select(_EQ_COLUMNS)


def parse_udiff(raw: bytes) -> pl.DataFrame:
    """bhavcopy_eq-shaped rows (prices in rupees, volume in shares) from a UDiFF cash-market zip."""
    df = read_strings(csv_from_zip(raw))
    df = df.rename(
        {
            "TckrSymb": "symbol", "SctySrs": "series", "TradDt": "date", "OpnPric": "open",
            "HghPric": "high", "LwPric": "low", "ClsPric": "close", "PrvsClsgPric": "prev_close",
            "LastPric": "last_price", "TtlTradgVol": "volume", "TtlTrfVal": "turnover_inr",
            "TtlNbOfTxsExctd": "n_trades", "ISIN": "isin", "FinInstrmNm": "name",
        }
    )  # fmt: skip
    df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    return _finish_eq(df, "udiff")


def parse_legacy(raw: bytes) -> pl.DataFrame:
    """bhavcopy_eq-shaped rows from a legacy cmDDMONYYYYbhav zip (trailing-comma column dropped)."""
    df = read_strings(csv_from_zip(raw))
    df = df.rename(
        {
            "SYMBOL": "symbol", "SERIES": "series", "OPEN": "open", "HIGH": "high", "LOW": "low",
            "CLOSE": "close", "LAST": "last_price", "PREVCLOSE": "prev_close", "TOTTRDQTY": "volume",
            "TOTTRDVAL": "turnover_inr", "TIMESTAMP": "date",
        }
    )  # fmt: skip
    if "TOTALTRADES" in df.columns:
        df = df.rename({"TOTALTRADES": "n_trades"})
    if "ISIN" in df.columns:
        df = df.rename({"ISIN": "isin"})
    df = df.with_columns(pl.col("date").str.to_titlecase().str.strptime(pl.Date, "%d-%b-%Y"))
    return _finish_eq(df, "legacy")


def parse_delivery(text: str, d: date) -> pl.DataFrame | None:
    """Delivery rows (symbol, series, date, deliv_qty, deliv_pct, avg_price), or None if not for `d`.

    Returns None when any DATE1 differs from `d` (NSE serves the previous session's file on a holiday).
    `-` becomes null. deliv_pct is 0-100.
    """
    df = read_strings(text.encode("utf-8"))
    dates = df.select(pl.col("DATE1").str.to_titlecase().str.strptime(pl.Date, "%d-%b-%Y"))["DATE1"]
    if dates.is_null().any() or (dates != d).any() or df.height == 0:
        return None

    return df.select(
        pl.col("SYMBOL").alias("symbol"),
        pl.col("SERIES").alias("series"),
        pl.lit(d).alias("date"),
        num("DELIV_QTY").alias("deliv_qty"),
        num("DELIV_PER").alias("deliv_pct"),
        num("AVG_PRICE").alias("avg_price"),
    )


# ---- stored-state lookups -----------------------------------------------------------------------


@dataclass
class _Counts:
    eq: dict[date, int] = field(default_factory=dict)
    deliv: dict[date, int] = field(default_factory=dict)

    @classmethod
    def load(cls, settings: Settings, a: date, b: date) -> _Counts:
        def per_day(dataset: str) -> dict[date, int]:
            g = (
                read(settings, dataset, start=a, end=b, columns=["date"])
                .group_by("date")
                .agg(pl.len().alias("n"))
                .collect()
            )
            return dict(zip(g["date"].to_list(), g["n"].to_list(), strict=True))

        return cls(per_day("bhavcopy_eq"), per_day("bhavcopy_delivery"))


# ---- one day ------------------------------------------------------------------------------------


def _eq_part(ctx: IngestContext, d: date, counts: _Counts) -> IngestResult:
    name = "bhavcopy_eq"
    have = counts.eq.get(d, 0)
    if have >= MIN_SESSION_ROWS:
        return IngestResult(name, d, have, "skipped")
    if ctx.state.kv_get(marker("eq", d)) is not None:
        return IngestResult(name, d, 0, "skipped")
    try:
        raw = fetch_cached(ctx, eq_url(d), _eq_cache(ctx.settings, d))
    except (DataNotAvailable, httpx.HTTPError) as e:
        log.warning("eq_fetch_failed", date=str(d), error=str(e))
        return IngestResult(name, d, 0, "failed")
    if raw is None:
        return absent(ctx, name, "eq", d)
    try:
        df = (parse_udiff if d >= UDIFF_START else parse_legacy)(raw)
    except (zipfile.BadZipFile, pl.exceptions.PolarsError, DataNotAvailable, ValueError) as e:
        log.warning("eq_parse_failed", date=str(d), error=str(e))
        return IngestResult(name, d, 0, "failed")
    if df.height == 0 or (df["date"] != d).any():
        log.warning("eq_wrong_date", date=str(d))
        return IngestResult(name, d, 0, "no_session")
    if df.height < MIN_SESSION_ROWS:
        # e.g. gold-ETF-only Sunday trading: real data, but not an equity session, so it must not
        # become a calendar session (the calendar is derived from the stored dates)
        log.warning("eq_too_few_rows", date=str(d), rows=df.height)
        res = absent(ctx, name, "eq", d)
        return IngestResult(name, d, df.height, "no_session" if res.status == "no_session" else res.status)
    upsert(ctx.settings, name, df, clock=ctx.clock)
    counts.eq[d] = df.height
    log.info("ingest_done", dataset=name, date=str(d), rows=df.height)
    return IngestResult(name, d, df.height, "ok")


def _delivery_part(ctx: IngestContext, d: date, counts: _Counts) -> IngestResult:
    name = "bhavcopy_delivery"
    have = counts.deliv.get(d, 0)
    if have > 0:
        return IngestResult(name, d, have, "skipped")
    if ctx.state.kv_get(marker("delivery", d)) is not None:
        return IngestResult(name, d, 0, "skipped")
    cache = _delivery_cache(ctx.settings, d)
    try:
        raw = _fetch_raw_delivery(ctx, d, cache)
    except (DataNotAvailable, httpx.HTTPError) as e:
        log.warning("delivery_fetch_failed", date=str(d), error=str(e))
        return IngestResult(name, d, 0, "failed")
    if raw is None:
        return absent(ctx, name, "delivery", d)
    try:
        df = parse_delivery(raw.decode("utf-8", errors="replace"), d)
    except (pl.exceptions.PolarsError, ValueError) as e:
        log.warning("delivery_parse_failed", date=str(d), error=str(e))
        return IngestResult(name, d, 0, "failed")
    if df is None:  # holiday trap: another day's file
        cache.unlink(missing_ok=True)
        return _absent_no_session(ctx, name, d)
    dv = df.select("symbol", "series", "date", "deliv_qty", "deliv_pct")
    upsert(ctx.settings, name, dv, clock=ctx.clock)
    _merge_avg_price(ctx, d, df)
    counts.deliv[d] = df.height
    log.info("ingest_done", dataset=name, date=str(d), rows=df.height)
    return IngestResult(name, d, df.height, "ok")


def _fetch_raw_delivery(ctx: IngestContext, d: date, cache) -> bytes | None:
    """Cached delivery bytes, else download; a downloaded file is cached only after DATE1 validation."""
    if cache.is_file():
        return cache.read_bytes()
    raw = ctx.nse.get_archive(delivery_url(d))
    if raw is not None and parse_delivery_ok(raw, d):
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
    return raw


def parse_delivery_ok(raw: bytes, d: date) -> bool:
    """True if the raw delivery file parses and every DATE1 equals `d`."""
    try:
        return parse_delivery(raw.decode("utf-8", errors="replace"), d) is not None
    except (pl.exceptions.PolarsError, ValueError):
        return False


def _absent_no_session(ctx: IngestContext, dataset: str, d: date) -> IngestResult:
    age = (ctx.clock.today() - d).days
    if age > MARKER_AGE_DAYS:
        ctx.state.kv_set(marker("delivery", d), "1")
    return IngestResult(dataset, d, 0, "no_session")


def _merge_avg_price(ctx: IngestContext, d: date, df: pl.DataFrame) -> None:
    """Write AVG_PRICE into bhavcopy_eq.avg_price for day d (join on symbol, series; other columns kept)."""
    eq = read(ctx.settings, "bhavcopy_eq", start=d, end=d).collect()
    if eq.height == 0:
        return
    avg = df.select("symbol", "series", pl.col("avg_price").alias("_avg")).unique(
        subset=["symbol", "series"], keep="last"
    )
    merged = (
        eq.join(avg, on=["symbol", "series"], how="left")
        .with_columns(pl.coalesce("_avg", "avg_price").alias("avg_price"))
        .drop("_avg")
    )
    upsert(ctx.settings, "bhavcopy_eq", merged, clock=ctx.clock)


def ingest_eq_day(ctx: IngestContext, d: date, counts: _Counts | None = None) -> list[IngestResult]:
    """Ingest the cash-market bhavcopy and (from 2019-08-23) delivery for one calendar day."""
    if counts is None:
        counts = _Counts.load(ctx.settings, d, d)
    out = [_eq_part(ctx, d, counts)]
    if d >= DELIVERY_START and out[0].status in ("ok", "skipped") and counts.eq.get(d, 0) > 0:
        out.append(_delivery_part(ctx, d, counts))
    return out


# ---- ranges and repair --------------------------------------------------------------------------


def ingest_eq_range(
    ctx: IngestContext,
    a: date,
    b: date,
    days: Iterable[date] | None = None,
    on_result: Callable[[IngestResult], None] | None = None,
) -> list[IngestResult]:
    """Ingest every calendar day in [a, b] (weekends included), or the explicit `days` list."""
    day_list = list(days) if days is not None else daterange(a, b)
    if not day_list:
        return []
    counts = _Counts.load(ctx.settings, min(day_list), max(day_list))
    results: list[IngestResult] = []
    for d in day_list:
        for r in ingest_eq_day(ctx, d, counts):
            results.append(r)
            if on_result is not None:
                on_result(r)
    return results


def repair_days(ctx: IngestContext, to: date) -> list[date]:
    """Repair order: (a) the 19 known specials, (b) 2026-09-30..to, (c) 2010-01-01..2026-09-29 (rest)."""
    a = list(MISSING_SPECIAL_SESSIONS)
    seen = set(a)
    b = daterange(date(2026, 9, 30), to)
    c = [d for d in daterange(date(2010, 1, 1), date(2026, 9, 29)) if d not in seen]
    return a + b + c


# ---- MCP cross-check ----------------------------------------------------------------------------

MCP_URL = "https://mcp.nseindia.in/bhavcopy/cm/mcp"
_TOLERANCE = 0.01

Pair = tuple[str, date]
ClosesFn = Callable[[list[Pair]], Awaitable[dict[Pair, float | None]]]


@dataclass(frozen=True)
class McpCheck:
    """status: ok | mismatch | unreachable. mismatches: (symbol, date, stored_close, mcp_close)."""

    status: str
    checked: int
    mismatches: list[tuple[str, date, float, float]] = field(default_factory=list)


def _number_from(obj: Any, want: date) -> float | None:
    """Best-effort close extraction from a tool result (dict/list/JSON text) for date `want`."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except ValueError:
            return None
    if isinstance(obj, int | float) and not isinstance(obj, bool):
        return float(obj)  # e.g. get_ltp_by_date answers a bare number
    if isinstance(obj, list):
        for item in obj:
            v = _number_from(item, want)
            if v is not None:
                return v
        return None
    if isinstance(obj, dict):
        ds = next((str(obj[k])[:10] for k in ("date", "tradDt", "TradDt", "timestamp") if k in obj), None)
        if ds is None or ds == want.isoformat():
            for k in ("close", "Close", "CLOSE", "closePrice", "ltp", "LTP", "ClsPric"):
                if k in obj and isinstance(obj[k], int | float):
                    return float(obj[k])
        for v in obj.values():
            if isinstance(v, dict | list):
                r = _number_from(v, want)
                if r is not None:
                    return r
    return None


async def _mcp_closes(pairs: list[Pair]) -> dict[Pair, float | None]:
    """Ask the official NSE bhavcopy MCP server for each pair's close (tool and arguments from list_tools)."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    out: dict[Pair, float | None] = {}
    async with (
        streamable_http_client(MCP_URL) as streams,
        ClientSession(streams[0], streams[1]) as session,
    ):
        await session.initialize()
        tools = {t.name: t for t in (await session.list_tools()).tools}
        tool = tools.get("get_ltp_by_date")
        if tool is None:
            raise DataNotAvailable(f"no single-date price tool on the MCP server; tools: {sorted(tools)}")
        schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
        props = schema.get("properties", {})
        for sym, d in pairs:
            res = await session.call_tool(tool.name, _build_args(tool.name, props, sym, d))
            out[(sym, d)] = _number_from("".join(getattr(c, "text", "") for c in res.content), d)
    return out


def _build_args(tool: str, props: dict[str, Any], sym: str, d: date) -> dict[str, Any]:
    """Arguments for `tool` from its input-schema property names."""
    args: dict[str, Any] = {}
    for p in props:
        low = p.lower()
        if low in ("symbol", "ticker", "stock", "tckrsymb"):
            args[p] = sym
        elif (
            low in ("date", "trade_date", "tradedate", "on_date", "as_of")
            or low in ("from_date", "start_date", "start", "from", "fromdate", "startdate")
            or low in ("to_date", "end_date", "end", "to", "todate", "enddate")
        ):
            args[p] = d.isoformat()
    return args


def cross_check_mcp(settings: Settings, pairs: list[Pair], closes_fn: ClosesFn | None = None) -> McpCheck:
    """Compare stored closes of (symbol, date) pairs with the official NSE MCP server (within 0.01).

    An unreachable server (or any transport failure) gives status 'unreachable' and never raises;
    a reachable server with any |difference| > 0.01 gives 'mismatch'. Pairs the server has no value
    for are not counted.
    """
    fn = closes_fn if closes_fn is not None else _mcp_closes
    try:
        got = asyncio.run(fn(list(pairs)))
    except Exception as e:  # noqa: BLE001 - the cross-check is non-blocking when the server is down
        log.warning("mcp_unreachable", error=f"{type(e).__name__}: {e}")
        return McpCheck("unreachable", 0, [])
    dates = sorted({d for _, d in pairs})
    stored = (
        read(
            settings,
            "bhavcopy_eq",
            start=dates[0],
            end=dates[-1],
            columns=["symbol", "series", "date", "close"],
        )
        .filter(pl.col("series") == "EQ")
        .collect()
    )
    close_of = {(s, d): c for s, d, c in zip(stored["symbol"], stored["date"], stored["close"], strict=True)}
    checked = 0
    bad: list[tuple[str, date, float, float]] = []
    for pair in pairs:
        theirs = got.get(pair)
        mine = close_of.get(pair)
        if theirs is None or mine is None:
            continue
        checked += 1
        if abs(theirs - mine) > _TOLERANCE:
            bad.append((pair[0], pair[1], mine, theirs))
    return McpCheck("mismatch" if bad else "ok", checked, bad)


def sample_pairs(settings: Settings, days: list[date], n: int = 20, seed: int = 0) -> list[Pair]:
    """n random (symbol, date) pairs (series EQ, positive volume) from the given ingested weekdays.

    Weekend sessions are left out: the MCP server answers a non-weekday date with the previous close.
    """
    days = [d for d in days if d.weekday() < 5]
    if not days:
        return []
    df = (
        read(
            settings,
            "bhavcopy_eq",
            start=min(days),
            end=max(days),
            columns=["symbol", "series", "date", "volume"],
        )
        .filter((pl.col("series") == "EQ") & (pl.col("volume") > 0) & pl.col("date").is_in(days))
        .collect()
    )
    pairs = sorted(zip(df["symbol"].to_list(), df["date"].to_list(), strict=True))
    return random.Random(seed).sample(pairs, min(n, len(pairs)))
