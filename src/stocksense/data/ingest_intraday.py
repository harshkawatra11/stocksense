from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from datetime import date, datetime, time, timedelta
from typing import Any

import polars as pl

from stocksense.brokers.angel import AngelClient
from stocksense.brokers.instruments import InstrumentMap
from stocksense.brokers.upstox import UpstoxClient
from stocksense.core.clock import IST
from stocksense.core.errors import BrokerError
from stocksense.core.log import get_logger
from stocksense.data.ingest_common import IngestContext, IngestResult
from stocksense.data.io import list_partitions, read, upsert

log = get_logger(__name__)

BARS_5M = "intraday_bars"
BARS_1M = "bars_1m"
TOP_PER_YEAR = 300
FIRST_1M_DATE = date(2022, 1, 3)
FLUSH_SYMBOLS = 25
MAX_CONSECUTIVE_FAILURES = 25
FULL_SESSION_BARS = {"1m": 375, "5m": 75}


# ---- load_bars ----------------------------------------------------------------------------------


def load_bars(
    settings: Any,
    symbols: Iterable[str],
    start: date,
    end: date,
    interval: str = "5m",
    basis: str = "adjusted",
) -> pl.DataFrame:
    """Intraday bars (symbol, ts, open, high, low, close, volume) for [start, end].

    basis="raw_ratio": the stored bars untouched (only ratios inside one session are meaningful).
    basis="adjusted": each (symbol, session) is rescaled by k = panel adj_open / first bar's open
    (prices x k, volume / k), so bars sit on the panel's adjusted price scale whatever adjustment the
    download carried; if the 09:15 bar is missing k = panel adj_close / last bar's close. Sessions with
    no panel row are dropped.
    """
    if interval not in ("5m", "1m"):
        raise ValueError(f"interval must be '5m' or '1m', got {interval!r}")
    if basis not in ("adjusted", "raw_ratio"):
        raise ValueError(f"basis must be 'adjusted' or 'raw_ratio', got {basis!r}")
    syms = list(symbols)
    ds = BARS_5M if interval == "5m" else BARS_1M
    cols = ["symbol", "ts", "open", "high", "low", "close", "volume"]
    bars = (
        read(settings, ds, start=start, end=end, columns=cols).filter(pl.col("symbol").is_in(syms)).collect()
    )
    bars = bars.with_columns(pl.col("volume").cast(pl.Float64)).sort("symbol", "ts")
    if basis == "raw_ratio" or bars.height == 0:
        return bars
    bars = bars.with_columns(pl.col("ts").dt.date().alias("date"))
    first_open = bars.filter(pl.col("ts").dt.time() == time(9, 15)).select(
        "symbol", "date", pl.col("open").alias("_fo")
    )
    last_close = bars.group_by("symbol", "date", maintain_order=True).agg(pl.col("close").last().alias("_lc"))
    panel = (
        read(
            settings, "panel_daily", start=start, end=end, columns=["symbol", "date", "adj_open", "adj_close"]
        )
        .filter(pl.col("symbol").is_in(syms))
        .collect()
    )
    j = (
        bars.join(panel, on=["symbol", "date"], how="inner")
        .join(first_open, on=["symbol", "date"], how="left")
        .join(last_close, on=["symbol", "date"], how="left")
    )
    k = pl.coalesce(pl.col("adj_open") / pl.col("_fo"), pl.col("adj_close") / pl.col("_lc"))
    j = j.with_columns(k.alias("_k")).filter(pl.col("_k").is_not_null() & pl.col("_k").is_finite())
    return j.select(
        "symbol",
        "ts",
        (pl.col("open") * pl.col("_k")).alias("open"),
        (pl.col("high") * pl.col("_k")).alias("high"),
        (pl.col("low") * pl.col("_k")).alias("low"),
        (pl.col("close") * pl.col("_k")).alias("close"),
        (pl.col("volume") / pl.col("_k")).alias("volume"),
    ).sort("symbol", "ts")


# ---- 5-minute extension (Angel) -------------------------------------------------------------------


def stored_5m_symbols(settings: Any) -> list[str]:
    """Symbols that have 5-minute bars in the store."""
    if not list_partitions(settings, BARS_5M):
        return []
    df = read(settings, BARS_5M, columns=["symbol"]).select(pl.col("symbol").unique()).collect()
    return sorted(df["symbol"].to_list())


def _bars_frame(
    df: pl.DataFrame, symbol: str, interval: str | None, source: str, adj_asof: date
) -> pl.DataFrame:
    out = df.select(
        pl.lit(symbol).alias("symbol"),
        "ts",
        "open",
        "high",
        "low",
        "close",
        pl.col("volume").round(0).cast(pl.Int64),
        pl.lit(source).alias("source"),
        pl.lit(adj_asof).alias("adj_asof"),
    )
    if interval:
        out = out.with_columns(pl.lit(interval).alias("interval"))
    return out


def extend_5m(
    ctx: IngestContext,
    symbols: list[str] | None = None,
    *,
    angel: AngelClient | None = None,
    instruments: InstrumentMap | None = None,
    on_result: Callable[[IngestResult], None] | None = None,
) -> list[IngestResult]:
    """Extend each symbol's 5-minute bars from its last stored bar to now via Angel FIVE_MINUTE
    (<= 100 days per request); source 'angel', adj_asof today."""
    settings = ctx.settings
    syms = symbols if symbols is not None else stored_5m_symbols(settings)
    own_angel, own_map = angel is None, instruments is None
    angel = angel or AngelClient(settings, ctx.clock)
    instruments = instruments or InstrumentMap(settings, ctx.clock)
    last = (
        read(settings, BARS_5M, columns=["symbol", "ts"]).group_by("symbol").agg(pl.col("ts").max()).collect()
        if list_partitions(settings, BARS_5M)
        else pl.DataFrame(schema={"symbol": pl.Utf8, "ts": pl.Datetime("us", "Asia/Kolkata")})
    )
    last_ts = dict(zip(last["symbol"].to_list(), last["ts"].to_list(), strict=True))
    now = ctx.clock.now()
    today = ctx.clock.today()
    results: list[IngestResult] = []
    pending: list[pl.DataFrame] = []

    def flush() -> None:
        if pending:
            upsert(settings, BARS_5M, pl.concat(pending), clock=ctx.clock)
            pending.clear()

    try:
        for n, sym in enumerate(syms, 1):
            token = instruments.angel_token(sym)
            if token is None:
                results.append(IngestResult(BARS_5M, today, 0, "skipped"))
                continue
            start = last_ts.get(sym)
            start_dt = (
                (start + timedelta(minutes=5))
                if start
                else datetime.combine(today - timedelta(days=90), time(9, 15), IST)
            )
            if start_dt >= now:
                results.append(IngestResult(BARS_5M, today, 0, "skipped"))
                continue
            try:
                df = angel.candles(token, "FIVE_MINUTE", start_dt, now)
            except BrokerError as e:
                log.warning("extend_5m_failed", symbol=sym, error=str(e))
                res = IngestResult(BARS_5M, today, 0, "failed")
            else:
                df = df.filter(pl.col("ts") >= start_dt)
                if df.height:
                    pending.append(_bars_frame(df, sym, "FIVE_MINUTE", "angel", today))
                res = IngestResult(BARS_5M, today, df.height, "ok")
            results.append(res)
            if on_result:
                on_result(res)
            if n % FLUSH_SYMBOLS == 0:
                flush()
    finally:
        flush()
        if own_angel:
            angel.close()
        if own_map:
            instruments.close()
    return results


# ---- 1-minute backfill (Upstox) -------------------------------------------------------------------


def top_symbols_by_year(settings: Any, first_year: int, last_year: int, n: int = TOP_PER_YEAR) -> list[str]:
    """Union over calendar years of the `n` EQ symbols with the highest median turnover_inr that year."""
    df = (
        read(
            settings,
            "bhavcopy_eq",
            start=date(first_year, 1, 1),
            end=date(last_year, 12, 31),
            columns=["symbol", "series", "date", "turnover_inr"],
        )
        .filter(pl.col("series") == "EQ")
        .with_columns(pl.col("date").dt.year().alias("y"))
        .group_by("y", "symbol")
        .agg(pl.col("turnover_inr").median().alias("med"))
        .collect()
    )
    out: set[str] = set()
    for _, g in df.partition_by("y", as_dict=True).items():
        out |= set(g.sort("med", descending=True).head(n)["symbol"].to_list())
    return sorted(out)


def month_starts(first: date, last: date) -> list[date]:
    """First day of every month from first's month to last's month, inclusive."""
    out: list[date] = []
    y, m = first.year, first.month
    while (y, m) <= (last.year, last.month):
        out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def month_end(d: date) -> date:
    return (date(d.year + (d.month == 12), d.month % 12 + 1, 1)) - timedelta(days=1)


def job_name(symbol: str, month: date) -> str:
    return f"bars_1m:{symbol}:{month:%Y-%m}"


def validate_month(df: pl.DataFrame, bhav: pl.DataFrame, symbol: str) -> dict[str, int]:
    """Per symbol-session checks: a full session is 375 bars; summed bar volume within 5% of bhavcopy.

    Violations are counted, never dropped."""
    if df.height == 0:
        return {"sessions": 0, "short_sessions": 0, "volume_violations": 0}
    per = (
        df.with_columns(pl.col("ts").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.len().alias("n"), pl.col("volume").sum().alias("v"))
    )
    ref = bhav.filter(pl.col("symbol") == symbol).select("date", pl.col("volume").alias("bv"))
    j = per.join(ref, on="date", how="left")
    viol = j.filter(
        pl.col("bv").is_not_null() & (pl.col("bv") > 0) & ((pl.col("v") / pl.col("bv") - 1).abs() > 0.05)
    )
    return {
        "sessions": per.height,
        "short_sessions": per.filter(pl.col("n") < FULL_SESSION_BARS["1m"]).height,
        "volume_violations": viol.height,
    }


def backfill_1m(
    ctx: IngestContext,
    symbols: list[str],
    start: date = FIRST_1M_DATE,
    *,
    end: date | None = None,
    upstox: UpstoxClient | None = None,
    instruments: InstrumentMap | None = None,
    on_result: Callable[[IngestResult], None] | None = None,
) -> list[IngestResult]:
    """1-minute Upstox candles for each (symbol, month), newest month first, into `bars_1m`.

    Resumable: job_runs(job="bars_1m:<symbol>:<YYYY-MM>", trade_date=<first of month>) is 'done' only
    after the rows are written, and done jobs are skipped. Months of the current session are final
    only once they end before today.
    """
    settings = ctx.settings
    own_up, own_map = upstox is None, instruments is None
    upstox = upstox or UpstoxClient(settings, ctx.clock)
    instruments = instruments or InstrumentMap(settings, ctx.clock)
    today = ctx.clock.today()
    last_day = end or (today - timedelta(days=1))
    results: list[IngestResult] = []
    failures = 0
    pending: list[tuple[str, int, pl.DataFrame, dict[str, int]]] = []
    try:
        for m0 in reversed(month_starts(start, last_day)):
            m1 = min(month_end(m0), last_day)
            a = max(m0, start)
            todo = [s for s in symbols if ctx.state.job_status(job_name(s, m0), m0) != "done"]
            if not todo:
                continue
            bhav = (
                read(
                    settings, "bhavcopy_eq", start=m0, end=m1, columns=["symbol", "series", "date", "volume"]
                )
                .filter(pl.col("series") == "EQ")
                .collect()
            )

            def flush(m0: date = m0) -> None:
                if not pending:
                    return
                frames = [p[2] for p in pending if p[2].height]
                if frames:
                    upsert(settings, BARS_1M, pl.concat(frames), clock=ctx.clock)
                for sym, attempt, df, stats in pending:
                    ctx.state.job_finish(
                        job_name(sym, m0), m0, attempt, "done", json.dumps({"rows": df.height, **stats})
                    )
                pending.clear()

            for sym in todo:
                key = instruments.upstox_key(sym)
                if key is None:
                    results.append(IngestResult(BARS_1M, m0, 0, "skipped"))
                    continue
                attempt = ctx.state.job_start(job_name(sym, m0), m0)
                try:
                    df = upstox.candles(key, "minutes", 1, a, m1)
                except BrokerError as e:
                    log.warning("bars_1m_failed", symbol=sym, month=f"{m0:%Y-%m}", error=str(e))
                    ctx.state.job_finish(job_name(sym, m0), m0, attempt, "failed", str(e)[:200])
                    failures += 1
                    res = IngestResult(BARS_1M, m0, 0, "failed")
                    results.append(res)
                    if on_result:
                        on_result(res)
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        flush()
                        raise
                    continue
                failures = 0
                stats = validate_month(df, bhav, sym)
                pending.append((sym, attempt, _bars_frame(df, sym, None, "upstox", today), stats))
                res = IngestResult(BARS_1M, m0, df.height, "ok")
                results.append(res)
                if on_result:
                    on_result(res)
                if len(pending) >= FLUSH_SYMBOLS:
                    flush()
            flush()
    finally:
        if own_up:
            upstox.close()
        if own_map:
            instruments.close()
    return results
