from __future__ import annotations

import io
from datetime import UTC, date, datetime, time, timedelta

import httpx
import polars as pl

from stocksense.core.clock import IST
from stocksense.core.log import get_logger
from stocksense.data.ingest_common import IngestContext, IngestResult
from stocksense.data.io import upsert

log = get_logger(__name__)

DATASET = "macro_daily"
#: series -> (stooq symbol, Yahoo chart symbol). stooq is the card's source; on 2026-11-11 it answered
#: every symbol with an HTML page (it now wants an API key), so Yahoo's chart endpoint is the fallback.
SERIES: dict[str, tuple[str, str]] = {
    "USDINR": ("usdinr", "USDINR=X"),
    "BRENT": ("cb.f", "BZ=F"),
    "SPX": ("^spx", "^GSPC"),
    "NDX": ("^ndx", "^NDX"),
    "US10Y": ("10usy.b", "^TNX"),
    "DXY": ("dx.f", "DX-Y.NYB"),
    "GOLD": ("xauusd", "GC=F"),
}
US_SESSION = {"SPX", "NDX", "US10Y", "DXY"}
_UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36"
}
_SCHEMA = {"series": pl.Utf8, "date": pl.Date, "close": pl.Float64}


def available_ts(series: str, d: date) -> datetime:
    """When the day's value is usable: US series next day 06:00 IST, the rest same day 23:30."""
    if series in US_SESSION:
        return datetime.combine(d + timedelta(days=1), time(6, 0), IST)
    return datetime.combine(d, time(23, 30), IST)


def parse_stooq(text: str) -> pl.DataFrame:
    """(date, close) rows from stooq's `Date,Open,High,Low,Close,Volume` CSV; empty for HTML / 'No data'."""
    head = text.lstrip()[:20].lower()
    if not head.startswith("date,") or text.strip().lower().startswith("no data"):
        return pl.DataFrame(schema={"date": pl.Date, "close": pl.Float64})
    df = pl.read_csv(io.StringIO(text), infer_schema_length=0)
    if "Date" not in df.columns or "Close" not in df.columns:
        return pl.DataFrame(schema={"date": pl.Date, "close": pl.Float64})
    return (
        df.select(
            pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("date"),
            pl.col("Close").cast(pl.Float64, strict=False).alias("close"),
        )
        .drop_nulls()
        .unique(subset=["date"], keep="last")
        .sort("date")
    )


def parse_yahoo(payload: dict) -> pl.DataFrame:
    """(date, close) rows from Yahoo's chart JSON; the date is the exchange-local calendar date."""
    try:
        res = payload["chart"]["result"][0]
        stamps = res["timestamp"]
        closes = res["indicators"]["quote"][0]["close"]
        off = int(res["meta"].get("gmtoffset", 0))
    except (KeyError, IndexError, TypeError):
        return pl.DataFrame(schema={"date": pl.Date, "close": pl.Float64})
    rows = [
        (datetime.fromtimestamp(t + off, UTC).date(), float(c))
        for t, c in zip(stamps, closes, strict=False)
        if c is not None
    ]
    if not rows:
        return pl.DataFrame(schema={"date": pl.Date, "close": pl.Float64})
    return (
        pl.DataFrame(rows, schema=["date", "close"], orient="row")
        .unique(subset=["date"], keep="last")
        .sort("date")
    )


def _fetch_series(http: httpx.Client, series: str) -> tuple[pl.DataFrame, str]:
    stooq, yahoo = SERIES[series]
    try:
        r = http.get(f"https://stooq.com/q/d/l/?s={stooq}&i=d", headers=_UA)
        df = parse_stooq(r.text) if r.status_code == 200 else parse_stooq("")
    except httpx.HTTPError:
        df = parse_stooq("")
    if df.height:
        return df, "stooq"
    try:
        r = http.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo}",
            params={"period1": 0, "period2": 4102444800, "interval": "1d"},  # range=max is coarse
            headers=_UA,
        )
        if r.status_code == 200:
            return parse_yahoo(r.json()), "yahoo"
    except (httpx.HTTPError, ValueError):
        pass
    return parse_stooq(""), "none"


def ingest_macro(ctx: IngestContext, http: httpx.Client | None = None) -> list[IngestResult]:
    """Fetch every macro series (stooq, then Yahoo) and upsert `macro_daily`; one result per series.

    Never raises: a series that no source serves is recorded in kv `macro_unavailable:<series>`.
    """
    own = http is None
    http = http or httpx.Client(timeout=httpx.Timeout(30, connect=15), follow_redirects=True)
    today = ctx.clock.today()
    out: list[IngestResult] = []
    try:
        for name in SERIES:
            df, src = _fetch_series(http, name)
            if df.height == 0:
                ctx.state.kv_set(f"macro_unavailable:{name}", today.isoformat())
                log.warning("macro_series_unavailable", series=name)
                out.append(IngestResult(DATASET, today, 0, "failed"))
                continue
            rows = df.filter(pl.col("date") <= today).with_columns(
                pl.lit(name).alias("series"),
                pl.col("date")
                .map_elements(
                    lambda d, n=name: available_ts(n, d), return_dtype=pl.Datetime("us", "Asia/Kolkata")
                )
                .alias("available_ts"),
            )
            upsert(
                ctx.settings, DATASET, rows.select("series", "date", "close", "available_ts"), clock=ctx.clock
            )
            ctx.state.kv_set(f"macro_source:{name}", src)
            out.append(IngestResult(DATASET, today, rows.height, "ok"))
    finally:
        if own:
            http.close()
    return out
