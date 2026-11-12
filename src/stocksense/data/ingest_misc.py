from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

from stocksense.core.clock import IST
from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.data.ingest_common import (
    NSE_ARCHIVES,
    IngestContext,
    IngestResult,
    absent,
    fetch_cached,
    marker,
    num,
    read_strings,
)
from stocksense.data.io import upsert

log = get_logger(__name__)

# ---- participant-wise open interest ---------------------------------------------------------------

OI = "participant_oi"
OI_MISS_LIMIT = 30
_OI_COLS = {
    "Future Index Long": "fut_idx_long",
    "Future Index Short": "fut_idx_short",
    "Future Stock Long": "fut_stk_long",
    "Future Stock Short": "fut_stk_short",
    "Option Index Call Long": "opt_idx_call_long",
    "Option Index Put Long": "opt_idx_put_long",
    "Option Index Call Short": "opt_idx_call_short",
    "Option Index Put Short": "opt_idx_put_short",
    "Option Stock Call Long": "opt_stk_call_long",
    "Option Stock Put Long": "opt_stk_put_long",
    "Option Stock Call Short": "opt_stk_call_short",
    "Option Stock Put Short": "opt_stk_put_short",
    "Total Long Contracts": "total_long",
    "Total Short Contracts": "total_short",
}
_RE_TITLE_DATE = re.compile(r"as on\s+([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})", re.IGNORECASE)
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def parse_participant_oi(raw: bytes) -> tuple[date | None, pl.DataFrame]:
    """(date in the title, rows). The first line is a title (`... as on Nov 09, 2026`), a blank line may
    follow, then the header; the TOTAL row is dropped. Columns are the card's plus stock-option and totals."""
    text = raw.decode("utf-8", "replace")
    lines = text.splitlines()
    title_date = None
    m = _RE_TITLE_DATE.search(lines[0]) if lines else None
    if m:
        mon = m[1][:3].lower()
        if mon in _MONTHS:
            title_date = date(int(m[3]), _MONTHS.index(mon) + 1, int(m[2]))
    start = next((i for i, ln in enumerate(lines) if ln.strip().lower().startswith("client type")), None)
    if start is None:
        raise DataNotAvailable("participant OI file has no 'Client Type' header")
    df = read_strings("\n".join(ln for ln in lines[start:] if ln.strip()).encode("utf-8"))
    df = df.rename({"Client Type": "client_type"}) if "Client Type" in df.columns else df
    df = df.filter(~pl.col("client_type").str.to_uppercase().is_in(["TOTAL", ""]))
    exprs = [pl.col("client_type").str.to_uppercase()]
    exprs += [num(src).alias(dst) for src, dst in _OI_COLS.items() if src in df.columns]
    return title_date, df.select(exprs)


def ingest_participant_oi_day(ctx: IngestContext, d: date) -> IngestResult:
    """One day's participant OI; a file whose title date is not `d` (holiday trap) is `no_session`."""
    if ctx.state.kv_get(marker(OI, d)) is not None:
        return IngestResult(OI, d, 0, "skipped")
    url = f"{NSE_ARCHIVES}/content/nsccl/fao_participant_oi_{d:%d%m%Y}.csv"
    cache = ctx.settings.paths.data_root / "cache" / OI / f"{d:%Y}" / f"{d:%Y-%m-%d}.csv"
    try:
        raw = fetch_cached(ctx, url, cache)
    except Exception as e:  # noqa: BLE001 - non-blocking dataset: any failure is a status, not a crash
        log.warning("participant_oi_failed", date=str(d), error=str(e))
        return IngestResult(OI, d, 0, "failed")
    if raw is None:
        return absent(ctx, OI, OI, d)
    try:
        title_date, df = parse_participant_oi(raw)
    except DataNotAvailable as e:
        log.warning("participant_oi_unparsable", date=str(d), error=str(e))
        return IngestResult(OI, d, 0, "failed")
    if title_date is not None and title_date != d:
        cache.unlink(missing_ok=True)
        return absent(ctx, OI, OI, d)
    upsert(ctx.settings, OI, df.with_columns(pl.lit(d).alias("date")), clock=ctx.clock)
    return IngestResult(OI, d, df.height, "ok")


def ingest_participant_oi_range(
    ctx: IngestContext, a: date, b: date, on_result: Callable[[IngestResult], None] | None = None
) -> list[IngestResult]:
    """Participant OI for every calendar day in [a, b]; non-blocking."""
    out = []
    d = a
    while d <= b:
        out.append(ingest_participant_oi_day(ctx, d))
        if on_result:
            on_result(out[-1])
        d += timedelta(days=1)
    return out


def backfill_participant_oi(
    ctx: IngestContext, end: date, on_result: Callable[[IngestResult], None] | None = None
) -> list[IngestResult]:
    """Walk backward from `end` over every calendar day until 30 consecutive days without a file."""
    out: list[IngestResult] = []
    misses = 0
    d = end
    while misses < OI_MISS_LIMIT:
        r = ingest_participant_oi_day(ctx, d)
        out.append(r)
        if on_result:
            on_result(r)
        if r.status == "ok":
            misses = 0
        elif r.status in ("no_session", "skipped", "not_published"):
            misses += 1
        else:
            misses += 1
        d -= timedelta(days=1)
    return out


# ---- FII / DII cash flows --------------------------------------------------------------------------

FII = "fii_dii"
_FII_CATEGORY = {"FII/FPI": "FII", "FII": "FII", "FPI": "FII", "DII": "DII"}


def _month_date(s: str) -> date | None:
    try:
        d, m, y = s.strip().split("-")
        return date(int(y), _MONTHS.index(m[:3].lower()) + 1, int(d))
    except (ValueError, AttributeError):
        return None


def parse_fii_dii(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """fii_dii rows from the API's list (`category, date 11-Nov-2026, buyValue, sellValue, netValue`)."""
    out = []
    for r in rows:
        d = _month_date(str(r.get("date", "")))
        cat = _FII_CATEGORY.get(str(r.get("category", "")).strip().upper())
        if d is None or cat is None:
            continue
        out.append(
            {
                "date": d,
                "category": cat,
                "buy_value": float(str(r.get("buyValue")).replace(",", "")),
                "sell_value": float(str(r.get("sellValue")).replace(",", "")),
                "net_value": float(str(r.get("netValue")).replace(",", "")),
            }
        )
    schema = {
        "date": pl.Date,
        "category": pl.Utf8,
        "buy_value": pl.Float64,
        "sell_value": pl.Float64,
        "net_value": pl.Float64,
    }
    df = pl.DataFrame(out, schema=schema) if out else pl.DataFrame(schema=schema)
    return df.unique(subset=["date", "category"], keep="last", maintain_order=True)


def ingest_fii_dii(ctx: IngestContext) -> IngestResult:
    """Append the latest day (the API serves no history). Non-blocking."""
    today = ctx.clock.today()
    try:
        payload = ctx.nse.get_api("/api/fiidiiTradeReact", None, "reports/fii-dii")
        df = parse_fii_dii(payload if isinstance(payload, list) else [])
    except Exception as e:  # noqa: BLE001
        log.warning("fii_dii_failed", error=str(e))
        return IngestResult(FII, today, 0, "failed")
    if df.height:
        upsert(ctx.settings, FII, df, clock=ctx.clock)
    return IngestResult(FII, today, df.height, "ok" if df.height else "no_session")


def seed_fii_dii_from_snapshots(ctx: IngestContext) -> int:
    """Seed `fii_dii` from the adopted nse_snapshots/fii_dii files (latest snapshot per date and category)."""
    import glob

    files = sorted(
        glob.glob(str(ctx.settings.paths.data_root / "parquet" / "nse_snapshots" / "fii_dii" / "*.parquet"))
    )
    frames = [pl.read_parquet(f) for f in files]
    if not frames:
        return 0
    df = pl.concat(frames, how="diagonal_relaxed")
    df = df.sort("snapshot_ts").group_by(["date", "category"], maintain_order=True).last()
    parsed = parse_fii_dii(df.select("category", "date", "buyValue", "sellValue", "netValue").to_dicts())
    if parsed.height:
        upsert(ctx.settings, FII, parsed, clock=ctx.clock)
    return parsed.height


# ---- bulk and block deals (today) ------------------------------------------------------------------

DEALS = "deals"
_DEAL_URLS = {"bulk": "/content/equities/bulk.csv", "block": "/content/equities/block.csv"}


def parse_deals(raw: bytes, kind: str) -> pl.DataFrame:
    """deals rows from bulk.csv / block.csv (`Date, Symbol, Security Name, Client Name, Buy/Sell, ...`)."""
    schema = {
        "date": pl.Date,
        "kind": pl.Utf8,
        "symbol": pl.Utf8,
        "client": pl.Utf8,
        "side": pl.Utf8,
        "qty": pl.Float64,
        "price": pl.Float64,
    }
    text = raw.decode("utf-8-sig", "replace")
    first = text.splitlines()[0] if text.strip() else ""
    if "Symbol" not in first:
        return pl.DataFrame(schema=schema)
    df = read_strings(text.encode("utf-8"))
    qty = [c for c in df.columns if c.lower().startswith("quantity")][0]
    price = [c for c in df.columns if c.lower().startswith("trade price")][0]
    out = df.select(
        pl.col("Date").str.to_titlecase().str.strptime(pl.Date, "%d-%b-%Y", strict=False).alias("date"),
        pl.lit(kind).alias("kind"),
        pl.col("Symbol").alias("symbol"),
        pl.col("Client Name").alias("client"),
        pl.col("Buy/Sell").str.to_uppercase().alias("side"),
        pl.col(qty).str.replace_all(",", "").cast(pl.Float64, strict=False).alias("qty"),
        pl.col(price).str.replace_all(",", "").cast(pl.Float64, strict=False).alias("price"),
    ).drop_nulls(["date", "symbol", "client", "side", "qty", "price"])
    return out.unique(maintain_order=True)


def ingest_deals_today(ctx: IngestContext) -> list[IngestResult]:
    """Append today's bulk and block deals (history starts now). The raw file is cached under its own date."""
    today = ctx.clock.today()
    out: list[IngestResult] = []
    for kind, path in _DEAL_URLS.items():
        try:
            raw = ctx.nse.get_archive(NSE_ARCHIVES + path)
            if raw is None:
                out.append(IngestResult(DEALS, today, 0, "not_published"))
                continue
            df = parse_deals(raw, kind)
            if df.height:
                d0 = df["date"].max()
                cache = ctx.settings.paths.data_root / "cache" / "deals" / f"{kind}-{d0:%Y-%m-%d}.csv"
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(raw)
                upsert(ctx.settings, DEALS, df, clock=ctx.clock)
            out.append(IngestResult(DEALS, today, df.height, "ok"))
        except Exception as e:  # noqa: BLE001
            log.warning("deals_failed", kind=kind, error=str(e))
            out.append(IngestResult(DEALS, today, 0, "failed"))
    return out


# ---- live captures: pre-open auction, spreads, quotes ----------------------------------------------

PREOPEN = "preopen"
_PREOPEN_REF = "market-data/pre-open-market-cm-and-emerge-market"
_PREOPEN_KEYS = ("NIFTY", "FO", "OTHERS")
_PRE_SCHEMA = {
    "symbol": pl.Utf8,
    "ts": pl.Datetime("us", "Asia/Kolkata"),
    "prev_close": pl.Float64,
    "indicative_price": pl.Float64,
    "final_price": pl.Float64,
    "buy_qty": pl.Float64,
    "sell_qty": pl.Float64,
    "source": pl.Utf8,
}


def _f(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def parse_preopen(payload: dict[str, Any], ts: datetime) -> pl.DataFrame:
    """preopen rows from `/api/market-data-pre-open` (`data[i].metadata` + `detail.preOpenMarket`).

    iep -> indicative_price; finalPrice (detail, else metadata; 0 -> null) -> final_price. Probe
    2026-11-11: `metadata` has no finalPrice (it is in `detail.preOpenMarket.finalPrice`) and carries
    `previousClose`, `iep`, `finalQuantity`; quantities are `detail.preOpenMarket.totalBuyQuantity` and
    `totalSellQuantity`.
    """
    rows = []
    for item in payload.get("data") or []:
        meta = item.get("metadata") or {}
        det = (item.get("detail") or {}).get("preOpenMarket") or {}
        sym = meta.get("symbol")
        if not sym:
            continue
        final = _f(det.get("finalPrice"))
        if final is None:
            final = _f(meta.get("finalPrice"))
        iep = _f(meta.get("iep"))
        if iep is None:
            iep = _f(det.get("IEP"))
        rows.append(
            {
                "symbol": str(sym),
                "ts": ts,
                "prev_close": _f(meta.get("previousClose")),
                "indicative_price": iep,
                "final_price": final if final else None,
                "buy_qty": _f(det.get("totalBuyQuantity")),
                "sell_qty": _f(det.get("totalSellQuantity")),
                "source": "nse_api",
            }
        )
    return pl.DataFrame(rows, schema=_PRE_SCHEMA) if rows else pl.DataFrame(schema=_PRE_SCHEMA)


def capture_preopen(ctx: IngestContext) -> int:
    """Store one pre-open snapshot (all symbols) at clock.now(); returns rows (live 09:00-09:08 IST)."""
    try:
        payload = ctx.nse.get_api("/api/market-data-pre-open", {"key": "ALL"}, _PREOPEN_REF)
        df = parse_preopen(payload, ctx.clock.now())
    except DataNotAvailable:
        frames = []
        for key in _PREOPEN_KEYS:
            payload = ctx.nse.get_api("/api/market-data-pre-open", {"key": key}, _PREOPEN_REF)
            frames.append(parse_preopen(payload, ctx.clock.now()))
        df = pl.concat(frames).unique(subset=["symbol", "ts"], keep="last")
    if df.height:
        upsert(ctx.settings, PREOPEN, df, clock=ctx.clock)
    return df.height


def seed_preopen_from_snapshots(ctx: IngestContext) -> int:
    """Seed `preopen` from the adopted nse_snapshots/pre_open files (ts = source or snapshot time)."""
    import glob

    files = sorted(
        glob.glob(str(ctx.settings.paths.data_root / "parquet" / "nse_snapshots" / "pre_open" / "*.parquet"))
    )
    total = 0
    for f in files:
        df = pl.read_parquet(f)
        for c in ("prev_close", "indicative_price", "final_price", "buy_quantity", "sell_quantity"):
            if c not in df.columns:  # the earliest snapshots lack the auction columns
                df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
        ts = (
            pl.coalesce(pl.col("source_timestamp"), pl.col("snapshot_ts"))
            if "source_timestamp" in df.columns
            else pl.col("snapshot_ts")
        )
        out = df.select(
            pl.col("symbol"),
            ts.dt.replace_time_zone("Asia/Kolkata").dt.cast_time_unit("us").alias("ts"),
            pl.col("prev_close"),
            pl.col("indicative_price"),
            pl.when(pl.col("final_price") > 0)
            .then(pl.col("final_price"))
            .otherwise(None)
            .alias("final_price"),
            pl.col("buy_quantity").alias("buy_qty"),
            pl.col("sell_quantity").alias("sell_qty"),
            pl.lit("nse_snapshots").alias("source"),
        ).drop_nulls(["symbol", "ts"])
        if out.height:
            upsert(ctx.settings, PREOPEN, out, clock=ctx.clock)
            total += out.height
    return total


def capture_spreads(
    ctx: IngestContext, symbols: Sequence[str], quotes_source: Callable[[Sequence[str]], list]
) -> int:
    """Append bid/ask spread rows for the symbols whose quote has both sides > 0; returns rows."""
    rows = [
        {
            "symbol": q.symbol,
            "ts": q.ts,
            "bid": q.bid,
            "ask": q.ask,
            "ltp": q.ltp,
            "bid_qty": q.bid_qty,
            "ask_qty": q.ask_qty,
            "source": q.source,
        }
        for q in quotes_source(list(symbols))
        if q.bid and q.ask and q.bid > 0 and q.ask > 0
    ]
    if not rows:
        return 0
    schema = {
        "symbol": pl.Utf8,
        "ts": pl.Datetime("us", "Asia/Kolkata"),
        "bid": pl.Float64,
        "ask": pl.Float64,
        "ltp": pl.Float64,
        "bid_qty": pl.Float64,
        "ask_qty": pl.Float64,
        "source": pl.Utf8,
    }
    upsert(ctx.settings, "spreads", pl.DataFrame(rows, schema=schema), clock=ctx.clock)
    return len(rows)


def capture_quotes(ctx: IngestContext, quotes: Sequence) -> int:
    """Append quotes (floored to the minute) to `quotes_1m`; returns rows."""
    rows = [
        {
            "symbol": q.symbol,
            "ts": q.ts.astimezone(IST).replace(second=0, microsecond=0),
            "ltp": q.ltp,
            "open": q.open,
            "high": q.high,
            "low": q.low,
            "close_prev": q.prev_close,
            "volume": q.volume,
            "bid": q.bid,
            "ask": q.ask,
            "source": q.source,
        }
        for q in quotes
    ]
    if not rows:
        return 0
    schema = {
        "symbol": pl.Utf8,
        "ts": pl.Datetime("us", "Asia/Kolkata"),
        "ltp": pl.Float64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close_prev": pl.Float64,
        "volume": pl.Float64,
        "bid": pl.Float64,
        "ask": pl.Float64,
        "source": pl.Utf8,
    }
    upsert(ctx.settings, "quotes_1m", pl.DataFrame(rows, schema=schema), clock=ctx.clock)
    return len(rows)
