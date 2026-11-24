"""Open-interest buildup: price and OI direction of the near-month future, basis and rollover."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import SystemClock
from stocksense.core.config import Settings
from stocksense.data import io
from stocksense.features.derivs import oi_price_quad

ROLL_WINDOW = 5  # rollover % is reported in the last 5 sessions before the nearest expiry
WARMUP_DAYS = 10  # calendar days read before `start` for the previous session's OI and settle
LABELS = {1.0: "long buildup", -1.0: "short buildup", 0.5: "short covering", -0.5: "long unwinding"}

_COLS = ["date", "symbol", "fut_close", "fut_oi", "oi_chg_pct", "price_chg_pct", "basis_pct", "rollover_pct"]


def _roll_cut(calendar: TradingCalendar, expiry: date) -> date:
    """First session on which a contract stops being the near month (the session before its expiry)."""
    try:
        return calendar.prev_session(expiry)
    except ValueError:
        return expiry


def futures_daily(fo: pl.DataFrame, calendar: TradingCalendar) -> pl.DataFrame:
    """Per (date, symbol): near-month close, total OI, OI/price change %, basis %, rollover %, quad.

    Near month = nearest expiry >= date, switched to the next expiry on the session before expiry.
    price_chg_pct compares the near contract with its own previous settle (no roll jump); oi_chg_pct is
    total OI over all expiries vs the previous session; rollover_pct = next-expiry OI / total OI x 100
    in the 1..5 sessions before the nearest expiry, else null.
    """
    fut = (
        fo.filter(pl.col("instrument_type").is_in(["STF", "IDF"]))
        .filter(pl.col("expiry_date") >= pl.col("date"))
        .select("symbol", "expiry_date", "date", "settle_price", "oi", "underlying_price")
        .with_columns(pl.col("oi").cast(pl.Float64))
        .sort("symbol", "expiry_date", "date")
        .with_columns(pl.col("settle_price").shift(1).over("symbol", "expiry_date").alias("_prev_settle"))
    )
    if fut.height == 0:
        schema = dict.fromkeys(_COLS, pl.Float64) | {"date": pl.Date, "symbol": pl.Utf8}
        return pl.DataFrame(schema=schema | {"quad": pl.Float64, "quad_label": pl.Utf8})
    expiries = fut["expiry_date"].unique().sort().to_list()
    cut = pl.DataFrame(
        {"expiry_date": expiries, "_cut": [_roll_cut(calendar, e) for e in expiries]},
        schema={"expiry_date": pl.Date, "_cut": pl.Date},
    )
    keys = ["symbol", "date"]
    fut = fut.join(cut, on="expiry_date").with_columns(
        (pl.col("date") < pl.col("_cut")).alias("_elig"),
        pl.col("expiry_date").rank("dense").over(keys).alias("_rk"),
    )
    near_e = pl.coalesce(
        pl.col("expiry_date").filter(pl.col("_elig")).min().over(keys), pl.col("expiry_date").min().over(keys)
    )
    tot = (
        fut.group_by(keys)
        .agg(
            pl.col("oi").sum().alias("fut_oi"),
            pl.col("expiry_date").min().alias("_e0"),
            pl.col("oi").filter(pl.col("_rk") == 2).first().alias("_oi_next"),
        )
        .sort("symbol", "date")
        .with_columns(pl.col("fut_oi").shift(1).over("symbol").alias("_prev_oi"))
    )
    pairs = tot.select("date", "_e0").unique()
    gap = pl.DataFrame(
        {
            "date": pairs["date"],
            "_e0": pairs["_e0"],
            "_gap": [len(calendar.sessions(d, e)) - 1 for d, e in pairs.iter_rows()],
        },
        schema={"date": pl.Date, "_e0": pl.Date, "_gap": pl.Int64},
    )
    tot = tot.join(gap, on=["date", "_e0"], how="left")
    near = fut.filter(pl.col("expiry_date") == near_e)
    out = (
        near.join(tot, on=keys)
        .with_columns(
            pl.col("settle_price").alias("fut_close"),
            pl.when(pl.col("_prev_oi") > 0)
            .then((pl.col("fut_oi") / pl.col("_prev_oi") - 1) * 100)
            .alias("oi_chg_pct"),
            pl.when(pl.col("_prev_settle") > 0)
            .then((pl.col("settle_price") / pl.col("_prev_settle") - 1) * 100)
            .alias("price_chg_pct"),
            pl.when(pl.col("underlying_price") > 0)
            .then((pl.col("settle_price") / pl.col("underlying_price") - 1) * 100)
            .alias("basis_pct"),
            pl.when(pl.col("_gap").is_between(1, ROLL_WINDOW) & (pl.col("fut_oi") > 0))
            .then(pl.col("_oi_next") / pl.col("fut_oi") * 100)
            .alias("rollover_pct"),
        )
        .with_columns(oi_price_quad(pl.col("price_chg_pct"), pl.col("oi_chg_pct")).alias("quad"))
        .with_columns(
            pl.col("quad").replace_strict(LABELS, default=None, return_dtype=pl.Utf8).alias("quad_label")
        )
    )
    return out.select(*_COLS, "quad", "quad_label").sort(keys)


def build_oi_daily(
    settings: Settings, start: date, end: date, calendar: TradingCalendar | None = None
) -> int:
    """Compute and upsert flow_oi_daily for start..end; returns rows written."""
    cal = calendar or TradingCalendar.from_store(settings, SystemClock())
    fo = io.read(settings, "fo_bhavcopy", start - timedelta(days=WARMUP_DAYS), end).collect()
    out = futures_daily(fo, cal).filter(pl.col("date").is_between(start, end))
    if out.height:
        io.upsert(settings, "flow_oi_daily", out)
    return out.height
