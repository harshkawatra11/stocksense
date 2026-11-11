"""Intraday features from 5-minute bars (PLAN T4.3k): within-session ratios only, so a split or bonus
between the download date and the session never matters (the `raw_ratio` basis). Null where a stock has
no bars; session-wide statistics need the full 75-bar session.

`data/ingest_intraday.py` does not exist yet, so `intraday_bars` is read through the context directly.
A bar is labelled by its START time: close_15:25 is the last bar, close_14:25 the close at 14:30.
"""

from __future__ import annotations

from datetime import time

import polars as pl

from stocksense.features.base import FeatureContext, feature, log_ret, roll_std, span_ok

FULL_SESSION_BARS = 75
INTERVAL = "FIVE_MINUTE"
_T = {
    k: time(*v)
    for k, v in {
        "open": (9, 15),
        "m40": (9, 40),
        "h1": (10, 10),
        "pre_last": (14, 25),
        "last": (15, 25),
        "lh": (14, 30),
    }.items()
}


def _at(col: str, t: time) -> pl.Expr:
    """The value of `col` on the bar that starts at `t` (null if absent)."""
    return pl.col(col).filter(pl.col("_t") == t).first()


def _session_stats(ctx: FeatureContext) -> pl.DataFrame:
    bars = (
        ctx.raw("intraday_bars", ["symbol", "ts", "interval", "open", "high", "low", "close", "volume"])
        .filter(pl.col("interval") == INTERVAL)
        .sort("symbol", "ts")
        .with_columns(pl.col("ts").dt.date().alias("date"), pl.col("ts").dt.time().alias("_t"))
        .with_columns(
            pl.col("close").shift(1).over("symbol", "date").alias("_pc"),
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0 * pl.col("volume")).alias("_pv"),
        )
        .with_columns(
            pl.when(pl.col("_pc").is_null()).then(pl.col("open")).otherwise(pl.col("_pc")).alias("_ref")
        )
    )
    full = pl.len() == FULL_SESSION_BARS
    vol = pl.col("volume").sum().cast(pl.Float64)
    return (
        bars.group_by("symbol", "date")
        .agg(
            (_at("close", _T["last"]) / _at("close", _T["pre_last"])).log().alias("last_hour_ret"),
            (_at("close", _T["h1"]) / _at("open", _T["open"])).log().alias("first_hour_ret"),
            (_at("close", _T["open"]) / _at("open", _T["open"])).log().alias("first5_ret"),
            pl.when(full)
            .then(_at("close", _T["last"]) / (pl.col("_pv").sum() / vol) - 1.0)
            .alias("vwap_dev"),
            pl.when(full)
            .then((pl.col("close") / pl.col("_ref")).log().pow(2).sum().sqrt())
            .alias("intraday_rv"),
            pl.when(full)
            .then(pl.col("volume").filter(pl.col("_t") >= _T["lh"]).sum().cast(pl.Float64) / vol)
            .alias("vol_last_hour_share"),
            pl.when(full).then(_at("volume", _T["open"]).cast(pl.Float64) / vol).alias("vol_first5_share"),
            (_at("close", _T["m40"]) / _at("open", _T["open"])).log().alias("_c40"),
        )
        .collect()
    )


def _intraday(ctx: FeatureContext) -> pl.DataFrame:
    """All nine features per (date, sid) panel row; memoised on the context."""
    cache = ctx.__dict__.setdefault("_intraday_cache", {})
    if "df" in cache:
        return cache["df"]
    panel = ctx.panel().collect().sort("sid", "date")
    pc = pl.col("adj_close").shift(1).over("sid")
    tr = pl.when(pc.is_not_null()).then(
        pl.max_horizontal(
            pl.col("adj_high") - pl.col("adj_low"),
            (pl.col("adj_high") - pc).abs(),
            (pl.col("adj_low") - pc).abs(),
        )
    )
    p = (
        panel.with_columns(log_ret("ret").alias("_la"), tr.alias("_tr"))
        .with_columns(
            roll_std("_la", 21).alias("_vol21"),
            pl.when(span_ok(15))
            .then(
                pl.col("_tr").ewm_mean(alpha=1 / 14, adjust=False, min_samples=14).over("sid")
                / pl.col("adj_close")
            )
            .alias("_atr_pct"),
        )
        .select("date", "sid", "symbol", "_vol21", "_atr_pct")
    )
    st = _session_stats(ctx)
    j = p.join(st, on=["symbol", "date"], how="left")
    out = j.select(
        "date",
        "sid",
        "last_hour_ret",
        "first_hour_ret",
        "first5_ret",
        "vwap_dev",
        "intraday_rv",
        (pl.col("intraday_rv") / pl.col("_vol21")).alias("rv_ratio"),
        "vol_last_hour_share",
        "vol_first5_share",
        (pl.col("_c40") / pl.col("_atr_pct")).alias("open_drive"),
    )
    cache["df"] = out
    return out


_SPECS = (
    ("last_hour_ret", 1, "How the stock moved in the last trading hour (log return, 14:30 to close)."),
    ("first_hour_ret", 1, "How the stock moved in the first trading hour (log return, 09:15 to 10:15)."),
    ("first5_ret", 1, "How the stock moved in the first five minutes after the open (log return)."),
    ("vwap_dev", 1, "Where the close sits relative to the day's volume-weighted average price."),
    (
        "intraday_rv",
        1,
        "Realised volatility inside the day from 5-minute returns (square root of summed squares).",
    ),
    ("rv_ratio", 21, "Today's intraday realised volatility relative to the stock's usual 21-day volatility."),
    ("vol_last_hour_share", 1, "Share of the day's volume traded in the last hour."),
    ("vol_first5_share", 1, "Share of the day's volume traded in the first five minutes."),
    ("open_drive", 14, "Move in the first 30 minutes divided by the stock's average daily range (ATR)."),
)


def _make(name: str):
    def fn(ctx: FeatureContext) -> pl.LazyFrame:
        return _intraday(ctx).select("date", "sid", name).lazy()

    fn.__doc__ = f"{name} from the shared intraday frame (null where the stock has no bars that day)."
    return fn


for _n, _w, _l in _SPECS:
    feature(_n, group="intraday", kind="xs", window=_w, label=_l)(_make(_n))
