"""Seed features that exercise the framework. T4.3 moves them into their library modules
(`price.py`, `risk.py`, `market.py`); delete the matching function here when it lands there."""

from __future__ import annotations

import polars as pl

from stocksense.features.base import FeatureContext, feature, log_ret, roll_std


@feature(
    "ret_1",
    group="price",
    kind="xs",
    window=1,
    label="Yesterday's total price move (log return, adjusted for splits and bonuses).",
)
def ret_1(ctx: FeatureContext) -> pl.LazyFrame:
    """Adjusted log return of the session, ln(1 + ret)."""
    return ctx.panel().select("date", "sid", log_ret("ret").alias("ret_1"))


@feature(
    "vol_21",
    group="risk",
    kind="xs",
    window=21,
    label="How jumpy the stock has been over the last 21 sessions (std of daily log returns).",
)
def vol_21(ctx: FeatureContext) -> pl.LazyFrame:
    """Sample std (ddof 1) of the adjusted log return over 21 rows."""
    return (
        ctx.panel()
        .with_columns(log_ret("ret").alias("_la"))
        .select("date", "sid", roll_std("_la", 21).alias("vol_21"))
    )
