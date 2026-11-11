"""Seed features that exercise the framework. T4.3 moves them into their library modules
(`price.py`, `risk.py`, `market.py`); delete the matching function here when it lands there."""

from __future__ import annotations

import polars as pl

from stocksense.features.base import FeatureContext, feature, log_ret


@feature(
    "mkt_ret_1",
    group="market",
    kind="mkt",
    window=1,
    label="How the broad market (equal-weight top 500) moved on the day, log return.",
)
def mkt_ret_1(ctx: FeatureContext) -> pl.LazyFrame:
    """ln(1 + ew500_ret) per date."""
    return ctx.market().select("date", log_ret("ew500_ret").alias("mkt_ret_1"))
