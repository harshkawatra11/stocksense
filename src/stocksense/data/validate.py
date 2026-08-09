"""
Data validation — the checks from docs/02-data-layer.md's validation
table, implemented for what Phase 0 can actually check with a single
source (yfinance). `flag_adjustment_anomalies` exists because Phase 0's
own stress testing (research/phase0_stress.py) found a real instance:
yfinance's adj_close for a handful of thin, early-2000s-listed symbols
jumps discontinuously (observed: an 8.6x day-over-day adjustment-factor
change for ADANIENT on 2003-09-04) while the raw close barely moves —
an adjustment-factor bug, not a stock split or a real price move. Since
features and labels are computed on adj_close (docs/03-feature-
engineering.md), an unflagged instance of this silently fabricates
extreme "returns" that contaminate both training and evaluation.

This is exactly the "Corporate action application" and "Price
continuity" checks named in docs/02-data-layer.md's validation table,
scoped to what a single-source pipeline can detect: a discontinuity in
the *adjustment factor itself* (adj_close / close) is a strong, source-
agnostic signal of a bad adjustment, independent of whether we have a
second source to cross-check against yet.
"""

from __future__ import annotations

import pandas as pd


def flag_adjustment_anomalies(candles: pd.DataFrame, jump_threshold: float = 1.5) -> pd.DataFrame:
    """Return the subset of rows where the day-over-day change in
    (adj_close / close) exceeds `jump_threshold` in either direction.
    These rows mark points where the adjustment series itself is
    discontinuous — not necessarily bad raw prices, but a broken
    adjustment factor from that point forward for the affected symbol.
    """
    df = candles.sort_values(["symbol", "date"]).copy()
    df["adj_factor"] = df["adj_close"] / df["close"].replace(0, pd.NA)
    df["factor_ratio"] = df.groupby("symbol")["adj_factor"].transform(lambda x: x / x.shift(1))
    anomalous = df[
        (df["factor_ratio"] > jump_threshold) | (df["factor_ratio"] < 1 / jump_threshold)
    ].dropna(subset=["factor_ratio"])
    return anomalous[["symbol", "date", "close", "adj_close", "factor_ratio"]]


def quarantine_symbols(candles: pd.DataFrame, jump_threshold: float = 1.5) -> tuple[pd.DataFrame, list[str]]:
    """Drop every row for any symbol that has at least one adjustment
    anomaly anywhere in its history. Coarse (quarantines the whole
    symbol, not just the affected date range) but correct: an adjustment
    bug at one date means the cumulative adjustment is suspect for the
    entire series before that point too. Returns (clean_df, quarantined_symbols).
    """
    anomalies = flag_adjustment_anomalies(candles, jump_threshold)
    bad_symbols = sorted(anomalies["symbol"].unique().tolist())
    clean = candles[~candles["symbol"].isin(bad_symbols)].copy()
    return clean, bad_symbols
