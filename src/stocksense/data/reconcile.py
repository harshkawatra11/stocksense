"""
Cross-source reconciliation (docs/17-data-spine.md, closes audit finding
HIGH-6: single data source, no provenance tracking).

bhavcopy (data/nse_archive.py) gives RAW exchange-printed prices, sourced
directly from NSE — independent of yfinance's own data pipeline.
yfinance's `close` is also meant to be a raw print, and `adj_close` is
corporate-action-adjusted. Comparing bhavcopy's raw close against
yfinance's raw close is a genuine cross-source check (two independent
pipelines should agree on what actually printed); comparing bhavcopy's
raw close against yfinance's *adjusted* close reproduces
data/validate.py's adjustment-anomaly detector using a completely
different raw-price source than the one that detector already checks
against itself — an independent second detector for the exact
ADANIENT-class bug, not a restatement of the first one.
"""

from __future__ import annotations

import pandas as pd


def reconcile_close_prices(bhavcopy_eq: pd.DataFrame, candles: pd.DataFrame, tolerance_pct: float = 0.5) -> pd.DataFrame:
    """Per (symbol, date) where BOTH sources have a row: compares
    bhavcopy's raw close to yfinance's raw close. A divergence beyond
    `tolerance_pct` flags a genuine data-quality question -- which
    source is wrong, or are they using different corporate-action
    treatment -- that a single-source pipeline has no way to even
    notice, let alone answer."""
    bhav = bhavcopy_eq[bhavcopy_eq["series"] == "EQ"][["symbol", "date", "close"]].rename(columns={"close": "bhavcopy_close"})
    yf = candles[["symbol", "date", "close"]].rename(columns={"close": "yfinance_close"})

    merged = bhav.merge(yf, on=["symbol", "date"], how="inner")
    merged["pct_diff"] = (merged["yfinance_close"] - merged["bhavcopy_close"]).abs() / merged["bhavcopy_close"].replace(0, pd.NA) * 100
    merged["flagged"] = merged["pct_diff"] > tolerance_pct
    return merged.sort_values(["symbol", "date"])


def flag_cross_source_adjustment_anomalies(bhavcopy_eq: pd.DataFrame, candles: pd.DataFrame, jump_threshold: float = 1.5) -> pd.DataFrame:
    """The independent second detector: computes an implied adjustment
    factor as (yfinance.adj_close / bhavcopy.close) -- using bhavcopy's
    raw price as the reference instead of yfinance's own `close` column
    (which is what data/validate.flag_adjustment_anomalies uses). A
    day-over-day discontinuity in THIS ratio is a bug signal that does
    not depend on yfinance's raw-close column being reliable at all,
    which matters if yfinance's own close and adj_close both drift
    together during a bad ingestion."""
    bhav = bhavcopy_eq[bhavcopy_eq["series"] == "EQ"][["symbol", "date", "close"]].rename(columns={"close": "bhavcopy_close"})
    yf = candles[["symbol", "date", "adj_close"]]

    merged = bhav.merge(yf, on=["symbol", "date"], how="inner").sort_values(["symbol", "date"])
    merged["implied_adj_factor"] = merged["adj_close"] / merged["bhavcopy_close"].replace(0, pd.NA)
    merged["factor_ratio"] = merged.groupby("symbol")["implied_adj_factor"].transform(lambda x: x / x.shift(1))

    anomalous = merged[
        (merged["factor_ratio"] > jump_threshold) | (merged["factor_ratio"] < 1 / jump_threshold)
    ].dropna(subset=["factor_ratio"])
    return anomalous[["symbol", "date", "bhavcopy_close", "adj_close", "implied_adj_factor", "factor_ratio"]]


def provenance_report(bhavcopy_eq: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, date): which source(s) had a row. This is the
    per-field provenance docs/02-data-layer.md promised and this project
    never built -- coarse (source presence, not per-field), but it
    answers the question that matters: for any given number, was it
    corroborated by an independent source or trusted from just one."""
    bhav_dates = bhavcopy_eq[bhavcopy_eq["series"] == "EQ"][["symbol", "date"]].drop_duplicates()
    bhav_dates["has_bhavcopy"] = True
    yf_dates = candles[["symbol", "date"]].drop_duplicates()
    yf_dates["has_yfinance"] = True

    merged = bhav_dates.merge(yf_dates, on=["symbol", "date"], how="outer")
    merged["has_bhavcopy"] = merged["has_bhavcopy"].fillna(False)
    merged["has_yfinance"] = merged["has_yfinance"].fillna(False)
    merged["corroborated"] = merged["has_bhavcopy"] & merged["has_yfinance"]
    return merged.sort_values(["symbol", "date"])
