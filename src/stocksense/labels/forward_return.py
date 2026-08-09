"""
Forward-return labels.

Per docs/03-feature-engineering.md: "labels are the only thing permitted
to look forward, and only during training." This module is the one place
in the codebase where `.shift(-k)` (a forward-looking operation) is
allowed to exist. It must never be imported by anything in
stocksense.features.

The label is *cross-sectional relative* forward return, not raw forward
return — v1's Phase 0 finding (docs of this session) showed positive
beta-removed alpha; training directly against relative return teaches the
model to predict outperformance vs the universe rather than direction of
the market, which is what the alpha signal actually is.
"""

from __future__ import annotations

import pandas as pd


def add_forward_return_labels(
    candles: pd.DataFrame, horizon_bars: int, label_col: str | None = None
) -> pd.DataFrame:
    """Add a forward-return column to a candles-like frame indexed by
    (symbol, date), computed strictly from FUTURE adj_close values.

    Rows where the horizon extends past the end of that symbol's history
    get NaN — never a fabricated value, per the missing-data discipline.
    """
    col = label_col or f"fwd_ret_{horizon_bars}b"
    df = candles.sort_values(["symbol", "date"]).copy()

    def _fwd(g: pd.DataFrame) -> pd.Series:
        px = g["adj_close"]
        return (px.shift(-horizon_bars) / px) - 1.0

    df[col] = df.groupby("symbol", group_keys=False).apply(_fwd)
    return df


def add_relative_forward_return(
    df_with_fwd: pd.DataFrame, horizon_bars: int, label_col: str | None = None
) -> pd.DataFrame:
    """Convert a raw forward return column into cross-sectional relative
    forward return (return minus that date's cross-sectional mean).

    This is the label the alpha model (stocksense.models) actually trains
    against — see module docstring.
    """
    raw_col = label_col or f"fwd_ret_{horizon_bars}b"
    rel_col = f"{raw_col}_rel"
    df = df_with_fwd.copy()
    xsec_mean = df.groupby("date")[raw_col].transform("mean")
    df[rel_col] = df[raw_col] - xsec_mean
    return df
