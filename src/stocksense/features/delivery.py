"""
Delivery-percentage features (docs/17-data-spine.md), built from
`bhavcopy_delivery` (data/nse_archive.fetch_delivery, ~2021+). Unlike
yfinance, this is genuinely India-specific information: the fraction of
traded volume actually taken to delivery vs. squared off intraday is a
classic conviction signal with no US-market equivalent baked into this
codebase's existing feature set (docs/03).

Per the plan: admitted only if it beats an ablation against the
existing Baseline 8 (price/volume-only) model, via the same
pre-registered gate machinery already governing the monthly track. This
module only computes the features; the ablation decision happens in
research, not here.

Same point-in-time discipline as features/engine.py: every value at
(symbol, date) uses only trailing history through that date.
"""

from __future__ import annotations

import pandas as pd


def build_delivery_features(bhavcopy_delivery: pd.DataFrame) -> pd.DataFrame:
    """Input: bhavcopy_delivery's schema (symbol, series, date,
    delivery_qty, delivery_pct). Output: one row per (symbol, date) with
    trailing-window delivery-conviction features."""
    df = bhavcopy_delivery.sort_values(["symbol", "date"]).copy()

    def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=g.index)
        pct = g["delivery_pct"]

        out["deliv_pct"] = pct
        out["deliv_pct_ma_20"] = pct.rolling(20, min_periods=5).mean()
        deliv_std_20 = pct.rolling(20, min_periods=5).std()
        out["deliv_pct_zscore_20"] = (pct - out["deliv_pct_ma_20"]) / deliv_std_20.replace(0, pd.NA)
        out["deliv_pct_trend_5"] = pct.rolling(5, min_periods=2).mean() - pct.rolling(20, min_periods=5).mean()
        out["deliv_qty_ma_20"] = g["delivery_qty"].rolling(20, min_periods=5).mean()
        return out

    feats = df.groupby("symbol", group_keys=False).apply(_per_symbol)
    return pd.concat([df[["symbol", "date"]], feats], axis=1)


def delivery_weighted_return(bhavcopy_delivery: pd.DataFrame, bhavcopy_eq: pd.DataFrame) -> pd.DataFrame:
    """Divergence between price move and delivery conviction: a price
    rally on LOW delivery % is churn (likely to reverse); a price rally
    on HIGH delivery % is conviction accumulation. Requires joining
    delivery data to price data (bhavcopy_eq) since delivery_pct alone
    has no price-direction information."""
    prices = bhavcopy_eq[["symbol", "date", "close", "prev_close"]].copy()
    prices["ret_1b"] = (prices["close"] - prices["prev_close"]) / prices["prev_close"]

    merged = bhavcopy_delivery.merge(prices, on=["symbol", "date"], how="inner")
    merged["deliv_weighted_ret"] = merged["ret_1b"] * (merged["delivery_pct"] / 100.0)
    return merged[["symbol", "date", "ret_1b", "delivery_pct", "deliv_weighted_ret"]]
