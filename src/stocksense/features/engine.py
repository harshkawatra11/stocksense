"""
Feature engine — Phase 0 scope.

Implements the Price, Candlestick, Volume, and Market-Context categories
from docs/03-feature-engineering.md. F&O and News/Events are omitted in
Phase 0 (yfinance carries neither, and Upstox credentials are not yet
available in this environment) — this is a scope reduction, not a design
change; both slot in later without touching this contract.

Hard invariant, restated from docs/03-feature-engineering.md: every
feature at row (symbol, date) may use only information dated <= date.
Windows are expressed in bars (trailing `.rolling(n)`), never centered,
never using `.shift(-k)` for anything but label construction (which lives
in stocksense.labels, not here). The leakage test suite in tests/ exists
specifically to catch violations of this rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# columns: symbol, date, open, high, low, close, adj_close, volume
FeatureFrame = pd.DataFrame


def _per_symbol_features(g: pd.DataFrame) -> pd.DataFrame:
    """All-backward-looking features computed within one symbol's own
    history. `g` must already be sorted by date ascending."""
    px = g["adj_close"]
    out = pd.DataFrame(index=g.index)

    # ---- Price / momentum ----
    for w in (1, 2, 3, 5, 10, 20, 60):
        out[f"ret_{w}b"] = px.pct_change(w)

    out["accel_5_20"] = out["ret_5b"] - out["ret_20b"]

    for w in (5, 10, 20, 50, 200):
        ma = px.rolling(w, min_periods=max(3, w // 4)).mean()
        out[f"dist_sma{w}"] = (px / ma) - 1.0

    ema20 = px.ewm(span=20, min_periods=5).mean()
    out["dist_ema20"] = (px / ema20) - 1.0

    roll_high_20 = g["high"].rolling(20, min_periods=5).max()
    roll_low_20 = g["low"].rolling(20, min_periods=5).min()
    out["dist_from_20d_high"] = (px / roll_high_20) - 1.0
    out["dist_from_20d_low"] = (px / roll_low_20) - 1.0
    out["range_pos_20"] = (px - roll_low_20) / (roll_high_20 - roll_low_20 + 1e-9)

    roll_high_252 = g["high"].rolling(252, min_periods=40).max()
    roll_low_252 = g["low"].rolling(252, min_periods=40).min()
    out["dist_from_252d_high"] = (px / roll_high_252) - 1.0
    out["dist_from_252d_low"] = (px / roll_low_252) - 1.0

    # ---- Candlestick / volatility ----
    body = (g["close"] - g["open"]).abs()
    rng = (g["high"] - g["low"]).replace(0, np.nan)
    out["body_pct_of_range"] = body / rng
    upper_wick = g["high"] - g[["open", "close"]].max(axis=1)
    lower_wick = g[["open", "close"]].min(axis=1) - g["low"]
    out["upper_wick_pct"] = upper_wick / rng
    out["lower_wick_pct"] = lower_wick / rng
    out["candle_dir"] = np.sign(g["close"] - g["open"])
    out["consec_up"] = (out["candle_dir"] > 0).astype(int).groupby(
        (out["candle_dir"] <= 0).cumsum()
    ).cumsum()

    true_range = pd.concat(
        [
            g["high"] - g["low"],
            (g["high"] - g["close"].shift(1)).abs(),
            (g["low"] - g["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.rolling(14, min_periods=5).mean()
    out["atr14_pct"] = atr14 / px
    realized_vol_20 = px.pct_change().rolling(20, min_periods=10).std()
    realized_vol_60 = px.pct_change().rolling(60, min_periods=20).std()
    out["realized_vol_20"] = realized_vol_20
    out["vol_expansion"] = realized_vol_20 / (realized_vol_60 + 1e-9)

    # ---- Volume ----
    vol_ma20 = g["volume"].rolling(20, min_periods=5).mean()
    out["rel_volume_20"] = g["volume"] / (vol_ma20 + 1e-9)
    out["volume_accel"] = g["volume"].pct_change(5)
    turnover = g["volume"] * px
    out["turnover_ma20"] = turnover.rolling(20, min_periods=5).mean()

    # up-day vs down-day volume balance over a trailing window — always
    # defined (unlike a per-row up/down split, which is NaN every other
    # row by construction and would poison the "all features present"
    # training mask).
    up_day = (out["candle_dir"] > 0).astype(float)
    up_vol = (g["volume"] * up_day).rolling(20, min_periods=5).sum()
    down_vol = (g["volume"] * (1 - up_day)).rolling(20, min_periods=5).sum()
    out["up_down_vol_ratio_20"] = up_vol / (down_vol + 1e-9)

    return out


def build_features(candles: pd.DataFrame) -> FeatureFrame:
    """Build the full feature matrix from a raw candles DataFrame
    (symbol, date, open, high, low, close, adj_close, volume, source).

    Returns one row per (symbol, date) with all feature columns plus
    the join keys, sorted by (symbol, date).
    """
    df = candles.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    per_symbol = df.groupby("symbol", group_keys=False).apply(
        lambda g: _per_symbol_features(g.sort_values("date"))
    )
    feats = pd.concat([df[["symbol", "date"]], per_symbol], axis=1)

    # ---- Market context: cross-sectional, computed per date across the
    # universe using only that date's already-computed per-symbol returns.
    # This is still point-in-time correct: on date D we only use data up
    # to and including D from every symbol, never a future cross-section. ----
    mkt_ret = feats.groupby("date")["ret_1b"].transform("mean")
    feats["mkt_ret_1b"] = mkt_ret
    feats["rel_strength_1b"] = feats["ret_1b"] - feats["mkt_ret_1b"]

    mkt_ret_5 = feats.groupby("date")["ret_5b"].transform("mean")
    feats["rel_strength_5b"] = feats["ret_5b"] - mkt_ret_5

    feats["xsec_rank_ret_5b"] = feats.groupby("date")["ret_5b"].rank(pct=True)
    feats["xsec_rank_rel_vol"] = feats.groupby("date")["rel_volume_20"].rank(pct=True)

    return feats.sort_values(["symbol", "date"]).reset_index(drop=True)


def feature_columns(feats: FeatureFrame) -> list[str]:
    """Names of the actual feature columns (excludes join keys)."""
    exclude = {"symbol", "date"}
    return [c for c in feats.columns if c not in exclude]
