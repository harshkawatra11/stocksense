"""Delivery feature tests, on hand-constructed scenarios where the
expected value is known -- e.g. a rising delivery-pct trend must show a
positive deliv_pct_trend_5, not just "some number"."""

from __future__ import annotations

import pandas as pd

from stocksense.features.delivery import build_delivery_features, delivery_weighted_return


def _delivery_rows(symbol, pcts, start="2023-01-02"):
    dates = pd.bdate_range(start, periods=len(pcts))
    return pd.DataFrame({
        "symbol": symbol, "series": "EQ", "date": dates,
        "delivery_qty": [p * 1000 for p in pcts], "delivery_pct": pcts,
    })


def test_deliv_pct_passthrough() -> None:
    df = _delivery_rows("AAA", [50.0] * 25)
    feats = build_delivery_features(df)
    assert (feats["deliv_pct"] == 50.0).all()


def test_rising_trend_produces_positive_trend_feature() -> None:
    # delivery % ramps steadily from 20 to 80 over 25 days
    pcts = [20.0 + i * 2.5 for i in range(25)]
    df = _delivery_rows("AAA", pcts)
    feats = build_delivery_features(df)
    last_trend = feats["deliv_pct_trend_5"].iloc[-1]
    assert last_trend > 0  # recent 5-day average above the 20-day average, since it's been rising


def test_falling_trend_produces_negative_trend_feature() -> None:
    pcts = [80.0 - i * 2.5 for i in range(25)]
    df = _delivery_rows("AAA", pcts)
    feats = build_delivery_features(df)
    last_trend = feats["deliv_pct_trend_5"].iloc[-1]
    assert last_trend < 0


def test_zscore_flags_unusual_spike() -> None:
    pcts = [50.0] * 24 + [95.0]  # a sharp spike after a stable baseline
    df = _delivery_rows("AAA", pcts)
    feats = build_delivery_features(df)
    assert feats["deliv_pct_zscore_20"].iloc[-1] > 2.0  # a genuine outlier vs. baseline


def test_features_computed_independently_per_symbol() -> None:
    df = pd.concat([_delivery_rows("AAA", [10.0] * 25), _delivery_rows("BBB", [90.0] * 25)])
    feats = build_delivery_features(df)
    assert feats[feats.symbol == "AAA"]["deliv_pct_ma_20"].iloc[-1] < feats[feats.symbol == "BBB"]["deliv_pct_ma_20"].iloc[-1]


def test_delivery_weighted_return_scales_by_delivery_conviction() -> None:
    deliv = pd.DataFrame({
        "symbol": ["AAA", "BBB"], "series": ["EQ", "EQ"], "date": ["2023-01-05", "2023-01-05"],
        "delivery_qty": [1000, 1000], "delivery_pct": [90.0, 10.0],  # AAA high conviction, BBB low
    })
    eq = pd.DataFrame({
        "symbol": ["AAA", "BBB"], "date": ["2023-01-05", "2023-01-05"],
        "close": [110.0, 110.0], "prev_close": [100.0, 100.0],  # both up 10%
    })
    result = delivery_weighted_return(deliv, eq)
    aaa_weighted = result[result.symbol == "AAA"]["deliv_weighted_ret"].iloc[0]
    bbb_weighted = result[result.symbol == "BBB"]["deliv_weighted_ret"].iloc[0]
    assert aaa_weighted > bbb_weighted  # same price move, but AAA's is backed by much higher delivery conviction
