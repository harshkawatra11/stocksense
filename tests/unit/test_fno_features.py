"""F&O feature tests: OI aggregation, the long-buildup/short-covering
quadrant classification (verified against all four hand-computed
combinations, not just one), and PCR."""

from __future__ import annotations

import pandas as pd

from stocksense.features.fno import build_oi_features, build_put_call_ratio, classify_oi_quadrant, days_to_expiry


def _fo_row(symbol, instrument, option_type, oi, chg_oi, close=100.0, date="2023-06-01", strike=0.0):
    return {
        "symbol": symbol, "instrument": instrument, "expiry_date": "2023-06-29", "strike": strike,
        "option_type": option_type, "date": date, "open": close, "high": close, "low": close,
        "close": close, "open_interest": oi, "chg_in_oi": chg_oi, "era": "legacy",
    }


def test_build_oi_features_aggregates_only_futures_not_options() -> None:
    df = pd.DataFrame([
        _fo_row("AAA", "FUTSTK", "XX", oi=1000, chg_oi=100),
        _fo_row("AAA", "OPTSTK", "CE", oi=500, chg_oi=50, strike=100.0),  # should be excluded from OI aggregation
    ])
    feats = build_oi_features(df)
    assert len(feats) == 1
    assert feats.iloc[0]["total_oi"] == 1000  # only the future's OI, not the option's


def test_oi_pct_change_computed_correctly() -> None:
    df = pd.DataFrame([_fo_row("AAA", "FUTSTK", "XX", oi=1100, chg_oi=100)])  # was 1000, now 1100
    feats = build_oi_features(df)
    assert abs(feats.iloc[0]["oi_pct_change"] - 0.1) < 1e-9  # +100/1000 = 10%


def test_classify_oi_quadrant_all_four_combinations() -> None:
    oi_features = pd.DataFrame({"oi_pct_change": [0.1, -0.1, 0.1, -0.1]}, index=[0, 1, 2, 3])
    returns = pd.Series([0.05, 0.05, -0.05, -0.05], index=[0, 1, 2, 3])

    quadrant = classify_oi_quadrant(oi_features, returns)
    assert quadrant[0] == "long_buildup"      # price up, OI up
    assert quadrant[1] == "short_covering"    # price up, OI down
    assert quadrant[2] == "short_buildup"     # price down, OI up
    assert quadrant[3] == "long_unwinding"    # price down, OI down


def test_put_call_ratio_computed_from_oi() -> None:
    df = pd.DataFrame([
        _fo_row("AAA", "OPTSTK", "CE", oi=1000, chg_oi=0, strike=100.0),
        _fo_row("AAA", "OPTSTK", "PE", oi=1500, chg_oi=0, strike=100.0),
        _fo_row("AAA", "FUTSTK", "XX", oi=5000, chg_oi=0),  # futures must not pollute PCR
    ])
    pcr = build_put_call_ratio(df)
    assert pcr.iloc[0]["put_oi"] == 1500
    assert pcr.iloc[0]["call_oi"] == 1000
    assert abs(pcr.iloc[0]["pcr_oi"] - 1.5) < 1e-9


def test_put_call_ratio_handles_missing_call_side() -> None:
    df = pd.DataFrame([_fo_row("AAA", "OPTSTK", "PE", oi=1000, chg_oi=0, strike=100.0)])  # puts only
    pcr = build_put_call_ratio(df)
    assert pcr.iloc[0]["call_oi"] == 0.0
    assert pd.isna(pcr.iloc[0]["pcr_oi"])  # division by zero -> NA, not inf or a crash


def test_days_to_expiry_computed_correctly() -> None:
    df = pd.DataFrame([{"date": "2023-06-01", "expiry_date": "2023-06-29"}])
    days = days_to_expiry(df)
    assert days.iloc[0] == 28
