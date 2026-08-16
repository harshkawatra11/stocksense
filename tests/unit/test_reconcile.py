"""Cross-source reconciliation tests. The load-bearing property, tested
explicitly: flag_cross_source_adjustment_anomalies must independently
re-detect an ADANIENT-class adjustment-factor discontinuity using
bhavcopy's raw close as the reference, WITHOUT depending on yfinance's
own close column being reliable -- a synthetic reproduction of the real
bug pattern found earlier in this project's history."""

from __future__ import annotations

import pandas as pd

from stocksense.data.reconcile import (
    flag_cross_source_adjustment_anomalies,
    provenance_report,
    reconcile_close_prices,
)


def test_reconcile_close_prices_flags_genuine_divergence() -> None:
    bhav = pd.DataFrame({"symbol": ["AAA"], "series": ["EQ"], "date": ["2024-01-01"], "close": [100.0]})
    candles = pd.DataFrame({"symbol": ["AAA"], "date": ["2024-01-01"], "close": [105.0]})  # 5% off

    result = reconcile_close_prices(bhav, candles, tolerance_pct=0.5)
    assert result.iloc[0]["flagged"] == True
    assert abs(result.iloc[0]["pct_diff"] - 5.0) < 1e-9


def test_reconcile_close_prices_no_flag_within_tolerance() -> None:
    bhav = pd.DataFrame({"symbol": ["AAA"], "series": ["EQ"], "date": ["2024-01-01"], "close": [100.0]})
    candles = pd.DataFrame({"symbol": ["AAA"], "date": ["2024-01-01"], "close": [100.1]})  # 0.1% off

    result = reconcile_close_prices(bhav, candles, tolerance_pct=0.5)
    assert result.iloc[0]["flagged"] == False


def test_reconcile_close_prices_only_matches_eq_series() -> None:
    bhav = pd.DataFrame({"symbol": ["AAA"], "series": ["GB"], "date": ["2024-01-01"], "close": [100.0]})  # government bond, not equity
    candles = pd.DataFrame({"symbol": ["AAA"], "date": ["2024-01-01"], "close": [200.0]})

    result = reconcile_close_prices(bhav, candles, tolerance_pct=0.5)
    assert result.empty  # GB series excluded, so no spurious cross-match


def test_cross_source_detector_reproduces_adanient_class_bug() -> None:
    """Synthetic reproduction of the real 2003-09-04 ADANIENT bug: an
    8.6x day-over-day jump in the adjustment factor while the raw price
    barely moves. Here the raw reference is bhavcopy's close (NOT
    yfinance's own close, which the original detector in data/validate.py
    uses) -- proving this is a genuinely independent detection path."""
    dates = pd.bdate_range("2003-09-01", periods=5)
    bhav = pd.DataFrame({
        "symbol": ["ADANIENT"] * 5, "series": ["EQ"] * 5, "date": dates,
        "close": [1.65, 1.64, 1.645, 1.607, 1.61],  # raw price barely moves throughout
    })
    # yfinance's adj_close jumps 8.6x on the 4th day (index 3) while
    # bhavcopy's raw close (above) shows only an ordinary small decline
    candles = pd.DataFrame({
        "symbol": ["ADANIENT"] * 5, "date": dates,
        "adj_close": [0.069, 0.0695, 0.0691, 0.5833, 0.585],
    })

    anomalies = flag_cross_source_adjustment_anomalies(bhav, candles, jump_threshold=1.5)
    assert len(anomalies) >= 1
    assert (anomalies["date"] == dates[3]).any()  # the exact date the real bug occurred


def test_cross_source_detector_clean_data_flags_nothing() -> None:
    dates = pd.bdate_range("2020-01-01", periods=10)
    bhav = pd.DataFrame({
        "symbol": ["CLEANCO"] * 10, "series": ["EQ"] * 10, "date": dates,
        "close": [100.0 + i * 0.5 for i in range(10)],
    })
    # adj_close tracks close with a stable, constant adjustment factor -- no corporate action
    candles = pd.DataFrame({
        "symbol": ["CLEANCO"] * 10, "date": dates,
        "adj_close": [(100.0 + i * 0.5) * 0.98 for i in range(10)],
    })
    anomalies = flag_cross_source_adjustment_anomalies(bhav, candles, jump_threshold=1.5)
    assert len(anomalies) == 0


def test_provenance_report_flags_corroborated_vs_single_source() -> None:
    bhav = pd.DataFrame({"symbol": ["AAA", "BBB"], "series": ["EQ", "EQ"], "date": ["2024-01-01", "2024-01-01"], "close": [100.0, 50.0]})
    candles = pd.DataFrame({"symbol": ["AAA"], "date": ["2024-01-01"], "close": [100.0]})  # BBB only in bhavcopy

    report = provenance_report(bhav, candles)
    aaa = report[report.symbol == "AAA"].iloc[0]
    bbb = report[report.symbol == "BBB"].iloc[0]
    assert aaa["corroborated"] == True
    assert bbb["corroborated"] == False
    assert bbb["has_bhavcopy"] == True
    assert bbb["has_yfinance"] == False
