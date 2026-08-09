"""
Regression test for a real bug found during Phase 0 stress testing: a
yfinance adjustment-factor discontinuity (ADANIENT, 2003-09-04, adj_close
jumped 8.6x day-over-day while close barely moved) that fabricated a
~750% "return" and inflated one walk-forward fold's alpha. This test
encodes that exact case so it can never silently regress.
"""

from __future__ import annotations

import pandas as pd

from stocksense.data.validate import flag_adjustment_anomalies, quarantine_symbols


def _candles_with_adjustment_bug() -> pd.DataFrame:
    dates = pd.bdate_range("2003-08-25", periods=6)
    # CLEAN: close and adj_close move together (no split, no bug)
    clean_rows = [
        {"symbol": "CLEAN", "date": d, "close": 100 + i, "adj_close": 100 + i, "volume": 1000, "source": "yfinance"}
        for i, d in enumerate(dates)
    ]
    # BUGGY: adj_close jumps 8.6x on day 3 while close moves normally —
    # the exact real-world shape found for ADANIENT.
    buggy_rows = [
        {"symbol": "BUGGY", "date": dates[0], "close": 1.60, "adj_close": 0.069, "volume": 1000, "source": "yfinance"},
        {"symbol": "BUGGY", "date": dates[1], "close": 1.62, "adj_close": 0.070, "volume": 1000, "source": "yfinance"},
        {"symbol": "BUGGY", "date": dates[2], "close": 1.61, "adj_close": 0.070, "volume": 1000, "source": "yfinance"},
        {"symbol": "BUGGY", "date": dates[3], "close": 1.61, "adj_close": 0.583, "volume": 1000, "source": "yfinance"},  # the bug
        {"symbol": "BUGGY", "date": dates[4], "close": 1.60, "adj_close": 0.581, "volume": 1000, "source": "yfinance"},
        {"symbol": "BUGGY", "date": dates[5], "close": 1.59, "adj_close": 0.578, "volume": 1000, "source": "yfinance"},
    ]
    df = pd.DataFrame(clean_rows + buggy_rows)
    for c in ("open", "high", "low"):
        df[c] = df["close"]
    return df


def test_flags_the_real_adanient_shaped_anomaly() -> None:
    candles = _candles_with_adjustment_bug()
    anomalies = flag_adjustment_anomalies(candles)
    assert set(anomalies["symbol"]) == {"BUGGY"}
    assert len(anomalies) == 1


def test_clean_series_produces_no_flags() -> None:
    candles = _candles_with_adjustment_bug()
    clean_only = candles[candles["symbol"] == "CLEAN"]
    anomalies = flag_adjustment_anomalies(clean_only)
    assert anomalies.empty


def test_quarantine_removes_whole_symbol_not_just_bad_row() -> None:
    candles = _candles_with_adjustment_bug()
    clean, bad_symbols = quarantine_symbols(candles)
    assert bad_symbols == ["BUGGY"]
    assert set(clean["symbol"]) == {"CLEAN"}
    assert len(clean) == 6  # all CLEAN rows retained, all BUGGY rows dropped
