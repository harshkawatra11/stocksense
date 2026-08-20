"""Upstox intraday fetcher tests (Phase E1). Network calls are mocked --
the endpoint itself (no-auth-required, 31-day max range, 2022-01 history
floor) was verified against the live API during planning. What's under
test here: instrument-symbol resolution (including the unresolved case,
which must be recorded not dropped), month-window chunking staying under
the verified 31-day cap, candle-to-frame shaping, and resumable caching --
the properties that must hold regardless of what Upstox happens to
return on a given call."""

from __future__ import annotations

import gzip
import json
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from stocksense.data.upstox_intraday import (
    EARLIEST_1MIN_DATE,
    MAX_DAYS_PER_REQUEST,
    FetchError,
    _candles_to_frame,
    _month_windows,
    fetch_instrument_master,
    fetch_range,
    resolve_symbol_map,
)


def _mock_response(status_code=200, json_body=None, content=b""):
    m = MagicMock()
    m.status_code = status_code
    m.content = content
    m.text = json.dumps(json_body) if json_body is not None else ""
    m.json.return_value = json_body or {}
    return m


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    import stocksense.data.upstox_intraday as mod

    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path / "upstox_intraday")
    monkeypatch.setattr(mod, "_POLITE_DELAY_S", 0.0)
    return tmp_path


def _sample_instrument(symbol: str, isin: str) -> dict:
    return {
        "segment": "NSE_EQ", "instrument_type": "EQ", "isin": isin,
        "instrument_key": f"NSE_EQ|{isin}", "trading_symbol": symbol,
    }


# ---- month windowing ----

def test_month_windows_never_exceed_the_verified_31_day_cap() -> None:
    windows = list(_month_windows(date(2022, 1, 1), date(2022, 3, 15)))
    for start, end in windows:
        assert (end - start).days <= MAX_DAYS_PER_REQUEST


def test_month_windows_cover_the_full_range_with_no_gaps() -> None:
    start, end = date(2023, 11, 10), date(2024, 2, 5)
    windows = list(_month_windows(start, end))
    assert windows[0][0] == start
    assert windows[-1][1] == end
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert next_start == prev_end + pd.Timedelta(days=1)


def test_month_windows_single_day_range() -> None:
    windows = list(_month_windows(date(2024, 5, 1), date(2024, 5, 1)))
    assert windows == [(date(2024, 5, 1), date(2024, 5, 1))]


# ---- candle -> frame shaping ----

def test_candles_to_frame_shapes_upstox_response_correctly() -> None:
    candles = [
        ["2026-08-13T09:15:00+05:30", 1327.8, 1327.8, 1310.4, 1313.4, 250665, 0],
        ["2026-08-13T09:16:00+05:30", 1313.4, 1315.0, 1313.0, 1314.0, 9040, 0],
    ]
    df = _candles_to_frame("RELIANCE", candles)
    assert list(df.columns) == ["symbol", "ts", "interval", "open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert (df["symbol"] == "RELIANCE").all()
    assert (df["interval"] == "1minute").all()
    # tz-naive IST wall-clock, not converted to UTC
    assert df["ts"].dt.tz is None
    assert df.iloc[0]["ts"] == pd.Timestamp("2026-08-13 09:15:00")


def test_candles_to_frame_empty_list_returns_empty_typed_frame() -> None:
    df = _candles_to_frame("RELIANCE", [])
    assert df.empty
    assert list(df.columns) == ["symbol", "ts", "interval", "open", "high", "low", "close", "volume"]


# ---- instrument master + symbol resolution ----

@patch("stocksense.data.upstox_intraday.requests.get")
def test_fetch_instrument_master_filters_to_nse_eq(mock_get, cache_dir) -> None:
    raw = [
        _sample_instrument("RELIANCE", "INE002A01018"),
        {"segment": "NSE_FO", "instrument_type": "FUT", "trading_symbol": "RELIANCE26AUGFUT"},
        {"segment": "NSE_EQ", "instrument_type": "ETF", "trading_symbol": "NIFTYBEES"},
    ]
    mock_get.return_value = _mock_response(200, content=gzip.compress(json.dumps(raw).encode()))
    master = fetch_instrument_master()
    assert len(master) == 1
    assert master[0]["trading_symbol"] == "RELIANCE"


@patch("stocksense.data.upstox_intraday.requests.get")
def test_fetch_instrument_master_is_cached_to_disk(mock_get, cache_dir) -> None:
    raw = [_sample_instrument("TCS", "INE467B01029")]
    mock_get.return_value = _mock_response(200, content=gzip.compress(json.dumps(raw).encode()))
    fetch_instrument_master()
    fetch_instrument_master()
    assert mock_get.call_count == 1  # second call served from disk cache


@patch("stocksense.data.upstox_intraday.requests.get")
def test_resolve_symbol_map_records_unresolved_symbols_not_dropping_them(mock_get, cache_dir) -> None:
    raw = [_sample_instrument("RELIANCE", "INE002A01018")]
    mock_get.return_value = _mock_response(200, content=gzip.compress(json.dumps(raw).encode()))

    out = resolve_symbol_map(["RELIANCE", "SOMEDELISTEDCO"])
    assert len(out) == 2
    reliance = out[out["symbol"] == "RELIANCE"].iloc[0]
    assert reliance["resolved"] and reliance["instrument_key"] == "NSE_EQ|INE002A01018"
    missing = out[out["symbol"] == "SOMEDELISTEDCO"].iloc[0]
    assert not missing["resolved"]
    assert pd.isna(missing["instrument_key"])  # pandas coerces None -> NaN in a mixed object column


# ---- fetch_range: resumability, caching, error handling ----

@patch("stocksense.data.upstox_intraday.requests.get")
def test_fetch_range_caches_each_window_to_disk_and_skips_on_rerun(mock_get, cache_dir) -> None:
    candles = [["2022-01-03T09:15:00+05:30", 100.0, 101.0, 99.0, 100.5, 1000, 0]]
    mock_get.return_value = _mock_response(200, {"status": "success", "data": {"candles": candles}})

    instrument_map = pd.DataFrame([
        {"symbol": "RELIANCE", "isin": "INE002A01018", "instrument_key": "NSE_EQ|INE002A01018", "resolved": True},
    ])

    results = list(fetch_range(instrument_map, date(2022, 1, 1), date(2022, 1, 5)))
    assert len(results) == 1
    symbol, wstart, wend, df = results[0]
    assert symbol == "RELIANCE" and not df.empty
    n_calls_first_run = mock_get.call_count

    # re-run over the identical range: fully served from disk, zero new network calls
    mock_get.reset_mock()
    results2 = list(fetch_range(instrument_map, date(2022, 1, 1), date(2022, 1, 5)))
    assert len(results2) == 1
    assert mock_get.call_count == 0
    assert n_calls_first_run == 1


@patch("stocksense.data.upstox_intraday.requests.get")
def test_fetch_range_yields_none_on_fetch_error_rather_than_raising(mock_get, cache_dir) -> None:
    mock_get.return_value = _mock_response(500, content=b"server error")
    instrument_map = pd.DataFrame([
        {"symbol": "BADCO", "isin": "INE000000001", "instrument_key": "NSE_EQ|INE000000001", "resolved": True},
    ])
    results = list(fetch_range(instrument_map, date(2022, 1, 1), date(2022, 1, 5)))
    assert len(results) == 1
    symbol, _, _, df = results[0]
    assert symbol == "BADCO"
    assert df is None  # caller must know to retry this window, not silently treat it as "no data"


@patch("stocksense.data.upstox_intraday.requests.get")
def test_fetch_range_skips_unresolved_symbols(mock_get, cache_dir) -> None:
    mock_get.return_value = _mock_response(200, {"status": "success", "data": {"candles": []}})
    instrument_map = pd.DataFrame([
        {"symbol": "RESOLVED", "isin": "INE1", "instrument_key": "NSE_EQ|INE1", "resolved": True},
        {"symbol": "UNRESOLVED", "isin": None, "instrument_key": None, "resolved": False},
    ])
    results = list(fetch_range(instrument_map, date(2022, 1, 1), date(2022, 1, 5)))
    assert {r[0] for r in results} == {"RESOLVED"}


def test_fetch_range_clamps_start_to_earliest_1min_date(cache_dir) -> None:
    with patch("stocksense.data.upstox_intraday.requests.get") as mock_get:
        mock_get.return_value = _mock_response(200, {"status": "success", "data": {"candles": []}})
        instrument_map = pd.DataFrame([
            {"symbol": "X", "isin": "INE1", "instrument_key": "NSE_EQ|INE1", "resolved": True},
        ])
        results = list(fetch_range(instrument_map, date(2015, 1, 1), date(2022, 1, 15)))
        first_window_start = results[0][1]
        assert first_window_start >= EARLIEST_1MIN_DATE
