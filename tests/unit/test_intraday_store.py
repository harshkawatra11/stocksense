"""Store-layer tests for Phase E1's new tables: intraday_bars (upsert
idempotency, interval-scoped reads) and upstox_instrument_map (records
unresolved symbols as a fact, not a gap)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from stocksense.data.store import Store


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def _bar_row(symbol, ts, close, interval="1minute"):
    return {
        "symbol": symbol, "ts": ts, "interval": interval,
        "open": close, "high": close, "low": close, "close": close, "volume": 1000.0,
    }


def test_write_and_read_intraday_bars_roundtrip(tmp_store) -> None:
    rows = [
        _bar_row("RELIANCE", datetime(2026, 1, 5, 9, 15), 100.0),
        _bar_row("RELIANCE", datetime(2026, 1, 5, 9, 16), 100.5),
        _bar_row("TCS", datetime(2026, 1, 5, 9, 15), 3500.0),
    ]
    n = tmp_store.write_intraday_bars(pd.DataFrame(rows))
    assert n == 3

    out = tmp_store.read_intraday_bars()
    assert len(out) == 3
    assert set(out["symbol"]) == {"RELIANCE", "TCS"}


def test_write_intraday_bars_is_idempotent_upsert(tmp_store) -> None:
    ts = datetime(2026, 1, 5, 9, 15)
    tmp_store.write_intraday_bars(pd.DataFrame([_bar_row("RELIANCE", ts, 100.0)]))
    tmp_store.write_intraday_bars(pd.DataFrame([_bar_row("RELIANCE", ts, 105.0)]))  # re-write same (symbol, ts, interval)

    out = tmp_store.read_intraday_bars()
    assert len(out) == 1  # not duplicated
    assert out.iloc[0]["close"] == 105.0  # updated, not ignored


def test_read_intraday_bars_filters_by_symbol_and_interval(tmp_store) -> None:
    ts = datetime(2026, 1, 5, 9, 15)
    tmp_store.write_intraday_bars(pd.DataFrame([
        _bar_row("RELIANCE", ts, 100.0, interval="1minute"),
        _bar_row("RELIANCE", ts, 100.0, interval="5minute"),
        _bar_row("TCS", ts, 3500.0, interval="1minute"),
    ]))
    out = tmp_store.read_intraday_bars(symbols=["RELIANCE"], interval="1minute")
    assert len(out) == 1
    assert out.iloc[0]["symbol"] == "RELIANCE"
    assert out.iloc[0]["interval"] == "1minute"


def test_write_and_read_upstox_instrument_map_records_unresolved(tmp_store) -> None:
    df = pd.DataFrame([
        {"symbol": "RELIANCE", "isin": "INE002A01018", "instrument_key": "NSE_EQ|INE002A01018", "resolved": True},
        {"symbol": "DELISTEDCO", "isin": None, "instrument_key": None, "resolved": False},
    ])
    tmp_store.write_upstox_instrument_map(df)

    out = tmp_store.read_upstox_instrument_map()
    assert len(out) == 2
    unresolved = out[out["symbol"] == "DELISTEDCO"].iloc[0]
    assert not unresolved["resolved"]
    assert pd.isna(unresolved["instrument_key"])


def test_write_upstox_instrument_map_upserts_on_symbol(tmp_store) -> None:
    tmp_store.write_upstox_instrument_map(pd.DataFrame([
        {"symbol": "X", "isin": None, "instrument_key": None, "resolved": False},
    ]))
    tmp_store.write_upstox_instrument_map(pd.DataFrame([
        {"symbol": "X", "isin": "INE999", "instrument_key": "NSE_EQ|INE999", "resolved": True},
    ]))
    out = tmp_store.read_upstox_instrument_map()
    assert len(out) == 1
    assert bool(out.iloc[0]["resolved"]) is True
