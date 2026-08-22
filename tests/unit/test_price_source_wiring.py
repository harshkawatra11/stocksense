"""Tests for the price_source config switch (Phase D2, docs/17-data-
spine.md): 'candles' must be byte-for-byte the Phase 0 path (so those
numbers stay reproducible), 'bhavcopy' must route through the
corporate-action adjustment layer and, when requested, the point-in-time
universe filter -- never both un-adjusted raw prices AND the full
7,556-symbol history landing in a model untouched."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stocksense.cli.main import _load_candles, _load_features_and_labels
from stocksense.core.config import get_settings
from stocksense.data.store import Store


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def _bhav_row(symbol, d, close=100.0, prev_close=None):
    return {
        "symbol": symbol, "series": "EQ", "date": d, "open": close, "high": close,
        "low": close, "close": close, "prev_close": prev_close if prev_close is not None else close,
        "volume": 1000.0, "turnover_inr": close * 1000.0, "era": "udiff",
    }


def test_candles_source_reads_the_candles_table_unchanged(tmp_store, monkeypatch) -> None:
    monkeypatch.setenv("STOCKSENSE_PRICE_SOURCE", "candles")
    tmp_store.upsert_candles(pd.DataFrame([{
        "symbol": "X", "date": date(2020, 1, 5), "open": 10, "high": 11, "low": 9,
        "close": 10.5, "adj_close": 10.5, "volume": 100.0, "source": "yfinance",
    }]))
    settings = get_settings()
    out = _load_candles(settings, tmp_store)
    assert list(out["symbol"]) == ["X"]
    assert out.iloc[0]["source"] == "yfinance"


def test_bhavcopy_source_routes_through_adjustment_layer(tmp_store, monkeypatch) -> None:
    monkeypatch.setenv("STOCKSENSE_PRICE_SOURCE", "bhavcopy")
    monkeypatch.setenv("STOCKSENSE_RETURN_BASIS", "price")
    rows = [
        _bhav_row("SPLITCO", date(2026, 1, 5), close=200.0),
        _bhav_row("SPLITCO", date(2026, 1, 6), close=100.0, prev_close=200.0),  # ex-date, raw halved
    ]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))
    tmp_store.write_corporate_actions(pd.DataFrame([{
        "symbol": "SPLITCO", "ex_date": date(2026, 1, 6), "action_type": "split",
        "ratio_num": None, "ratio_den": None, "factor_price": 0.5, "dividend_amount": None,
        "face_before": None, "face_after": None, "subject_raw": "test", "parse_status": "ok",
    }]))

    settings = get_settings()
    out = _load_candles(settings, tmp_store)
    out = out.sort_values("date").reset_index(drop=True)

    assert (out["source"] == "bhavcopy").all()
    # the pre-split raw close (200) must come out adjusted to 100, matching
    # the post-split scale -- an unadjusted feature pipeline would see a
    # fake -50% return here instead of continuity
    assert out.iloc[0]["adj_close"] == pytest.approx(100.0)
    # raw OHLC stays untouched (candlestick features need the actual printed bar)
    assert out.iloc[0]["close"] == pytest.approx(200.0)


def test_bhavcopy_source_with_point_in_time_universe_drops_illiquid_rows(tmp_store, monkeypatch) -> None:
    monkeypatch.setenv("STOCKSENSE_PRICE_SOURCE", "bhavcopy")
    monkeypatch.setenv("STOCKSENSE_USE_POINT_IN_TIME_UNIVERSE", "true")
    rows = [
        _bhav_row("LIQUID", date(2020, 1, 15)),
        _bhav_row("THIN", date(2020, 1, 15)),
    ]
    df = pd.DataFrame(rows)
    df.loc[df["symbol"] == "LIQUID", "turnover_inr"] = 10_000_000.0  # above the 5M default floor
    df.loc[df["symbol"] == "THIN", "turnover_inr"] = 1000.0  # below the 5M default floor
    tmp_store.write_bhavcopy_eq(df)

    settings = get_settings()
    out = _load_candles(settings, tmp_store)
    assert "LIQUID" in set(out["symbol"])
    assert "THIN" not in set(out["symbol"])


def test_bhavcopy_source_with_turnover_rank_band_selects_a_cap_slice(tmp_store, monkeypatch) -> None:
    """turnover_rank_band must reach filter_to_point_in_time_universe
    through _load_candles -- the actual wiring point Phase G's mid/
    small-cap gate re-run depends on."""
    monkeypatch.setenv("STOCKSENSE_PRICE_SOURCE", "bhavcopy")
    monkeypatch.setenv("STOCKSENSE_USE_POINT_IN_TIME_UNIVERSE", "true")
    rows = [_bhav_row(f"SYM{i}", date(2020, 1, 15)) for i in range(1, 6)]
    df = pd.DataFrame(rows)
    for i in range(1, 6):
        df.loc[df["symbol"] == f"SYM{i}", "turnover_inr"] = 10_000_000.0 * i
    tmp_store.write_bhavcopy_eq(df)

    settings = get_settings()
    out = _load_candles(settings, tmp_store, turnover_rank_band=(0.8, 1.0))
    assert set(out["symbol"]) == {"SYM5"}  # top quintile by turnover only


def test_bhavcopy_source_turnover_rank_band_ignored_without_pit_universe(tmp_store, monkeypatch) -> None:
    """No point-in-time filter at all -> a rank band has nothing to
    slice and must not silently drop rows."""
    monkeypatch.setenv("STOCKSENSE_PRICE_SOURCE", "bhavcopy")
    monkeypatch.setenv("STOCKSENSE_USE_POINT_IN_TIME_UNIVERSE", "false")
    rows = [_bhav_row(f"SYM{i}", date(2020, 1, 15)) for i in range(1, 6)]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))

    settings = get_settings()
    out = _load_candles(settings, tmp_store, turnover_rank_band=(0.8, 1.0))
    assert set(out["symbol"]) == {f"SYM{i}" for i in range(1, 6)}


def test_unknown_price_source_raises(tmp_store, monkeypatch) -> None:
    monkeypatch.setenv("STOCKSENSE_PRICE_SOURCE", "bogus")
    settings = get_settings()
    with pytest.raises(ValueError):
        _load_candles(settings, tmp_store)


def test_bhavcopy_source_with_no_data_returns_empty_not_crash(tmp_store, monkeypatch) -> None:
    monkeypatch.setenv("STOCKSENSE_PRICE_SOURCE", "bhavcopy")
    settings = get_settings()
    out = _load_candles(settings, tmp_store)
    assert out.empty


def test_load_features_and_labels_uses_bhavcopy_appropriate_quarantine(tmp_path, monkeypatch) -> None:
    """Regression: _load_features_and_labels used to apply the
    yfinance-appropriate quarantine_symbols unconditionally, which
    quarantined a symbol with a genuine 1:1 bonus (factor exactly 2.0)
    for a real corporate action -- found live in Phase D2, where it
    wiped out RELIANCE, TCS, and ~600 other blue-chip symbols. A
    symbol with a REAL, CA-explained split must survive into the
    feature/label frame when price_source='bhavcopy'."""
    db_path = tmp_path / "test.duckdb"
    store = Store(db_path)
    dates = [date(2024, 1, d) for d in range(2, 30) if date(2024, 1, d).weekday() < 5]
    rows = []
    for i, d in enumerate(dates):
        if i < 10:
            close, prev = 800.0 + i, None
        elif i == 10:
            close, prev = 409.0, 809.0  # ex-date: raw close halves
        else:
            close, prev = 409.0 + (i - 10), None
        rows.append(_bhav_row("GENUINESPLIT", d, close=close, prev_close=prev))
        rows.append(_bhav_row("PLAINCO2", d, close=50.0 + i * 0.1))  # second symbol: sidesteps a pandas single-group groupby.apply edge case, unrelated to this test
    df = pd.DataFrame(rows)
    df.loc[:, "turnover_inr"] = 10_000_000.0  # clear the point-in-time liquidity floor if ever applied
    store.write_bhavcopy_eq(df)
    store.write_corporate_actions(pd.DataFrame([{
        "symbol": "GENUINESPLIT", "ex_date": dates[10], "action_type": "bonus",
        "ratio_num": 1.0, "ratio_den": 1.0, "factor_price": 0.5, "dividend_amount": None,
        "face_before": None, "face_after": None, "subject_raw": "Bonus 1:1", "parse_status": "ok",
    }]))
    store.close()

    monkeypatch.setenv("STOCKSENSE_DUCKDB_PATH", str(db_path))
    monkeypatch.setenv("STOCKSENSE_PRICE_SOURCE", "bhavcopy")
    monkeypatch.setenv("STOCKSENSE_USE_POINT_IN_TIME_UNIVERSE", "false")

    candles, feats, fcols, labeled = _load_features_and_labels(horizon=5)
    assert "GENUINESPLIT" in set(candles["symbol"])
    assert "GENUINESPLIT" in set(feats["symbol"])
