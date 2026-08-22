"""Point-in-time universe tests. Survivorship bias is invisible by
construction -- a broken point-in-time filter and a correct one look
identical on a single date, differing only across many dates checked
against history. These tests are written to catch exactly that: a
symbol delisted partway through the sample must appear in an early
universe and disappear from a later one, and a symbol that only becomes
liquid AFTER a given date must never appear in that date's universe no
matter how liquid it later becomes."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stocksense.data.store import Store
from stocksense.data.universe_pit import filter_to_point_in_time_universe, universe_as_of, universe_membership_table


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def _bhavcopy_row(symbol, d, close=100.0, turnover=10_000_000.0, series="EQ"):
    return {
        "symbol": symbol, "series": series, "date": d, "open": close, "high": close,
        "low": close, "close": close, "prev_close": close, "volume": 100000.0,
        "turnover_inr": turnover, "era": "legacy" if d < date(2024, 7, 8) else "udiff",
    }


def test_delisted_symbol_present_in_early_universe_absent_from_late_universe(tmp_store) -> None:
    rows = []
    # DELISTEDCO: liquid throughout 2010-2014, then simply stops trading
    for month in range(1, 13):
        rows.append(_bhavcopy_row("DELISTEDCO", date(2010, month, 15)))
    # STILLLISTED: liquid throughout, including 2020
    for year in (2010, 2020):
        rows.append(_bhavcopy_row("STILLLISTED", date(year, 1, 15)))

    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))

    universe_2010 = universe_as_of(tmp_store, date(2010, 1, 20), lookback_days=60)
    universe_2020 = universe_as_of(tmp_store, date(2020, 1, 20), lookback_days=60)

    assert "DELISTEDCO" in universe_2010
    assert "DELISTEDCO" not in universe_2020  # no data in 2020 -> correctly excluded
    assert "STILLLISTED" in universe_2010
    assert "STILLLISTED" in universe_2020


def test_universe_as_of_excludes_future_liquidity(tmp_store) -> None:
    """The property that matters most: a symbol that becomes liquid
    only AFTER the query date must never appear, no matter how liquid
    it eventually becomes -- this is the literal definition of
    survivorship bias, tested directly rather than trusted."""
    rows = []
    # FUTURECO: only starts trading (and only becomes liquid) after 2015
    for month in range(1, 13):
        rows.append(_bhavcopy_row("FUTURECO", date(2015, month, 15), turnover=50_000_000.0))

    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))

    universe_2010 = universe_as_of(tmp_store, date(2010, 6, 15), lookback_days=60)
    assert "FUTURECO" not in universe_2010  # doesn't exist yet as of 2010 -- must not appear

    universe_2015 = universe_as_of(tmp_store, date(2015, 12, 20), lookback_days=60)
    assert "FUTURECO" in universe_2015  # now liquid, now correctly included


def test_low_turnover_symbol_excluded(tmp_store) -> None:
    rows = [_bhavcopy_row("THINLYTRADED", date(2020, 1, 15), turnover=100_000.0)]  # below default 5M threshold
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))

    universe = universe_as_of(tmp_store, date(2020, 1, 20), min_turnover_inr=5_000_000.0, lookback_days=60)
    assert "THINLYTRADED" not in universe


def test_penny_stock_excluded_by_price_floor(tmp_store) -> None:
    rows = [_bhavcopy_row("PENNYCO", date(2020, 1, 15), close=2.0, turnover=10_000_000.0)]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))

    universe = universe_as_of(tmp_store, date(2020, 1, 20), min_price_inr=5.0, lookback_days=60)
    assert "PENNYCO" not in universe


def test_non_eq_series_excluded(tmp_store) -> None:
    rows = [_bhavcopy_row("BONDCO", date(2020, 1, 15), turnover=10_000_000.0, series="GB")]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))

    universe = universe_as_of(tmp_store, date(2020, 1, 20), lookback_days=60, series="EQ")
    assert "BONDCO" not in universe


def test_universe_membership_table_builds_across_multiple_dates(tmp_store) -> None:
    rows = [_bhavcopy_row("A", date(2020, 1, d)) for d in (5, 10, 15)]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))

    table = universe_membership_table(tmp_store, [date(2020, 1, 20)], lookback_days=60)
    assert len(table) == 1
    assert table.iloc[0]["symbol"] == "A"
    assert bool(table.iloc[0]["is_tradeable"]) is True


def test_filter_to_point_in_time_universe_drops_future_liquidity(tmp_store) -> None:
    """The wiring point that actually closes HIGH-4: a feature/label
    frame containing a symbol that only becomes liquid AFTER a given
    row's date must have that row dropped, even though the SAME symbol
    has other rows (at later dates) that survive the filter."""
    rows = []
    for month in range(1, 13):
        rows.append(_bhavcopy_row("FUTURECO", date(2015, month, 15), turnover=50_000_000.0))
    for year in (2010, 2015):
        rows.append(_bhavcopy_row("STEADY", date(year, 6, 15)))
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))

    frame = pd.DataFrame([
        {"symbol": "FUTURECO", "date": date(2010, 6, 20), "value": 1},  # before FUTURECO exists -> must be dropped
        {"symbol": "FUTURECO", "date": date(2015, 12, 20), "value": 2},  # now liquid -> must survive
        {"symbol": "STEADY", "date": date(2010, 6, 20), "value": 3},
        {"symbol": "STEADY", "date": date(2015, 6, 20), "value": 4},
    ])

    out = filter_to_point_in_time_universe(tmp_store, frame, lookback_days=60)

    kept = set(zip(out["symbol"], out["value"]))
    assert (2010, 6, 20) not in [(r["date"].year, r["date"].month, r["date"].day) for _, r in out.iterrows() if r["symbol"] == "FUTURECO"]
    assert ("FUTURECO", 1) not in kept
    assert ("FUTURECO", 2) in kept
    assert ("STEADY", 3) in kept
    assert ("STEADY", 4) in kept


def test_filter_to_point_in_time_universe_preserves_non_universe_columns(tmp_store) -> None:
    rows = [_bhavcopy_row("A", date(2020, 1, d)) for d in (5, 10, 15)]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))
    frame = pd.DataFrame([{"symbol": "A", "date": date(2020, 1, 20), "some_feature": 42.0, "other": "x"}])

    out = filter_to_point_in_time_universe(tmp_store, frame, lookback_days=60)
    assert list(out.columns) == ["symbol", "date", "some_feature", "other"]
    assert out.iloc[0]["some_feature"] == 42.0


def test_filter_to_point_in_time_universe_empty_input(tmp_store) -> None:
    frame = pd.DataFrame(columns=["symbol", "date", "value"])
    out = filter_to_point_in_time_universe(tmp_store, frame, lookback_days=60)
    assert out.empty


def test_turnover_rank_band_selects_a_liquidity_slice(tmp_store) -> None:
    """10 symbols with distinct turnover, all otherwise identical. The
    top band (0.8, 1.0] must contain only the 2 highest-turnover names,
    the bottom band (0.0, 0.2] only the 2 lowest -- proving the band is
    a rank slice of the qualifying set, not an absolute threshold."""
    rows = []
    for i in range(1, 11):
        rows.append(_bhavcopy_row(f"SYM{i:02d}", date(2020, 1, 15), turnover=10_000_000.0 * i))
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))

    top_band = universe_as_of(tmp_store, date(2020, 1, 20), lookback_days=60, turnover_rank_band=(0.8, 1.0))
    assert set(top_band) == {"SYM09", "SYM10"}

    bottom_band = universe_as_of(tmp_store, date(2020, 1, 20), lookback_days=60, turnover_rank_band=(0.0, 0.2))
    assert set(bottom_band) == {"SYM01", "SYM02"}

    mid_band = universe_as_of(tmp_store, date(2020, 1, 20), lookback_days=60, turnover_rank_band=(0.4, 0.6))
    assert set(mid_band) == {"SYM05", "SYM06"}


def test_turnover_rank_band_none_returns_full_liquidity_filtered_set(tmp_store) -> None:
    rows = [_bhavcopy_row(f"SYM{i}", date(2020, 1, 15), turnover=10_000_000.0 * i) for i in range(1, 6)]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))

    full = universe_as_of(tmp_store, date(2020, 1, 20), lookback_days=60)
    assert len(full) == 5


def test_turnover_rank_band_is_point_in_time_safe(tmp_store) -> None:
    """The band must rank only names/turnover known as of `d` -- a
    future-only high-turnover name must not distort an earlier date's
    band membership, the same invariant every other rule in this module
    enforces."""
    rows = []
    for month in range(1, 13):
        rows.append(_bhavcopy_row("EARLYLOW", date(2010, month, 15), turnover=10_000_000.0))
        rows.append(_bhavcopy_row("EARLYHIGH", date(2010, month, 15), turnover=90_000_000.0))
    for month in range(1, 13):
        rows.append(_bhavcopy_row("FUTUREHUGE", date(2015, month, 15), turnover=500_000_000.0))

    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))

    top_band_2010 = universe_as_of(tmp_store, date(2010, 6, 20), lookback_days=60, turnover_rank_band=(0.5, 1.0))
    assert top_band_2010 == ["EARLYHIGH"]
    assert "FUTUREHUGE" not in top_band_2010


def test_filter_to_point_in_time_universe_passes_through_turnover_rank_band(tmp_store) -> None:
    rows = [_bhavcopy_row(f"SYM{i}", date(2020, 1, 15), turnover=10_000_000.0 * i) for i in range(1, 6)]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))
    frame = pd.DataFrame([
        {"symbol": f"SYM{i}", "date": date(2020, 1, 20), "value": i} for i in range(1, 6)
    ])

    out = filter_to_point_in_time_universe(tmp_store, frame, lookback_days=60, turnover_rank_band=(0.8, 1.0))
    assert set(out["symbol"]) == {"SYM5"}
