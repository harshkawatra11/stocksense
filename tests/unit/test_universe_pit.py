"""Tests for the point-in-time tradeable universe.

THE anti-survivorship control. Every test here is ultimately checking one
property: a filter computed for date `d` must never be able to see a row
dated on or after `d`.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from stocksense.data.store import Reader, Store
from stocksense.data.universe_pit import (
    UniverseFilter,
    filter_panel,
    universe_as_of,
    universe_membership,
)


def _bhav_row(symbol: str, d: date, close: float = 100.0, turnover: float = 10_000_000.0) -> dict:
    return dict(
        symbol=symbol, series="EQ", date=d, open=close, high=close, low=close,
        close=close, prev_close=close, last_price=close, volume=1000.0,
        turnover_inr=turnover, n_trades=100.0, era="udiff",
    )


def _daily(start: date, n: int) -> list[date]:
    """n consecutive weekday dates from start (skipping weekends), matching
    what a real trading calendar looks like."""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


@pytest.fixture()
def reader(tmp_path):
    db, pq = tmp_path / "hot.duckdb", tmp_path / "parquet"
    with Store(db, pq):
        pass  # just create the store; individual tests seed their own data
    r = Reader(pq)
    yield r
    r.close()


def _seed(tmp_path, rows):
    db, pq = tmp_path / "hot.duckdb", tmp_path / "parquet"
    with Store(db, pq) as s:
        s.write_bhavcopy_eq(pd.DataFrame(rows))
    return Reader(pq)


# --------------------------------------------------------------- universe_as_of
def test_symbol_illiquid_before_and_liquid_after_is_excluded_from_the_earlier_date(tmp_path):
    """The core anti-survivorship property. A symbol whose liquidity only
    develops in 2015 must be ABSENT from a 2010 universe, even though it is
    present in the full history -- and present once it has earned it."""
    early = _daily(date(2010, 1, 4), 60)  # 60 low-turnover days
    late = _daily(date(2015, 6, 1), 60)  # 60 high-turnover days
    rows = [_bhav_row("LATECOMER", d, turnover=1_000.0) for d in early]  # far below floor
    rows += [_bhav_row("LATECOMER", d, turnover=10_000_000.0) for d in late]  # well above
    reader = _seed(tmp_path, rows)

    as_of_2010 = early[-1] + timedelta(days=1)
    as_of_2015 = late[-1] + timedelta(days=1)

    assert "LATECOMER" not in universe_as_of(reader, as_of_2010)
    assert "LATECOMER" in universe_as_of(reader, as_of_2015)
    reader.close()


def test_window_end_is_exclusive_no_lookahead(tmp_path):
    """A row dated EXACTLY as_of must not influence that date's universe --
    the single '<' that stops tomorrow's liquidity leaking into today."""
    early = _daily(date(2020, 1, 1), 60)
    rows = [_bhav_row("X", d, turnover=1_000.0) for d in early]  # illiquid throughout
    as_of = early[-1] + timedelta(days=1)
    # One huge-turnover row dated EXACTLY as_of -- must not count.
    rows.append(_bhav_row("X", as_of, turnover=10_000_000.0))
    reader = _seed(tmp_path, rows)

    assert "X" not in universe_as_of(reader, as_of)
    reader.close()


def test_price_bounds_default_to_the_account_band(tmp_path):
    """With no explicit price bounds and a live equity figure, the universe is
    defined by what THIS account can trade -- tradeable_price_band's output,
    not a hardcoded constant."""
    days = _daily(date(2024, 1, 1), 60)
    rows = [_bhav_row("CHEAP", d, close=50.0) for d in days]  # below the tick-drag floor
    rows += [_bhav_row("MID", d, close=900.0) for d in days]  # inside the band
    rows += [_bhav_row("DEAR", d, close=50_000.0) for d in days]  # above the divisibility ceiling
    as_of = days[-1] + timedelta(days=1)
    reader = _seed(tmp_path, rows)

    names = universe_as_of(reader, as_of, equity_inr=17_500.0)
    assert "MID" in names
    assert "CHEAP" not in names
    assert "DEAR" not in names
    reader.close()


def test_explicit_price_bounds_override_the_account_band(tmp_path):
    days = _daily(date(2024, 1, 1), 60)
    rows = [_bhav_row("X", d, close=50.0) for d in days]
    as_of = days[-1] + timedelta(days=1)
    reader = _seed(tmp_path, rows)

    flt = UniverseFilter(min_price_inr=10.0, max_price_inr=100.0)
    assert "X" in universe_as_of(reader, as_of, flt, equity_inr=17_500.0)
    reader.close()


def test_no_capital_and_no_explicit_bounds_means_no_price_filter(tmp_path):
    """Research/backtest callers with no live capital must not have a price
    band invented for them -- only liquidity filtering applies."""
    days = _daily(date(2024, 1, 1), 60)
    rows = [_bhav_row("PENNY", d, close=1.0) for d in days]
    as_of = days[-1] + timedelta(days=1)
    reader = _seed(tmp_path, rows)

    assert "PENNY" in universe_as_of(reader, as_of)  # no equity_inr passed
    reader.close()


def test_low_turnover_symbol_is_excluded(tmp_path):
    days = _daily(date(2024, 1, 1), 60)
    rows = [_bhav_row("THIN", d, turnover=100.0) for d in days]
    as_of = days[-1] + timedelta(days=1)
    reader = _seed(tmp_path, rows)
    assert "THIN" not in universe_as_of(reader, as_of)
    reader.close()


def test_too_few_observations_is_excluded_even_with_high_turnover(tmp_path):
    """Two prints of huge turnover is not a liquid history -- min_observations
    guards against a symbol that briefly spiked once or twice."""
    days = _daily(date(2024, 1, 1), 3)
    rows = [_bhav_row("FLASH", d, turnover=50_000_000.0) for d in days]
    as_of = days[-1] + timedelta(days=1)
    reader = _seed(tmp_path, rows)
    assert "FLASH" not in universe_as_of(reader, as_of)
    reader.close()


def test_universe_as_of_on_empty_data_returns_empty_list(reader):
    assert universe_as_of(reader, date(2024, 1, 1)) == []


def test_universe_as_of_returns_sorted_symbols(tmp_path):
    days = _daily(date(2024, 1, 1), 60)
    rows = [_bhav_row(sym, d) for d in days for sym in ("ZETA", "ALPHA", "MID")]
    as_of = days[-1] + timedelta(days=1)
    reader = _seed(tmp_path, rows)
    assert universe_as_of(reader, as_of) == ["ALPHA", "MID", "ZETA"]
    reader.close()


# ------------------------------------------------------------- universe_membership
def test_membership_is_one_query_per_date(tmp_path, monkeypatch):
    """One SQL query per unique date, never per (symbol, date) pair -- for a
    multi-year daily frame that is the difference between one query per
    trading day and millions of tiny queries."""
    import stocksense.data.universe_pit as up

    days = _daily(date(2024, 1, 1), 65)
    rows = [_bhav_row(sym, d) for d in days for sym in ("A", "B", "C")]
    reader = _seed(tmp_path, rows)

    query_dates = days[60:63]  # 3 distinct dates
    calls = {"n": 0}
    real = up.universe_as_of

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(up, "universe_as_of", counted)
    up.universe_membership(reader, query_dates)
    assert calls["n"] == len(set(query_dates))
    reader.close()


def test_membership_covers_multiple_dates_correctly(tmp_path):
    days = _daily(date(2024, 1, 1), 65)
    rows = [_bhav_row("A", d) for d in days]
    reader = _seed(tmp_path, rows)

    query_dates = [days[61], days[62]]
    m = universe_membership(reader, query_dates)
    assert set(m["date"].dt.date) == set(query_dates)
    assert set(m["symbol"]) == {"A"}
    reader.close()


def test_membership_on_no_dates_returns_empty_frame(reader):
    m = universe_membership(reader, [])
    assert m.empty
    assert list(m.columns) == ["date", "symbol"]


# ------------------------------------------------------------------ filter_panel
def test_filter_panel_preserves_extra_columns(tmp_path):
    membership = pd.DataFrame(
        {"date": pd.to_datetime([date(2024, 1, 5)]), "symbol": ["A"]}
    )
    panel = pd.DataFrame(
        [
            {"date": date(2024, 1, 5), "symbol": "A", "feature_1": 1.23, "feature_2": "x"},
            {"date": date(2024, 1, 5), "symbol": "B", "feature_1": 9.99, "feature_2": "y"},
        ]
    )
    out = filter_panel(panel, membership)
    assert len(out) == 1
    assert out.iloc[0]["symbol"] == "A"
    assert out.iloc[0]["feature_1"] == pytest.approx(1.23)
    assert list(out.columns) == ["date", "symbol", "feature_1", "feature_2"]


def test_filter_panel_drops_non_member_rows(tmp_path):
    membership = pd.DataFrame({"date": pd.to_datetime([date(2024, 1, 5)]), "symbol": ["A"]})
    panel = pd.DataFrame(
        [
            {"date": date(2024, 1, 5), "symbol": "A", "v": 1},
            {"date": date(2024, 1, 6), "symbol": "A", "v": 2},  # different date, not a member
            {"date": date(2024, 1, 5), "symbol": "Z", "v": 3},  # different symbol
        ]
    )
    out = filter_panel(panel, membership)
    assert len(out) == 1
    assert out.iloc[0]["v"] == 1


def test_filter_panel_on_empty_inputs(tmp_path):
    empty = pd.DataFrame(columns=["date", "symbol"])
    panel = pd.DataFrame([{"date": date(2024, 1, 1), "symbol": "A", "v": 1}])
    assert filter_panel(panel, empty).empty
    assert filter_panel(empty, empty).empty
