"""Tests for the corporate-action price adjustment layer.

Acceptance criteria from the plan's R1: a known split (ECLERX 1:2, PASHUPATI
1:10) yields a continuous adjusted series with no residual jump; a symbol with
a genuine bonus is NOT quarantined; every unexplained jump is enumerated, never
aggregated away.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stocksense.data.adjust import (
    adjusted_prices,
    apply_factors,
    cumulative_factors,
    flag_unexplained_jumps,
    quarantine_unexplained,
    with_prev_adjusted_close,
)
from stocksense.data.store import Reader, Store


def _bhav_row(symbol: str, d: date, close: float, **over) -> dict:
    row = dict(
        symbol=symbol, series="EQ", date=d, open=close, high=close, low=close,
        close=close, prev_close=close, last_price=close, volume=10_000.0,
        turnover_inr=close * 10_000.0, n_trades=500.0, era="udiff",
    )
    row.update(over)
    return row


def _ca_row(symbol: str, ex_date: date, factor_price: float, action_type="split", **over) -> dict:
    row = dict(
        symbol=symbol, ex_date=ex_date, action_type=action_type, ratio_num=None,
        ratio_den=None, factor_price=factor_price, dividend_amount=None,
        face_before=None, face_after=None, subject_raw="test", parse_status="ok",
    )
    row.update(over)
    return row


@pytest.fixture()
def paths(tmp_path):
    return tmp_path / "hot.duckdb", tmp_path / "parquet"


def _seeded_reader(paths, bhav_rows, ca_rows=()):
    db, pq = paths
    with Store(db, pq) as s:
        s.write_bhavcopy_eq(pd.DataFrame(bhav_rows))
        if ca_rows:
            s.write_corporate_actions(pd.DataFrame(ca_rows))
    return Reader(pq)


# ---------------------------------------------------------- cumulative_factors
def test_cumulative_factors_applies_only_before_the_ex_date():
    ca = pd.DataFrame([_ca_row("X", date(2024, 6, 15), 0.5)])
    factors = cumulative_factors(ca)
    assert len(factors) == 1
    assert factors.iloc[0].cum_factor == pytest.approx(0.5)


def test_compound_events_on_one_ex_date_multiply():
    """Bonus 1:1 (0.5) and a 1:5 split (0.2) sharing an ex-date compose to 0.1."""
    ca = pd.DataFrame(
        [
            _ca_row("X", date(2024, 6, 15), 0.5, action_type="bonus", subject_raw="Bonus 1:1"),
            _ca_row("X", date(2024, 6, 15), 0.2, action_type="split", subject_raw="Split 5:1"),
        ]
    )
    factors = cumulative_factors(ca)
    assert factors.iloc[0].cum_factor == pytest.approx(0.1)


def test_dividends_do_not_affect_the_price_basis():
    """factor_price == 1.0 rows (dividends) are dropped -- they carry no
    price-return adjustment."""
    ca = pd.DataFrame([_ca_row("X", date(2024, 1, 1), 1.0, action_type="dividend")])
    assert cumulative_factors(ca).empty


def test_no_corporate_actions_returns_empty_factors():
    assert cumulative_factors(pd.DataFrame(columns=["symbol", "ex_date", "factor_price"])).empty


def test_invalid_basis_rejected():
    with pytest.raises(ValueError, match="basis must be"):
        cumulative_factors(pd.DataFrame(), basis="nonsense")


# --------------------------------------------------------------- apply_factors
def test_apply_factors_produces_a_continuous_series_across_a_split():
    """THE acceptance test. ECLERX-shaped 1:2 split: pre-split raw close 3000,
    post-split raw close 1500 -- economically identical. The adjusted series
    must show NO jump at the ex-date."""
    dates = [date(2024, 6, d) for d in (10, 11, 12, 13, 14)]
    # ex-date is the 12th: raw halves that day.
    closes = [3000.0, 3010.0, 1500.0, 1505.0, 1510.0]
    prices = pd.DataFrame([_bhav_row("ECLERX", d, c) for d, c in zip(dates, closes)])
    factors = cumulative_factors(pd.DataFrame([_ca_row("ECLERX", dates[2], 0.5)]))

    out = apply_factors(prices, factors)
    adj = out.sort_values("date")["adj_close"].to_numpy()

    # Continuous: no adjacent-day ratio deviates from a "normal" day's drift.
    day_over_day = adj[1:] / adj[:-1]
    assert np.all(np.abs(day_over_day - 1.0) < 0.02), f"residual jump in {day_over_day}"
    # Pre-split prices scaled down by the factor; post-split untouched.
    assert adj[0] == pytest.approx(3000.0 * 0.5)
    assert adj[-1] == pytest.approx(1510.0)


def test_apply_factors_handles_a_1_to_10_split_pashupati_shaped():
    dates = [date(2024, 3, d) for d in (18, 19, 20, 21)]
    closes = [1000.0, 1010.0, 100.0, 101.0]  # ex-date on the 20th, raw /10
    prices = pd.DataFrame([_bhav_row("PASHUPATI", d, c) for d, c in zip(dates, closes)])
    factors = cumulative_factors(pd.DataFrame([_ca_row("PASHUPATI", dates[2], 0.1)]))

    out = apply_factors(prices, factors)
    adj = out.sort_values("date")["adj_close"].to_numpy()
    day_over_day = adj[1:] / adj[:-1]
    assert np.all(np.abs(day_over_day - 1.0) < 0.02)


def test_apply_factors_with_no_factors_is_a_no_op():
    prices = pd.DataFrame([_bhav_row("X", date(2024, 1, 1), 100.0)])
    out = apply_factors(prices, pd.DataFrame(columns=["symbol", "ex_date", "cum_factor"]))
    assert out.iloc[0].adj_close == pytest.approx(100.0)
    assert out.iloc[0].cum_factor == 1.0


def test_all_four_price_columns_are_adjusted():
    prices = pd.DataFrame(
        [_bhav_row("X", date(2024, 1, 1), 100.0, open=98.0, high=102.0, low=97.0)]
    )
    factors = cumulative_factors(pd.DataFrame([_ca_row("X", date(2024, 1, 2), 0.5)]))
    out = apply_factors(prices, factors)
    row = out.iloc[0]
    assert row.adj_open == pytest.approx(49.0)
    assert row.adj_high == pytest.approx(51.0)
    assert row.adj_low == pytest.approx(48.5)
    assert row.adj_close == pytest.approx(50.0)


# ------------------------------------------------------------ adjusted_prices
def test_adjusted_prices_end_to_end_through_the_reader(paths):
    dates = [date(2024, 6, d) for d in (10, 11, 12, 13)]
    closes = [3000.0, 3010.0, 1500.0, 1505.0]
    rows = [_bhav_row("ECLERX", d, c) for d, c in zip(dates, closes)]
    reader = _seeded_reader(paths, rows, [_ca_row("ECLERX", dates[2], 0.5)])

    out = adjusted_prices(reader, symbols=["ECLERX"])
    adj = out.sort_values("date")["adj_close"].to_numpy()
    day_over_day = adj[1:] / adj[:-1]
    assert np.all(np.abs(day_over_day - 1.0) < 0.02)
    reader.close()


def test_adjusted_prices_on_empty_reader_returns_empty(paths):
    _, pq = paths
    reader = Reader(pq)
    out = adjusted_prices(reader, symbols=["NOPE"])
    assert out.empty
    reader.close()


def test_total_basis_reinvests_dividends(paths):
    """On the total-return basis a dividend factor < 1.0 is folded in, so the
    total-return series sits BELOW the price-only series before the ex-date."""
    dates = [date(2024, 1, d) for d in (10, 11, 12, 13)]
    closes = [100.0, 100.0, 99.0, 99.0]  # small ex-div drop on the 12th
    rows = [_bhav_row("X", d, c) for d, c in zip(dates, closes)]
    ca_rows = [
        dict(
            symbol="X", ex_date=dates[2], action_type="dividend", ratio_num=None,
            ratio_den=None, factor_price=1.0, dividend_amount=1.0, face_before=None,
            face_after=None, subject_raw="Dividend Rs 1", parse_status="ok",
        )
    ]
    reader = _seeded_reader(paths, rows, ca_rows)

    price_basis = adjusted_prices(reader, symbols=["X"], basis="price")
    total_basis = adjusted_prices(reader, symbols=["X"], basis="total")

    pre_ex_price = price_basis.sort_values("date").iloc[0].adj_close
    pre_ex_total = total_basis.sort_values("date").iloc[0].adj_close
    assert pre_ex_total < pre_ex_price
    reader.close()


# ------------------------------------------------------- with_prev_adjusted_close
def test_prev_adjusted_close_uses_the_adjusted_series_not_raw():
    """THE trap this function exists to close: a split ex-date must not show up
    as a fabricated -50% overnight return."""
    dates = [date(2024, 6, d) for d in (10, 11)]
    prices = pd.DataFrame([_bhav_row("X", dates[0], 2000.0), _bhav_row("X", dates[1], 1000.0)])
    factors = cumulative_factors(pd.DataFrame([_ca_row("X", dates[1], 0.5)]))
    adjusted = apply_factors(prices, factors)

    out = with_prev_adjusted_close(adjusted)
    row = out.sort_values("date").iloc[1]
    overnight_ret = row.adj_close / row.prev_adj_close - 1.0
    assert abs(overnight_ret) < 0.02, "split leaked into the overnight return"


def test_prev_gap_sessions_detects_a_multi_week_gap():
    """THE INDOTECH trap: an illiquid symbol's previous ROW is not necessarily
    the previous trading DAY. groupby.shift(1) alone would fabricate a huge
    one-day move across the gap."""
    calendar = [date(2024, 1, d) for d in range(1, 22)]  # 21 consecutive sessions
    # INDOTECH only trades on day 1 and day 21 -- everything else is silent.
    prices = pd.DataFrame(
        [
            _bhav_row("INDOTECH", calendar[0], 100.0),
            _bhav_row("INDOTECH", calendar[20], 150.0),
        ]
    )
    prices["adj_close"] = prices["close"]  # pretend already adjusted
    out = with_prev_adjusted_close(prices, calendar=calendar)
    row = out.sort_values("date").iloc[1]
    assert row.prev_gap_sessions == 20, "the gap must be counted in real sessions"


def test_prev_gap_sessions_is_one_for_consecutive_trading_days():
    calendar = [date(2024, 1, d) for d in (2, 3, 4)]
    prices = pd.DataFrame([_bhav_row("X", d, 100.0) for d in calendar])
    prices["adj_close"] = prices["close"]
    out = with_prev_adjusted_close(prices, calendar=calendar).sort_values("date")
    assert out.iloc[1].prev_gap_sessions == 1
    assert out.iloc[2].prev_gap_sessions == 1


# --------------------------------------------------------- flag_unexplained_jumps
def test_a_genuine_bonus_is_not_quarantined():
    """THE regression from a previous build: RELIANCE-style genuine splits were
    quarantined by comparing adj_close to raw close. That check is gone; this
    one must let a properly-adjusted, CA-explained jump through untouched."""
    dates = [date(2024, 10, d) for d in (25, 26, 27, 28, 29)]
    closes = [3000.0, 3010.0, 1500.0, 1505.0, 1510.0]  # 1:2 bonus on the 27th
    prices = pd.DataFrame([_bhav_row("RELIANCE", d, c) for d, c in zip(dates, closes)])
    ca = pd.DataFrame([_ca_row("RELIANCE", dates[2], 0.5, action_type="bonus")])
    factors = cumulative_factors(ca)
    adjusted = apply_factors(prices, factors)

    flagged = flag_unexplained_jumps(adjusted, ca, calendar=dates)
    assert flagged.empty, "a CA-explained, correctly-adjusted jump must not be flagged"

    clean, quarantined = quarantine_unexplained(adjusted, ca, calendar=dates)
    assert quarantined == []
    assert len(clean) == len(prices)


def test_an_unexplained_jump_is_flagged_and_enumerated():
    """No matching CA record -- this must surface as a named, dated finding,
    never silently aggregated away."""
    dates = [date(2024, 5, d) for d in (10, 11, 12, 13)]
    closes = [100.0, 101.0, 40.0, 41.0]  # -60% with nothing to explain it
    prices = pd.DataFrame([_bhav_row("BUGGY", d, c) for d, c in zip(dates, closes)])
    prices["adj_close"] = prices["close"]
    prices["cum_factor"] = 1.0

    empty_ca = pd.DataFrame(columns=Store.CA_COLS)
    flagged = flag_unexplained_jumps(prices, empty_ca, calendar=dates)
    assert not flagged.empty
    assert set(flagged["symbol"]) == {"BUGGY"}
    assert flagged.iloc[0].date == pd.Timestamp(dates[2])


def test_unexplained_jumps_are_ignored_across_a_calendar_gap():
    """A big move across a multi-week gap in an illiquid name is not a one-day
    jump and must not be flagged."""
    calendar = [date(2024, 1, d) for d in range(1, 15)]
    prices = pd.DataFrame(
        [_bhav_row("THIN", calendar[0], 100.0), _bhav_row("THIN", calendar[13], 200.0)]
    )
    prices["adj_close"] = prices["close"]
    prices["cum_factor"] = 1.0
    flagged = flag_unexplained_jumps(prices, pd.DataFrame(columns=Store.CA_COLS), calendar=calendar)
    assert flagged.empty


def test_a_jump_within_the_explain_window_of_an_unrelated_ca_is_still_explained():
    """The explain window tolerates the announced ex-date and the actual price
    gap differing by a day or two around holidays."""
    dates = [date(2024, 6, d) for d in (10, 11, 12, 13)]
    closes = [3000.0, 3010.0, 1500.0, 1505.0]
    prices = pd.DataFrame([_bhav_row("X", d, c) for d, c in zip(dates, closes)])
    prices["adj_close"] = prices["close"]
    prices["cum_factor"] = 1.0
    # CA record dated 2 days after the actual jump.
    ca = pd.DataFrame([_ca_row("X", dates[2] + pd.Timedelta(days=2), 0.5)])
    flagged = flag_unexplained_jumps(prices, ca, explain_window_days=7, calendar=dates)
    assert flagged.empty


def test_quarantine_drops_the_whole_symbol_not_just_the_bad_row():
    """Coarse but safe: one unexplained discontinuity taints the whole
    symbol's history, because other rows may share the same missing action."""
    dates = [date(2024, 5, d) for d in (10, 11, 12, 13)]
    closes = [100.0, 101.0, 40.0, 41.0]
    prices = pd.DataFrame([_bhav_row("BUGGY", d, c) for d, c in zip(dates, closes)])
    prices["adj_close"] = prices["close"]
    prices["cum_factor"] = 1.0
    clean, quarantined = quarantine_unexplained(prices, pd.DataFrame(columns=Store.CA_COLS), calendar=dates)
    assert quarantined == ["BUGGY"]
    assert clean.empty


def test_empty_prices_short_circuit_cleanly():
    empty = pd.DataFrame(columns=["symbol", "date", "adj_close"])
    assert flag_unexplained_jumps(empty, pd.DataFrame(columns=Store.CA_COLS)).empty
    clean, bad = quarantine_unexplained(empty, pd.DataFrame(columns=Store.CA_COLS))
    assert bad == []
    assert clean.empty
