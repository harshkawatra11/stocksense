"""Price adjustment tests. The whole point of data/adjust.py is that a
confirmed split (verified live: ECLERX 1:2, PASHUPATI 1:10 -- bhavcopy's
own prev_close equals the prior RAW close across both, factor 1.0, no
adjustment) must produce a CONTINUOUS adjusted series -- no residual
jump on the ex-date -- while an unrelated symbol with no corporate
action is left untouched."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stocksense.data.adjust import (
    adjusted_prices,
    adjustment_factors,
    flag_unexplained_adjustment_jumps,
    quarantine_unexplained_jumps,
)
from stocksense.data.store import Store


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def _bhav_row(symbol, d, close, prev_close=None):
    return {
        "symbol": symbol, "series": "EQ", "date": d, "open": close, "high": close,
        "low": close, "close": close, "prev_close": prev_close if prev_close is not None else close,
        "volume": 1000.0, "turnover_inr": close * 1000.0, "era": "udiff",
    }


def _ca_row(symbol, ex_date, action_type, factor_price, dividend_amount=None):
    return {
        "symbol": symbol, "ex_date": ex_date, "action_type": action_type,
        "ratio_num": None, "ratio_den": None, "factor_price": factor_price,
        "dividend_amount": dividend_amount, "face_before": None, "face_after": None,
        "subject_raw": f"test {action_type}", "parse_status": "ok",
    }


def test_split_produces_continuous_adjusted_series_no_jump_on_ex_date(tmp_store) -> None:
    # A 1:1 split (ECLERX-style): raw close halves on the ex-date, but
    # NO real change in shareholder value occurred.
    rows = [
        _bhav_row("SPLITCO", date(2026, 3, 10), close=3151.8),  # day before ex-date
        _bhav_row("SPLITCO", date(2026, 3, 13), close=1576.6, prev_close=3151.8),  # ex-date, raw halved
        _bhav_row("SPLITCO", date(2026, 3, 16), close=1580.0),  # continues trading post-split
    ]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))
    tmp_store.write_corporate_actions(pd.DataFrame([_ca_row("SPLITCO", date(2026, 3, 13), "split", 0.5)]))

    out = adjusted_prices(tmp_store, ["SPLITCO"], date(2026, 3, 1), date(2026, 3, 31), basis="price")
    out = out.sort_values("date").reset_index(drop=True)

    pre_split_adj = out.iloc[0]["adj_close"]   # 2026-03-10, before ex-date -> gets multiplied by 0.5
    post_split_adj = out.iloc[1]["adj_close"]  # 2026-03-13, ex-date itself -> already at new scale, factor 1.0

    assert pre_split_adj == pytest.approx(3151.8 * 0.5)
    assert post_split_adj == pytest.approx(1576.6)
    # continuity: no residual one-day "return" from the split itself
    day_over_day_ret = abs(post_split_adj / pre_split_adj - 1)
    assert day_over_day_ret < 0.02


def test_symbol_with_no_corporate_action_is_unadjusted(tmp_store) -> None:
    rows = [_bhav_row("PLAINCO", date(2026, 1, 5), close=100.0), _bhav_row("PLAINCO", date(2026, 1, 6), close=101.0)]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))

    out = adjusted_prices(tmp_store, ["PLAINCO"], date(2026, 1, 1), date(2026, 1, 31), basis="price")
    assert (out["adj_close"] == out["close"]).all()


def test_adjustment_factors_cumulative_across_multiple_events(tmp_store) -> None:
    # Two splits on the same symbol: a price before BOTH ex-dates must
    # be multiplied by the product of both factors.
    rows = [
        _bhav_row("DOUBLE", date(2020, 1, 1), close=1000.0),
        _bhav_row("DOUBLE", date(2021, 1, 1), close=500.0, prev_close=1000.0),
        _bhav_row("DOUBLE", date(2022, 1, 1), close=250.0, prev_close=500.0),
    ]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))
    tmp_store.write_corporate_actions(pd.DataFrame([
        _ca_row("DOUBLE", date(2021, 1, 1), "split", 0.5),
        _ca_row("DOUBLE", date(2022, 1, 1), "split", 0.5),
    ]))

    out = adjusted_prices(tmp_store, ["DOUBLE"], date(2019, 1, 1), date(2022, 12, 31), basis="price").sort_values("date")
    first = out.iloc[0]["adj_close"]  # 2020-01-01: before both ex-dates
    assert first == pytest.approx(1000.0 * 0.5 * 0.5)


def test_dividend_only_affects_total_basis_not_price_basis(tmp_store) -> None:
    rows = [
        _bhav_row("DIVCO", date(2026, 1, 5), close=100.0),
        _bhav_row("DIVCO", date(2026, 1, 6), close=98.0, prev_close=100.0),  # ex-dividend drop
    ]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))
    tmp_store.write_corporate_actions(pd.DataFrame([_ca_row("DIVCO", date(2026, 1, 6), "dividend", 1.0, dividend_amount=2.0)]))

    price_basis = adjusted_prices(tmp_store, ["DIVCO"], date(2026, 1, 1), date(2026, 1, 31), basis="price")
    total_basis = adjusted_prices(tmp_store, ["DIVCO"], date(2026, 1, 1), date(2026, 1, 31), basis="total")

    assert (price_basis["adj_close"] == price_basis["close"]).all()
    pre_div_total = total_basis.sort_values("date").iloc[0]["adj_close"]
    assert pre_div_total < 100.0  # total-return basis backs out the dividend from the pre-ex-date price


def test_adjustment_factors_empty_when_no_actions(tmp_store) -> None:
    factors = adjustment_factors(tmp_store, ["ANYTHING"], basis="price")
    assert factors.empty


def test_adjustment_factors_rejects_invalid_basis(tmp_store) -> None:
    with pytest.raises(ValueError):
        adjustment_factors(tmp_store, ["X"], basis="bogus")


# ---- flag_unexplained_adjustment_jumps / quarantine_unexplained_jumps ----
#
# Regression coverage for the real bug found live in Phase D2:
# data/validate.quarantine_symbols (built for yfinance, where `close` is
# already split-adjusted) was applied unconditionally to bhavcopy-sourced
# adjusted prices and quarantined RELIANCE, TCS, and ~600 other blue-chip
# symbols for their GENUINE 1:1 bonuses (factor exactly 2.0), because
# bhavcopy's raw `close` legitimately produces a step in adj_close/close
# at every real corporate action. These tests pin down the correct,
# source-appropriate behavior: a jump WITH a matching corporate_actions
# record must pass through untouched; a jump with NO matching record
# (an unparsed action, a genuine data error) must still be caught.


def _trading_calendar(store, dates: list[date]) -> None:
    """Registers a full trading calendar in bhavcopy_eq (via a throwaway
    liquid symbol) so flag_unexplained_adjustment_jumps' calendar-join
    sees every date as a genuine trading day, matching how the real
    table is populated."""
    rows = [_bhav_row("_CAL_", d, close=100.0) for d in dates]
    store.write_bhavcopy_eq(pd.DataFrame(rows))


def test_genuine_split_with_matching_ca_record_is_not_flagged(tmp_store) -> None:
    dates = [date(2026, 1, d) for d in (5, 6, 7, 8, 9)]
    _trading_calendar(tmp_store, dates)
    tmp_store.write_corporate_actions(pd.DataFrame([_ca_row("RELIANCE_LIKE", dates[2], "bonus", 0.5)]))

    # simulate the raw-close jump a real 1:1 bonus produces (close halves on ex-date)
    rows = [
        _bhav_row("RELIANCE_LIKE", dates[0], close=800.0),
        _bhav_row("RELIANCE_LIKE", dates[1], close=810.0),
        _bhav_row("RELIANCE_LIKE", dates[2], close=409.0, prev_close=818.0),  # ex-date: raw halves
        _bhav_row("RELIANCE_LIKE", dates[3], close=412.0),
        _bhav_row("RELIANCE_LIKE", dates[4], close=415.0),
    ]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))
    adjusted = adjusted_prices(tmp_store, ["RELIANCE_LIKE"], dates[0], dates[-1], basis="price")

    flagged = flag_unexplained_adjustment_jumps(tmp_store, adjusted, jump_threshold=0.35)
    assert flagged.empty

    clean, bad = quarantine_unexplained_jumps(tmp_store, adjusted, jump_threshold=0.35)
    assert bad == []
    assert "RELIANCE_LIKE" in set(clean["symbol"])


def test_jump_with_no_matching_ca_record_is_flagged_and_quarantined(tmp_store) -> None:
    dates = [date(2026, 2, d) for d in (2, 3, 4, 5, 6)]
    _trading_calendar(tmp_store, dates)
    # NO corporate_actions record written -- this jump is unexplained
    rows = [
        _bhav_row("BUGGYCO", dates[0], close=800.0),
        _bhav_row("BUGGYCO", dates[1], close=810.0),
        _bhav_row("BUGGYCO", dates[2], close=400.0, prev_close=810.0),  # unexplained ~50% drop
        _bhav_row("BUGGYCO", dates[3], close=405.0),
        _bhav_row("BUGGYCO", dates[4], close=410.0),
    ]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))
    adjusted = adjusted_prices(tmp_store, ["BUGGYCO"], dates[0], dates[-1], basis="price")

    flagged = flag_unexplained_adjustment_jumps(tmp_store, adjusted, jump_threshold=0.35)
    assert not flagged.empty
    assert "BUGGYCO" in set(flagged["symbol"])

    clean, bad = quarantine_unexplained_jumps(tmp_store, adjusted, jump_threshold=0.35)
    assert bad == ["BUGGYCO"]
    assert clean.empty


def test_non_consecutive_market_days_are_not_falsely_flagged(tmp_store) -> None:
    """An illiquid symbol whose rows skip several calendar/trading days
    must not have that gap misread as a same-day jump -- the same
    LAG-across-gaps pitfall found earlier when analyzing bhavcopy
    directly (INDOTECH's apparent 'jump' was really a multi-week gap)."""
    dates = [date(2026, 3, d) for d in range(2, 12)]  # full calendar, 10 days
    _trading_calendar(tmp_store, dates)
    # THINCO only trades on day 1 and day 10 -- a large price difference
    # across that gap is NOT a same-day jump and must not be flagged.
    rows = [
        _bhav_row("THINCO", dates[0], close=100.0),
        _bhav_row("THINCO", dates[-1], close=180.0),  # +80% but across a 9-day gap, not consecutive
    ]
    tmp_store.write_bhavcopy_eq(pd.DataFrame(rows))
    adjusted = adjusted_prices(tmp_store, ["THINCO"], dates[0], dates[-1], basis="price")

    flagged = flag_unexplained_adjustment_jumps(tmp_store, adjusted, jump_threshold=0.35)
    assert flagged.empty
