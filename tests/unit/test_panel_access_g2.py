"""G2-C3: the u500 hysteresis flag and the real days-since-result in EnginePanelAccess."""

from __future__ import annotations

from datetime import timedelta

import polars as pl
import pytest

from stocksense.backtest.engine import PanelData
from stocksense.core.calendar import TradingCalendar
from stocksense.portfolio.construct import EnginePanelAccess
from tests.fixtures.engine_panel import dates, frame, stock_rows

DS = dates(40)
CAL = TradingCalendar.from_dates(DS)
T = DS[20]
D1 = DS[21]


def panel(ranks: dict[int, int]) -> PanelData:
    stocks = [stock_rows(s, DS, [100.0] * len(DS), [100.0] * len(DS)) for s in ranks]
    daily = frame(*stocks)
    uni = pl.DataFrame(
        [(d, s, r) for d in DS for s, r in ranks.items()],
        schema={"date": pl.Date, "sid": pl.Int64, "turnover_rank": pl.Int64},
        orient="row",
    )
    return PanelData.from_frame(daily, uni)


def test_rank_540_with_u500_true_stays_in_universe_only_with_the_flag():
    p = panel({1: 540, 2: 10})
    u500 = {d: {1, 2} for d in DS}  # sid 1 sits above rank 500 but the hysteresis flag still holds it
    off = EnginePanelAccess(p, CAL).day(T, D1)
    on = EnginePanelAccess(p, CAL, u500=u500, universe_flag=True).day(T, D1)
    assert off.rows[1].in_universe is False and off.rows[2].in_universe is True
    assert on.rows[1].in_universe is True and on.rows[2].in_universe is True


def test_flag_false_name_leaves_even_with_a_good_rank():
    p = panel({1: 100, 2: 10})
    u500 = {d: {2} for d in DS}  # sid 1 dropped out of the flag
    acc = EnginePanelAccess(p, CAL, u500=u500, universe_flag=True)
    rows = acc.day(T, D1).rows
    assert rows[1].in_universe is False and rows[2].in_universe is True


def test_flag_is_looked_up_on_the_decision_date_only():
    p = panel({1: 540})
    u500 = {T: {1}}  # member on T, not on the next day
    acc = EnginePanelAccess(p, CAL, u500=u500, universe_flag=True)
    assert acc.day(T, D1).rows[1].in_universe is True
    assert acc.day(DS[22], DS[23]).rows[1].in_universe is False


def test_universe_flag_needs_the_u500_map():
    with pytest.raises(ValueError, match="u500"):
        EnginePanelAccess(panel({1: 10}), CAL, universe_flag=True)


def test_days_since_result_counts_sessions_since_the_latest_result():
    p = panel({1: 10, 2: 10, 3: 10})
    ev = {
        1: [DS[10], DS[17]],  # latest on or before T is DS[17]: 3 sessions before T = DS[20]
        2: [DS[20]],  # a result on T itself
        3: [DS[21]],  # in the future: no past result, so 90
    }
    rows = EnginePanelAccess(p, CAL, result_dates=ev).day(T, D1).rows
    assert rows[1].days_since_result == 3
    assert rows[2].days_since_result == 0
    assert rows[3].days_since_result == 90
    # a name with no result dates at all is 90 as before
    assert EnginePanelAccess(panel({9: 10}), CAL).day(T, D1).rows[9].days_since_result == 90


def test_days_since_result_is_capped_at_90():
    ds = dates(200)
    cal = TradingCalendar.from_dates(ds)
    daily = frame(stock_rows(1, ds, [100.0] * 200, [100.0] * 200))
    p = PanelData.from_frame(daily)
    acc = EnginePanelAccess(p, cal, result_dates={1: [ds[5]]})
    assert acc.day(ds[50], ds[51]).rows[1].days_since_result == 45
    assert acc.day(ds[150], ds[151]).rows[1].days_since_result == 90  # 145 sessions ago: capped


def test_future_result_dates_do_not_leak_into_days_since_result():
    p = panel({1: 10})
    rows = EnginePanelAccess(p, CAL, result_dates={1: [T + timedelta(days=1)]}).day(T, D1).rows
    assert rows[1].days_since_result == 90


def test_universe_flag_is_a_spec_key_only_when_true():
    from dataclasses import replace

    from stocksense.validation.g1 import G1Config, _spec_s1

    cfg = G1Config(capitals=(5e5,), reference_capital=5e5)
    assert "universe_flag" not in _spec_s1(cfg, "y_c2o_10", 10, False, [])
    assert _spec_s1(replace(cfg, universe_flag=True), "y_c2o_10", 10, False, [])["universe_flag"] is True


# ---- G2-E1: HAR / GARCH forecasts on the rows ---------------------------------------------------------


def test_vol_forecasts_reach_the_name_rows_and_default_to_none():
    import numpy as np

    p = panel({1: 10, 2: 20})
    n = len(p.sid)
    har = np.full(n, np.nan)
    garch = np.full(n, np.nan)
    j = p.find(T, 1)
    har[j], garch[j] = 0.021, 0.034  # sid 1 has both on T; sid 2 has none
    rows = EnginePanelAccess(p, CAL, volfc=(har, garch)).day(T, D1).rows
    assert rows[1].vol_har == pytest.approx(0.021) and rows[1].vol_garch == pytest.approx(0.034)
    assert rows[2].vol_har is None and rows[2].vol_garch is None
    plain = EnginePanelAccess(p, CAL).day(T, D1).rows
    assert plain[1].vol_har is None and plain[1].vol_garch is None
    with pytest.raises(ValueError):
        EnginePanelAccess(p, CAL, volfc=(har[:-1], garch))
