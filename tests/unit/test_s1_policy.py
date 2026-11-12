from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings as hsettings
from hypothesis import strategies as st

from stocksense.costs.india import min_position_for_fixed_cost
from stocksense.portfolio.construct import (
    Holding,
    NameRow,
    PanelDay,
    S1Book,
    S1Config,
    allocate_weights,
    build_s1_orders,
    plan_s1,
    portfolio_vol,
)
from tests.fixtures.engine_panel import settings, zero_slippage

S = settings()
T, D1 = date(2020, 3, 2), date(2020, 3, 3)
H = 5


def cfg(k_max: int = 10, overlay: bool = False, hysteresis: bool = True) -> S1Config:
    return S1Config.from_settings(S, H, zero_slippage(), k_max=k_max, overlay=overlay, hysteresis=hysteresis)


def row(**kw) -> NameRow:
    base = dict(close=100.0, vol_21=0.005, turnover_rank=10, med_turn_20=1e10, industry="X")
    base.update(kw)
    return NameRow(**base)


def world(n: int = 14, industries=None, **row_kw):
    """n names, sid i has score 0.4 - 0.05 i (descending) and er 0.05; each in its own industry by default."""
    scores = {i: 0.4 - 0.05 * i for i in range(n)}
    er = {i: 0.05 for i in range(n)}
    rows = {i: row(industry=(industries[i] if industries else f"I{i}"), **row_kw) for i in range(n)}
    return scores, er, rows


def day(rows, **kw) -> PanelDay:
    return PanelDay(T, D1, rows, **kw)


def book(holdings=(), cash=1e9, dd=0.0, stopped=False) -> S1Book:
    return S1Book({h.sid: h for h in holdings}, cash, dd, stopped)


def buys(plan):
    return [o for o in plan.orders if o.side == "BUY"]


def sells(plan):
    return [o for o in plan.orders if o.side == "SELL"]


# ---- breadth ------------------------------------------------------------------------------------------


def test_minimum_position_is_48000_with_the_d7_costs():
    c = cfg()
    assert min_position_for_fixed_cost("delivery", S.costs, S.risk.max_fixed_cost_bps) == 48_000.0
    assert c.min_pos == 48_000.0


def test_k_is_zero_at_30000_rupees():
    sc, er, rows = world()
    plan = plan_s1(T, sc, er, book(), 30_000.0, day(rows), cfg())
    assert plan.k_raw == 0 and plan.k == 0
    assert plan.orders == []
    assert "capital below one minimum position" in plan.note


def test_k_is_two_at_one_lakh():
    sc, er, rows = world()
    plan = plan_s1(T, sc, er, book(), 100_000.0, day(rows), cfg())
    assert plan.gross_target == pytest.approx(1.0)  # low vol: the cap max_gross binds
    assert plan.k_raw == 2 and plan.k == 2
    assert [o.sid for o in buys(plan)] == [0, 1]
    # one name is at least one minimum position (the name cap never goes below it)
    assert all(o.qty * 100.0 >= 48_000.0 - 100.0 for o in buys(plan))


def test_exposure_target_scales_with_portfolio_vol():
    sc, er, rows = world(vol_21=0.02)  # 32% annual vol per name: 12% target needs gross < 1
    plan = plan_s1(T, sc, er, book(), 5_000_000.0, day(rows), cfg(k_max=10))
    expected = S.risk.target_vol_annual / plan.port_vol
    assert plan.gross_target == pytest.approx(expected) and expected < 1.0
    assert plan.port_vol == pytest.approx(0.02 * np.sqrt(0.3 + 0.7 / 10) * np.sqrt(252))


def test_ledoit_wolf_used_with_enough_history_and_fallback_without():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, (3, 126))
    with_hist = [row(vol_21=0.01, rets=r[i]) for i in range(3)]
    v_lw = portfolio_vol(with_hist)
    short = [row(vol_21=0.01, rets=r[i][:50]) for i in range(3)]  # fewer than 60 common rows
    v_fb = portfolio_vol(short)
    assert v_fb == pytest.approx(0.01 * np.sqrt(0.3 + 0.7 / 3) * np.sqrt(252))
    assert v_lw != pytest.approx(v_fb, rel=1e-3) and 0.05 < v_lw < 0.3
    # rows with a null are dropped before the 60-row test
    rn = r.copy()
    rn[:, ::2] = np.nan  # leaves 63 common rows -> still Ledoit-Wolf
    assert portfolio_vol([row(vol_21=0.01, rets=rn[i]) for i in range(3)]) != pytest.approx(v_fb, rel=1e-3)
    rn[:, ::1] = np.nan
    assert portfolio_vol([row(vol_21=0.01, rets=rn[i]) for i in range(3)]) == pytest.approx(v_fb)


def test_overlay_halves_gross_in_a_bad_market_only_when_on():
    sc, er, rows = world()
    bad = day(rows, breadth_200=0.2, mkt_ret_21=-0.05)
    on = plan_s1(T, sc, er, book(), 5e6, bad, cfg(overlay=True))
    off = plan_s1(T, sc, er, book(), 5e6, bad, cfg(overlay=False))
    ok = plan_s1(T, sc, er, book(), 5e6, day(rows, breadth_200=0.5, mkt_ret_21=-0.05), cfg(overlay=True))
    assert off.gross_target == pytest.approx(1.0) and ok.gross_target == pytest.approx(1.0)
    assert on.gross_target == pytest.approx(0.5)


def test_drawdown_12_percent_halves_and_16_percent_zeroes_gross():
    sc, er, rows = world()
    base = plan_s1(T, sc, er, book(), 5e6, day(rows), cfg())
    half = plan_s1(T, sc, er, book(dd=0.12), 5e6, day(rows), cfg())
    stop = plan_s1(T, sc, er, book(dd=0.16), 5e6, day(rows), cfg())
    flag = plan_s1(T, sc, er, book(dd=0.0, stopped=True), 5e6, day(rows), cfg())
    assert base.gross_target == pytest.approx(1.0)
    assert half.gross_target == pytest.approx(0.5)
    assert stop.gross_target == 0.0 and stop.k == 0 and buys(stop) == []
    assert flag.gross_target == 0.0 and buys(flag) == []


# ---- exits --------------------------------------------------------------------------------------------


def test_holding_at_h_minus_1_in_top_k_is_extended_not_sold():
    sc, er, rows = world()
    h = Holding(0, 500, 100.0, held=H - 1, vol_at_entry=0.01)  # sid 0 is the best name
    plan = plan_s1(T, sc, er, book([h]), 5e6, day(rows), cfg())
    assert plan.extended == [0]
    assert 0 not in plan.exits and all(o.sid != 0 for o in plan.orders)


def test_holding_at_h_minus_1_outside_top_k_is_sold_at_the_horizon():
    sc, er, rows = world()
    h = Holding(8, 500, 100.0, held=H - 1, vol_at_entry=0.01)  # rank 9 of 14, K = 10: in top K but...
    # make it clearly outside the top K: lower its score below everyone
    sc[8] = -0.5
    plan = plan_s1(T, sc, er, book([h]), 5e6, day(rows), cfg(k_max=5))
    assert plan.exits == {8: "horizon"}
    o = sells(plan)[0]
    assert (o.sid, o.when, o.trade_date, o.qty) == (8, "open", D1, 500)


def test_extension_needs_the_name_to_stay_eligible():
    sc, er, rows = world()
    er[0] = 0.0001  # below costs: no longer eligible, so it is not extended
    h = Holding(0, 500, 100.0, held=H - 1, vol_at_entry=0.01)
    plan = plan_s1(T, sc, er, book([h]), 5e6, day(rows), cfg())
    assert plan.exits == {0: "horizon"} and plan.extended == []


def test_rank_decay_after_two_sessions_outside_3k():
    sc, er, rows = world(n=14)
    k = 3  # k_max
    deep = Holding(12, 500, 100.0, held=2, vol_at_entry=0.01)  # rank 13 > 3K = 9
    fresh = Holding(13, 500, 100.0, held=1, vol_at_entry=0.01)  # rank 14 but held only 1
    plan = plan_s1(T, sc, er, book([deep, fresh]), 5e6, day(rows), cfg(k_max=k))
    assert plan.exits == {12: "rank_decay"}


def test_stop_loss_uses_the_larger_of_the_floor_and_the_vol_rule():
    sc, er, rows = world()
    # vol_at_entry 0.02 * sqrt(5) * 3 = 0.134 -> stop at 86.6; floor 0.08 -> 92
    rows[0] = row(close=90.0, industry="I0")
    h = Holding(0, 100, 100.0, held=1, vol_at_entry=0.02)
    assert 0 not in plan_s1(T, sc, er, book([h]), 5e6, day(rows), cfg()).exits
    rows[0] = row(close=86.0, industry="I0")
    assert plan_s1(T, sc, er, book([h]), 5e6, day(rows), cfg()).exits == {0: "stop"}
    h2 = Holding(0, 100, 100.0, held=1, vol_at_entry=0.001)  # floor 8% binds
    rows[0] = row(close=91.9, industry="I0")
    assert plan_s1(T, sc, er, book([h2]), 5e6, day(rows), cfg()).exits == {0: "stop"}


def test_universe_exit_when_dropped_or_stale():
    sc, er, rows = world()
    gone = Holding(99, 10, 100.0, held=0, vol_at_entry=0.01)
    stale_h = Holding(1, 10, 100.0, held=0, vol_at_entry=0.01)
    rows[1] = row(stale=True, industry="I1")
    plan = plan_s1(T, sc, er, book([gone, stale_h]), 5e6, day(rows), cfg())
    assert plan.exits == {99: "universe_exit", 1: "universe_exit"}


def test_a_name_sold_today_is_not_bought_back():
    sc, er, rows = world()
    h = Holding(0, 500, 100.0, held=H - 1, vol_at_entry=0.01)
    er[0] = 0.0001
    plan = plan_s1(T, sc, er, book([h]), 5e6, day(rows), cfg())
    assert plan.exits == {0: "horizon"} and all(o.sid != 0 for o in buys(plan))


# ---- entries ------------------------------------------------------------------------------------------


def test_industry_cap_blocks_the_fourth_name_of_one_industry_at_k10():
    inds = ["Banks"] * 6 + [f"I{i}" for i in range(6, 20)]
    sc, er, rows = world(n=20, industries=inds)
    plan = plan_s1(T, sc, er, book(), 5e7, day(rows), cfg(k_max=10))
    assert plan.k == 10
    chosen = [o.sid for o in buys(plan)]
    assert sum(1 for s in chosen if s < 6) == 3  # 30% of 10
    assert len(chosen) == 10
    assert chosen[:3] == [0, 1, 2] and 3 not in chosen  # the 4th bank is skipped, the next industry fills in


def test_unknown_industry_is_capped_like_any_other():
    sc, er, rows = world(n=20, industries=["UNKNOWN"] * 20)
    plan = plan_s1(T, sc, er, book(), 5e7, day(rows), cfg(k_max=10))
    assert len(buys(plan)) == 3


def test_name_with_results_tomorrow_is_not_entered():
    sc, er, rows = world()
    rows[0] = row(industry="I0", days_to_results=1)
    plan = plan_s1(T, sc, er, book(), 5e6, day(rows), cfg())
    assert 0 not in [o.sid for o in buys(plan)] and 1 in [o.sid for o in buys(plan)]


def test_name_with_a_split_ex_date_within_two_sessions_is_not_entered():
    sc, er, rows = world()
    rows[0] = row(industry="I0", days_to_split=2)
    rows[1] = row(industry="I1", days_to_split=3)
    got = [o.sid for o in buys(plan_s1(T, sc, er, book(), 5e6, day(rows), cfg()))]
    assert 0 not in got and 1 in got


def test_locked_up_and_stale_names_are_not_entered():
    sc, er, rows = world()
    rows[0] = row(industry="I0", locked_up=True)
    rows[1] = row(industry="I1", stale=True)
    got = [o.sid for o in buys(plan_s1(T, sc, er, book(), 5e6, day(rows), cfg()))]
    assert 0 not in got and 1 not in got


def test_expected_return_below_costs_is_not_entered():
    sc, er, rows = world()
    er[0] = 0.002  # a 0.4% round trip at Rs 5 lakh costs more than that
    plan = plan_s1(T, sc, er, book(), 5e6, day(rows), cfg())
    assert 0 not in [o.sid for o in buys(plan)]
    # and with every er below costs: nothing, with the message
    none = plan_s1(T, sc, {i: 0.001 for i in er}, book(), 5e6, day(rows), cfg())
    assert buys(none) == [] and none.note == "no stock clears costs today"


def test_two_times_slippage_is_part_of_the_entry_test():
    sc, er, rows = world()
    from datetime import date as d

    from stocksense.costs.slippage import SlippageRow, SlippageTable

    wide = SlippageTable([SlippageRow(d(2000, 1, 1), 0.0, 0.0, 0.0, ((500, 200.0),))])  # 2% half spread
    c = S1Config.from_settings(S, H, wide, k_max=10)
    assert buys(plan_s1(T, sc, {i: 0.05 for i in er}, book(), 5e6, day(rows), c))  # 5% > 4% + costs
    assert not buys(plan_s1(T, sc, {i: 0.04 for i in er}, book(), 5e6, day(rows), c))  # 4% < 4% + costs


def test_held_names_are_not_bought_again_and_use_up_slots():
    sc, er, rows = world()
    h = Holding(0, 500, 100.0, held=1, vol_at_entry=0.005)
    plan = plan_s1(T, sc, er, book([h]), 5e6, day(rows), cfg(k_max=3))
    assert [o.sid for o in buys(plan)] == [1, 2]  # K = 3, one slot is taken


# ---- weights and orders ---------------------------------------------------------------------------------


@given(
    raw=st.lists(st.floats(0.0, 400.0, allow_nan=False), min_size=1, max_size=30),
    room=st.floats(0.01, 1.0),
    cap=st.floats(0.01, 0.5),
)
@hsettings(max_examples=300, deadline=None)
def test_allocated_weights_never_exceed_the_cap_or_the_room(raw, room, cap):
    w = allocate_weights({i: v for i, v in enumerate(raw)}, room, cap)
    assert all(0.0 <= x <= cap + 1e-12 for x in w.values())
    assert sum(w.values()) <= room + 1e-9


@given(
    scores=st.lists(st.floats(-0.5, 0.5), min_size=12, max_size=25),
    vols=st.lists(st.floats(0.002, 0.08), min_size=12, max_size=25),
    equity=st.floats(5e4, 5e7),
)
@hsettings(max_examples=150, deadline=None)
def test_plan_weights_respect_caps_and_gross(scores, vols, equity):
    n = min(len(scores), len(vols))
    sc = {i: scores[i] for i in range(n)}
    er = {i: 0.2 for i in range(n)}  # always clears costs
    rows = {i: row(vol_21=vols[i], industry=f"I{i}") for i in range(n)}
    plan = plan_s1(T, sc, er, book(), equity, day(rows), cfg(k_max=10))
    assert sum(plan.weights.values()) <= plan.gross_target + 1e-9
    assert all(w <= plan.cap + 1e-12 for w in plan.weights.values())
    assert plan.cap <= max(S.risk.max_name_weight, plan.min_pos / equity) + 1e-12
    assert len(plan.weights) <= plan.k


def test_weights_are_inverse_vol_tilted_by_score():
    sc, er, rows = world(n=14)
    rows[0] = row(industry="I0", vol_21=0.01)
    rows[1] = row(industry="I1", vol_21=0.005)
    plan = plan_s1(T, sc, er, book(), 1e8, day(rows), cfg(k_max=10))
    w0, w1 = plan.weights[0], plan.weights[1]
    # raw = (0.5 + score) / vol: sid0 0.9/0.01 = 90, sid1 0.85/0.005 = 170 (before any cap)
    assert w1 / w0 == pytest.approx(170 / 90, rel=1e-6) or w0 == pytest.approx(plan.cap)


def test_orders_are_integer_close_buys_for_the_next_session_within_expected_cash():
    sc, er, rows = world(n=14, close=137.3)
    h = Holding(13, 1000, 137.3, held=H - 1, vol_at_entry=0.01)  # will be sold at the horizon
    cash = 400_000.0
    plan = plan_s1(T, sc, er, book([h], cash=cash), 1_000_000.0, day(rows), cfg(k_max=10))
    bs = buys(plan)
    assert bs and all(isinstance(o.qty, int) and o.qty > 0 for o in bs)
    assert all(o.when == "close" and o.trade_date == D1 and o.product == "delivery" for o in bs)
    assert all(o.priority == pytest.approx(sc[o.sid]) for o in bs)
    expected_cash = cash + 1000 * 137.3
    assert sum(o.qty * 137.3 for o in bs) <= expected_cash
    # and the same with cash far too small: the buy list shrinks
    tight = plan_s1(T, sc, er, book([h], cash=1_000.0), 1_000_000.0, day(rows), cfg(k_max=10))
    assert sum(o.qty * 137.3 for o in buys(tight)) <= 1_000.0 + 1000 * 137.3


def test_names_whose_weight_is_below_one_minimum_position_are_dropped_to_cash():
    sc, er, rows = world(n=14)
    # Rs 1 lakh, K = 2 -> each about Rs 48,000; give sid 1 a tiny score so its share falls below min_pos
    sc[0], sc[1] = 0.5, -0.45
    plan = plan_s1(T, sc, er, book(), 100_000.0, day(rows), cfg())
    assert 1 not in plan.weights and 0 in plan.weights


def test_no_scores_means_no_entries_but_stops_and_horizon_still_exit():
    _, _, rows = world()
    h = Holding(0, 100, 100.0, held=H - 1, vol_at_entry=0.01)
    plan = plan_s1(T, {}, {}, book([h]), 5e6, day(rows), cfg())
    assert buys(plan) == [] and plan.exits == {0: "horizon"}


def test_entries_can_be_blocked_by_the_risk_checks_while_exits_run():
    sc, er, rows = world()
    h = Holding(0, 100, 100.0, held=H - 1, vol_at_entry=0.01)
    er0 = dict(er)
    er0[0] = 0.0001
    plan = plan_s1(T, sc, er0, book([h]), 5e6, day(rows), cfg(), allow_entries=False)
    assert buys(plan) == [] and plan.exits == {0: "horizon"}


def test_build_s1_orders_returns_the_plans_orders():
    sc, er, rows = world()
    o = build_s1_orders(T, sc, er, book(), 5e6, day(rows), cfg())
    assert o == plan_s1(T, sc, er, book(), 5e6, day(rows), cfg()).orders


def test_naive_daily_rebalancing_sells_anything_outside_the_top_k():
    sc, er, rows = world()
    h = Holding(6, 100, 100.0, held=1, vol_at_entry=0.01)  # rank 7: inside 3K = 15 but outside K = 5
    kw = dict(k_max=5)
    assert plan_s1(T, sc, er, book([h]), 5e6, day(rows), cfg(**kw)).exits == {}
    assert plan_s1(T, sc, er, book([h]), 5e6, day(rows), cfg(hysteresis=False, **kw)).exits == {
        6: "rank_decay"
    }
