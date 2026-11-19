"""G2-C1: S1 policy options. Every option defaults to the pre-G2 behaviour (golden test)."""

from __future__ import annotations

from datetime import date

import pytest

from stocksense.costs.india import round_trip_cost_frac
from stocksense.portfolio.construct import (
    Holding,
    NameRow,
    PanelDay,
    S1Book,
    S1Config,
    plan_s1,
)
from tests.fixtures.engine_panel import settings, zero_slippage

S = settings()
T, D1 = date(2020, 3, 2), date(2020, 3, 3)
H = 5
N = 30
EQUITY = 250_000.0  # k = floor(250000 / 48000) = 5


def cfg(**opts) -> S1Config:
    return S1Config.from_settings(S, H, zero_slippage(), k_max=5, **opts)


def world(n: int = N, er_default: float = 0.05):
    """sid i has score 0.45 - 0.03 i, so rank = sid + 1; every name in its own industry."""
    scores = {i: 0.45 - 0.03 * i for i in range(n)}
    er = {i: er_default for i in range(n)}
    rows = {
        i: NameRow(close=100.0, vol_21=0.005, turnover_rank=10, med_turn_20=1e10, industry=f"I{i}")
        for i in range(n)
    }
    return scores, er, rows


def hold(sid: int, held: int) -> Holding:
    return Holding(sid, 500, 100.0, held, 0.01)  # Rs 50,000 each


def book(holdings=(), recent_sells=None, cash=1e9) -> S1Book:
    b = S1Book({h.sid: h for h in holdings}, cash)
    if recent_sells is not None:
        b = S1Book(b.holdings, cash, recent_sells=recent_sells)
    return b


def plan(holdings, sc, er, rows, cfg_=None, equity=EQUITY, recent_sells=None):
    return plan_s1(T, sc, er, book(holdings, recent_sells), equity, PanelDay(T, D1, rows), cfg_ or cfg())


# ---- (1) defaults reproduce the pre-change plan -----------------------------------------------------


def test_defaults_reproduce_the_golden_plan():
    """Golden computed with the code as it was before C1 (k=5, 30 names, six holdings)."""
    sc, er, rows = world()
    hs = [hold(2, 3), hold(14, 3), hold(16, 3), hold(20, 1), hold(1, 4), hold(8, 4)]
    p = plan(hs, sc, er, rows)
    assert p.k == 5
    assert p.exits == {8: "horizon", 16: "rank_decay"}
    assert p.extended == [1]
    assert list(p.weights) == [0] and p.weights[0] == pytest.approx(0.192)
    got = [(o.sid, o.side, o.when, o.qty, o.reason) for o in p.orders]
    assert got == [
        (8, "SELL", "open", 500, "horizon"),
        (16, "SELL", "open", 500, "rank_decay"),
        (0, "BUY", "close", 480, "entry"),
    ]


# ---- (2) exit_rank ----------------------------------------------------------------------------------


def test_exit_rank_replaces_three_k():
    sc, er, rows = world()
    p = plan([hold(7, 3), hold(11, 3)], sc, er, rows, cfg(exit_rank=10))  # ranks 8 and 12
    assert p.exits == {11: "rank_decay"}
    # default 3k = 15 keeps both
    assert plan([hold(7, 3), hold(11, 3)], sc, er, rows).exits == {}


# ---- (3) min_hold -----------------------------------------------------------------------------------


def test_min_hold_delays_rank_decay():
    sc, er, rows = world()
    assert plan([hold(24, 3)], sc, er, rows, cfg(min_hold=5)).exits == {}  # rank 25, held 3
    p = plan([hold(24, 5)], sc, er, rows, cfg(min_hold=5))
    assert p.exits == {24: "rank_decay"}  # rank_decay is tested before the horizon


# ---- (4) free_extension -----------------------------------------------------------------------------


def test_free_extension_skips_the_cost_hurdle_but_needs_positive_er():
    sc, er, rows = world()
    cost = round_trip_cost_frac("delivery", 50_000.0, S.costs)
    assert cost > 0.003
    er[0] = 0.001
    h = [hold(0, H - 1)]  # best name, at the horizon
    assert plan(h, sc, er, rows).exits == {0: "horizon"}  # er 0.001 < round trip: not extended
    p = plan(h, sc, er, rows, cfg(free_extension=True))
    assert p.extended == [0] and p.exits == {}
    er[0] = -0.001
    p = plan(h, sc, er, rows, cfg(free_extension=True))
    assert p.exits == {0: "horizon"} and p.extended == []


def test_free_extension_still_respects_non_cost_filters():
    sc, er, rows = world()
    rows[0] = NameRow(100.0, 0.005, 10, 1e10, "I0", days_to_results=1)
    p = plan([hold(0, H - 1)], sc, er, rows, cfg(free_extension=True))
    assert p.exits == {0: "horizon"}


def test_extend_rank_widens_the_extension_window():
    sc, er, rows = world()
    h = [hold(6, H - 1)]  # rank 7 > k = 5
    assert plan(h, sc, er, rows).exits == {6: "horizon"}
    assert plan(h, sc, er, rows, cfg(extend_rank=8)).extended == [6]


# ---- (5) rebuy_block --------------------------------------------------------------------------------


def test_rebuy_block_skips_recently_sold_names():
    sc, er, rows = world()
    open_ = plan([], sc, er, rows)
    assert 0 in [o.sid for o in open_.orders if o.side == "BUY"]
    blocked = plan([], sc, er, rows, cfg(rebuy_block=5), recent_sells={0: 3})
    buys = [o.sid for o in blocked.orders if o.side == "BUY"]
    assert 0 not in buys and buys  # sid 0 skipped, the next names are bought
    old = plan([], sc, er, rows, cfg(rebuy_block=5), recent_sells={0: 6})
    assert 0 in [o.sid for o in old.orders if o.side == "BUY"]
    # off by default even when the book carries the info
    off = plan([], sc, er, rows, cfg(), recent_sells={0: 1})
    assert 0 in [o.sid for o in off.orders if o.side == "BUY"]


# ---- (6)/(7) switch_hurdle --------------------------------------------------------------------------


def _switch_world(k_hold: int, n_cand: int, er_cand: float, er_weak: float):
    """k_hold holdings (rank 1..k_hold, all weak), then candidates right below them."""
    sc, _, rows = world()
    er = {i: 0.001 for i in range(N)}  # below the round trip: ineligible unless set
    for i in range(k_hold):
        er[i] = er_weak
    for i in range(k_hold, k_hold + n_cand):
        er[i] = er_cand
    return sc, er, rows


def test_switch_when_the_gain_beats_cost_and_slippage():
    cost = round_trip_cost_frac("delivery", 50_000.0, S.costs)  # zero slippage in this config
    sc, er, rows = _switch_world(5, 1, er_cand=0.002 + cost + 0.004, er_weak=0.002)
    for i in range(4):
        er[i] = 0.05  # only sid 4 is weak
    h = [hold(i, 3) for i in range(5)]
    p = plan(h, sc, er, rows, cfg(switch_hurdle=True))
    assert p.exits == {4: "switch"}
    assert [o.sid for o in p.orders if o.side == "SELL"] == [4]
    assert [o.sid for o in p.orders if o.side == "BUY"] == [5]
    assert [o.when for o in p.orders if o.side == "SELL"] == ["open"]


def test_no_switch_when_the_gain_is_below_the_hurdle():
    cost = round_trip_cost_frac("delivery", 50_000.0, S.costs)
    sc, er, rows = _switch_world(5, 1, er_cand=0.002 + cost - 0.001, er_weak=0.002)
    for i in range(4):
        er[i] = 0.05
    h = [hold(i, 3) for i in range(5)]
    p = plan(h, sc, er, rows, cfg(switch_hurdle=True))
    assert p.exits == {} and p.orders == []
    # and off by default even with a huge gain
    sc, er, rows = _switch_world(5, 1, er_cand=0.05, er_weak=0.002)
    assert plan(h, sc, er, rows).orders == []


def test_switch_respects_min_hold():
    cost = round_trip_cost_frac("delivery", 50_000.0, S.costs)
    sc, er, rows = _switch_world(5, 1, er_cand=0.002 + cost + 0.004, er_weak=0.002)
    h = [hold(i, 1) for i in range(5)]  # all held 1 < min_hold 2
    assert plan(h, sc, er, rows, cfg(switch_hurdle=True)).exits == {}


def test_at_most_k_over_four_switches_a_day():
    k = 8
    equity = 400_000.0  # floor(400000 / 48000) = 8
    sc, er, rows = _switch_world(k, 4, er_cand=0.05, er_weak=0.001)
    h = [hold(i, 3) for i in range(k)]
    c8 = S1Config.from_settings(S, H, zero_slippage(), k_max=8, switch_hurdle=True)
    p = plan(h, sc, er, rows, c8, equity=equity)
    assert p.k == 8
    assert sorted(s for s, r in p.exits.items() if r == "switch") == [0, 1]  # 8 // 4 = 2, weakest first
    assert len([o for o in p.orders if o.side == "BUY"]) == 2


# ---- policy: the sell memory behind rebuy_block ------------------------------------------------------


def test_policy_remembers_sells_for_rebuy_block_sessions():
    from types import SimpleNamespace

    from stocksense.core.calendar import TradingCalendar
    from stocksense.portfolio.construct import DayScores, S1Policy
    from tests.fixtures.handmade import weekdays

    sess = weekdays(30)
    cal = TradingCalendar.from_dates(sess)
    sc, er, rows = world()

    class Src:
        def scores(self, t):
            return DayScores(sc, er)

    class Access:
        def day(self, t, nd):
            return PanelDay(t, nd, rows)

    pol = S1Policy(
        SimpleNamespace(target="y_c2o_5"), S, Src(), Access(), S.costs, zero_slippage(), k_max=5,
        policy_opts={"rebuy_block": 3},
    )  # fmt: skip
    held = SimpleNamespace(product="delivery", qty=500, entry_date=sess[0], entry_price=100.0)

    def run(i, positions):
        b = SimpleNamespace(calendar=cal, positions=positions, cash=1e9, reserved=0.0)
        pol.decide(sess[i], b, EQUITY)
        return [o.sid for o in pol.plans[sess[i]].orders if o.side == "BUY"]

    run(1, {0: held})  # sid 0 held
    assert 0 not in run(2, {})  # seen gone at session 2 (sold at its open): blocked
    assert 0 not in run(4, {})  # 2 sessions later
    assert 0 not in run(5, {})  # 3 sessions later: still within the block
    assert 0 in run(6, {})  # 4 sessions later: eligible again
