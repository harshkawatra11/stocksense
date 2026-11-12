from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from stocksense.costs.india import min_position_for_fixed_cost
from stocksense.portfolio.construct import (
    AuctionRow,
    S2Config,
    _split_sides,
    build_s2_orders,
    intraday_stop_hit,
    plan_s2,
)
from tests.fixtures.engine_panel import settings, zero_slippage

S = settings()
D = date(2020, 3, 3)


def cfg(k_max: int = 10, fno: bool = True) -> S2Config:
    c = S2Config.from_settings(S, zero_slippage(), k_max=k_max)
    return c if fno else S2Config(c.k_max, c.risk, c.costs, c.slippage, shorts_require_fno=False)


def arow(**kw) -> AuctionRow:
    base = dict(open=100.0, gap=0.0, turnover_rank=20, med_turn_20=1e10, vol_21=0.01, in_fno=True)
    base.update(kw)
    return AuctionRow(**base)


def book_of(n_long: int, n_short: int, er: float = 0.02):
    """sids 0.. are long candidates (er > 0), then short candidates (er < 0); all F&O."""
    auc, sc, ers = {}, {}, {}
    for i in range(n_long):
        auc[i], sc[i], ers[i] = arow(), 0.4 - 0.01 * i, er
    for j in range(n_short):
        s = 100 + j
        auc[s], sc[s], ers[s] = arow(), -0.4 + 0.01 * j, -er
    return sc, ers, auc


def test_min_position_for_intraday_is_about_31500():
    assert min_position_for_fixed_cost("intraday", S.costs, S.risk.max_fixed_cost_bps) == pytest.approx(
        31_500, abs=2000
    )
    assert cfg().min_pos >= 10_000


def test_budget_below_one_min_pos_gives_no_orders():
    sc, er, auc = book_of(5, 5)
    p = plan_s2(D, sc, er, auc, cfg().min_pos - 1.0, 1e6, cfg())
    assert p.orders == [] and p.n == 0 and "budget" in p.note


def test_n_is_min_of_kmax_and_budget_over_min_pos():
    sc, er, auc = book_of(10, 10)
    c = cfg()
    assert plan_s2(D, sc, er, auc, 3.2 * c.min_pos, 1e7, c).n == 3
    assert plan_s2(D, sc, er, auc, 100 * c.min_pos, 1e8, c).n == 10
    assert plan_s2(D, sc, er, auc, 100 * c.min_pos, 1e8, cfg(k_max=4)).n == 4


def test_budget_uses_free_cash_fraction_of_risk_config():
    sc, er, auc = book_of(10, 0)
    c = cfg()
    half_risk = S.risk.model_copy(update={"s2_free_cash_frac": 0.5})
    c2 = S2Config(c.k_max, half_risk, c.costs, c.slippage)
    assert plan_s2(D, sc, er, auc, 10e5, 1e8, c).budget == pytest.approx(10e5)
    assert plan_s2(D, sc, er, auc, 10e5, 1e8, c2).budget == pytest.approx(5e5)


def test_no_short_for_a_non_fno_name():
    sc, er, auc = book_of(0, 3)
    auc[100] = arow(in_fno=False)
    orders = build_s2_orders(D, sc, er, auc, 1e7, 1e8, cfg())
    assert sorted(o.sid for o in orders) == [101, 102] and all(o.side == "SELL" for o in orders)
    # with the F&O requirement off the non-F&O name is allowed
    assert 100 in [o.sid for o in build_s2_orders(D, sc, er, auc, 1e7, 1e8, cfg(fno=False))]


def test_nothing_when_no_name_clears_costs():
    sc, er, auc = book_of(5, 5, er=0.0002)  # 2 bps against a ~4 bp round trip at Rs 10 lakh
    p = plan_s2(D, sc, er, auc, 1e7, 1e8, cfg())
    assert p.orders == [] and p.note == "no stock clears costs today"


def test_two_times_open_slippage_is_part_of_the_cost():
    from stocksense.costs.slippage import SlippageRow, SlippageTable

    wide = SlippageTable([SlippageRow(date(2000, 1, 1), 0.0, 0.0, 0.0, ((500, 100.0),))])  # 1% half spread
    c = S2Config.from_settings(S, wide)
    sc, er, auc = book_of(3, 0, er=0.015)  # 1.5% < 2 x 1% + costs; zero-slip config would take it
    assert plan_s2(D, sc, er, auc, 1e7, 1e8, c).orders == []
    assert plan_s2(D, sc, er, auc, 1e7, 1e8, cfg()).orders


def test_orders_are_intraday_at_the_open_of_d_with_integer_quantities():
    sc, er, auc = book_of(2, 2)
    orders = build_s2_orders(D, sc, er, auc, 1e7, 1e8, cfg())
    assert len(orders) == 4
    for o in orders:
        assert (o.when, o.product, o.trade_date) == ("open", "intraday", D)
        assert isinstance(o.qty, int) and o.qty > 0
    assert {o.sid for o in orders if o.side == "BUY"} == {0, 1}
    assert {o.sid for o in orders if o.side == "SELL"} == {100, 101}


def test_universe_is_u200_only():
    sc, er, auc = book_of(3, 0)
    auc[0] = arow(turnover_rank=201)
    assert 0 not in [o.sid for o in build_s2_orders(D, sc, er, auc, 1e7, 1e8, cfg())]


def test_shorts_skip_names_at_the_upper_band_or_gapping_up_18_percent():
    sc, er, auc = book_of(0, 4)
    auc[100] = arow(gap=0.18)
    auc[101] = arow(upper_circuit=101.0)  # open within 2% of the upper circuit
    auc[102] = arow(upper_circuit=110.0)
    got = sorted(o.sid for o in build_s2_orders(D, sc, er, auc, 1e7, 1e8, cfg()))
    assert got == [102, 103]


def test_side_split_rule():
    assert _split_sides(10, 8, 2) == (5, 2)  # at most ceil(n/2) per side
    assert _split_sides(10, 4, 4) == (4, 4)  # everyone fits
    assert _split_sides(10, 20, 0) == (10, 0)  # the other side is empty: no per-side limit
    assert _split_sides(10, 0, 3) == (0, 3)
    assert _split_sides(4, 10, 10) == (2, 2)
    assert _split_sides(5, 6, 2) == (3, 2)
    assert _split_sides(7, 30, 10) == (4, 3)  # proportional share 5.25 -> 5, limited to ceil(7/2) = 4
    assert _split_sides(0, 3, 3) == (0, 0)


def test_split_in_a_plan_is_proportional_and_capped_per_side():
    sc, er, auc = book_of(8, 2)
    p = plan_s2(D, sc, er, auc, 100 * cfg().min_pos, 1e8, cfg())
    assert (p.n_long_qualify, p.n_short_qualify) == (8, 2)
    assert (p.n_long, p.n_short) == (5, 2)


def test_positions_are_equal_value_and_cash_covers_shorts_at_120_percent():
    sc, er, auc = book_of(3, 3)
    c = cfg()
    budget = 10 * c.min_pos
    orders = build_s2_orders(D, sc, er, auc, budget, 1e9, c)
    values = [o.qty * 100.0 for o in orders]
    assert max(values) - min(values) <= 100.0  # equal up to one share
    locked = sum(v * (1.2 if o.side == "SELL" else 1.0) for o, v in zip(orders, values, strict=True))
    assert locked <= budget + 1e-6


def test_a_lone_name_is_capped_at_the_name_weight():
    sc, er, auc = book_of(1, 0)
    o = build_s2_orders(D, sc, er, auc, 1e7, 1e7, cfg())[0]
    assert o.qty * 100.0 <= S.risk.max_name_weight * 1e7 + 100.0


def test_intraday_stop_hit_long_and_short():
    long_ = SimpleNamespace(qty=100, entry_price=100.0)
    short = SimpleNamespace(qty=-100, entry_price=100.0)
    assert intraday_stop_hit(long_, 97.0) and intraday_stop_hit(long_, 90.0)
    assert not intraday_stop_hit(long_, 97.01) and not intraday_stop_hit(long_, 110.0)
    assert intraday_stop_hit(short, 103.0) and not intraday_stop_hit(short, 102.99)
    assert not intraday_stop_hit(short, 90.0)
    assert not intraday_stop_hit(SimpleNamespace(qty=0, entry_price=100.0), 50.0)
