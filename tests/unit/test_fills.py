from __future__ import annotations

from datetime import date

import pytest

from stocksense.backtest.fills import DayBar, Fill, FillContext, Order, Reject, try_fill
from stocksense.core.config import load_settings
from stocksense.costs.slippage import SlippageTable
from tests.fixtures.synth import make_market


@pytest.fixture
def ctx(tmp_data_root):
    return FillContext(
        turnover_rank=120,
        vol_21=0.02,
        med_turn_20=1e8,
        slippage=SlippageTable.load_latest(tmp_data_root),
        costs=tmp_data_root.costs,
        capital=500_000.0,
        max_adv_participation=0.01,
    )


def bar(o=100.0, h=105.0, lo=99.0, c=102.0, prev=98.0, **flags) -> DayBar:
    return DayBar(
        open=o, high=h, low=lo, close=c, prev_base=prev, ret_co=o / prev - 1, ret=c / prev - 1, **flags
    )


def order(side="BUY", when="open", qty=500, product="delivery") -> Order:
    return Order(sid=1, side=side, product=product, when=when, qty=qty, trade_date=date(2026, 11, 12))


def test_buy_open_fill_price_cost_and_tick(ctx):
    # slip = 0.0004 + 0.5*0.02*sqrt(50000/1e8) + 0.0003 = 0.000923607 -> 100 * (1+slip) = 100.0923607
    # BUY rounds UP to the 0.05 tick -> 100.10; value 500 * 100.10 = 50,050
    # cost: 20 + stt 50.05 + stamp 7.5075 + exch 1.486485 + sebi 0.05005 + gst 0.18*21.536535 (3.8765763)
    # = 82.9706113
    f = try_fill(order(), bar(), ctx)
    assert isinstance(f, Fill)
    assert f.qty == 500
    assert f.price == pytest.approx(100.10)
    assert f.slip == pytest.approx(0.000923607, abs=1e-8)
    assert f.cost.total == pytest.approx(82.9706113, abs=1e-4)


def test_sell_rounds_down(ctx):
    # 100 * (1 - 0.000923607) = 99.9076393 -> floor to tick 99.90
    f = try_fill(order("SELL"), bar(), ctx)
    assert isinstance(f, Fill) and f.price == pytest.approx(99.90)


def test_close_uses_close_price_and_penalty(ctx):
    # close 102, penalty 2 bps: slip = 0.0004 + 0.01*sqrt(51000/1e8)(= 0.000225832) + 0.0002 = 0.000825832
    # 102 * 1.000825832 = 102.0842 -> up to 102.10
    f = try_fill(order(when="close"), bar(), ctx)
    assert isinstance(f, Fill) and f.price == pytest.approx(102.10)


def test_no_trade(ctx):
    r = try_fill(order(), None, ctx)
    assert isinstance(r, Reject) and r.reason == "no_trade"


def test_one_price_both_sides(ctx):
    b = bar(o=105.0, h=105.0, lo=105.0, c=105.0, flag_one_price=True, flag_locked_up=True)
    for side in ("BUY", "SELL"):
        for when in ("open", "close"):
            r = try_fill(order(side, when), b, ctx)
            assert isinstance(r, Reject) and r.reason == "one_price"


def test_locked_up_open_blocks_buy_only(ctx):
    # prev 100, open = high = 105 -> ret_co = 0.05, |0.05 - 0.05| = 0 <= 0.0025
    b = bar(o=105.0, h=105.0, lo=100.0, c=103.0, prev=100.0)
    r = try_fill(order("BUY"), b, ctx)
    assert isinstance(r, Reject) and r.reason == "locked_up_open"
    assert isinstance(try_fill(order("SELL"), b, ctx), Fill)


def test_locked_up_open_tolerance(ctx):
    # ret_co = 0.0524: |0.0524 - 0.05| = 0.0024 -> still locked; 0.053 -> 0.003 > 0.0025 -> fills
    locked = bar(o=105.24, h=105.24, lo=100.0, c=103.0, prev=100.0)
    free = bar(o=105.3, h=105.3, lo=100.0, c=103.0, prev=100.0)
    assert isinstance(try_fill(order("BUY"), locked, ctx), Reject)
    assert isinstance(try_fill(order("BUY"), free, ctx), Fill)


def test_locked_dn_open_blocks_sell_only(ctx):
    # prev 100, open = low = 95 -> ret_co = -0.05
    b = bar(o=95.0, h=97.0, lo=95.0, c=96.0, prev=100.0)
    r = try_fill(order("SELL"), b, ctx)
    assert isinstance(r, Reject) and r.reason == "locked_dn_open"
    assert isinstance(try_fill(order("BUY"), b, ctx), Fill)


def test_locked_close_flags(ctx):
    up = bar(flag_locked_up=True)
    dn = bar(flag_locked_dn=True)
    r1 = try_fill(order("BUY", "close"), up, ctx)
    r2 = try_fill(order("SELL", "close"), dn, ctx)
    assert isinstance(r1, Reject) and r1.reason == "locked_up_close"
    assert isinstance(r2, Reject) and r2.reason == "locked_dn_close"
    assert isinstance(try_fill(order("SELL", "close"), up, ctx), Fill)
    assert isinstance(try_fill(order("BUY", "close"), dn, ctx), Fill)


def test_participation_cap_reduces_quantity(ctx):
    # med_turn_20 = 1e6 -> max value 1e4 -> floor(1e4 / 100) = 100 shares
    c = FillContext(**{**ctx.__dict__, "med_turn_20": 1e6})
    f = try_fill(order(qty=500), bar(), c)
    assert isinstance(f, Fill) and f.qty == 100


def test_too_small(ctx):
    # med_turn_20 = 5,000 -> max value 50 -> floor(50 / 100) = 0
    c = FillContext(**{**ctx.__dict__, "med_turn_20": 5_000.0})
    r = try_fill(order(), bar(), c)
    assert isinstance(r, Reject) and r.reason == "too_small"


def test_second_sell_has_no_dp(ctx):
    c = FillContext(**{**ctx.__dict__, "first_sell_of_scrip_today": False})
    f = try_fill(order("SELL"), bar(), c)
    assert isinstance(f, Fill) and f.cost.dp == 0.0


def test_s15_locked_sessions_block_buys():
    # S15 trades with open = high = low = close (upper circuit) on sessions 450-452
    m = make_market()
    for k in (450, 451, 452):
        rows = m.eq.filter((m.eq["symbol"] == "S15") & (m.eq["date"] == m.sessions[k]))
        r = rows.row(0, named=True)
        assert r["open"] == r["high"] == r["low"] == r["close"]
        b = DayBar(
            open=r["open"],
            high=r["high"],
            low=r["low"],
            close=r["close"],
            prev_base=r["prev_close"],
            ret_co=r["open"] / r["prev_close"] - 1,
            ret=r["close"] / r["prev_close"] - 1,
            flag_one_price=r["high"] == r["low"],
            flag_locked_up=True,
        )
        s = load_settings()
        c = FillContext(120, 0.02, 1e8, SlippageTable.load_latest(s), s.costs, 5e5)
        for when in ("open", "close"):
            res = try_fill(order("BUY", when), b, c)
            assert isinstance(res, Reject) and res.reason == "one_price"
