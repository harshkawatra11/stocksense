from __future__ import annotations

import pytest

from stocksense.costs.india import min_position_for_fixed_cost, round_trip_cost_frac, trade_cost


@pytest.fixture
def sched(tmp_data_root):
    return tmp_data_root.costs


def approx(x):
    return pytest.approx(x, abs=0.001)


def test_delivery_buy_100k(sched):
    # brokerage min(20, 100) = 20; stt 100; stamp 15; exchange 2.97; sebi 0.10;
    # gst 0.18 * (20 + 2.97 + 0.10) = 4.1526; dp 0 -> 142.2226
    c = trade_cost("BUY", "delivery", 100_000, sched)
    assert c.brokerage == approx(20.0)
    assert c.stt == approx(100.0)
    assert c.stamp == approx(15.0)
    assert c.exchange == approx(2.97)
    assert c.sebi == approx(0.10)
    assert c.gst == approx(4.1526)
    assert c.dp == 0.0
    assert c.total == approx(142.2226)


def test_delivery_sell_105k(sched):
    # 20 + 105 + 0 + 3.1185 + 0.105 + dp 20 + gst 0.18 * 43.2235 (= 7.78023) -> 156.00373
    c = trade_cost("SELL", "delivery", 105_000, sched)
    assert c.stt == approx(105.0)
    assert c.stamp == 0.0
    assert c.exchange == approx(3.1185)
    assert c.sebi == approx(0.105)
    assert c.dp == approx(20.0)
    assert c.gst == approx(7.78023)
    assert c.total == approx(156.00373)


def test_brokerage_minimum_and_percentage(sched):
    # 0.1% of 4,000 = 4 -> raised to the minimum 5
    assert trade_cost("BUY", "delivery", 4_000, sched).brokerage == approx(5.0)
    # 0.1% of 10,000 = 10 (between min and cap)
    assert trade_cost("BUY", "delivery", 10_000, sched).brokerage == approx(10.0)


def test_intraday_buy_and_sell_50k(sched):
    # buy: 20 + 0 + 1.50 + 1.485 + 0.05 + gst 0.18 * 21.535 (= 3.8763) -> 26.9113
    b = trade_cost("BUY", "intraday", 50_000, sched)
    assert b.stt == 0.0
    assert b.stamp == approx(1.5)
    assert b.total == approx(26.9113)
    # sell: 20 + stt 12.50 + 1.485 + 0.05 + 3.8763 -> 37.9113, no dp for intraday
    s = trade_cost("SELL", "intraday", 50_000, sched)
    assert s.stt == approx(12.5)
    assert s.dp == 0.0
    assert s.total == approx(37.9113)
    # round trip 64.8226 / 50,000 = 12.96 bps
    assert round_trip_cost_frac("intraday", 50_000, sched) == pytest.approx(64.8226 / 50_000, abs=1e-7)


def test_second_sell_of_scrip_same_day_has_no_dp(sched):
    c = trade_cost("SELL", "delivery", 105_000, sched, first_sell_of_scrip_today=False)
    # 156.00373 - dp 20 - gst on dp 3.6 = 132.40373
    assert c.dp == 0.0
    assert c.total == approx(132.40373)


def test_round_trip_falls_with_size(sched):
    # 10,000: buy 23.66226 + sell 45.76226 = 69.42452 -> 0.0069425
    # 100,000: 142.2226 + 150.8226 (sell: 20+100+2.97+0.1+20+gst 7.7526) = 293.0452 -> 0.0029305
    small = round_trip_cost_frac("delivery", 10_000, sched)
    big = round_trip_cost_frac("delivery", 100_000, sched)
    assert small == pytest.approx(0.0069425, abs=1e-6)
    assert big == pytest.approx(0.0029305, abs=1e-6)
    assert small > big


def test_min_position_for_fixed_cost(sched):
    # (2*20 + 20) * 1.18 = 70.8; 70.8 / 0.0015 = 47,200 -> first 1,000 step is 48,000
    v = min_position_for_fixed_cost("delivery", sched, 15)
    assert v == 48_000
    assert 45_000 <= v <= 50_000


def test_bad_side(sched):
    with pytest.raises(ValueError):
        trade_cost("HOLD", "delivery", 1000, sched)  # type: ignore[arg-type]
