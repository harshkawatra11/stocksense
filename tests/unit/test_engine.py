from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as hsettings
from hypothesis import strategies as st

from stocksense.backtest.engine import (
    BuyAndHold,
    EqualWeightUniverse,
    PanelData,
    run_backtest,
    run_grid,
)
from stocksense.backtest.fills import Order
from stocksense.core.calendar import TradingCalendar
from tests.fixtures.engine_panel import dates, frame, settings, stock_rows, zero_slippage

S = settings()
CAP = 1_000_000.0
DS = dates(12)
CAL = TradingCalendar.from_dates(DS)
FLAT = [100.0] * 12
DS_IDX: dict[date, int] = {}


class Scripted:
    """Policy that returns canned orders keyed by the index of the decision date."""

    def __init__(self, script, auction=None):
        self.script = script
        self.auction = auction or {}

    def decide(self, t, book, capital_inr):
        return self.script.get(DS_IDX.get(t, -1), [])

    def decide_auction(self, d, book, capital_inr, view):
        return self.auction.get(DS_IDX.get(d, -1), [])


def run(panel_df, script, *, ds=DS, cap=CAP, auction=None, slip=None):
    DS_IDX.clear()
    DS_IDX.update({d: i for i, d in enumerate(ds)})
    return run_backtest(
        PanelData.from_frame(panel_df),
        TradingCalendar.from_dates(ds),
        Scripted(script, auction),
        ds[0],
        ds[-1],
        cap,
        S.costs,
        slip or zero_slippage(),
        S.risk,
        debug=True,
    )


def buy(sid, qty, day, when="open", product="delivery", priority=0.0):
    return Order(sid, "BUY", product, when, qty, trade_date=DS[day], priority=priority)


def sell(sid, qty, day, when="open", product="delivery"):
    return Order(sid, "SELL", product, when, qty, trade_date=DS[day])


def daily_at(res, day, ds=DS):
    return res.daily.filter(pl.col("date") == ds[day]).row(0, named=True)


def test_buy_and_hold_ten_sessions_exact():
    closes = [100.0 + i for i in range(12)]
    opens = [100.0] + closes[:-1]
    df = frame(stock_rows(1, DS, opens, closes))
    res = run(df, {0: [buy(1, 1000, 1)]})
    # buy 1000 @ open day1 = 100.00 (zero slippage) -> value 100,000; delivery BUY cost 142.2226 (T5.1 card)
    # day 11 close = 111 -> equity = 1,000,000 - 142.2226 + 1000 * (111 - 100) = 1,010,857.7774
    last = res.daily.row(-1, named=True)
    assert last["equity"] == pytest.approx(1_010_857.7774, abs=0.01)
    assert last["nav"] == pytest.approx(101.08577774, abs=1e-6)
    assert last["n_pos"] == 1
    assert res.fills.height == 1
    assert res.daily["costs_inr"].sum() == pytest.approx(142.2226, abs=1e-3)


def test_split_mid_holding_keeps_equity_apart_from_real_return():
    # flat 100, then a split with gap_factor 0.3 on day 6: raw open 30, close 31
    opens = [100.0] * 6 + [30.0] * 6
    closes = [100.0] * 6 + [31.0] * 6
    df = frame(stock_rows(1, DS, opens, closes, gap={6: 0.3}))
    res = run(df, {0: [buy(1, 1000, 1)]})
    # qty 1000 / 0.3 = 3333.33 -> keep 3333, remainder 0.3333 * prev_base (100 * 0.3 = 30) = 10.0 to cash
    # equity day 6 - day 5 = 3333 * 31 + 10 - 100000 = 3333.0 (the real return is 3333.33; the 0.33 is the
    # fractional share valued at 30 instead of 31)
    e5, e6 = daily_at(res, 5)["equity"], daily_at(res, 6)["equity"]
    assert e6 - e5 == pytest.approx(3333.0, abs=0.01)
    assert daily_at(res, 6)["n_pos"] == 1


def test_dividend_adds_cash():
    df = frame(stock_rows(1, DS, FLAT, FLAT, div={5: 5.0}))
    res = run(df, {0: [buy(1, 100, 1)]})
    # 100 shares x Rs 5 = 500 cash on day 5; the close is flat so equity rises by exactly 500
    assert daily_at(res, 5)["dividends_inr"] == pytest.approx(500.0, abs=1e-6)
    assert daily_at(res, 5)["equity"] - daily_at(res, 4)["equity"] == pytest.approx(500.0, abs=0.01)
    assert res.daily["dividends_inr"].sum() == pytest.approx(500.0, abs=1e-6)


def test_locked_sell_retried_until_first_unlocked_session():
    df = frame(stock_rows(1, DS, FLAT, FLAT, one_price={6}))
    res = run(df, {0: [buy(1, 100, 1)], 5: [sell(1, 100, 6)]})
    # day 6 is one-price: the open sell is rejected, retried at the close (rejected again), then it
    # fills at the day-7 open
    assert res.rejects["reason"].to_list() == ["one_price", "one_price"]
    assert res.rejects["when"].to_list() == ["open", "close"]
    sells = res.fills.filter(pl.col("side") == "SELL")
    assert sells.height == 1
    assert sells["date"].item() == DS[7]
    assert daily_at(res, 7)["n_pos"] == 0


def test_buy_scaled_down_to_cash():
    df = frame(stock_rows(1, DS, FLAT, FLAT))
    res = run(df, {0: [buy(1, 1000, 1)]}, cap=10_000.0)
    # 100 shares would cost 10,000 + 23.4 of fees > cash; 99 shares: 9,900 + 23.33 fits -> qty 99
    assert res.fills["qty"].item() == 99


def test_sells_run_before_buys_at_the_same_open():
    a = stock_rows(1, DS, FLAT, FLAT)
    b = stock_rows(2, DS, FLAT, FLAT)
    res = run(frame(a, b), {0: [buy(1, 99, 1)], 1: [buy(2, 99, 2), sell(1, 99, 2)]}, cap=10_000.0)
    # all cash is in sid 1 after day 1; the day-2 buy of sid 2 is only affordable with the sale proceeds
    f2 = res.fills.filter((pl.col("sid") == 2) & (pl.col("date") == DS[2]))
    assert f2.height == 1 and f2["qty"].item() >= 90


def test_buys_by_descending_priority_when_cash_short():
    a = stock_rows(1, DS, FLAT, FLAT)
    b = stock_rows(2, DS, FLAT, FLAT)
    res = run(
        frame(a, b),
        {0: [buy(1, 99, 1, priority=1.0), buy(2, 99, 1, priority=5.0)]},
        cap=10_000.0,
    )
    # the higher-priority sid 2 takes all the cash; sid 1 cannot buy a single share
    assert res.fills["sid"].to_list() == [2]


def test_intraday_short_pnl():
    opens = [100.0, 100.0, 100.0, 200.0] + [100.0] * 8
    closes = [100.0, 100.0, 100.0, 198.0] + [100.0] * 8
    df = frame(stock_rows(1, DS, opens, closes))
    res = run(df, {2: [sell(1, 100, 3, product="intraday")]})
    # short 100 @ 200 (value 20,000): brokerage 20, stt 5.00, exchange 0.594, sebi 0.02, gst 3.71052
    #   = 29.32452. Squared off at the 198 close (value 19,800): brokerage 19.8, stamp 0.594, exchange
    #   0.58806, sebi 0.0198, gst 3.6734148 = 24.6752748. Gross +200 - 53.9997948 = +146.0002052
    assert daily_at(res, 3)["equity"] == pytest.approx(CAP + 146.0002052, abs=0.01)
    assert daily_at(res, 3)["n_pos"] == 0
    assert res.trades.filter(pl.col("reason") == "squareoff").height == 1
    assert res.trades["net_pnl_inr"].item() == pytest.approx(146.0002052, abs=0.01)


def test_intraday_long_squared_off_at_close():
    closes = [100.0] * 3 + [104.0] + [100.0] * 8
    df = frame(stock_rows(1, DS, [100.0] * 12, closes))
    res = run(df, {2: [buy(1, 100, 3, product="intraday")]})
    assert daily_at(res, 3)["n_pos"] == 0
    assert res.fills.filter(pl.col("side") == "SELL")["when"].to_list() == ["close"]
    assert res.trades["reason"].to_list() == ["squareoff"]


def test_delisting_written_off_at_80pct_after_20_sessions():
    ds = dates(40)
    a = stock_rows(1, ds, [50.0] * 10 + [None] * 30, [50.0] * 10 + [None] * 30)
    b = stock_rows(2, ds, [10.0] * 40, [10.0] * 40)
    DS_IDX.clear()
    DS_IDX.update({d: i for i, d in enumerate(ds)})
    res = run_backtest(
        PanelData.from_frame(frame(a, b)),
        TradingCalendar.from_dates(ds),
        Scripted({0: [Order(1, "BUY", "delivery", "open", 100, trade_date=ds[1])]}),
        ds[0], ds[-1], CAP, S.costs, zero_slippage(), S.risk, debug=True,
    )  # fmt: skip
    # last row day 9; days 10..29 are 20 sessions without a row -> written off on day 29 at 0.8 * 50 = 40
    p28, p29 = daily_at(res, 28, ds), daily_at(res, 29, ds)
    assert p28["n_pos"] == 1 and p29["n_pos"] == 0
    assert p28["equity"] - p29["equity"] == pytest.approx(1000.0, abs=0.01)  # 100 * (50 - 40)
    d = res.trades.filter(pl.col("reason") == "delisted")
    assert d.height == 1 and d["exit_price"].item() == pytest.approx(40.0)


def test_same_inputs_give_identical_frames():
    df = frame(stock_rows(1, DS, FLAT, FLAT), stock_rows(2, DS, FLAT, [100.0 + i for i in range(12)]))
    outs = [
        run_backtest(
            PanelData.from_frame(df),
            CAL,
            EqualWeightUniverse(rebalance_every=3),
            DS[0],
            DS[-1],
            CAP,
            S.costs,
            zero_slippage(),
            S.risk,
        )  # fmt: skip
        for _ in range(2)
    ]
    assert outs[0].daily.equals(outs[1].daily)
    assert outs[0].trades.equals(outs[1].trades)
    assert outs[0].rejects.equals(outs[1].rejects)


def test_small_capital_is_worse_than_reference_capital_for_20_names():
    stocks = [stock_rows(i, DS, FLAT, [100.0 * 1.01 ** (t + 1) for t in range(12)]) for i in range(1, 21)]
    grid = run_grid(
        PanelData.from_frame(frame(*stocks)), CAL, lambda: BuyAndHold(list(range(1, 21))), DS[0], DS[-1],
        S.costs, zero_slippage(), S.risk, capitals=[25_000.0, 500_000.0],
    )  # fmt: skip
    small, big = grid[25_000.0], grid[500_000.0]
    # 1,250 per name pays the Rs 5 brokerage floor plus STT and stamp (~0.6% of the position) against
    # ~0.2% at 25,000 per name
    assert small.daily["nav"][-1] < big.daily["nav"][-1]
    assert small.summary["total_costs_inr"] / 25_000 > big.summary["total_costs_inr"] / 500_000


def test_participation_cap_limits_quantity():
    df = frame(stock_rows(1, DS, FLAT, FLAT, turnover=1e6))
    res = run(df, {0: [buy(1, 500, 1)]})
    # cap = 1% * 1e6 = 10,000 -> floor(10000 / 100) = 100 shares
    assert res.fills["qty"].item() == 100


_order = st.builds(
    lambda sid, side, when, qty, prod, day: Order(
        sid, side, prod, when, qty, trade_date=DS[day], priority=float(qty % 7)
    ),
    st.integers(1, 3),
    st.sampled_from(["BUY", "SELL"]),
    st.sampled_from(["open", "close"]),
    st.integers(1, 600),
    st.sampled_from(["delivery", "intraday"]),
    st.integers(1, 11),
)


@hsettings(max_examples=200, deadline=None, suppress_health_check=list(HealthCheck))
@given(orders=st.lists(_order, max_size=30), cap=st.sampled_from([5_000.0, 50_000.0, 1_000_000.0]))
def test_cash_never_negative_with_random_orders(orders, cap):
    stocks = [
        stock_rows(
            i,
            DS,
            [100.0 + 3 * i + (t % 4) for t in range(12)],
            [100.0 + 3 * i + ((t + 1) % 4) for t in range(12)],
        )
        for i in (1, 2, 3)
    ]
    res = run(frame(*stocks), {0: orders}, cap=cap)  # debug=True asserts the invariants every session
    assert (res.daily["cash_frac"] * res.daily["equity"]).min() >= -0.01


def test_auction_policy_fills_on_the_same_open():
    df = frame(stock_rows(1, DS, FLAT, FLAT))
    res = run(df, {}, auction={3: [Order(1, "BUY", "delivery", "open", 10, trade_date=DS[3])]})
    assert res.fills["date"].item() == DS[3]


def test_buy_and_hold_and_equal_weight_policies_trade():
    df = frame(stock_rows(1, DS, FLAT, FLAT), stock_rows(2, DS, FLAT, FLAT))
    p = PanelData.from_frame(df)
    r1 = run_backtest(p, CAL, BuyAndHold([1]), DS[0], DS[-1], CAP, S.costs, zero_slippage(), S.risk)
    assert r1.daily["n_pos"][-1] == 1
    r2 = run_backtest(
        p, CAL, EqualWeightUniverse(rebalance_every=5), DS[0], DS[-1], CAP, S.costs, zero_slippage(), S.risk
    )
    assert r2.daily["n_pos"][-1] == 2
    assert r2.summary["final_nav"] < 100.0  # flat prices: only costs
