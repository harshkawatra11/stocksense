from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from stocksense.brokers.types import Quote
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, SteppingClock
from stocksense.core.state import State
from stocksense.costs.slippage import SlippageTable
from stocksense.paper.broker import CorpAction, PaperBroker
from stocksense.paper.ledger import Ledger, NewOrder

D = date(2026, 11, 12)
CAP = 1_000_000.0


def at(h, m, s=0) -> datetime:
    return datetime(2026, 11, 12, h, m, s, tzinfo=IST)


def quote(sym="AAA", ts=None, ltp=101.0, open_=100.0, **kw) -> Quote:
    base = dict(
        symbol=sym, ts=ts or at(9, 15, 30), ltp=ltp, open=open_, high=105.0, low=99.0, prev_close=99.0,
        volume=1e5, bid=None, ask=None, bid_qty=100, ask_qty=100, upper_circuit=120.0, lower_circuit=80.0,
        source="scripted",
    )
    base.update(kw)
    return Quote(**base)


class Hub:
    """Scripted quotes: `script` maps symbol -> callable(clock) -> Quote | None."""

    def __init__(self, clock):
        self.clock, self.script, self.degraded = clock, {}, False

    def get(self, symbols, held=None):
        out = {}
        for s in symbols:
            f = self.script.get(s)
            q = f(self.clock) if f else None
            if q is not None:
                out[s] = q
        return out


@pytest.fixture
def env(tmp_data_root):
    clock = SteppingClock(at(9, 15, 10))
    st = State(tmp_data_root, clock)
    st.migrate()
    led = Ledger(st, clock)
    hub = Hub(clock)
    cal = TradingCalendar.from_dates([date(2026, 11, 9), date(2026, 11, 11), D])
    br = PaperBroker(led, hub, tmp_data_root.costs, SlippageTable.load_latest(tmp_data_root), clock, cal)
    return br, led, hub, clock


def submit(led, book, side="BUY", when="open", qty=100, product="delivery", sid=1, sym="AAA"):
    dec = led.record_decision(book, "test", D, CAP, False, {}, {})
    led.add_orders(dec, book, [NewOrder(sid, sym, side, product, when, D, qty, 100.0, "t")])


def fills(led):
    return led.fills_for_date(D)


def test_auction_fills_at_quote_open_and_strict_at_0916_ltp(env):
    br, led, hub, clock = env
    hub.script["AAA"] = lambda c: quote(ts=c.now(), ltp=103.0, open_=100.0)
    submit(led, "auction")
    submit(led, "auction_strict")
    r = br.process_open(D)
    assert len(r.filled) == 1 and len(r.pending) == 1
    (f,) = fills(led)
    assert f.book == "auction" and 100.0 <= f.price < 100.5 and f.quote_ltp == 103.0
    clock.advance(timedelta(seconds=40))  # 09:15:50: strict still waits
    assert br.process_open(D).filled == []
    clock.advance(timedelta(seconds=20))  # 09:16:10
    r = br.process_open(D)
    assert len(r.filled) == 1
    s = [x for x in fills(led) if x.book == "auction_strict"][0]
    assert s.price >= 103.0 and s.price < 103.6


def test_open_price_missing_waits_until_0925_then_ltp(env):
    br, led, hub, clock = env
    hub.script["AAA"] = lambda c: quote(ts=c.now(), ltp=102.0, open_=None)
    submit(led, "main")
    assert br.process_open(D).filled == []
    clock.advance(timedelta(minutes=5))  # 09:20:10
    assert br.process_open(D).filled == []
    clock.advance(timedelta(minutes=5))  # 09:25:10
    assert len(br.process_open(D).filled) == 1
    assert fills(led)[0].price >= 102.0


def test_idempotent_rerun(env):
    br, led, hub, clock = env
    hub.script["AAA"] = lambda c: quote(ts=c.now())
    submit(led, "main")
    br.process_open(D)
    again = br.process_open(D)
    assert again.filled == [] and len(fills(led)) == 1


def test_blocked_buy_retried_then_cancelled(env):
    br, led, hub, clock = env
    hub.script["AAA"] = lambda c: quote(ts=c.now(), ltp=119.97, open_=118.0)  # >= upper - 0.05
    submit(led, "main")
    for _ in range(3):
        assert br.process_open(D).filled == []
        clock.advance(timedelta(minutes=1))
    (o,) = led.pending_orders(D, "open")
    clock.advance(timedelta(hours=6))  # 15:19:xx
    assert br.process_open(D).cancelled == []
    clock.advance(timedelta(minutes=2))
    r = br.process_open(D)
    assert r.cancelled == [(o.order_id, "circuit_blocked_buy")]
    (o2,) = led.orders_for_date(D)
    assert o2.status == "cancelled" and o2.cancel_reason == "circuit_blocked_buy"
    # no depth on the ask side also blocks
    hub.script["BBB"] = lambda c: quote("BBB", ts=c.now(), ask_qty=0)
    submit(led, "main", sid=2, sym="BBB")
    r = br.process_open(D)  # past the cut-off already
    assert r.filled == [] and r.cancelled[0][1] == "circuit_blocked_buy"


def test_touch_is_respected_for_ltp_fills(env):
    br, led, hub, clock = env
    hub.script["AAA"] = lambda c: quote(ts=c.now(), ltp=100.0, ask=105.0, bid=95.0)
    submit(led, "main", when="now")
    submit(led, "auction", side="BUY", when="now", sid=1)
    br.process_now(D)
    for f in fills(led):
        assert f.price >= 105.0


def test_squareoff_closes_shorts(env):
    br, led, hub, clock = env
    hub.script["AAA"] = lambda c: quote(ts=c.now(), ltp=98.0, open_=100.0)
    submit(led, "auction", side="SELL", product="intraday", qty=200)
    br.process_open(D)
    (p,) = led.positions("auction")
    assert p.qty == -200
    clock.advance(timedelta(hours=6, minutes=2))  # 15:17:10 -> 15:18 call
    clock.advance(timedelta(minutes=1))
    r = br.squareoff_intraday(D)
    assert len(r.filled) == 1 and led.positions("auction") == []
    assert br.squareoff_intraday(D).filled == []  # idempotent


def test_close_orders_wait_for_1522(env):
    br, led, hub, clock = env
    hub.script["AAA"] = lambda c: quote(ts=c.now(), ltp=100.0)
    submit(led, "main", when="close")
    clock.advance(timedelta(hours=6))  # 15:15:10
    assert br.process_close(D).filled == []
    clock.advance(timedelta(minutes=8))  # 15:23:10
    assert len(br.process_close(D).filled) == 1


def test_expired_orders_cancelled(env):
    br, led, hub, clock = env
    submit(led, "main")
    clock.advance(timedelta(days=1))
    assert br.process_open(D).cancelled[0][1] == "expired"


def test_corp_action_before_open(env):
    br, led, hub, clock = env
    prev = date(2026, 11, 11)
    dec = led.record_decision("main", "p", prev, CAP, False, {}, {})
    led.add_orders(dec, "main", [NewOrder(1, "AAA", "BUY", "delivery", "close", prev, 100, 100.0)])
    (o,) = led.pending_orders(prev, "close")
    from stocksense.costs.india import CostBreakdown

    led.fill_order(o.order_id, "2026-11-11T15:25:00+05:30", 100.0, 100, 0.0, None, None, 100.0, "t",
                   CostBreakdown(0, 0, 0, 0, 0, 0, 0, 0))
    hub.script["AAA"] = lambda c: quote(ts=c.now(), ltp=50.0, open_=50.0, prev_close=50.0)
    br.process_open(D, corp_actions=[CorpAction(1, 0.5, 1.0)])
    br.process_open(D, corp_actions=[CorpAction(1, 0.5, 1.0)])
    (p,) = led.positions("main")
    assert p.qty == pytest.approx(200) and p.last_price == pytest.approx(50.0)
    m = led.mark("main", D, {1: 50.0})
    assert m.day_ret == pytest.approx(100 * 1.0 / CAP)


def panel(rows):
    return pl.DataFrame(rows)


def official(sid=1, o=100.0, h=105.0, lo=99.0, c=102.0, prev=98.0, **kw):
    r = dict(sid=sid, open=o, high=h, low=lo, close=c, prev_close=prev, gap_factor=1.0, turnover_rank=120,
             vol_21=0.02, med_turn_20=1e8, flag_one_price=False, flag_locked_up=False, flag_locked_dn=False)
    r.update(kw)
    return r


def test_reconcile_reprices_main_and_keeps_strict(env):
    br, led, hub, clock = env
    hub.script["AAA"] = lambda c: quote(ts=c.now(), ltp=104.0, open_=100.4)
    submit(led, "main")
    submit(led, "auction_strict")
    br.process_open(D)
    clock.advance(timedelta(seconds=60))
    br.process_open(D)
    rep = br.reconcile(D, panel([official()]))
    assert len(rep.repriced) == 1 and len(rep.kept) == 1
    by = {f.book: f for f in fills(led)}
    # official open 100 at rank 120: slip 0.000923607 on 100 shares -> 100.0923 -> 100.10
    assert by["main"].price == pytest.approx(100.10) and by["main"].reconciled
    assert by["main"].official_ref == 100.0
    assert by["auction_strict"].price >= 104.0 and by["auction_strict"].official_ref == 100.0
    con = led.state.connect()
    assert con.execute("SELECT reason FROM corrections").fetchall() == [("official_reprice",)]
    again = br.reconcile(D, panel([official()]))
    assert again.repriced == [] and again.kept == []
    assert con.execute("SELECT count(*) FROM corrections").fetchone()[0] == 1


def test_reconcile_reverses_fill_official_bar_rejects(env):
    br, led, hub, clock = env
    hub.script["AAA"] = lambda c: quote(ts=c.now(), ltp=101.0, open_=100.0)
    submit(led, "main")
    br.process_open(D)
    assert len(led.positions("main")) == 1
    # official open == high at a 5% gap: locked up at the open
    row = official(o=103.9, h=103.9, lo=101.0, c=103.0, prev=99.0)
    rep = br.reconcile(D, panel([row]))
    assert rep.reversed and rep.reversed[0][1] == "locked_up_open"
    (o,) = led.orders_for_date(D)
    assert o.status == "cancelled" and o.cancel_reason == "locked_up_open"
    assert led.positions("main") == [] and fills(led)[0].qty == 0


def test_reconcile_clamps_strict_to_official_range(env):
    br, led, hub, clock = env
    hub.script["AAA"] = lambda c: quote(ts=c.now(), ltp=110.0)
    clock.advance(timedelta(seconds=60))
    submit(led, "events")
    br.process_open(D)
    rep = br.reconcile(D, panel([official(h=105.0)]))
    assert len(rep.clamped) == 1
    (f,) = fills(led)
    assert f.price == 105.0 and f.reconciled


def test_reconcile_refuses_empty_panel(env):
    br, *_ = env
    with pytest.raises(ValueError):
        br.reconcile(D, pl.DataFrame([]))
