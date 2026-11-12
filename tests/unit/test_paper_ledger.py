from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from stocksense.core.state import State
from stocksense.costs.india import CostBreakdown
from stocksense.paper.ledger import Ledger, NewOrder

D1, D2, D3 = date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 4)
CAP = 1_000_000.0


def fees(total: float) -> CostBreakdown:
    return CostBreakdown(total, 0, 0, 0, 0, 0, 0, total)


@pytest.fixture
def led(tmp_data_root, fixed_clock):
    st = State(tmp_data_root, fixed_clock)
    st.migrate()
    return Ledger(st, fixed_clock)


def new_order(side="BUY", when="open", qty=1000, d=D1, sid=7, product="delivery") -> NewOrder:
    return NewOrder(
        sid=sid,
        symbol="AAA",
        side=side,
        product=product,
        when_kind=when,
        trade_date=d,
        qty=qty,
        ref_price=100.0,
        reason="t",
    )


def do_fill(led, book, d, side, qty, price, fee, sid=7, when="open", product="delivery"):
    dec = led.record_decision(book, f"s-{side}-{d}", d, CAP, False, {}, {})
    led.add_orders(dec, book, [new_order(side, when, qty, d, sid, product)])
    (o,) = led.pending_orders(d, when, [book])
    return led.fill_order(
        o.order_id, f"{d}T09:15:30+05:30", price, qty, 0.001, None, None, price, "t", fees(fee)
    )


def test_hand_computed_nav_path(led):
    do_fill(led, "main", D1, "BUY", 1000, 100.0, 100.0)  # w = 0.10, cost_frac = 1e-4
    (p,) = led.positions("main")
    assert p.weight == pytest.approx(0.10) and p.qty == 1000
    m1 = led.mark("main", D1, {7: 102.0})
    r1 = 0.10 * 0.02 - 1e-4
    nav1 = 100.0 * (1 + r1)
    w1 = 0.10 * 1.02 / (1 + r1)
    assert m1.day_ret == pytest.approx(r1) and m1.nav == pytest.approx(nav1)
    assert m1.cash_frac == pytest.approx(1 - w1) and m1.gross == pytest.approx(w1) and m1.n_pos == 1
    m2 = led.mark("main", D2, {7: 101.0})
    r2 = w1 * (101 / 102 - 1)
    nav2 = nav1 * (1 + r2)
    w2 = w1 * (101 / 102) / (1 + r2)
    assert m2.nav == pytest.approx(nav2)
    do_fill(led, "main", D3, "SELL", 1000, 103.0, 110.0)
    r3 = w2 * (103 / 101 - 1) - 110.0 / CAP
    nav3 = nav2 * (1 + r3)
    m3 = led.mark("main", D3, {})
    assert m3.nav == pytest.approx(nav3) and m3.n_pos == 0 and m3.cash_frac == pytest.approx(1.0)
    assert led.positions("main") == []
    ns = led.nav_series("main")
    assert ns["nav"].to_list()[:3] == [pytest.approx(nav1), pytest.approx(nav2), pytest.approx(nav3)]
    assert ns["nav"].to_list()[-1] == pytest.approx(nav3)
    # re-marking a day is a no-op
    assert led.mark("main", D2, {7: 50.0}).nav == pytest.approx(nav2)


def test_duplicate_add_orders_ignored(led):
    dec = led.record_decision("main", "s1", D1, CAP, False, {}, {})
    assert led.add_orders(dec, "main", [new_order()]) == 1
    assert led.add_orders(dec, "main", [new_order()]) == 0
    assert led.record_decision("main", "s1", D1, CAP, False, {}, {}) == dec
    assert len(led.pending_orders(D1, "open")) == 1


def test_fill_is_idempotent_and_cash_events(led):
    f1 = do_fill(led, "main", D1, "BUY", 1000, 100.0, 100.0)
    (o,) = [x for x in led.orders_for_date(D1)]
    assert o.status == "filled"
    f2 = led.fill_order(o.order_id, f"{D1}T09:16:00+05:30", 99.0, 1000, 0.0, None, None, 99.0, "t", fees(1))
    assert f1 == f2
    con = led.state.connect()
    rows = con.execute("SELECT kind, amount_frac FROM cash_events WHERE book='main'").fetchall()
    assert rows == [("trade", pytest.approx(-0.10 - 1e-4))]


def test_split_keeps_position_value(led):
    do_fill(led, "main", D1, "BUY", 1000, 100.0, 0.0)
    led.mark("main", D1, {7: 100.0})
    assert led.apply_corp_action("main", 7, D2, 0.5, 0.0) is True
    assert led.apply_corp_action("main", 7, D2, 0.5, 0.0) is False
    (p,) = led.positions("main")
    assert p.qty == pytest.approx(2000) and p.last_price == pytest.approx(50.0)
    m = led.mark("main", D2, {7: 50.0})
    assert m.day_ret == pytest.approx(0.0, abs=1e-12)


def test_dividend_adds_to_return(led):
    do_fill(led, "main", D1, "BUY", 1000, 100.0, 0.0)
    led.mark("main", D1, {7: 100.0})
    led.apply_corp_action("main", 7, D2, 1.0, 0.002)
    m = led.mark("main", D2, {7: 100.0})
    assert m.day_ret == pytest.approx(0.002)


def test_short_position_marks(led):
    do_fill(led, "auction", D1, "SELL", 500, 100.0, 0.0, when="open", product="intraday")
    (p,) = led.positions("auction")
    assert p.qty == -500 and p.weight == pytest.approx(-0.05)
    m = led.mark("auction", D1, {7: 98.0})
    assert m.day_ret == pytest.approx(-0.05 * -0.02)


def test_correction_changes_price_and_logs(led):
    fid = do_fill(led, "main", D1, "BUY", 1000, 100.0, 100.0)
    led.add_correction(fid, 101.0, "official_reprice", official_ref=100.5)
    con = led.state.connect()
    assert con.execute("SELECT price, official_ref, reconciled FROM fills").fetchone() == (101.0, 100.5, 1)
    assert con.execute("SELECT old_price, new_price, reason FROM corrections").fetchall() == [
        (100.0, 101.0, "official_reprice")
    ]
    (p,) = led.positions("main")
    assert p.weight == pytest.approx(0.101)
    assert led.cash_frac("main") == pytest.approx(1 - 0.101)


def test_correction_to_zero_qty_cancels_order(led):
    fid = do_fill(led, "main", D1, "BUY", 1000, 100.0, 100.0)
    led.add_correction(fid, 100.0, "official_reject:locked_up_open", new_qty=0, new_fees=fees(0.0))
    (o,) = led.orders_for_date(D1)
    assert o.status == "cancelled" and o.cancel_reason == "locked_up_open"
    assert led.positions("main") == []


def test_cancel_pending(led):
    dec = led.record_decision("main", "s1", D1, CAP, False, {}, {})
    led.add_orders(dec, "main", [new_order()])
    (o,) = led.pending_orders(D1, "open")
    led.cancel_pending(o.order_id, "circuit")
    (o2,) = led.orders_for_date(D1)
    assert o2.status == "cancelled" and o2.cancel_reason == "circuit"


def test_module_is_append_only():
    src = (Path(__file__).resolve().parents[2] / "src/stocksense/paper/ledger.py").read_text(encoding="utf-8")
    assert not re.search(r"\bDELETE\b", src, re.IGNORECASE)
    assert not re.search(r"INSERT\s+OR\s+REPLACE|REPLACE\s+INTO", src, re.IGNORECASE)
    for m in re.finditer(r"UPDATE\s+(\w+)\s+SET\s+([^\"]+?)\s+WHERE", src, re.IGNORECASE | re.DOTALL):
        table, cols = m.group(1), m.group(2)
        names = set(re.findall(r"(\w+)\s*=", cols))
        allowed = {
            "orders": {"status", "cancel_reason"},
            "fills": {"reconciled", "price", "official_ref", "qty"},
            "fees": {"brokerage", "stt", "exchange", "sebi", "stamp", "gst", "dp", "total"},
        }
        assert table in allowed and names <= allowed[table], (table, names)
