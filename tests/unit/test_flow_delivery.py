from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from stocksense.data import io
from stocksense.flow.deals import build_deals_daily, deals_summary, load_clients
from stocksense.flow.delivery import build_delivery_daily, delivery_daily


def _panel(n: int = 21, last_pct: float = 50.0, last_close: float = 11.0) -> pl.DataFrame:
    d0 = date(2026, 1, 1)
    pcts = [38.0 if i % 2 == 0 else 42.0 for i in range(n - 1)] + [last_pct]
    return pl.DataFrame(
        {
            "sid": [1] * n,
            "date": [d0 + timedelta(days=i) for i in range(n)],
            "open": [10.0] * n,
            "close": [10.0] * (n - 1) + [last_close],
            "deliv_qty": [1_000_000.0] * n,
            "deliv_pct": pcts,
        }
    )


def test_delivery_z20_hand_number():
    out = delivery_daily(_panel()).sort("date")
    last = out.row(-1, named=True)
    assert last["deliv_z20"] == pytest.approx(4.873, abs=1e-3)
    assert last["deliv_spike"] is True
    assert last["deliv_value_cr"] == pytest.approx(1_000_000 * 11.0 / 1e7)
    assert out["deliv_z20"][:15].null_count() == 15  # fewer than 15 prior obs


def test_delivery_spike_needs_close_above_open():
    last = delivery_daily(_panel(last_close=9.0)).sort("date").row(-1, named=True)
    assert last["deliv_z20"] > 2
    assert last["deliv_spike"] is False


def test_delivery_truncation():
    p = _panel()
    full = delivery_daily(p).sort("date")
    for k in (16, 18, 20):
        cut = delivery_daily(p.head(k)).sort("date")
        assert cut.equals(full.head(k))


def test_delivery_build_idempotent(tmp_data_root):
    p = _panel().with_columns(
        symbol=pl.lit("AAA"), series=pl.lit("EQ"), volume=pl.lit(1, pl.Int64), prev_close=pl.lit(10.0)
    )
    io.upsert(tmp_data_root, "panel_daily", p)
    a, b = date(2026, 1, 1), date(2026, 1, 21)
    n1 = build_delivery_daily(tmp_data_root, a, b)
    first = io.read(tmp_data_root, "flow_delivery_daily").collect()
    n2 = build_delivery_daily(tmp_data_root, a, b)
    second = io.read(tmp_data_root, "flow_delivery_daily").collect()
    assert n1 == n2 == 21
    assert first.equals(second)


def _deals() -> pl.DataFrame:
    d = date(2026, 11, 12)
    return pl.DataFrame(
        {
            "date": [d, d, d],
            "kind": ["bulk", "bulk", "block"],
            "symbol": ["AAA", "AAA", "BBB"],
            "client": ["ALPHA", "BETA", "ALPHA"],
            "side": ["BUY", "SELL", "BUY"],
            "qty": [1000.0, 400.0, 100.0],
            "price": [100.0, 100.0, 50.0],
        }
    )


def test_deals_net_hand_number():
    out = deals_summary(_deals(), set()).sort("symbol")
    a = out.row(0, named=True)
    assert a["buy_value_cr"] == pytest.approx(0.01)
    assert a["sell_value_cr"] == pytest.approx(0.004)
    assert a["net_value_cr"] == pytest.approx(0.006)
    assert a["n_deals"] == 2
    assert a["top_client"] == "ALPHA"
    assert a["flagged"] is False
    assert out.row(1, named=True)["net_value_cr"] == pytest.approx(100 * 50 / 1e7)


def test_deals_flagged_client():
    out = deals_summary(_deals(), {"BETA"}).sort("symbol")
    assert out["flagged"].to_list() == [True, False]


def test_deals_empty():
    out = deals_summary(_deals().head(0), set())
    assert out.height == 0 and "net_value_cr" in out.columns


def test_load_clients(tmp_path):
    f = tmp_path / "flow_clients.toml"
    f.write_text('clients = ["A B", "C"]\n')
    assert load_clients(f) == {"A B", "C"}
    assert load_clients(tmp_path / "missing.toml") == set()


def test_deals_build_idempotent(tmp_data_root):
    io.upsert(tmp_data_root, "deals", _deals())
    d = date(2026, 11, 12)
    assert build_deals_daily(tmp_data_root, d, d) == 2
    first = io.read(tmp_data_root, "flow_deals_daily").collect()
    assert build_deals_daily(tmp_data_root, d, d) == 2
    assert io.read(tmp_data_root, "flow_deals_daily").collect().equals(first)
