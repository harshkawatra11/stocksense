"""G2-H4: cm-market helpers, staleness gate, live index, call-auction exclusion (recorded fixtures)."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from stocksense.brokers import nse_mcp as M
from stocksense.brokers.quotes import NseMcpQuotes, QuoteHub
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, SteppingClock
from stocksense.core.errors import DataNotAvailable
from stocksense.core.state import State
from stocksense.live import jobs as J
from stocksense.live.jobs import JobContext
from stocksense.portfolio.construct import (
    AuctionRow,
    NameRow,
    S1Config,
    S2Config,
    _s1_entry_ok,
    plan_s2,
)
from tests.fixtures.engine_panel import settings as eng_settings
from tests.fixtures.engine_panel import zero_slippage

FILES = Path(__file__).resolve().parents[1] / "fixtures" / "files" / "mcp_cm"
D = date(2026, 11, 20)  # a Friday


def fx(name: str) -> dict:
    return json.loads((FILES / name).read_text(encoding="utf-8"))


def at(h, m, s=0):
    return datetime(2026, 11, 20, h, m, s, tzinfo=IST)


def fake_call(status="allstocks_status.json", calls=None):
    answers = {
        "cm_get_allstocks_status": lambda a: fx(status),
        "cm_get_data_status": lambda a: fx("data_status.json"),
        "cm_get_index_quote": lambda a: fx(
            "index_nifty_50.json" if a["indexName"] == "NIFTY 50" else "index_india_vix.json"
        ),
        "cm_get_call_auction_stocks": lambda a: fx("call_auction.json"),
    }

    def call(tool, args):
        if calls is not None:
            calls.append((tool, args))
        return json.dumps(answers[tool](args))

    return call


# ---- parsing ----------------


def test_parse_index_quote_recorded():
    q = M.parse_index_quote(json.dumps(fx("index_india_vix.json")))
    assert q["name"] == "INDIA VIX" and q["last"] == 17.27 and q["prev_close"] == 16.11
    assert q["ts"] == "20-Nov-2026 15:05:34"


def test_error_payload_and_non_json_raise_mcp_error():
    with pytest.raises(M.McpError):
        M.parse_index_quote({"hint": "Example: NIFTY 50", "error": "indexName is required"})
    with pytest.raises(M.McpError):
        M.as_json("java.lang.NullPointerException: x")


def test_parse_call_auction_symbols():
    assert M.parse_call_auction(fx("call_auction.json")) == sorted(
        s["symbol"] for s in fx("call_auction.json")["stocks"]
    )


def test_last_crawled_both_status_shapes_with_nanoseconds():
    a = M.last_crawled(fx("allstocks_status.json"))
    assert a is not None and a.utcoffset() == timedelta(0)
    assert (a.hour, a.minute, a.second, a.microsecond) == (9, 36, 11, 619682)
    b = M.last_crawled(fx("data_status.json"))  # the older of gainers / losers
    assert b is not None and b.minute == 36


# ---- staleness gate ----------------


def test_fresh_status_passes_and_stale_raises_inside_market_hours():
    st = fx("allstocks_status.json")  # crawled 15:06:11 IST
    M.assert_fresh(st, at(15, 20))  # 14 min
    with pytest.raises(DataNotAvailable, match="stale"):
        M.assert_fresh(st, at(15, 22))  # 16 min


def test_stale_status_is_ignored_outside_market_hours():
    M.assert_fresh(fx("allstocks_status.json"), at(18, 0))
    M.assert_fresh(fx("allstocks_status.json"), datetime(2026, 11, 21, 11, 0, tzinfo=IST))  # Saturday


def test_nse_mcp_quotes_raise_when_the_crawl_is_stale_and_hub_falls_through():
    clk = SteppingClock(at(15, 25))
    stock = json.dumps(
        {"stock": {"symbol": "A", "lastTradedPrice": 11, "latestTimestamp": "2026-11-20 15:24:00"}}
    )
    stale = NseMcpQuotes(clk, call=lambda s: stock, status_call=lambda: fx("allstocks_status.json"))
    with pytest.raises(DataNotAvailable):
        stale.quotes(["A"])
    hub = QuoteHub([stale], clk)
    assert hub.get(["A"], held={"A"}) == {} and "nse_mcp_delayed" in hub.last_errors
    fresh_clk = SteppingClock(at(15, 10))
    ok = NseMcpQuotes(fresh_clk, call=lambda s: stock, status_call=lambda: fx("allstocks_status.json"))
    assert ok.quotes(["A"])[0].ltp == 11.0


def test_no_status_call_means_no_gate_so_old_behaviour_is_kept():
    clk = SteppingClock(at(15, 25))
    stock = {"stock": {"symbol": "A", "lastTradedPrice": 11, "latestTimestamp": "2026-11-20 15:24:00"}}
    assert NseMcpQuotes(clk, call=lambda s: stock).quotes(["A"])[0].ltp == 11.0


def test_bulk_equity_probe_fixture_has_the_recorded_field_names():
    row = fx("equity_stocks.json")["stocks"][0]
    assert {"symbol", "series", "lastTradedPrice", "preClosePrice", "latestTimestamp"} <= set(row)


# ---- live index ----------------


def test_fetch_live_index_returns_both_and_gates_stale_quotes():
    out = M.fetch_live_index(fake_call(), at(15, 6))
    assert set(out) == {"NIFTY 50", "INDIA VIX"} and out["NIFTY 50"]["last"] == 25800.45
    with pytest.raises(DataNotAvailable, match="stale"):
        M.fetch_live_index(fake_call(), at(15, 25))  # quote stamped 15:05:34
    assert M.fetch_live_index(fake_call(), at(19, 0))["INDIA VIX"]["last"] == 17.27


# ---- jobs ----------------


@pytest.fixture
def ctx_of(tmp_data_root):
    def make(now, hooks):
        clock = SteppingClock(now)
        st = State(tmp_data_root, clock)
        st.migrate()
        cal = TradingCalendar.from_dates([date(2026, 11, 19)])
        return JobContext(tmp_data_root, clock, st, cal, steps=[], nse=object(), hooks=hooks)

    return make


def test_index_capture_stores_kv_by_date_and_minute(ctx_of):
    ctx = ctx_of(at(15, 6), {"cm_market_call": fake_call()})
    out = J.index_capture(ctx, D)
    assert out == "indices=2"
    snap = json.loads(ctx.state.kv_get("live_index:2026-11-20:1506"))
    assert snap["INDIA VIX"]["last"] == 17.27 and snap["NIFTY 50"]["change_pct"] == 1.12


def test_index_capture_unreachable_or_stale_is_skipped_never_raises(ctx_of):
    def boom(tool, args):
        raise M.McpError("down")

    ctx = ctx_of(at(15, 6), {"cm_market_call": boom})
    assert J.index_capture(ctx, D).startswith("skipped: unreachable")
    ctx = ctx_of(at(15, 29), {"cm_market_call": fake_call()})
    assert J.index_capture(ctx, D).startswith("skipped")
    assert ctx.state.kv_get("live_index:2026-11-20:1529") is None


def test_call_auction_capture_stores_json_list_and_failure_is_skipped(ctx_of):
    ctx = ctx_of(at(8, 50), {"cm_market_call": fake_call()})
    assert J.call_auction_capture(ctx, D) == "symbols=3"
    assert M.call_auction_symbols(ctx.state, D) == frozenset(
        s["symbol"] for s in fx("call_auction.json")["stocks"]
    )

    def boom(tool, args):
        raise RuntimeError("x")

    d2 = date(2026, 11, 19)
    ctx2 = ctx_of(at(8, 50), {"cm_market_call": boom})
    assert J.call_auction_capture(ctx2, d2).startswith("skipped: unreachable")
    assert M.call_auction_symbols(ctx2.state, d2) == frozenset()


def test_new_jobs_are_registered_and_scheduled():
    from stocksense.core.config import load_settings
    from stocksense.live.schedule import load_schedule

    assert {"index_capture", "call_auction_capture"} <= set(J.JOBS)
    keys = [j.key for j in load_schedule(load_settings())]
    assert "call_auction_capture" in keys and "index_capture@09:07" in keys
    assert "index_capture@15:05" in keys


# ---- exclusion in the planners ----------------

S = eng_settings()


def test_s1_entry_ok_default_unchanged_and_excluded_sid_refused():
    cfg = S1Config.from_settings(S, 5, zero_slippage(), k_max=5)
    row = NameRow(close=100.0, vol_21=0.01, turnover_rank=10, med_turn_20=1e10)
    assert cfg.excluded_sids == frozenset()
    assert _s1_entry_ok(7, row, 0.05, 50_000.0, cfg) is True
    ex = S1Config.from_settings(S, 5, zero_slippage(), k_max=5, excluded_sids=frozenset({7}))
    assert _s1_entry_ok(7, row, 0.05, 50_000.0, ex) is False
    assert _s1_entry_ok(8, row, 0.05, 50_000.0, ex) is True


def test_plan_s2_skips_excluded_names_only_when_given():
    base = S2Config.from_settings(S, zero_slippage(), k_max=3)
    auc = {i: AuctionRow(100.0, 0.0, 20, 1e10, 0.01, True) for i in range(6)}
    sc = {i: 0.4 - 0.01 * i for i in range(6)}
    er = dict.fromkeys(range(6), 0.02)
    free = 100 * base.min_pos
    before = {o.sid for o in plan_s2(date(2026, 11, 20), sc, er, auc, free, 1e8, base).orders}
    assert 0 in before
    ex = S2Config.from_settings(S, zero_slippage(), k_max=3, excluded_sids=frozenset({0}))
    after = {o.sid for o in plan_s2(date(2026, 11, 20), sc, er, auc, free, 1e8, ex).orders}
    assert 0 not in after and after


def test_wiring_resolves_excluded_symbols_to_sids(ctx_of):
    from stocksense.live import wiring

    ctx = ctx_of(at(15, 0), {})
    ctx.state.kv_set("call_auction:2026-11-20", json.dumps(["AAA", "ZZZ"]))
    assert wiring.excluded_sids(ctx, D, {1: "AAA", 2: "BBB"}) == frozenset({1})
    assert wiring.excluded_sids(ctx, date(2026, 11, 19), {1: "AAA"}) == frozenset()
    ctx.state.kv_set("call_auction:2026-11-19", "not json")
    assert wiring.excluded_sids(ctx, date(2026, 11, 19), {1: "AAA"}) == frozenset()
