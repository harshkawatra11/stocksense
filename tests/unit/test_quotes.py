from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from stocksense.brokers.quotes import (
    AngelQuotes,
    NseMcpQuotes,
    QuoteHub,
    UpstoxQuotes,
    parse_nse_mcp_quote,
)
from stocksense.brokers.types import Quote
from stocksense.core.clock import IST, SteppingClock

T0 = datetime(2026, 11, 12, 9, 20, tzinfo=IST)


def q(sym, ts=T0, ltp=100.0, source="x") -> Quote:
    return Quote(sym, ts, ltp, ltp, ltp, ltp, ltp, 1.0, ltp, ltp, 10, 10, ltp * 1.1, ltp * 0.9, source)


class Fake:
    held_only = False

    def __init__(self, name, clock, serves=None, fail=False, age=timedelta(0)):
        self.name, self.clock, self.serves, self.fail, self.age = name, clock, serves, fail, age
        self.calls = []

    def quotes(self, symbols):
        self.calls.append(list(symbols))
        if self.fail:
            raise RuntimeError("down")
        ok = symbols if self.serves is None else [s for s in symbols if s in self.serves]
        return [q(s, self.clock.now() - self.age, source=self.name) for s in ok]


def test_fallback_order_fills_missing_symbols():
    clk = SteppingClock(T0)
    a, b = Fake("a", clk, serves={"X"}), Fake("b", clk)
    hub = QuoteHub([a, b], clk)
    got = hub.get(["X", "Y"])
    assert set(got) == {"X", "Y"} and got["X"].source == "a" and got["Y"].source == "b"
    assert b.calls == [["Y"]] and not hub.degraded


def test_circuit_breaker_five_minutes():
    clk = SteppingClock(T0)
    a, b = Fake("a", clk, fail=True), Fake("b", clk)
    hub = QuoteHub([a, b], clk)
    got = hub.get(["X"])
    assert got["X"].source == "b" and hub.degraded
    assert len(a.calls) == 1
    clk.advance(timedelta(minutes=4, seconds=59))
    hub.get(["X"])
    assert len(a.calls) == 1  # still skipped
    clk.advance(timedelta(seconds=2))
    a.fail = False
    got = hub.get(["X"])
    assert len(a.calls) == 2 and got["X"].source == "a" and not hub.degraded


def test_stale_quote_dropped_and_next_source_used():
    clk = SteppingClock(T0)
    a = Fake("a", clk, age=timedelta(minutes=3, seconds=1))
    b = Fake("b", clk)
    hub = QuoteHub([a, b], clk)
    assert hub.get(["X"])["X"].source == "b"
    only_stale = QuoteHub([Fake("a", clk, age=timedelta(minutes=4))], clk)
    assert only_stale.get(["X"]) == {}
    fresh = QuoteHub([Fake("a", clk, age=timedelta(minutes=2, seconds=59))], clk)
    assert "X" in fresh.get(["X"])


def test_nse_mcp_only_for_held_when_brokers_fail():
    clk = SteppingClock(T0)
    a, b = Fake("a", clk, fail=True), Fake("b", clk, fail=True)
    calls = []

    def call(sym):
        calls.append(sym)
        return json.dumps(
            {"stock": {"symbol": sym, "openPrice": 10, "highPrice": 12, "lowPrice": 9, "preClosePrice": 9.5,
                       "lastTradedPrice": 11, "volume": 5, "latestTimestamp": "2026-11-12 09:19:00"}}
        )

    mcp = NseMcpQuotes(clk, call=call)
    hub = QuoteHub([a, b, mcp], clk)
    got = hub.get(["HELD", "OTHER"], held={"HELD"})
    assert set(got) == {"HELD"} and calls == ["HELD"]
    assert got["HELD"].source == "nse_mcp_delayed" and got["HELD"].ltp == 11 and hub.degraded


def test_parse_nse_mcp_quote():
    quote = parse_nse_mcp_quote(
        {"stock": {"symbol": "RELIANCE", "openPrice": 1333.5, "highPrice": 1349.4, "lowPrice": 1327.0,
                   "preClosePrice": 1324.4, "lastTradedPrice": 1335.4, "volume": 17554310,
                   "latestTimestamp": "2026-11-11 16:00:28"}}
    )
    assert quote.symbol == "RELIANCE" and quote.ltp == 1335.4 and quote.open == 1333.5
    assert quote.ts == datetime(2026, 11, 11, 16, 0, 28, tzinfo=IST) and quote.bid is None


def test_angel_and_upstox_adapters_map_symbols():
    clk = SteppingClock(T0)

    class Inst:
        def angel_token(self, s):
            return {"AAA": "1", "BBB": "2"}.get(s)

        def upstox_key(self, s):
            return {"AAA": "NSE_EQ|A"}.get(s)

    class AngelC:
        def quote_full(self, tokens):
            return [q("AAA") if t == "1" else q("BBB") for t in tokens]

    class UpC:
        def quotes(self, keys):
            return [q("AAA") for _ in keys]

    assert [x.symbol for x in AngelQuotes(AngelC(), Inst()).quotes(["AAA", "BBB", "ZZZ"])] == ["AAA", "BBB"]
    assert [x.symbol for x in UpstoxQuotes(UpC(), Inst()).quotes(["AAA", "BBB"])] == ["AAA"]
    assert AngelQuotes(AngelC(), Inst()).name == "angel" and UpstoxQuotes(UpC(), Inst()).name == "upstox"
    assert clk.now() == T0


def test_hub_empty_request():
    clk = SteppingClock(T0)
    hub = QuoteHub([Fake("a", clk)], clk)
    assert hub.get([]) == {} and not hub.degraded
    with pytest.raises(ValueError):
        QuoteHub([], clk)
