from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import httpx
import polars as pl

from stocksense.brokers.instruments import InstrumentMap, nfo_future_rows
from stocksense.brokers.types import Quote
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock
from stocksense.data import io
from stocksense.flow.oi import FuturesOiLive
from stocksense.marketlab.store import LabStore
from tests.unit.test_brokers import FULL, LOGIN_OK, _client, _ok, _settings

T0 = datetime(2026, 11, 12, 10, 0, tzinfo=IST)  # a Thursday, market open
TODAY = T0.date()


def row(token, symbol, name, expiry, kind="FUTSTK", seg="NFO", lot="500"):
    return {
        "token": token,
        "symbol": symbol,
        "name": name,
        "expiry": expiry,
        "strike": "-1.000000",
        "lotsize": lot,
        "instrumenttype": kind,
        "exch_seg": seg,
    }


MASTER = [
    row("1", "RELIANCE23NOV26FUT", "RELIANCE", "23NOV2026"),
    row("2", "RELIANCE29DEC26FUT", "RELIANCE", "29DEC2026"),
    row("3", "NIFTY29DEC26FUT", "NIFTY", "29DEC2026", "FUTIDX", lot="65"),
    row("4", "NIFTY10NOV26FUT", "NIFTY", "10NOV2026", "FUTIDX", lot="65"),  # expired before 2026-11-12
    row("5", "RELIANCE26NOV26CE1000", "RELIANCE", "26NOV2026", "OPTSTK"),  # an option, ignored
    row("6", "SBIN-EQ", "SBIN", "", "", "NSE"),
]


def test_parser_picks_nearest_expiry_on_or_after_asof():
    r = nfo_future_rows(MASTER, TODAY)
    assert r == {"RELIANCE": ("1", "RELIANCE23NOV26FUT"), "NIFTY": ("3", "NIFTY29DEC26FUT")}
    assert nfo_future_rows(MASTER, date(2026, 11, 23))["RELIANCE"][0] == "1"  # expiry day still counts
    assert nfo_future_rows(MASTER, date(2026, 11, 24))["RELIANCE"][0] == "2"
    assert nfo_future_rows(MASTER, date(2027, 1, 1)) == {}


def test_instrument_map_reads_nfo_from_the_cached_master(tmp_data_root):
    d = tmp_data_root.paths.data_root / "cache" / "instruments"
    d.mkdir(parents=True)
    (d / f"angel-{TODAY.isoformat()}.json").write_text(json.dumps(MASTER), encoding="utf-8")
    inst = InstrumentMap(
        tmp_data_root, FixedClock(T0), transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    assert inst.angel_nfo_futures(TODAY) == {"RELIANCE": "1", "NIFTY": "3"}
    assert inst.angel_nfo_trading_symbols(TODAY)["RELIANCE"] == "RELIANCE23NOV26FUT"
    assert inst.angel_token("SBIN") == "6"  # the equity map is unaffected


def test_quote_full_exchange_parameter_defaults_to_nse(tmp_data_root):
    bodies = []

    def quote(req, n):
        bodies.append(json.loads(req.content))
        return _ok({"fetched": [FULL], "unfetched": []})(req, n)

    c, _, _ = _client(_settings(tmp_data_root), {"loginByPassword": LOGIN_OK, "market/v1/quote/": quote})
    c.quote_full(["1"])
    c.quote_full(["2", "3"], exchange="NFO")
    assert bodies[0]["exchangeTokens"] == {"NSE": ["1"]}
    assert bodies[1]["exchangeTokens"] == {"NFO": ["2", "3"]} and bodies[1]["mode"] == "FULL"


class FakeAngel:
    def __init__(self, quotes):
        self.quotes, self.calls = quotes, []

    def quote_full(self, tokens, exchange="NSE"):
        self.calls.append((list(tokens), exchange))
        return self.quotes


class FakeInst:
    def angel_nfo_futures(self, asof):
        return {"AAA": "11", "BBB": "22", "CCC": "33"}

    def angel_nfo_trading_symbols(self, asof):
        return {"AAA": "AAA26NOV26FUT", "BBB": "BBB26NOV26FUT", "CCC": "CCC26NOV26FUT"}


def _q(sym, ltp, oi):
    return Quote(
        symbol=sym,
        ts=T0,
        ltp=ltp,
        open=None,
        high=None,
        low=None,
        prev_close=None,
        volume=None,
        bid=None,
        ask=None,
        bid_qty=None,
        ask_qty=None,
        upper_circuit=None,
        lower_circuit=None,
        source="angel",
        oi=oi,
    )


def _loop(tmp_data_root, clock, quotes, top=50):
    prev = TODAY - timedelta(days=1)
    io.upsert(
        tmp_data_root,
        "flow_oi_daily",
        pl.DataFrame(
            {
                "date": [prev] * 3,
                "symbol": ["AAA", "BBB", "CCC"],
                "fut_close": [100.0, 200.0, 50.0],
                "fut_oi": [1000.0, 2000.0, 500.0],
                "oi_chg_pct": [0.0] * 3,
                "price_chg_pct": [0.0] * 3,
                "basis_pct": [0.0] * 3,
                "rollover_pct": [None] * 3,
                "quad": [0.0] * 3,
                "quad_label": [None] * 3,
            },
            schema_overrides={"rollover_pct": pl.Float64, "quad_label": pl.Utf8},
        ),
    )
    store = LabStore()
    store.tape["stocks"] = {
        "AAA": {"value_cr": 300.0},
        "BBB": {"value_cr": 200.0},
        "CCC": {"value_cr": 100.0},
        "ZZZ": {"value_cr": 999.0},
    }
    cal = TradingCalendar.from_dates([TODAY - timedelta(days=i) for i in range(1, 30)])
    angel = FakeAngel(quotes)
    return FuturesOiLive(tmp_data_root, clock, cal, store, angel, FakeInst(), top=top), store, angel


def test_live_oi_quad_from_ltp_and_oi_vs_previous_eod(tmp_data_root):
    quotes = [
        _q("AAA26NOV26FUT", 101.0, 1100.0),
        _q("BBB26NOV26FUT", 190.0, 2400.0),
        _q("CCC26NOV26FUT", 55.0, 400.0),
    ]
    loop, store, angel = _loop(tmp_data_root, FixedClock(T0), quotes)
    assert loop.step() == 60.0
    rows = {r["symbol"]: r for r in store.flow["oi_live"]["rows"]}
    assert rows["AAA"]["quad_label"] == "long buildup" and rows["AAA"]["oi_chg_pct"] == 10.0
    assert rows["BBB"]["quad_label"] == "short buildup" and rows["BBB"]["price_chg_pct"] == -5.0
    assert rows["CCC"]["quad_label"] == "short covering"
    assert angel.calls == [(["11", "22", "33"], "NFO")]  # ZZZ has no futures token; tape order by value_cr


def test_live_oi_top_n_and_market_closed(tmp_data_root):
    quotes = [_q("AAA26NOV26FUT", 101.0, 1100.0)]
    loop, store, angel = _loop(tmp_data_root, FixedClock(T0), quotes, top=1)
    loop.step()
    assert angel.calls == [(["11"], "NFO")]
    closed, store2, angel2 = _loop(
        tmp_data_root, FixedClock(datetime(2026, 11, 12, 20, 0, tzinfo=IST)), quotes
    )
    assert closed.step() == 120.0 and angel2.calls == [] and "oi_live" not in store2.flow


def test_live_oi_unit_mismatch_is_flagged_not_reported(tmp_data_root):
    quotes = [_q("AAA26NOV26FUT", 101.0, 11.0)]  # 11 vs 1000 shares: looks like contracts, not shares
    loop, store, _ = _loop(tmp_data_root, FixedClock(T0), quotes)
    loop.step()
    r = store.flow["oi_live"]["rows"][0]
    assert r["oi_chg_pct"] is None and r["quad_label"] is None and r["note"] == "oi unit mismatch"
