from __future__ import annotations

from datetime import date, datetime

import httpx
import polars as pl

from stocksense.brokers.angel import parse_quote
from stocksense.brokers.types import Quote
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock, SteppingClock
from stocksense.core.state import State
from stocksense.data.ingest_common import IngestContext
from stocksense.data.ingest_misc import capture_spreads, depth_imbalance
from stocksense.data.io import read, write_partition
from stocksense.data.nse_http import NseClient

NOW = datetime(2026, 11, 12, 10, 0, tzinfo=IST)

# the recorded key list of a FULL quote (docs/DATA.md, 2026-11-12)
KEYS = [
    "52WeekHigh",
    "52WeekLow",
    "avgPrice",
    "close",
    "depth",
    "exchFeedTime",
    "exchTradeTime",
    "exchange",
    "high",
    "lastTradeQty",
    "low",
    "lowerCircuit",
    "ltp",
    "netChange",
    "open",
    "opnInterest",
    "percentChange",
    "referenceLimitPrice",
    "symbolToken",
    "totBuyQuan",
    "totSellQuan",
    "tradeVolume",
    "tradingSymbol",
    "upperCircuit",
]


def _level(p, q):
    return {"price": p, "quantity": q, "orders": 1}


def _full():
    q = dict.fromkeys(KEYS, 0)
    q.update(
        tradingSymbol="RELIANCE-EQ",
        exchFeedTime="11-Nov-2026 16:08:11",
        ltp=1335.4,
        open=1333.5,
        high=1349.4,
        low=1327.0,
        close=1324.4,
        tradeVolume=17554310,
        opnInterest=0,
        totBuyQuan=250000,
        totSellQuan=310000,
        avgPrice=1338.25,
    )
    q["52WeekHigh"], q["52WeekLow"] = 1612.0, 1115.5
    q["depth"] = {
        "buy": [_level(1335.3 - i * 0.1, 100 * (i + 1)) for i in range(5)],
        "sell": [_level(1335.5 + i * 0.1, 50) for i in range(5)],
    }
    return q


def test_parse_full_quote_all_fields():
    q = parse_quote(_full(), NOW)
    assert q.symbol == "RELIANCE" and q.bid == 1335.3 and q.ask == 1335.5
    assert q.oi == 0.0 and q.total_buy_qty == 250000.0 and q.total_sell_qty == 310000.0
    assert q.avg_price == 1338.25 and q.week52_high == 1612.0 and q.week52_low == 1115.5
    assert len(q.depth_bid) == 5 and len(q.depth_ask) == 5
    assert q.depth_bid[1] == (1335.2, 200.0) or abs(q.depth_bid[1][0] - 1335.2) < 1e-9
    assert q.depth_bid[4][1] == 500.0 and q.depth_ask[4][1] == 50.0


def test_parse_quote_without_depth_extras_keeps_defaults():
    q = parse_quote({"tradingSymbol": "X-EQ", "ltp": 1.0}, NOW)
    assert q.depth_bid == () and q.depth_ask == ()
    assert q.oi is None and q.total_buy_qty is None and q.week52_high is None


def test_quote_new_fields_default_none():
    q = Quote("A", NOW, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, "angel")
    assert q.oi is None and q.avg_price is None and q.depth_bid == ()


def test_depth_imbalance_by_hand():
    assert depth_imbalance(((10, 100), (10, 200), (10, 0)), ((10, 50), (10, 50))) == 0.5
    assert depth_imbalance((), ()) is None
    assert depth_imbalance(((10, 0),), ((10, 0),)) is None


def _ctx(settings):
    clock = FixedClock(NOW)
    nse = NseClient(SteppingClock(NOW), transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    state = State(settings, clock)
    state.migrate()
    cal = TradingCalendar.from_dates([date(2026, 11, 11)])
    return IngestContext(settings=settings, clock=clock, calendar=cal, nse=nse, state=state)


def test_capture_spreads_stores_depth_columns_and_old_partitions_read(tmp_data_root):
    old = pl.DataFrame(
        {
            "symbol": ["OLD"],
            "ts": [datetime(2026, 11, 2, 10, 0, tzinfo=IST)],
            "bid": [1.0],
            "ask": [1.1],
            "ltp": [1.05],
            "bid_qty": [5.0],
            "ask_qty": [6.0],
            "source": ["angel"],
        }
    ).with_columns(pl.col("ts").dt.cast_time_unit("us"))
    write_partition(tmp_data_root, "spreads", "2026-11", old)
    q = parse_quote(_full(), NOW)
    ctx = _ctx(tmp_data_root)
    assert capture_spreads(ctx, ["RELIANCE"], lambda s: [q]) == 1
    df = read(tmp_data_root, "spreads").collect().sort("ts")
    assert df["symbol"].to_list() == ["OLD", "RELIANCE"]
    assert df["depth_imbalance_5"][0] is None and df["oi"][0] is None
    # bids 100..500 = 1500, asks 5 x 50 = 250
    assert abs(df["depth_imbalance_5"][1] - (1500 - 250) / 1750) < 1e-12
    assert df["total_buy_qty"][1] == 250000.0 and df["total_sell_qty"][1] == 310000.0
    assert df["oi"][1] == 0.0


def test_feed_token_kept_in_memory_and_never_logged(tmp_data_root, caplog, capfd):
    from stocksense.brokers.angel import AngelClient
    from tests.unit.test_brokers import T0, _ok, _settings

    login = _ok({"jwtToken": "JWT-X", "refreshToken": "R", "feedToken": "FEED-SECRET-77"})
    c = AngelClient(
        _settings(tmp_data_root),
        SteppingClock(T0),
        transport=httpx.MockTransport(lambda req: login(req, 1)),
    )
    assert c.feed_token is None
    c.login()
    assert c.feed_token == "FEED-SECRET-77"
    out = capfd.readouterr()
    assert "FEED-SECRET-77" not in out.out + out.err + caplog.text
