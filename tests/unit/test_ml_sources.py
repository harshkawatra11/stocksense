from __future__ import annotations

from datetime import datetime

import httpx
import pytest

from stocksense.core.clock import IST, FixedClock
from stocksense.data.nse_http import NseClient
from stocksense.marketlab.sources import (
    Sources,
    SourceThrottled,
    parse_all_indices,
    parse_google_news,
    parse_spark,
    parse_symbol_data,
    parse_watch,
    to_float,
)
from stocksense.marketlab.store import LabStore

NOW = FixedClock(datetime(2026, 11, 12, 10, 0, tzinfo=IST))


def make(handler, tmp_path=None):
    store = LabStore()
    nse = NseClient(NOW, transport=httpx.MockTransport(handler))
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return Sources(None, NOW, store, nse=nse, http=http), store  # type: ignore[arg-type]


def test_to_float():
    assert to_float("1,234.5") == 1234.5 and to_float(None) is None and to_float("x") is None


def test_parse_all_indices_week_change():
    d = {
        "data": [
            {
                "index": "NIFTY 50",
                "last": "110",
                "oneWeekAgoVal": "100",
                "percentChange": "0.5",
                "advances": 30,
            }
        ]
    }
    r = parse_all_indices(d)["NIFTY 50"]
    assert r["w1"] == 10.0 and r["pct"] == 0.5 and r["adv"] == 30.0


def test_parse_watch_keeps_eq_only_and_converts_value():
    rows = [
        {"symbol": "ABC", "series": "EQ", "lastPrice": 10, "totalTradedValue": 2e8, "companyName": "ABC LTD"},
        {"symbol": "XYZ", "series": "BE", "lastPrice": 5},
    ]
    out = parse_watch(rows)
    assert list(out) == ["ABC"] and out["ABC"]["value_cr"] == 20.0 and out["ABC"]["name"] == "Abc Ltd"


def test_parse_symbol_data_fields():
    d = {
        "equityResponse": [
            {
                "secInfo": {"sector": "Power", "basicIndustry": "Power Generation", "pdSymbolPe": "12.5"},
                "tradeInfo": {"deliveryToTradedQuantity": "55.5", "totalMarketCap": 1e12},
                "priceInfo": {"yearHigh": 300},
                "metaData": {"companyName": "Foo", "lastPrice": 250},
            }
        ]
    }
    r = parse_symbol_data(d, 1.0)
    assert r["sector"] == "Power" and r["pe"] == 12.5 and r["delivery"] == 55.5 and r["ltp"] == 250


def test_parse_spark_drops_null_closes():
    p = {
        "spark": {
            "result": [
                {
                    "symbol": "A.NS",
                    "response": [
                        {
                            "meta": {
                                "regularMarketPrice": 5,
                                "chartPreviousClose": 4,
                                "regularMarketTime": 99,
                            },
                            "timestamp": [1, 2, 3],
                            "indicators": {"quote": [{"close": [1.0, None, 3.0]}]},
                        }
                    ],
                },
                {"symbol": "B.NS", "response": [None]},
            ]
        }
    }
    r = parse_spark(p)
    assert list(r) == ["A.NS"] and r["A.NS"]["ts"] == [1, 3] and r["A.NS"]["close"] == [1.0, 3.0]


RSS = (
    b"<rss><channel>"
    b"<item><title>Big result - Src</title><source>Src</source>"
    b"<pubDate>Wed, 11 Nov 2026 10:00:00 GMT</pubDate><link>u1</link></item>"
    b"<item><title>Big result - Src</title><source>Src</source>"
    b"<pubDate>Wed, 11 Nov 2026 09:00:00 GMT</pubDate><link>u2</link></item>"
    b"<item><title>Old - Src</title><source>Src</source>"
    b"<pubDate>Wed, 01 Jan 2020 09:00:00 GMT</pubDate><link>u3</link></item>"
    b"</channel></rss>"
)


def test_parse_google_news_strips_source_dedupes_and_cuts_old():
    out = parse_google_news(RSS, NOW.now(), days=30, limit=10)
    assert (
        [x["title"] for x in out] == ["Big result"] and out[0]["source"] == "Src" and out[0]["link"] == "u1"
    )


def test_yahoo_429_sets_backoff_and_raises():
    src, store = make(lambda r: httpx.Response(429))
    with pytest.raises(SourceThrottled):
        src.yahoo_spark(["A.NS"])
    assert store.slow_until["yahoo"] == NOW.now().timestamp() + 120
    with pytest.raises(SourceThrottled):  # inside the window: no network call at all
        src.yahoo_chart("A.NS", "1d", "1y")


def test_spark_chunks_of_twenty():
    calls = []

    def handler(r: httpx.Request) -> httpx.Response:
        calls.append(r.url.params["symbols"].count(",") + 1)
        return httpx.Response(200, json={"spark": {"result": []}})

    src, _ = make(handler)
    src.yahoo_spark([f"S{i}.NS" for i in range(45)])
    assert calls == [20, 20, 5]


def test_nse_watch_falls_back_to_legacy_path():
    seen = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append(r.url.path)
        if "marketWatchApi" in r.url.path:
            return httpx.Response(404)
        return httpx.Response(200, json={"data": [{"symbol": "ABC", "series": "EQ", "lastPrice": 1}]})

    src, _ = make(handler)
    assert list(src.nse_watch("NIFTY 500")) == ["ABC"]
    assert seen == ["/api/NextApi/apiClient/marketWatchApi", "/api/equity-stockIndices"]


def test_nifty500_csv():
    csv_bytes = b"Company Name,Industry,Symbol,Series,ISIN Code\nFoo Ltd,Power,FOO,EQ,X\n"
    src, _ = make(lambda r: httpx.Response(200, content=csv_bytes))
    assert src.nifty500_list() == {"FOO": {"name": "Foo Ltd", "industry": "Power"}}


def test_option_chain_uses_contract_info_then_v3():
    seen = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append((r.url.path, dict(r.url.params)))
        if r.url.path.endswith("contract-info"):
            return httpx.Response(200, json={"expiryDates": ["01-Dec-2026", "08-Dec-2026"]})
        return httpx.Response(
            200, json={"records": {"underlyingValue": 25000.0, "data": [{"strikePrice": 25000}]}}
        )

    src, _ = make(handler)
    rec = src.nse_option_chain("NIFTY", True)
    assert rec["underlyingValue"] == 25000.0
    assert seen[1] == (
        "/api/option-chain-v3",
        {"type": "Indices", "symbol": "NIFTY", "expiry": "01-Dec-2026"},
    )
    src.nse_option_chain("RELIANCE", False, expiry="29-Dec-2026")
    assert seen[2][1]["type"] == "Equity" and seen[2][1]["expiry"] == "29-Dec-2026"


def test_wiki_returns_none_when_blocked():
    src, _ = make(lambda r: httpx.Response(403))
    assert src.wiki_summary("Reliance Industries Limited") is None
