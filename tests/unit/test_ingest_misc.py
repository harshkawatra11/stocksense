from __future__ import annotations

import copy
import json
from datetime import date, datetime
from pathlib import Path

import httpx
import polars as pl

from stocksense.brokers.types import Quote
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock, SteppingClock
from stocksense.core.state import State
from stocksense.data.ingest_common import IngestContext
from stocksense.data.ingest_macro import available_ts, ingest_macro, parse_stooq, parse_yahoo
from stocksense.data.ingest_misc import (
    backfill_participant_oi,
    capture_preopen,
    capture_quotes,
    capture_spreads,
    ingest_deals_today,
    ingest_fii_dii,
    ingest_participant_oi_day,
    parse_deals,
    parse_fii_dii,
    parse_participant_oi,
    parse_preopen,
    seed_fii_dii_from_snapshots,
)
from stocksense.data.io import read
from stocksense.data.nse_http import NseClient

FILES = Path(__file__).resolve().parents[1] / "fixtures" / "files"
NOW = datetime(2026, 11, 12, 9, 5, tzinfo=IST)


def _ctx(settings, handler) -> IngestContext:
    clock = FixedClock(NOW)
    nse = NseClient(SteppingClock(NOW), transport=httpx.MockTransport(handler))
    state = State(settings, clock)
    state.migrate()
    cal = TradingCalendar.from_dates([date(2026, 11, 6), date(2026, 11, 9), date(2026, 11, 11)])
    return IngestContext(settings=settings, clock=clock, calendar=cal, nse=nse, state=state)


# ---- macro ---------------------------------------------------------------------------------------------


def test_available_ts_rule():
    # US-session series: next calendar day 06:00 IST; others: same day 23:30 IST
    assert available_ts("SPX", date(2026, 11, 6)) == datetime(2026, 11, 7, 6, 0, tzinfo=IST)
    assert available_ts("US10Y", date(2026, 11, 6)) == datetime(2026, 11, 7, 6, 0, tzinfo=IST)
    assert available_ts("USDINR", date(2026, 11, 6)) == datetime(2026, 11, 6, 23, 30, tzinfo=IST)
    assert available_ts("GOLD", date(2026, 11, 6)) == datetime(2026, 11, 6, 23, 30, tzinfo=IST)


def test_parse_stooq_csv_and_html_and_no_data():
    csv = "Date,Open,High,Low,Close,Volume\n2026-11-05,1,2,0.5,1.5,10\n2026-11-06,1,2,0.5,1.75,12\n"
    df = parse_stooq(csv)
    assert df["date"].to_list() == [date(2026, 11, 5), date(2026, 11, 6)] and df["close"].to_list() == [
        1.5,
        1.75,
    ]
    assert parse_stooq("<!DOCTYPE html><html></html>").height == 0
    assert parse_stooq("No data").height == 0


def test_parse_yahoo_uses_exchange_local_date():
    # 2026-11-06 09:30 New York = 14:30 UTC; gmtoffset -18000 -> local date 2026-11-06
    ts = int(datetime(2026, 11, 6, 14, 30, tzinfo=IST).timestamp())  # 09:00 UTC = 04:00 NY: same local date
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"gmtoffset": -18000},
                    "timestamp": [ts, ts + 86400],
                    "indicators": {"quote": [{"close": [100.5, None]}]},
                }
            ]
        }
    }
    df = parse_yahoo(payload)
    assert df["date"].to_list() == [date(2026, 11, 6)] and df["close"].to_list() == [100.5]


def test_ingest_macro_falls_back_to_yahoo_and_records_missing(tmp_data_root):
    ts = int(datetime(2026, 11, 6, 12, 0, tzinfo=IST).timestamp())

    def handler(req: httpx.Request) -> httpx.Response:
        if "stooq.com" in str(req.url):
            return httpx.Response(200, text="<!DOCTYPE html><html></html>")
        if "GC=F" in str(req.url):
            return httpx.Response(404, json={})
        body = {
            "chart": {
                "result": [
                    {
                        "meta": {"gmtoffset": 19800},
                        "timestamp": [ts],
                        "indicators": {"quote": [{"close": [88.0]}]},
                    }
                ]
            }
        }
        return httpx.Response(200, json=body)

    ctx = _ctx(tmp_data_root, lambda r: httpx.Response(404))
    res = ingest_macro(ctx, http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert sum(r.status == "ok" for r in res) == 6 and sum(r.status == "failed" for r in res) == 1
    df = read(tmp_data_root, "macro_daily").collect()
    assert df["series"].n_unique() == 6 and "GOLD" not in df["series"].to_list()
    assert ctx.state.kv_get("macro_unavailable:GOLD") is not None
    spx = df.filter(pl.col("series") == "SPX").row(0, named=True)
    assert spx["available_ts"] == datetime(2026, 11, 7, 6, 0, tzinfo=IST)


# ---- participant OI -------------------------------------------------------------------------------------


def test_parse_participant_oi_sample():
    title_date, df = parse_participant_oi((FILES / "participant_oi_sample.csv").read_bytes())
    assert title_date == date(2026, 11, 9)
    assert df["client_type"].to_list() == ["CLIENT", "DII", "FII", "PRO"]  # TOTAL dropped
    fii = df.filter(pl.col("client_type") == "FII").row(0, named=True)
    assert fii["fut_idx_long"] == 52382.0 and fii["fut_idx_short"] == 143123.0
    assert fii["opt_idx_put_short"] == 582033.0 and fii["opt_stk_call_long"] == 319289.0
    assert fii["total_long"] == 5924644.0


def test_participant_oi_day_stores_and_wrong_title_date_is_no_session(tmp_data_root):
    raw = (FILES / "participant_oi_sample.csv").read_bytes()
    calls = []

    def handler(req):
        calls.append(req.url.path)
        return httpx.Response(200, content=raw)

    ctx = _ctx(tmp_data_root, handler)
    assert ingest_participant_oi_day(ctx, date(2026, 11, 9)).status == "ok"
    df = read(tmp_data_root, "participant_oi").collect()
    assert df.height == 4 and df["date"].unique().to_list() == [date(2026, 11, 9)]
    # the server answers a holiday with the previous day's file: title says Nov 09 -> rejected
    assert ingest_participant_oi_day(ctx, date(2026, 11, 10)).status == "no_session"
    assert read(tmp_data_root, "participant_oi").collect().height == 4


def test_participant_oi_backfill_stops_after_30_misses(tmp_data_root):
    raw = (FILES / "participant_oi_sample.csv").read_bytes()
    ok_days = {"09112026", "06112026"}

    def handler(req):
        d = req.url.path.rsplit("_", 1)[1][:8]
        if d in ok_days:
            return httpx.Response(
                200, content=raw.replace(b"Nov 09", b"Nov 09" if d == "09112026" else b"Nov 06")
            )
        return httpx.Response(404)

    ctx = _ctx(tmp_data_root, handler)
    res = backfill_participant_oi(ctx, date(2026, 11, 9))
    # 11-09 ok, then 7 misses ... 11-06 ok (resets), then 30 misses
    assert sum(r.status == "ok" for r in res) == 2
    assert len(res) == 1 + 2 + 1 + 30


# ---- FII / DII ------------------------------------------------------------------------------------------


def test_parse_fii_dii_sample_maps_fpi_to_fii():
    rows = json.loads((FILES / "fii_dii_sample.json").read_text(encoding="utf-8"))
    df = parse_fii_dii(rows)
    assert sorted(df["category"].to_list()) == ["DII", "FII"]
    fii = df.filter(pl.col("category") == "FII").row(0, named=True)
    assert fii["date"] == date(2026, 11, 11) and fii["buy_value"] == 30087.21 and fii["net_value"] == 12563.33


def test_ingest_fii_dii_idempotent(tmp_data_root):
    rows = json.loads((FILES / "fii_dii_sample.json").read_text(encoding="utf-8"))
    ctx = _ctx(tmp_data_root, lambda r: httpx.Response(200, json=rows))
    assert ingest_fii_dii(ctx).rows == 2
    assert ingest_fii_dii(ctx).rows == 2
    assert read(tmp_data_root, "fii_dii").collect().height == 2


def test_seed_fii_dii_from_adopted_snapshots(tmp_data_root, tmp_data_dir):
    d = tmp_data_dir / "parquet" / "nse_snapshots" / "fii_dii"
    d.mkdir(parents=True)
    snap = pl.DataFrame(
        {
            "buyValue": ["1.5", "2.5", "9.0"],
            "category": ["DII", "FII/FPI", "DII"],
            "date": ["21-Sep-2026", "21-Sep-2026", "21-Sep-2026"],
            "netValue": ["0.5", "-1.0", "3.0"],
            "sellValue": ["1.0", "3.5", "6.0"],
            "snapshot_ts": [datetime(2026, 9, 22, 13), datetime(2026, 9, 22, 13), datetime(2026, 9, 22, 15)],
        }
    )
    snap.write_parquet(d / "2026-09-22.parquet")
    ctx = _ctx(tmp_data_root, lambda r: httpx.Response(404))
    assert seed_fii_dii_from_snapshots(ctx) == 2
    df = read(tmp_data_root, "fii_dii").collect().sort("category")
    # the later DII snapshot (15:00) wins
    assert df["buy_value"].to_list() == [9.0, 2.5] and df["category"].to_list() == ["DII", "FII"]


# ---- deals ----------------------------------------------------------------------------------------------


def test_parse_deals_samples():
    bulk = parse_deals((FILES / "bulk_sample.csv").read_bytes(), "bulk")
    assert (
        bulk.height >= 5
        and set(bulk["kind"]) == {"bulk"}
        and bulk["date"].unique().to_list() == [date(2026, 11, 11)]
    )
    first = bulk.row(0, named=True)
    assert (
        first["symbol"] == "AAATECH"
        and first["side"] == "BUY"
        and first["qty"] == 66829.0
        and first["price"] == 93.89
    )
    blk = parse_deals((FILES / "block_sample.csv").read_bytes(), "block")
    # the file repeats DIXON SELL 702481 @ 4140 twice in the real file; identical rows collapse
    assert blk.height == blk.unique().height and blk["price"].dtype == pl.Float64
    assert parse_deals(b"", "bulk").height == 0


def test_ingest_deals_today_appends_both_kinds(tmp_data_root):
    def handler(req):
        name = "bulk_sample.csv" if req.url.path.endswith("bulk.csv") else "block_sample.csv"
        return httpx.Response(200, content=(FILES / name).read_bytes())

    ctx = _ctx(tmp_data_root, handler)
    res = ingest_deals_today(ctx)
    assert [r.status for r in res] == ["ok", "ok"]
    df = read(tmp_data_root, "deals").collect()
    assert set(df["kind"]) == {"bulk", "block"}
    n = df.height
    ingest_deals_today(ctx)
    assert read(tmp_data_root, "deals").collect().height == n


# ---- pre-open / spreads / quotes -----------------------------------------------------------------------


def _preopen_payload():
    return json.loads((FILES / "preopen_sample.json").read_text(encoding="utf-8"))


def test_parse_preopen_sample_and_zero_final_price_is_null():
    p = _preopen_payload()
    df = parse_preopen(p, NOW)
    first = p["data"][0]
    assert df.height == 2 and df["ts"][0] == NOW
    assert df["indicative_price"][0] == first["metadata"]["iep"]
    assert df["final_price"][0] == first["detail"]["preOpenMarket"]["finalPrice"]
    assert df["buy_qty"][0] == first["detail"]["preOpenMarket"]["totalBuyQuantity"]
    assert df["prev_close"][0] == first["metadata"]["previousClose"]
    z = copy.deepcopy(p)
    z["data"][0]["detail"]["preOpenMarket"]["finalPrice"] = 0
    assert parse_preopen(z, NOW)["final_price"][0] is None


def test_capture_preopen_stores_rows_and_falls_back_to_keys(tmp_data_root):
    p = _preopen_payload()
    keys = []

    def handler(req):
        if "key" not in req.url.params:  # cookie warm-up pages
            return httpx.Response(200, text="")
        k = req.url.params["key"]
        keys.append(k)
        if k == "ALL":
            return httpx.Response(403, text="denied")  # persistent refusal -> DataNotAvailable after warm-ups
        one = copy.deepcopy(p)
        one["data"] = [p["data"][{"NIFTY": 0, "FO": 1}.get(k, 0)]]
        if k == "OTHERS":
            one["data"] = []
        return httpx.Response(200, json=one)

    ctx = _ctx(tmp_data_root, handler)
    assert capture_preopen(ctx) == 2
    assert {"NIFTY", "FO", "OTHERS"} <= set(keys)
    assert read(tmp_data_root, "preopen").collect().height == 2


def _q(sym, bid, ask, minute=1):
    return Quote(
        sym,
        datetime(2026, 11, 12, 10, minute, 37, tzinfo=IST),
        100.0,
        99.0,
        101.0,
        98.0,
        99.5,
        1000.0,
        bid,
        ask,
        10.0,
        20.0,
        110.0,
        90.0,
        "angel",
    )


def test_capture_spreads_keeps_only_two_sided_quotes(tmp_data_root):
    ctx = _ctx(tmp_data_root, lambda r: httpx.Response(404))
    qs = [_q("A", 99.9, 100.1), _q("B", 0.0, 100.1), _q("C", None, None)]
    assert capture_spreads(ctx, ["A", "B", "C"], lambda syms: qs) == 1
    df = read(tmp_data_root, "spreads").collect()
    assert df["symbol"].to_list() == ["A"] and df["ask"].to_list() == [100.1]


def test_capture_quotes_floors_to_the_minute(tmp_data_root):
    ctx = _ctx(tmp_data_root, lambda r: httpx.Response(404))
    # two quotes of A inside the same minute collapse to one row
    assert capture_quotes(ctx, [_q("A", 1, 2), _q("A", 1, 2), _q("A", 1, 2, minute=2)]) == 3
    df = read(tmp_data_root, "quotes_1m").collect()
    assert df.height == 2 and df["ts"].dt.second().unique().to_list() == [0]
