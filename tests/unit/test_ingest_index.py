from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import httpx
import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock, SteppingClock
from stocksense.core.state import State
from stocksense.data.ingest_common import IngestContext
from stocksense.data.ingest_index import (
    INDEX_NAME_MAP,
    backfill_index,
    canonical_index_name,
    index_url,
    ingest_index_day,
    ingest_index_range,
    parse_index,
    rebuild_index_from_cache,
)
from stocksense.data.io import read, upsert
from stocksense.data.nse_http import NseClient

FILES = Path(__file__).resolve().parents[1] / "fixtures" / "files"


def _t(name: str) -> str:
    return (FILES / name).read_text(encoding="utf-8")


def _for_day(d: date, base: str = "ind_close_all_2026_sample.csv") -> bytes:
    """The 2026 sample with its 09-11-2026 date replaced by `d`."""
    return _t(base).replace("09-11-2026", f"{d:%d-%m-%Y}").encode()


class Recorder:
    def __init__(self, routes: dict[str, bytes] | None = None) -> None:
        self.routes = routes or {}
        self.urls: list[str] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        u = str(req.url)
        self.urls.append(u)
        if u in self.routes:
            return httpx.Response(200, content=self.routes[u])
        return httpx.Response(404)


def _ctx(settings, handler, today=date(2026, 11, 11)) -> IngestContext:
    clock = FixedClock(datetime(today.year, today.month, today.day, 10, 0, tzinfo=IST))
    nse = NseClient(
        SteppingClock(datetime(2026, 11, 11, 10, 0, tzinfo=IST)), transport=httpx.MockTransport(handler)
    )
    state = State(settings, clock)
    state.migrate()
    cal = TradingCalendar.from_dates([date(2026, 9, 17), date(2026, 9, 18)])
    return IngestContext(settings=settings, clock=clock, calendar=cal, nse=nse, state=state)


# ---- names and parser ---------------------------------------------------------------------------


def test_canonical_name_maps_old_and_new_names_to_one():
    # CNX Nifty (2015) and Nifty 50 (2026) are the same index
    assert canonical_index_name("CNX Nifty") == "NIFTY 50"
    assert canonical_index_name("Nifty 50") == "NIFTY 50"
    assert canonical_index_name("  Nifty   Bank ") == "NIFTY BANK"
    assert canonical_index_name("CNX Bank") == "NIFTY BANK"
    assert canonical_index_name("CNX Nifty Junior") == "NIFTY NEXT 50"
    assert canonical_index_name("Nifty Next 50") == "NIFTY NEXT 50"
    assert canonical_index_name("India VIX") == "INDIA VIX"
    assert canonical_index_name("Nifty 50 Arbitrage") == "NIFTY 50 ARBITRAGE"  # unmapped: upper-cased only


def test_the_twenty_plan_renames_are_in_the_map():
    wanted = {
        "CNX NIFTY": "NIFTY 50", "CNX NIFTY JUNIOR": "NIFTY NEXT 50", "CNX 100": "NIFTY 100",
        "CNX 200": "NIFTY 200", "CNX 500": "NIFTY 500", "CNX MIDCAP": "NIFTY MIDCAP 100",
        "CNX SMALLCAP": "NIFTY SMALLCAP 100", "BANK NIFTY": "NIFTY BANK", "CNX IT": "NIFTY IT",
        "CNX FMCG": "NIFTY FMCG", "CNX PHARMA": "NIFTY PHARMA", "CNX AUTO": "NIFTY AUTO",
        "CNX METAL": "NIFTY METAL", "CNX REALTY": "NIFTY REALTY", "CNX ENERGY": "NIFTY ENERGY",
        "CNX INFRA": "NIFTY INFRASTRUCTURE", "CNX PSU BANK": "NIFTY PSU BANK", "CNX MEDIA": "NIFTY MEDIA",
        "CNX FINANCE": "NIFTY FINANCIAL SERVICES", "INDIA VIX": "INDIA VIX",
    }  # fmt: skip
    for k, v in wanted.items():
        assert INDEX_NAME_MAP[k] == v


def test_parse_2015_sample():
    df = parse_index(_t("ind_close_all_2015_sample.csv"), date(2015, 1, 5))
    # fixture: header + 5 rows; CNX Nifty O 8407.95 H 8445.6 L 8363.9 C 8378.4, volume 118160545,
    # turnover 5525.52 Cr, P/E 21.4, P/B 3.53, yield 1.26
    assert df is not None and df.height == 5
    r = df.filter(pl.col("index_name") == "NIFTY 50").row(0, named=True)
    assert r["date"] == date(2015, 1, 5)
    assert (r["open"], r["high"], r["low"], r["close"]) == (8407.95, 8445.6, 8363.9, 8378.4)
    assert (r["pe"], r["pb"], r["div_yield"]) == (21.4, 3.53, 1.26)
    assert (r["volume"], r["turnover_cr"]) == (118160545.0, 5525.52)
    vix = df.filter(pl.col("index_name") == "INDIA VIX").row(0, named=True)
    # India VIX has "-" for volume, turnover, P/E, P/B, yield
    assert vix["close"] == 14.15
    assert vix["volume"] is None and vix["pe"] is None and vix["div_yield"] is None
    assert df.schema["date"] == pl.Date and df.schema["close"] == pl.Float64


def test_2015_and_2026_nifty_share_a_canonical_name():
    a = parse_index(_t("ind_close_all_2015_sample.csv"), date(2015, 1, 5))
    b = parse_index(_t("ind_close_all_2026_sample.csv"), date(2026, 11, 9))
    assert a is not None and b is not None
    assert "NIFTY 50" in a["index_name"].to_list() and "NIFTY 50" in b["index_name"].to_list()
    assert "NIFTY BANK" in a["index_name"].to_list() and "NIFTY BANK" in b["index_name"].to_list()


def test_parse_2026_dash_becomes_null():
    df = parse_index(_t("ind_close_all_2026_sample.csv"), date(2026, 11, 9))
    assert df is not None and df.height == 5
    r = df.filter(pl.col("index_name") == "NIFTY50 DIVIDEND POINTS").row(0, named=True)
    # open, high, low, volume, turnover, pe, pb, yield are "-"; close is 212.78
    assert r["open"] is None and r["high"] is None and r["low"] is None
    assert r["close"] == 212.78 and r["volume"] is None and r["pe"] is None


def test_parse_rejects_a_file_of_another_day():
    # NSE serves the previous session's file on a holiday: every Index Date is 05-01-2015
    assert parse_index(_t("ind_close_all_2015_sample.csv"), date(2015, 1, 6)) is None


def test_parse_drops_stray_rows_of_another_date_but_keeps_the_file():
    text = _t("ind_close_all_2026_sample.csv") + "Stale Index,06-11-2026,1,1,1,1,0,0,-,-,-,-,-\n"
    df = parse_index(text, date(2026, 11, 9))
    assert df is not None and df.height == 5 and "STALE INDEX" not in df["index_name"].to_list()


# ---- ingest -------------------------------------------------------------------------------------


def test_ingest_day_ok_idempotent_and_skipped_without_http(tmp_data_root):
    d = date(2026, 9, 18)
    srv = Recorder({index_url(d): _for_day(d)})
    ctx = _ctx(tmp_data_root, srv)
    res = ingest_index_day(ctx, d)
    assert [(r.dataset, r.rows, r.status) for r in res] == [("index_daily", 5, "ok"), ("india_vix", 1, "ok")]
    assert read(tmp_data_root, "index_daily").collect().height == 5
    n = len(srv.urls)
    assert ingest_index_day(ctx, d)[0].status == "skipped" and len(srv.urls) == n
    ingest_index_day(ctx, d, counts={})  # forced re-parse from cache: same row count
    assert read(tmp_data_root, "index_daily").collect().height == 5
    assert len(srv.urls) == n


def test_vix_extension_appends_only_new_dates(tmp_data_root):
    old = pl.DataFrame({"date": [date(2026, 9, 16), date(2026, 9, 17)], "close": [10.0, 11.0]})
    upsert(tmp_data_root, "india_vix", old)
    srv = Recorder(
        {
            index_url(date(2026, 9, 17)): _for_day(date(2026, 9, 17)),
            index_url(date(2026, 9, 18)): _for_day(date(2026, 9, 18)),
        }
    )
    ctx = _ctx(tmp_data_root, srv)
    ingest_index_day(ctx, date(2026, 9, 17))  # not newer than the stored VIX: stored 11.0 is untouched
    ingest_index_day(ctx, date(2026, 9, 18))  # newer: appended, close 17.42
    v = read(tmp_data_root, "india_vix").collect().sort("date")
    assert v["date"].to_list() == [date(2026, 9, 16), date(2026, 9, 17), date(2026, 9, 18)]
    assert v["close"].to_list() == [10.0, 11.0, 17.42]


def test_holiday_file_of_another_day_is_no_session_and_not_cached(tmp_data_root):
    d = date(2026, 9, 21)  # a Monday, asked for; NSE answers with the 2026-09-18 file
    srv = Recorder({index_url(d): _for_day(date(2026, 9, 18))})
    ctx = _ctx(tmp_data_root, srv)
    res = ingest_index_day(ctx, d)
    assert [(r.dataset, r.status) for r in res] == [("index_daily", "no_session")]
    assert read(tmp_data_root, "index_daily").collect().height == 0
    cache = tmp_data_root.paths.data_root / "cache" / "index" / "2026" / "2026-09-21.csv"
    assert not cache.exists()
    assert ctx.state.kv_get("no_session:index:2026-09-21") == "1"  # older than 3 days: remembered


def test_404_old_day_marked_then_no_request(tmp_data_root):
    srv = Recorder()
    ctx = _ctx(tmp_data_root, srv)
    sat = date(2026, 9, 19)
    assert ingest_index_day(ctx, sat)[0].status == "no_session"
    n = len(srv.urls)
    assert ingest_index_day(ctx, sat)[0].status == "skipped" and len(srv.urls) == n


def test_range_every_calendar_day(tmp_data_root):
    d18 = date(2026, 9, 18)
    ctx = _ctx(tmp_data_root, Recorder({index_url(d18): _for_day(d18)}))
    res = ingest_index_range(ctx, date(2026, 9, 18), date(2026, 9, 20))
    got = [(r.date.day, r.dataset, r.status) for r in res]
    assert got == [
        (18, "index_daily", "ok"),
        (18, "india_vix", "ok"),
        (19, "index_daily", "no_session"),
        (20, "index_daily", "no_session"),
    ]


def test_backfill_walks_back_until_consecutive_misses(tmp_data_root):
    days = [date(2026, 9, 18), date(2026, 9, 17), date(2026, 9, 16)]
    srv = Recorder({index_url(d): _for_day(d) for d in days})
    ctx = _ctx(tmp_data_root, srv)
    res = backfill_index(ctx, date(2026, 9, 18), max_misses=3)
    idx = [r for r in res if r.dataset == "index_daily"]
    # 18, 17, 16 hit; 15, 14, 13 miss -> stop after 3 consecutive misses (6 days visited)
    assert [(r.date.day, r.status) for r in idx] == [
        (18, "ok"), (17, "ok"), (16, "ok"), (15, "no_session"), (14, "no_session"), (13, "no_session"),
    ]  # fmt: skip
    assert len(srv.urls) == 6
    first = read(tmp_data_root, "index_daily").select(pl.col("date").min()).collect().item()
    assert first == date(2026, 9, 16)


def test_backfill_is_resumable_and_stored_days_cost_no_http(tmp_data_root):
    days = [date(2026, 9, 18), date(2026, 9, 17)]
    srv = Recorder({index_url(d): _for_day(d) for d in days})
    ctx = _ctx(tmp_data_root, srv)
    backfill_index(ctx, date(2026, 9, 18), max_misses=2)
    n = len(srv.urls)  # 18, 17 hit + 16, 15 miss = 4
    assert n == 4
    backfill_index(ctx, date(2026, 9, 18), max_misses=2)
    assert len(srv.urls) == n  # stored days and marked misses are not fetched again


def test_rebuild_from_cache_applies_the_current_name_map(tmp_data_root):
    d = date(2015, 1, 5)
    ctx = _ctx(tmp_data_root, Recorder())
    cache = tmp_data_root.paths.data_root / "cache" / "index" / "2015" / "2015-01-05.csv"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(_t("ind_close_all_2015_sample.csv").encode())
    # stored with a stale (un-mapped) name for the same day
    stale = pl.DataFrame(
        {"index_name": ["CNX NIFTY"], "date": [d], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}
    )
    upsert(tmp_data_root, "index_daily", stale)
    n = rebuild_index_from_cache(ctx)
    assert n == {d: 5}
    df = read(tmp_data_root, "index_daily").collect()
    # 5 rows, the stale CNX NIFTY row replaced by NIFTY 50 with the real close 8378.4
    assert df.height == 5 and "CNX NIFTY" not in df["index_name"].to_list()
    assert df.filter(pl.col("index_name") == "NIFTY 50")["close"].to_list() == [8378.4]
