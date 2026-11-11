from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import httpx
import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock, SteppingClock
from stocksense.core.state import State
from stocksense.data.ingest_common import IngestContext
from stocksense.data.ingest_reference import (
    ingest_etf_list,
    ingest_fno_list,
    ingest_holidays,
    ingest_listed_equities,
    ingest_price_bands,
    ingest_reference,
    ingest_sectors,
    ingest_symbol_changes,
    parse_etf,
    parse_fno_symbols,
    parse_holidays,
    parse_listed,
    parse_price_bands,
    parse_sectors,
    parse_symbol_changes,
)
from stocksense.data.io import read, upsert
from stocksense.data.nse_http import NseClient

FILES = Path(__file__).resolve().parents[1] / "fixtures" / "files"
TODAY = date(2026, 11, 11)
A = "https://nsearchives.nseindia.com"
URLS = {
    "symbolchange": f"{A}/content/equities/symbolchange.csv",
    "listed": f"{A}/content/equities/EQUITY_L.csv",
    "etf": f"{A}/content/equities/eq_etfseclist.csv",
    "bands": f"{A}/content/equities/sec_list.csv",
    "lots": f"{A}/content/fo/fo_mktlots.csv",
    "n500": f"{A}/content/indices/ind_nifty500list.csv",
    "total": f"{A}/content/indices/ind_niftytotalmarket_list.csv",
}


def _t(name: str) -> str:
    return (FILES / name).read_text(encoding="utf-8")


class Server:
    """MockTransport handler: archive files by URL, holiday API by year; records every request."""

    def __init__(self, routes: dict[str, bytes] | None = None, holidays: dict[str, list] | None = None):
        self.routes = routes or {}
        self.holidays = holidays or {}
        self.urls: list[str] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        u = str(req.url)
        self.urls.append(u)
        if req.url.path == "/api/holiday-master":
            year = req.url.params.get("year", str(TODAY.year))
            return httpx.Response(200, json={"CM": self.holidays.get(year, [])})
        base = u.split("?")[0]
        if base in self.routes:
            return httpx.Response(200, content=self.routes[base])
        return httpx.Response(404)


def _ctx(settings, server: Server) -> IngestContext:
    clock = FixedClock(datetime(2026, 11, 11, 10, 0, tzinfo=IST))
    nse = NseClient(
        SteppingClock(datetime(2026, 11, 11, 10, 0, tzinfo=IST)), transport=httpx.MockTransport(server)
    )
    state = State(settings, clock)
    state.migrate()
    cal = TradingCalendar.from_dates([date(2026, 11, 6), date(2026, 11, 9)])
    return IngestContext(settings=settings, clock=clock, calendar=cal, nse=nse, state=state)


def _hol(day: str, desc: str) -> dict:
    return {"tradingDate": day, "weekDay": "x", "description": desc}


# ---- parsers ------------------------------------------------------------------------------------


def test_parse_symbol_changes_sample():
    df = parse_symbol_changes(_t("symbolchange_sample.csv"))
    # fixture: header + 10 rows; first is NIPPON ... GO, RDAXEDG -> NDAXEDG on 30-OCT-2019 (name stripped)
    assert df.height == 10
    r = df.row(0, named=True)
    assert r["old_symbol"] == "RDAXEDG" and r["new_symbol"] == "NDAXEDG"
    assert r["change_date"] == date(2019, 10, 30)
    assert r["company"].startswith("NIPPON INDIA MF") and r["company"] == r["company"].strip()
    assert df.schema["change_date"] == pl.Date


def test_parse_symbol_changes_mixed_case_month():
    text = "NAME OF THE COMPANY,OLD SYMBOL,NEW SYMBOL,EFFECTIVE DATE\n X ltd ,AAA ,BBB,05-Mar-2021\n"
    assert parse_symbol_changes(text)["change_date"].to_list() == [date(2021, 3, 5)]


def test_parse_listed_sample_strips_leading_space_headers():
    df = parse_listed(_t("equity_l_sample.csv"), date(2026, 11, 11))
    # fixture: header + 10 rows; first 20MICRONS EQ listed 06-OCT-2008, face 5, ISIN INE144J01027
    assert df.height == 10
    r = df.row(0, named=True)
    assert r["symbol"] == "20MICRONS" and r["series"] == "EQ" and r["name"] == "20 Microns Limited"
    assert r["listing_date"] == date(2008, 10, 6) and r["face_value"] == 5.0
    assert r["isin"] == "INE144J01027" and r["as_of"] == date(2026, 11, 11)


def test_parse_etf_sample():
    df = parse_etf(_t("eq_etfseclist_sample.csv"), date(2026, 11, 11))
    # fixture: header + 10 rows; first NIFTYBEES, underlying "Nifty 50", name NIPINDETFNIFTYBEES
    assert df.height == 10
    r = df.row(0, named=True)
    assert (r["symbol"], r["underlying"], r["name"]) == ("NIFTYBEES", "Nifty 50", "NIPINDETFNIFTYBEES")
    assert r["as_of"] == date(2026, 11, 11)


def test_parse_sectors_sample():
    df = parse_sectors(_t("ind_nifty500list_sample.csv"), "ind_nifty500list.csv", date(2026, 11, 11))
    # fixture: header + 10 rows; first 360ONE, Financial Services, "360 ONE WAM Ltd."
    assert df.height == 10
    r = df.row(0, named=True)
    assert (r["symbol"], r["industry"], r["company"]) == ("360ONE", "Financial Services", "360 ONE WAM Ltd.")
    assert r["isin"] == "INE466L01038" and r["source"] == "ind_nifty500list.csv"


def test_parse_price_bands_sample():
    df = parse_price_bands(_t("sec_list_sample.csv"), date(2026, 11, 11))
    # fixture: header + 10 rows; ABAN BE band "2"
    assert df.height == 10
    r = df.row(0, named=True)
    assert (r["symbol"], r["series"], r["band"], r["date"]) == ("ABAN", "BE", "2", date(2026, 11, 11))


def test_parse_holidays_sample():
    j = json.loads(_t("holiday_master_sample.json"))
    df = parse_holidays(j, fetched_at=datetime(2026, 11, 11, 10, 0, tzinfo=IST))
    # CM list of 6 rows; first 15-Jan-2026; the FO list is ignored
    assert df.height == 6
    assert df["date"].to_list()[:2] == [date(2026, 1, 15), date(2026, 1, 26)]
    assert df["description"][1] == "Republic Day"


def test_parse_fno_symbols_excludes_index_rows():
    syms = parse_fno_symbols(_t("fo_mktlots_sample.csv"))
    # sample: 6 index rows (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, NIFTYFPI, NIFTYNXT50), the section
    # header row (SYMBOL "Symbol") and two stocks
    assert syms == ["ADANIENSOL", "ANANDRATHI"]


# ---- ingesters ----------------------------------------------------------------------------------


def test_symbol_changes_replace_whole_dataset(tmp_data_root):
    srv = Server({URLS["symbolchange"]: _t("symbolchange_sample.csv").encode()})
    ctx = _ctx(tmp_data_root, srv)
    r = ingest_symbol_changes(ctx)
    assert (r.dataset, r.rows, r.status) == ("symbol_changes", 10, "ok")
    r2 = ingest_symbol_changes(ctx)  # rerun: whole dataset replaced, count unchanged
    assert r2.rows == 10
    assert read(tmp_data_root, "symbol_changes").collect().height == 10


def test_listed_equities_snapshot_only_when_content_changes(tmp_data_root):
    body = _t("equity_l_sample.csv").encode()
    srv = Server({URLS["listed"]: body})
    ctx = _ctx(tmp_data_root, srv)
    assert ingest_listed_equities(ctx).status == "ok"
    assert ingest_listed_equities(ctx).status == "skipped"  # same content: no new snapshot
    assert read(tmp_data_root, "listed_equities").collect().height == 10
    # content changes (one row removed) on a later day: a second snapshot is appended
    srv.routes[URLS["listed"]] = b"\n".join(body.split(b"\n")[:-2]) + b"\n"
    ctx.clock = FixedClock(datetime(2026, 11, 12, 10, 0, tzinfo=IST))
    assert ingest_listed_equities(ctx).status == "ok"
    df = read(tmp_data_root, "listed_equities").collect()
    # 10 rows (as_of 11-11) + 9 rows (as_of 11-12)
    assert df.height == 19
    assert sorted(df["as_of"].unique().to_list()) == [date(2026, 11, 11), date(2026, 11, 12)]


def test_etf_list_snapshot_change_detection(tmp_data_root):
    srv = Server({URLS["etf"]: _t("eq_etfseclist_sample.csv").encode()})
    ctx = _ctx(tmp_data_root, srv)
    assert ingest_etf_list(ctx).status == "ok"
    assert ingest_etf_list(ctx).status == "skipped"
    assert read(tmp_data_root, "etf_list").collect().height == 10


def test_sectors_nifty500_row_wins_over_total_market(tmp_data_root):
    n500 = "Company Name,Industry,Symbol,Series,ISIN Code\nA Ltd,Capital Goods,AAA,EQ,INE0000A0001\n"
    tot = (
        "Company Name,Industry,Symbol,Series,ISIN Code\n"
        "A Ltd,Services,AAA,EQ,INE0000A0001\nB Ltd,Chemicals,BBB,EQ,INE0000B0001\n"
    )
    srv = Server({URLS["n500"]: n500.encode(), URLS["total"]: tot.encode()})
    ctx = _ctx(tmp_data_root, srv)
    r = ingest_sectors(ctx)
    assert (r.dataset, r.rows, r.status) == ("sectors", 2, "ok")
    df = read(tmp_data_root, "sectors").collect().sort("symbol")
    # AAA is in both files: the Nifty 500 row (Capital Goods) is kept; BBB comes from the total-market file
    assert df["industry"].to_list() == ["Capital Goods", "Chemicals"]
    assert df["source"].to_list() == ["ind_nifty500list.csv", "ind_niftytotalmarket_list.csv"]
    assert ingest_sectors(ctx).status == "skipped"


def test_price_bands_append_daily_and_idempotent(tmp_data_root):
    srv = Server({URLS["bands"]: _t("sec_list_sample.csv").encode()})
    ctx = _ctx(tmp_data_root, srv)
    ctx.calendar = TradingCalendar.from_dates([date(2026, 11, 9), date(2026, 11, 11)])
    assert ingest_price_bands(ctx).rows == 10
    assert ingest_price_bands(ctx).rows == 10  # same day again: upsert by (symbol, series, date)
    assert read(tmp_data_root, "price_bands").collect().height == 10


def test_holidays_replace_rows_of_fetched_year(tmp_data_root):
    h = {"2026": [_hol("10-Nov-2026", "Diwali-Balipratipada"), _hol("24-Nov-2026", "Guru Nanak")]}
    srv = Server(holidays=h)
    ctx = _ctx(tmp_data_root, srv)
    r = ingest_holidays(ctx, years=[2026])
    assert (r.dataset, r.rows, r.status) == ("holidays", 2, "ok")
    # the API now lists only one 2026 holiday: the stale 24-Nov row must disappear
    h["2026"] = [_hol("10-Nov-2026", "Diwali-Balipratipada")]
    ingest_holidays(ctx, years=[2026])
    df = read(tmp_data_root, "holidays").collect()
    assert df["date"].to_list() == [date(2026, 11, 10)]


def test_holidays_keep_other_years(tmp_data_root):
    h = {"2025": [_hol("26-Feb-2025", "Mahashivratri")], "2026": [_hol("10-Nov-2026", "Diwali")]}
    ctx = _ctx(tmp_data_root, Server(holidays=h))
    ingest_holidays(ctx, years=[2025])
    ingest_holidays(ctx, years=[2026])
    got = read(tmp_data_root, "holidays").collect()["date"].to_list()
    assert got == [date(2025, 2, 26), date(2026, 11, 10)]


def test_fno_list_goes_to_kv(tmp_data_root):
    srv = Server({URLS["lots"]: _t("fo_mktlots_sample.csv").encode()})
    ctx = _ctx(tmp_data_root, srv)
    r = ingest_fno_list(ctx)
    assert (r.dataset, r.rows, r.status) == ("fno_symbols", 2, "ok")
    assert json.loads(ctx.state.kv_get("fno_symbols:2026-11-11")) == ["ADANIENSOL", "ANANDRATHI"]


def test_missing_file_is_failed_not_raised(tmp_data_root):
    ctx = _ctx(tmp_data_root, Server())
    assert ingest_symbol_changes(ctx).status == "failed"


def test_ingest_reference_runs_all_and_rebuilds_calendar(tmp_data_root):
    n500 = "Company Name,Industry,Symbol,Series,ISIN Code\nA Ltd,Capital Goods,AAA,EQ,INE0000A0001\n"
    srv = Server(
        {
            URLS["symbolchange"]: _t("symbolchange_sample.csv").encode(),
            URLS["listed"]: _t("equity_l_sample.csv").encode(),
            URLS["etf"]: _t("eq_etfseclist_sample.csv").encode(),
            URLS["bands"]: _t("sec_list_sample.csv").encode(),
            URLS["lots"]: _t("fo_mktlots_sample.csv").encode(),
            URLS["n500"]: n500.encode(),
            URLS["total"]: n500.encode(),
        },
        holidays={str(TODAY.year): [_hol("10-Nov-2026", "Diwali"), _hol("24-Nov-2026", "Guru Nanak")]},
    )
    ctx = _ctx(tmp_data_root, srv)
    eq = pl.DataFrame(
        {"symbol": ["X"] * 2, "series": ["EQ"] * 2, "date": [date(2026, 11, 6), date(2026, 11, 9)],
         "open": [1.0] * 2, "high": [1.0] * 2, "low": [1.0] * 2, "close": [1.0] * 2, "volume": [1] * 2}
    )  # fmt: skip
    upsert(tmp_data_root, "bhavcopy_eq", eq)
    assert ctx.calendar.is_session(date(2026, 11, 10))  # before holidays are known
    res = ingest_reference(ctx)
    by = {r.dataset: r for r in res}
    assert set(by) == {
        "symbol_changes", "listed_equities", "etf_list", "sectors", "price_bands", "holidays", "fno_symbols",
    }  # fmt: skip
    assert all(r.status == "ok" for r in res), res
    assert not ctx.calendar.is_session(date(2026, 11, 10))  # calendar rebuilt with the holidays
    assert ctx.calendar.is_session(date(2026, 11, 11))
