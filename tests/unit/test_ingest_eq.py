from __future__ import annotations

import io
import zipfile
from datetime import date, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock, SteppingClock
from stocksense.core.state import State
from stocksense.data import ingest_eq as ingest_mod
from stocksense.data.ingest_eq import (
    MISSING_SPECIAL_SESSIONS,
    IngestContext,
    IngestResult,
    McpCheck,
    cross_check_mcp,
    delivery_url,
    eq_url,
    ingest_eq_day,
    ingest_eq_range,
    parse_delivery,
    parse_legacy,
    parse_udiff,
    repair_days,
    sample_pairs,
)
from stocksense.data.io import read, upsert
from stocksense.data.nse_http import NseClient

FILES = Path(__file__).resolve().parents[1] / "fixtures" / "files"


def _zip(text: str, name: str = "x.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, text)
    return buf.getvalue()


def _udiff_zip() -> bytes:
    return _zip((FILES / "udiff_cm_sample.csv").read_text(encoding="utf-8"))


def _legacy_text(day: str = "03-FEB-2020") -> str:
    return (FILES / "legacy_cm_sample.csv").read_text(encoding="utf-8").replace("03-FEB-2020", day)


def _delivery_text(day: str = "29-Sep-2026") -> str:
    return (FILES / "sec_bhavdata_full_sample.csv").read_text(encoding="utf-8").replace("29-Sep-2026", day)


@pytest.fixture(autouse=True)
def _small_sessions(monkeypatch):
    """The real-file samples have 19 rows; shrink the 'is a session' bar so they count as sessions."""
    monkeypatch.setattr(ingest_mod, "MIN_SESSION_ROWS", 10)


class Recorder:
    """MockTransport handler: serves `routes` (url -> bytes), 404 otherwise, records every request."""

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
    cal = TradingCalendar.from_dates([date(2026, 9, 28), date(2026, 9, 29)])
    return IngestContext(settings=settings, clock=clock, calendar=cal, nse=nse, state=state)


def _eq_rows(d: date, n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [f"S{i:04d}" for i in range(n)],
            "series": ["EQ"] * n,
            "date": [d] * n,
            "open": [10.0] * n,
            "high": [11.0] * n,
            "low": [9.0] * n,
            "close": [10.5] * n,
            "prev_close": [10.0] * n,
            "volume": [100] * n,
            "era": ["udiff"] * n,
        }
    )


# ---- URLs and parsers ---------------------------------------------------------------------------


def test_urls_by_era():
    # udiff from 2024-07-08, legacy before; month is upper-case 3-letter in the legacy path
    assert eq_url(date(2026, 9, 29)).endswith("/content/cm/BhavCopy_NSE_CM_0_0_0_20260929_F_0000.csv.zip")
    assert eq_url(date(2024, 7, 8)).endswith("BhavCopy_NSE_CM_0_0_0_20240708_F_0000.csv.zip")
    assert eq_url(date(2024, 7, 5)).endswith("/content/historical/EQUITIES/2024/JUL/cm05JUL2024bhav.csv.zip")
    assert eq_url(date(2020, 2, 3)).endswith("/historical/EQUITIES/2020/FEB/cm03FEB2020bhav.csv.zip")
    assert delivery_url(date(2026, 9, 29)).endswith("/products/content/sec_bhavdata_full_29092026.csv")


def test_special_sessions_list_has_19_dates():
    assert len(MISSING_SPECIAL_SESSIONS) == 19
    assert MISSING_SPECIAL_SESSIONS[0] == date(2010, 2, 6)
    assert MISSING_SPECIAL_SESSIONS[-1] == date(2026, 2, 1)


def test_parse_udiff_sample():
    df = parse_udiff(_udiff_zip())
    # fixture: header + 19 rows; first row SGBJUN28 / GB
    assert df.height == 19
    r = df.row(0, named=True)
    assert r["symbol"] == "SGBJUN28" and r["series"] == "GB"
    assert r["date"] == date(2026, 9, 29)
    assert (r["open"], r["high"], r["low"], r["close"]) == (14700.0, 14700.0, 14700.0, 14700.0)
    assert r["prev_close"] == 14751.45 and r["last_price"] == 14700.0
    assert r["volume"] == 48 and r["turnover_inr"] == 705600.0 and r["n_trades"] == 10.0
    assert r["isin"] == "IN0020200104" and r["name"] == "2.5%GOLDBONDS2028SR-III"
    assert r["era"] == "udiff" and r["avg_price"] is None
    assert df.schema["volume"] == pl.Int64 and df.schema["date"] == pl.Date
    assert df.schema["open"] == pl.Float64 and df.schema["n_trades"] == pl.Float64


def test_parse_legacy_sample_drops_trailing_comma_column():
    df = parse_legacy(_zip(_legacy_text()))
    # fixture: header + 19 rows; the trailing comma makes an empty last column which must be gone
    assert df.height == 19
    assert set(df.columns) == {
        "symbol", "series", "date", "open", "high", "low", "close", "prev_close", "last_price",
        "volume", "turnover_inr", "n_trades", "era", "isin", "name", "avg_price",
    }  # fmt: skip
    r = df.row(0, named=True)
    assert r["symbol"] == "20MICRONS" and r["series"] == "EQ" and r["date"] == date(2020, 2, 3)
    assert (r["open"], r["high"], r["low"], r["close"]) == (34.5, 34.85, 33.5, 34.45)
    assert r["last_price"] == 34.15 and r["prev_close"] == 33.75
    assert r["volume"] == 13398 and r["turnover_inr"] == 461342.95 and r["n_trades"] == 304.0
    assert r["isin"] == "INE144J01027" and r["era"] == "legacy" and r["name"] is None


def test_parse_legacy_without_trades_and_isin_columns():
    text = "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,\n"
    text += "AAA,EQ,1,2,1,2,2,1,10,20.5,04-JAN-2010,\n"
    df = parse_legacy(_zip(text))
    assert df.height == 1
    r = df.row(0, named=True)
    assert r["n_trades"] is None and r["isin"] is None and r["date"] == date(2010, 1, 4)


def test_parse_delivery_sample_dash_is_null():
    df = parse_delivery(_delivery_text(), date(2026, 9, 29))
    assert df is not None and df.height == 19
    r = df.filter(pl.col("symbol") == "20MICRONS").row(0, named=True)
    # 20MICRONS, EQ: DELIV_QTY 24068, DELIV_PER 44.52, AVG_PRICE 212.99
    assert (r["series"], r["deliv_qty"], r["deliv_pct"], r["avg_price"]) == ("EQ", 24068.0, 44.52, 212.99)
    d = df.filter(pl.col("symbol") == "3IINFOLTD").row(0, named=True)
    assert d["series"] == "BE" and d["deliv_qty"] is None and d["deliv_pct"] is None


def test_parse_delivery_wrong_date_is_rejected():
    # the holiday trap: file rows are dated 29-Sep-2026 but 2026-09-30 was requested
    assert parse_delivery(_delivery_text(), date(2026, 9, 30)) is None


# ---- ingest_eq_day ------------------------------------------------------------------------------


def test_ingest_day_udiff_then_delivery(tmp_data_root):
    d = date(2026, 9, 29)
    rec = Recorder({eq_url(d): _udiff_zip(), delivery_url(d): _delivery_text().encode()})
    ctx = _ctx(tmp_data_root, rec)
    res = ingest_eq_day(ctx, d)
    # 19 eq rows and 19 delivery rows; delivery's AVG_PRICE lands in bhavcopy_eq.avg_price
    assert res == [
        IngestResult("bhavcopy_eq", d, 19, "ok"),
        IngestResult("bhavcopy_delivery", d, 19, "ok"),
    ]
    eq = read(tmp_data_root, "bhavcopy_eq").collect()
    assert eq.height == 19
    # 20MICRONS is not in the udiff sample (it is in the delivery sample): only SGBJUN28 etc. are, so
    # avg_price is null for every eq row that has no delivery counterpart
    assert eq["isin"].null_count() == 0
    assert (tmp_data_root.paths.data_root / "cache" / "cm_udiff" / "2026" / "2026-09-29.zip").is_file()
    assert (tmp_data_root.paths.data_root / "cache" / "delivery" / "2026" / "2026-09-29.csv").is_file()


def test_delivery_avg_price_joins_into_eq(tmp_data_root):
    d = date(2026, 9, 29)
    eq = _eq_rows(d, 500).with_columns(
        pl.when(pl.col("symbol") == "S0000")
        .then(pl.lit("20MICRONS"))
        .otherwise(pl.col("symbol"))
        .alias("symbol")
    )
    upsert(tmp_data_root, "bhavcopy_eq", eq)
    rec = Recorder({delivery_url(d): _delivery_text().encode()})
    ctx = _ctx(tmp_data_root, rec)
    res = ingest_eq_day(ctx, d)
    # eq already has 500 rows -> skipped; delivery ingested; 20MICRONS AVG_PRICE is 212.99
    assert [r.status for r in res] == ["skipped", "ok"]
    got = read(tmp_data_root, "bhavcopy_eq").collect()
    assert got.height == 500  # the join must not add or drop eq rows
    assert got.filter(pl.col("symbol") == "20MICRONS")["avg_price"].to_list() == [212.99]
    assert got.filter(pl.col("symbol") == "S0001")["avg_price"].to_list() == [None]
    assert got.filter(pl.col("symbol") == "S0001")["close"].to_list() == [10.5]  # other columns intact


def test_saturday_with_file_is_ingested(tmp_data_root):
    d = date(2020, 2, 1)  # a Saturday; calendar says it is no session, the file exists anyway
    assert d.weekday() == 5
    rec = Recorder({eq_url(d): _zip(_legacy_text("01-FEB-2020"))})
    ctx = _ctx(tmp_data_root, rec)
    res = ingest_eq_day(ctx, d)
    # delivery history starts 2019-08-23, so the Saturday also asks for delivery; it 404s
    assert res[0] == IngestResult("bhavcopy_eq", d, 19, "ok")
    assert read(tmp_data_root, "bhavcopy_eq").collect()["date"].unique().to_list() == [d]


def test_present_day_skipped_without_http(tmp_data_root):
    d = date(2026, 9, 29)
    upsert(tmp_data_root, "bhavcopy_eq", _eq_rows(d, 500))
    upsert(
        tmp_data_root,
        "bhavcopy_delivery",
        pl.DataFrame(
            {"symbol": ["S0000"], "series": ["EQ"], "date": [d], "deliv_qty": [1.0], "deliv_pct": [5.0]}
        ),
    )
    rec = Recorder()
    ctx = _ctx(tmp_data_root, rec)
    res = ingest_eq_day(ctx, d)
    assert rec.urls == []
    assert [r.status for r in res] == ["skipped", "skipped"]


def test_404_older_than_3_days_writes_marker_and_is_not_requested_again(tmp_data_root):
    d = date(2020, 2, 2)  # Sunday, 6 years old
    rec = Recorder()
    ctx = _ctx(tmp_data_root, rec)
    res = ingest_eq_day(ctx, d)
    assert res == [IngestResult("bhavcopy_eq", d, 0, "no_session")]
    assert len(rec.urls) == 1
    assert ctx.state.kv_get("no_session:eq:2020-02-02") is not None
    res2 = ingest_eq_day(ctx, d)
    assert len(rec.urls) == 1  # no new request
    assert res2[0].status == "skipped"


def test_404_on_recent_expected_weekday_is_not_published_without_marker(tmp_data_root):
    d = date(2026, 11, 10)  # Tuesday, one day before the fixed clock; calendar projects it as a session
    rec = Recorder()
    ctx = _ctx(tmp_data_root, rec)
    res = ingest_eq_day(ctx, d)
    assert res == [IngestResult("bhavcopy_eq", d, 0, "not_published")]
    assert ctx.state.kv_get("no_session:eq:2026-11-10") is None


def test_rerun_of_same_day_leaves_row_count_unchanged(tmp_data_root):
    d = date(2026, 9, 29)
    rec = Recorder({eq_url(d): _udiff_zip(), delivery_url(d): _delivery_text().encode()})
    ctx = _ctx(tmp_data_root, rec)
    ingest_eq_day(ctx, d)
    n_eq = read(tmp_data_root, "bhavcopy_eq").collect().height
    n_dv = read(tmp_data_root, "bhavcopy_delivery").collect().height
    ctx2 = _ctx(tmp_data_root, rec)
    ingest_eq_day(ctx2, d)
    assert read(tmp_data_root, "bhavcopy_eq").collect().height == n_eq == 19
    assert read(tmp_data_root, "bhavcopy_delivery").collect().height == n_dv == 19


def test_delivery_holiday_trap_gives_no_session_and_nothing_stored(tmp_data_root):
    d = date(2026, 9, 29)
    # delivery url for 2026-09-29 answers with a file dated 28-Sep-2026 (previous session served)
    rec = Recorder({eq_url(d): _udiff_zip(), delivery_url(d): _delivery_text("28-Sep-2026").encode()})
    ctx = _ctx(tmp_data_root, rec)
    res = ingest_eq_day(ctx, d)
    assert res[0].status == "ok"
    assert res[1] == IngestResult("bhavcopy_delivery", d, 0, "no_session")
    assert read(tmp_data_root, "bhavcopy_delivery").collect().height == 0


def test_cached_file_is_not_downloaded_again(tmp_data_root):
    d = date(2026, 9, 29)
    cache = tmp_data_root.paths.data_root / "cache" / "cm_udiff" / "2026"
    cache.mkdir(parents=True)
    (cache / "2026-09-29.zip").write_bytes(_udiff_zip())
    rec = Recorder()
    ctx = _ctx(tmp_data_root, rec)
    res = ingest_eq_day(ctx, d)
    assert res[0] == IngestResult("bhavcopy_eq", d, 19, "ok")
    assert all("BhavCopy_NSE_CM" not in u for u in rec.urls)


# ---- range, repair ------------------------------------------------------------------------------


def test_range_visits_every_calendar_day(tmp_data_root):
    rec = Recorder()
    ctx = _ctx(tmp_data_root, rec)
    res = ingest_eq_range(ctx, date(2020, 2, 1), date(2020, 2, 3))
    eq = [r for r in res if r.dataset == "bhavcopy_eq"]
    assert [r.date for r in eq] == [date(2020, 2, 1), date(2020, 2, 2), date(2020, 2, 3)]
    assert len([u for u in rec.urls if "cm0" in u]) == 3


def test_repair_days_order(tmp_data_root):
    upsert(tmp_data_root, "bhavcopy_eq", _eq_rows(date(2026, 9, 29), 500))
    ctx = _ctx(tmp_data_root, Recorder())
    days = repair_days(ctx, date(2026, 10, 2))
    # (a) the 19 specials first, then 2026-09-30..2026-10-02, then 2010-01-01.. except the specials
    assert days[:19] == list(MISSING_SPECIAL_SESSIONS)
    assert days[19:22] == [date(2026, 9, 30), date(2026, 10, 1), date(2026, 10, 2)]
    assert days[22] == date(2010, 1, 1)
    assert days[-1] == date(2026, 9, 29)
    assert len(days) == len(set(days))
    assert date(2010, 2, 6) not in days[22:]


# ---- MCP cross-check ----------------------------------------------------------------------------


def _stored_pairs(settings):
    d = date(2026, 9, 29)
    upsert(settings, "bhavcopy_eq", _eq_rows(d, 3))
    return [("S0000", d), ("S0001", d)]


def test_cross_check_ok_within_tolerance(tmp_data_root):
    pairs = _stored_pairs(tmp_data_root)

    async def closes(p):
        return {k: 10.505 for k in p}  # stored close 10.5; |diff| 0.005 <= 0.01

    res = cross_check_mcp(tmp_data_root, pairs, closes_fn=closes)
    assert res == McpCheck(status="ok", checked=2, mismatches=[])


def test_cross_check_mismatch_fails(tmp_data_root):
    pairs = _stored_pairs(tmp_data_root)

    async def closes(p):
        return {pairs[0]: 10.5, pairs[1]: 11.0}

    res = cross_check_mcp(tmp_data_root, pairs, closes_fn=closes)
    assert res.status == "mismatch" and res.checked == 2
    assert res.mismatches == [("S0001", date(2026, 9, 29), 10.5, 11.0)]


def test_cross_check_unreachable_is_non_blocking(tmp_data_root):
    pairs = _stored_pairs(tmp_data_root)

    async def closes(p):
        raise ConnectionError("down")

    res = cross_check_mcp(tmp_data_root, pairs, closes_fn=closes)
    assert res.status == "unreachable" and res.checked == 0


@pytest.mark.parametrize("bad", [None])
def test_cross_check_missing_value_is_not_a_mismatch(tmp_data_root, bad):
    pairs = _stored_pairs(tmp_data_root)

    async def closes(p):
        return {pairs[0]: 10.5, pairs[1]: bad}

    res = cross_check_mcp(tmp_data_root, pairs, closes_fn=closes)
    assert res.status == "ok" and res.checked == 1


def test_tiny_special_day_file_is_not_a_session(tmp_data_root, monkeypatch):
    # Gold-ETF-only Sunday trading (e.g. 2010-05-16, 7 rows) is not an equity session: nothing is stored
    monkeypatch.setattr(ingest_mod, "MIN_SESSION_ROWS", 500)
    d = date(2010, 5, 16)
    text = "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,\n"
    text += "GOLDBEES,EQ,1800,1829.5,1785,1800.6,1794,1795.7,482410,873916640.35,16-MAY-2010,\n" * 7
    rec = Recorder({eq_url(d): _zip(text)})
    ctx = _ctx(tmp_data_root, rec)
    res = ingest_eq_day(ctx, d)
    assert res == [IngestResult("bhavcopy_eq", d, 7, "no_session")]
    assert read(tmp_data_root, "bhavcopy_eq").collect().height == 0
    assert ctx.state.kv_get("no_session:eq:2010-05-16") is not None
    ingest_eq_day(ctx, d)
    assert len(rec.urls) == 1  # cached and marked: no second request


def test_sample_pairs_skips_weekend_sessions(tmp_data_root):
    # the MCP server forward-fills non-weekday dates (answers the previous close), so weekend special
    # sessions cannot be checked against it: only weekday rows are sampled
    sat, mon = date(2026, 2, 7), date(2026, 2, 9)
    upsert(tmp_data_root, "bhavcopy_eq", pl.concat([_eq_rows(sat, 3), _eq_rows(mon, 3)]))
    pairs = sample_pairs(tmp_data_root, [sat, mon], n=20, seed=1)
    assert len(pairs) == 3 and {d for _, d in pairs} == {mon}
