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
from stocksense.core.errors import DataNotAvailable
from stocksense.core.state import State
from stocksense.data import ingest_fo as fo_mod
from stocksense.data.ingest_common import IngestContext
from stocksense.data.ingest_fo import (
    fo_url,
    ingest_fo_day,
    ingest_fo_range,
    parse_fo,
    rebuild_fo_from_cache,
)
from stocksense.data.io import read
from stocksense.data.nse_http import NseClient

FILES = Path(__file__).resolve().parents[1] / "fixtures" / "files"


@pytest.fixture(autouse=True)
def _small_sessions(monkeypatch):
    """The real-file sample has 10 rows; shrink the 'is a session' bar so it counts as a session."""
    monkeypatch.setattr(fo_mod, "MIN_FO_ROWS", 5)


def _zip(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", text)
    return buf.getvalue()


def _sample_zip() -> bytes:
    return _zip((FILES / "udiff_fo_sample.csv").read_text(encoding="utf-8"))


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


def test_url():
    assert fo_url(date(2026, 9, 18)) == (
        "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_20260918_F_0000.csv.zip"
    )


def test_parse_fo_sample():
    df = parse_fo(_sample_zip())
    # fixture: header + 10 rows: 3 STF (ABCAPITAL), 3 STO (ABCAPITAL), 2 IDF and 2 IDO (BANKNIFTY)
    assert df.height == 10
    assert df["instrument_type"].to_list() == ["STF"] * 3 + ["STO"] * 3 + ["IDF"] * 2 + ["IDO"] * 2
    f = df.row(0, named=True)
    # ABCAPITAL Sep future: O 393.50 H 405.30 L 390.85 C 403.80 prev 390.65 settle 403.80, OI 36,161,500
    # (change -750,200), underlying 410.04, expiry 2026-09-29
    assert f["symbol"] == "ABCAPITAL" and f["date"] == date(2026, 9, 18)
    assert f["expiry_date"] == date(2026, 9, 29)
    assert (f["open"], f["high"], f["low"], f["close"]) == (393.5, 405.3, 390.85, 403.8)
    assert (f["prev_close"], f["settle_price"], f["underlying_price"]) == (390.65, 403.8, 410.04)
    assert (f["oi"], f["change_in_oi"]) == (36161500, -750200)
    # futures have no strike and no option type
    assert f["strike_price"] is None and f["option_type"] is None
    o = df.row(3, named=True)
    # ABCAPITAL 2026-09-29 480 PE: close 64.00, settle 69.24, OI 6200
    assert (o["strike_price"], o["option_type"], o["close"], o["settle_price"], o["oi"]) == (
        480.0,
        "PE",
        64.0,
        69.24,
        6200,
    )
    assert (
        df.schema["oi"] == pl.Int64 and df.schema["date"] == pl.Date and df.schema["expiry_date"] == pl.Date
    )
    assert df.schema["strike_price"] == pl.Float64


def test_ingest_day_ok_then_idempotent_then_skipped_without_http(tmp_data_root):
    d = date(2026, 9, 18)
    srv = Recorder({fo_url(d): _sample_zip()})
    ctx = _ctx(tmp_data_root, srv)
    r = ingest_fo_day(ctx, d)
    assert (r.dataset, r.rows, r.status) == ("fo_bhavcopy", 10, "ok")
    assert read(tmp_data_root, "fo_bhavcopy").collect().height == 10
    n = len(srv.urls)
    r2 = ingest_fo_day(ctx, d)  # already stored: skipped with no request
    assert r2.status == "skipped" and len(srv.urls) == n
    assert read(tmp_data_root, "fo_bhavcopy").collect().height == 10


def test_rerun_after_cache_leaves_row_count_unchanged(tmp_data_root):
    d = date(2026, 9, 18)
    srv = Recorder({fo_url(d): _sample_zip()})
    ctx = _ctx(tmp_data_root, srv)
    ingest_fo_day(ctx, d)
    n = len(srv.urls)
    r = ingest_fo_day(ctx, d, counts={})  # empty stored-count map: forces parse from cache and upsert again
    assert r.status == "ok" and len(srv.urls) == n  # cache hit: no download
    assert read(tmp_data_root, "fo_bhavcopy").collect().height == 10


def test_404_weekend_is_no_session_and_old_one_is_marked(tmp_data_root):
    srv = Recorder()
    ctx = _ctx(tmp_data_root, srv)
    sat = date(2026, 9, 19)
    assert ingest_fo_day(ctx, sat).status == "no_session"
    assert ctx.state.kv_get("no_session:fo:2026-09-19") == "1"
    n = len(srv.urls)
    assert ingest_fo_day(ctx, sat).status == "skipped"
    assert len(srv.urls) == n


def test_recent_weekday_404_is_not_published(tmp_data_root):
    ctx = _ctx(tmp_data_root, Recorder())
    ctx.calendar = TradingCalendar.from_dates([date(2026, 11, 9)])
    # 2026-11-11 is a weekday, projected as a session and 0 days old: not yet published
    assert ingest_fo_day(ctx, date(2026, 11, 11)).status == "not_published"
    assert ctx.state.kv_get("no_session:fo:2026-11-11") is None


def test_wrong_date_in_file_is_no_session(tmp_data_root):
    srv = Recorder({fo_url(date(2026, 9, 19)): _sample_zip()})  # file is dated 2026-09-18
    ctx = _ctx(tmp_data_root, srv)
    assert ingest_fo_day(ctx, date(2026, 9, 19)).status == "no_session"
    assert read(tmp_data_root, "fo_bhavcopy").collect().height == 0


def test_range_visits_every_calendar_day(tmp_data_root):
    srv = Recorder({fo_url(date(2026, 9, 18)): _sample_zip()})
    ctx = _ctx(tmp_data_root, srv)
    res = ingest_fo_range(ctx, date(2026, 9, 18), date(2026, 9, 21))
    # 18 ok, 19 and 20 are weekend (404, no_session), 21 is a Monday older than a day: no_session
    assert [(r.date.day, r.status) for r in res] == [
        (18, "ok"),
        (19, "no_session"),
        (20, "no_session"),
        (21, "no_session"),
    ]
    assert len(srv.urls) == 4


def _write_collapsed_store(settings, d: date) -> None:
    """Mimic the adopted data: expiry_date entirely null (Datetime), keys collapsed to fewer rows."""
    sample = parse_fo(_sample_zip())
    old = (
        sample.unique(subset=["symbol", "instrument_type", "strike_price", "option_type"], keep="last")
        .with_columns(pl.lit(None, dtype=pl.Datetime("us")).alias("expiry_date"))
        .with_columns(pl.col("date").cast(pl.Datetime("us")))
    )
    path = settings.paths.data_root / "parquet" / "fo_bhavcopy" / f"part-{d:%Y-%m}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    old.write_parquet(path)


def test_rebuild_from_cache_replaces_collapsed_rows(tmp_data_root):
    d = date(2026, 9, 18)
    _write_collapsed_store(tmp_data_root, d)
    # sample: 10 rows; without the expiry the three ABCAPITAL futures collapse to 1 and the 2 IDF to 1,
    # the 3 STO and 2 IDO strikes stay distinct: 1 + 3 + 1 + 2 = 7 collapsed rows
    assert read(tmp_data_root, "fo_bhavcopy").collect().height == 7
    ctx = _ctx(tmp_data_root, Recorder())
    cache = tmp_data_root.paths.data_root / "cache" / "fo_bhavcopy" / "2026" / "2026-09-18.zip"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(_sample_zip())
    n = rebuild_fo_from_cache(ctx, date(2024, 7, 8), date(2026, 9, 18))
    assert n == {d: 10}
    df = read(tmp_data_root, "fo_bhavcopy").collect()
    assert df.height == 10 and df["expiry_date"].null_count() == 0


def test_rebuild_refuses_when_a_stored_day_has_no_cache(tmp_data_root):
    _write_collapsed_store(tmp_data_root, date(2026, 9, 18))
    ctx = _ctx(tmp_data_root, Recorder())
    with pytest.raises(DataNotAvailable):
        rebuild_fo_from_cache(ctx, date(2024, 7, 8), date(2026, 9, 18))
    assert read(tmp_data_root, "fo_bhavcopy").collect().height == 7  # untouched
