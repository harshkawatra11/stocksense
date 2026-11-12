from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import httpx
import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock, SteppingClock
from stocksense.core.state import State
from stocksense.data.ingest_announcements import (
    announcement_id,
    backfill_announcements,
    backward_windows,
    ingest_board_meetings,
    ingest_results_index,
    parse_announcements,
    parse_board_meetings,
    parse_results,
    poll_announcements,
)
from stocksense.data.ingest_common import IngestContext
from stocksense.data.io import read
from stocksense.data.nse_http import NseClient

FILES = Path(__file__).resolve().parents[1] / "fixtures" / "files"


def _load(name: str) -> list[dict]:
    return json.loads((FILES / name).read_text(encoding="utf-8"))


class Api:
    """MockTransport handler: serves `rows_for(path, params)`; records every call."""

    def __init__(self, rows_for):
        self.rows_for = rows_for
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        self.calls.append((request.url.path, params))
        return httpx.Response(200, json=self.rows_for(request.url.path, params))


def _ctx(settings, handler) -> IngestContext:
    clock = FixedClock(datetime(2026, 11, 11, 10, 0, tzinfo=IST))
    nse = NseClient(
        SteppingClock(datetime(2026, 11, 11, 10, 0, tzinfo=IST)), transport=httpx.MockTransport(handler)
    )
    state = State(settings, clock)
    state.migrate()
    cal = TradingCalendar.from_dates([date(2026, 11, 6), date(2026, 11, 9)])
    return IngestContext(settings=settings, clock=clock, calendar=cal, nse=nse, state=state)


# ---- parsing (4 real rows each, trimmed) --------------------------------------------------------


def test_parse_announcements_sample():
    rows = _load("announcements_sample.json")
    df = parse_announcements(rows)
    assert df.height == 4
    first = df.row(0, named=True)
    r0 = rows[0]
    assert first["seq_id"] == r0["seq_id"] and first["symbol"] == r0["symbol"]
    # an_dt "06-Nov-2026 hh:mm:ss" is read in IST
    ts = first["ts"]
    assert ts.tzinfo is not None and (ts.year, ts.month, ts.day) == (2026, 11, 6)
    assert first["category"] == r0["desc"] and first["summary"] == r0["attchmntText"].strip()
    assert first["first_seen_ts"] is None
    assert df["has_xbrl"].dtype == pl.Boolean


def test_seq_id_fallback_is_sha1_prefix_and_dash_exchange_time_is_null():
    row = {
        "symbol": "ABC",
        "an_dt": "09-Nov-2026 23:45:59",
        "attchmntText": "hello",
        "exchdisstime": "-",
        "seq_id": "",
        "desc": "Updates",
    }
    df = parse_announcements([row])
    assert df["seq_id"].item() == announcement_id("ABC", "09-Nov-2026 23:45:59", "hello")
    assert len(df["seq_id"].item()) == 16
    assert df["exch_ts"].item() is None
    # 23:45:59 on the 9th, IST
    assert df["ts"].item().hour == 23 and df["ts"].item().second == 59


def test_rows_without_symbol_or_date_are_dropped():
    rows = [{"symbol": "", "an_dt": "09-Nov-2026 10:00:00", "seq_id": "1"}, {"symbol": "A", "an_dt": "x"}]
    assert parse_announcements(rows).height == 0


def test_parse_board_meetings_sample():
    rows = _load("board_meetings_sample.json")
    df = parse_board_meetings(rows)
    assert df.height == 4
    r = df.row(0, named=True)
    assert r["symbol"] == rows[0]["bm_symbol"] and r["purpose"] == rows[0]["bm_purpose"]
    assert isinstance(r["meeting_date"], date) and r["announced_ts"].tzinfo is not None


def test_parse_results_sample():
    rows = _load("results_sample.json")
    df = parse_results(rows)
    assert df.height == 4
    r = df.row(0, named=True)
    # filingDate has no seconds: "31-Jan-2025 21:13"
    assert r["filing_ts"].hour == 21 and r["filing_ts"].minute == 13 and r["filing_ts"].second == 0
    assert r["period_from"] == date(2024, 10, 1) and r["period_to"] == date(2024, 12, 31)
    assert r["seq_number"] == rows[0]["seqNumber"] and r["xbrl_url"] == rows[0]["xbrl"]


def test_backward_windows():
    w = backward_windows(date(2026, 11, 1), date(2026, 11, 15), 7)
    # newest first; the last window is clipped at `start`
    assert w == [
        (date(2026, 11, 9), date(2026, 11, 15)),
        (date(2026, 11, 2), date(2026, 11, 8)),
        (date(2026, 11, 1), date(2026, 11, 1)),
    ]


# ---- backfill ------------------------------------------------------------------------------------


def _row(i: int, d: date) -> dict:
    return {
        "symbol": f"S{i}",
        "an_dt": f"{d:%d-%b-%Y} 10:00:00",
        "attchmntText": f"t{i}",
        "seq_id": f"{d:%Y%m%d}{i}",
        "desc": "Outcome of Board Meeting",
    }


def _ann_api(days_with_data: set[date]):
    def rows_for(path, params):
        a = datetime.strptime(params["from_date"], "%d-%m-%Y").date()
        b = datetime.strptime(params["to_date"], "%d-%m-%Y").date()
        out = []
        d = a
        while d <= b:
            if d in days_with_data:
                out += [_row(1, d), _row(2, d)]
            d = date.fromordinal(d.toordinal() + 1)
        return out

    return Api(rows_for)


def test_backfill_stores_rows_and_caches_final_windows(tmp_data_root):
    api = _ann_api({date(2026, 10, 1), date(2026, 10, 9)})
    ctx = _ctx(tmp_data_root, api)
    res = backfill_announcements(ctx, date(2026, 9, 28), date(2026, 10, 11))
    # windows (newest first): 10/05-10/11, 09/28-10/04 ; two rows on each of the two days
    assert [r.rows for r in res] == [2, 2] and all(r.status == "ok" for r in res)
    df = read(tmp_data_root, "announcements").collect()
    assert df.height == 4 and df["seq_id"].n_unique() == 4
    assert len(api.calls) == 2
    cache = tmp_data_root.paths.data_root / "cache" / "announcements" / "2026-10-05_2026-10-11.json.gz"
    assert cache.is_file()
    # resumable: a second run reads every final window from the cache, zero HTTP calls
    api.calls.clear()
    res2 = backfill_announcements(ctx, date(2026, 9, 28), date(2026, 10, 11))
    assert api.calls == [] and [r.rows for r in res2] == [2, 2]
    assert read(tmp_data_root, "announcements").collect().height == 4


def test_backfill_stops_after_twelve_empty_windows(tmp_data_root):
    api = _ann_api({date(2026, 10, 9)})
    ctx = _ctx(tmp_data_root, api)
    # 2026-10-09 is in the newest window; the twelve windows before it are empty -> stop there
    res = backfill_announcements(ctx, date(2025, 1, 1), date(2026, 10, 11))
    assert len(res) == 1 + 12
    assert len(api.calls) == 13


def test_oversized_window_is_refetched_day_by_day(tmp_data_root, monkeypatch):
    import stocksense.data.ingest_announcements as m

    monkeypatch.setattr(m, "SPLIT_ABOVE", 3)
    api = _ann_api({date(2026, 10, 5), date(2026, 10, 6)})
    ctx = _ctx(tmp_data_root, api)
    rows = m.fetch_announcement_window(ctx, date(2026, 10, 5), date(2026, 10, 11))
    # the 7-day call returned 4 rows (>= 3) -> seven 1-day calls follow, same 4 rows after merging
    assert len(rows) == 4 and len(api.calls) == 1 + 7


# ---- live poll -----------------------------------------------------------------------------------


def test_poll_marks_only_new_ids_with_first_seen(tmp_data_root):
    today = date(2026, 11, 11)
    api = _ann_api({today})
    ctx = _ctx(tmp_data_root, api)
    assert poll_announcements(ctx) == 2
    df = read(tmp_data_root, "announcements").collect()
    assert df["first_seen_ts"].null_count() == 0
    assert df["first_seen_ts"][0] == datetime(2026, 11, 11, 10, 0, tzinfo=IST)
    # same payload again: nothing new, nothing rewritten
    assert poll_announcements(ctx) == 0
    # one extra id appears
    api2 = Api(lambda p, q: [_row(1, today), _row(2, today), _row(3, today)])
    ctx2 = _ctx(tmp_data_root, api2)
    assert poll_announcements(ctx2) == 1
    assert read(tmp_data_root, "announcements").collect().height == 3


# ---- board meetings and results --------------------------------------------------------------------


def test_board_meetings_ingest(tmp_data_root):
    sample = _load("board_meetings_sample.json")
    api = Api(lambda path, params: sample if params["from_date"] == "01-11-2026" else [])
    ctx = _ctx(tmp_data_root, api)
    res = ingest_board_meetings(ctx, date(2026, 11, 1), date(2026, 11, 30))
    assert [r.rows for r in res] == [4]
    assert read(tmp_data_root, "board_meetings").collect().height == 4


def test_results_index_ingest_sends_period_and_dates(tmp_data_root):
    sample = _load("results_sample.json")

    def rows_for(path, params):
        return sample if params["period"] == "Quarterly" else []

    api = Api(rows_for)
    ctx = _ctx(tmp_data_root, api)
    res = ingest_results_index(ctx, date(2025, 1, 1), date(2025, 1, 31))
    assert [r.rows for r in res] == [4]
    periods = [p["period"] for _, p in api.calls]
    assert periods == ["Quarterly", "Annual"]
    assert api.calls[0][1]["from_date"] == "01-01-2025" and api.calls[0][1]["to_date"] == "31-01-2025"
    assert read(tmp_data_root, "results_index").collect().height == 4
