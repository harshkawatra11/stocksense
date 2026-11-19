from __future__ import annotations

from datetime import date, datetime, timedelta

import httpx
import polars as pl
import pytest

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock, SteppingClock
from stocksense.core.errors import BrokerError
from stocksense.core.state import State
from stocksense.data.ingest_common import IngestContext
from stocksense.data.ingest_intraday import (
    backfill_1m,
    extend_5m,
    last_complete_bar_start,
    load_bars,
    month_end,
    month_starts,
    top_symbols_by_year,
    validate_month,
)
from stocksense.data.io import read, upsert

D1, D2 = date(2026, 3, 2), date(2026, 3, 3)


def _bars(symbol, day, prices, with_open_bar=True):
    """Three 5-minute bars at 09:15, 09:20, 09:25 with the given close prices (open = previous close)."""
    rows = []
    prev = prices[0]
    for i, p in enumerate(prices):
        if i == 0 and not with_open_bar:
            prev = p
            continue
        rows.append(
            {
                "symbol": symbol,
                "ts": datetime(day.year, day.month, day.day, 9, 15 + 5 * i, tzinfo=IST),
                "interval": "FIVE_MINUTE",
                "open": prev,
                "high": max(prev, p) + 0.5,
                "low": min(prev, p) - 0.5,
                "close": p,
                "volume": 100,
            }
        )
        prev = p
    return pl.DataFrame(rows)


def _panel(rows):
    base = {c: None for c in ("adj_high", "adj_low")}
    return pl.DataFrame(
        [{"sid": 1, "symbol": "AAA", "series": "EQ", **base, **r} for r in rows],
        schema_overrides={"sid": pl.Int32},
    )


def test_load_bars_adjusted_is_continuous_across_a_1_to_2_split(tmp_data_root):
    # raw (unadjusted) bars: day 1 around 100, day 2 around 50 after a 1:2 split
    upsert(
        tmp_data_root,
        "intraday_bars",
        pl.concat([_bars("AAA", D1, [100.0, 101.0, 102.0]), _bars("AAA", D2, [50.0, 50.5, 51.0])]),
    )
    # panel adjusted open: day 1 = 50.0 (adjusted by 0.5), day 2 = 50.0
    upsert(
        tmp_data_root,
        "panel_daily",
        _panel(
            [
                {"date": D1, "adj_open": 50.0, "adj_close": 51.0},
                {"date": D2, "adj_open": 50.0, "adj_close": 51.0},
            ]
        ),
    )
    out = load_bars(tmp_data_root, ["AAA"], D1, D2)
    d1 = out.filter(pl.col("ts").dt.date() == D1).sort("ts")
    d2 = out.filter(pl.col("ts").dt.date() == D2).sort("ts")
    # k1 = 50 / 100 = 0.5 -> closes 50, 50.5, 51 ; volume 100 / 0.5 = 200
    assert d1["close"].to_list() == [50.0, 50.5, 51.0]
    assert d1["volume"].to_list() == [200.0, 200.0, 200.0]
    # k2 = 50 / 50 = 1 -> unchanged; the series is continuous (51 -> 50 open)
    assert d2["close"].to_list() == [50.0, 50.5, 51.0]
    assert abs(d2["open"][0] / d1["close"][-1] - 1) < 0.02


def test_load_bars_adjusted_is_idempotent_when_bars_are_already_adjusted(tmp_data_root):
    upsert(tmp_data_root, "intraday_bars", _bars("AAA", D1, [50.0, 50.5, 51.0]))
    upsert(tmp_data_root, "panel_daily", _panel([{"date": D1, "adj_open": 50.0, "adj_close": 51.0}]))
    out = load_bars(tmp_data_root, ["AAA"], D1, D1)
    assert out["close"].to_list() == [50.0, 50.5, 51.0]


def test_load_bars_falls_back_to_close_when_open_bar_missing(tmp_data_root):
    upsert(tmp_data_root, "intraday_bars", _bars("AAA", D1, [100.0, 101.0, 102.0], with_open_bar=False))
    upsert(tmp_data_root, "panel_daily", _panel([{"date": D1, "adj_open": 55.0, "adj_close": 51.0}]))
    out = load_bars(tmp_data_root, ["AAA"], D1, D1)
    # no 09:15 bar: k = adj_close / last close = 51 / 102 = 0.5
    assert out["close"].to_list() == [50.5, 51.0]


def test_load_bars_drops_sessions_without_panel_row_and_raw_ratio_is_untouched(tmp_data_root):
    upsert(
        tmp_data_root,
        "intraday_bars",
        pl.concat([_bars("AAA", D1, [100.0, 101.0, 102.0]), _bars("AAA", D2, [50.0, 50.5, 51.0])]),
    )
    upsert(tmp_data_root, "panel_daily", _panel([{"date": D1, "adj_open": 50.0, "adj_close": 51.0}]))
    adj = load_bars(tmp_data_root, ["AAA"], D1, D2)
    assert adj["ts"].dt.date().unique().to_list() == [D1]
    raw = load_bars(tmp_data_root, ["AAA"], D1, D2, basis="raw_ratio")
    assert raw.height == 6 and raw["close"].to_list()[:3] == [100.0, 101.0, 102.0]


# ---- 1-minute backfill --------------------------------------------------------------------------------


class FakeUpstox:
    def __init__(self):
        self.calls: list[tuple[str, date, date]] = []

    def candles(self, key, unit, interval, start, end):
        self.calls.append((key, start, end))
        d = start
        rows = []
        while d <= end:
            if d.weekday() < 5:
                for m in range(375):
                    rows.append(
                        (
                            datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST) + timedelta(minutes=m),
                            10.0,
                            11.0,
                            9.0,
                            10.5,
                            2.0,
                        )
                    )
            d += timedelta(days=1)
        return pl.DataFrame(
            rows, schema=["ts", "open", "high", "low", "close", "volume"], orient="row"
        ).with_columns(pl.col("ts").dt.convert_time_zone("Asia/Kolkata").dt.cast_time_unit("us"))

    def close(self):
        pass


class FakeMap:
    def upstox_key(self, s):
        return None if s == "NOKEY" else f"NSE_EQ|{s}"

    def close(self):
        pass


def _ctx(settings):
    clock = FixedClock(datetime(2026, 11, 11, 20, 0, tzinfo=IST))
    state = State(settings, clock)
    state.migrate()
    return IngestContext(settings, clock, TradingCalendar.from_dates([date(2026, 11, 9)]), None, state)  # type: ignore[arg-type]


def test_month_helpers():
    assert month_starts(date(2025, 11, 20), date(2026, 2, 3)) == [
        date(2025, 11, 1),
        date(2025, 12, 1),
        date(2026, 1, 1),
        date(2026, 2, 1),
    ]
    assert month_end(date(2026, 2, 1)) == date(2026, 2, 28) and month_end(date(2025, 12, 1)) == date(
        2025, 12, 31
    )


def test_backfill_1m_is_resumable_and_stores_rows(tmp_data_root):
    ctx = _ctx(tmp_data_root)
    up = FakeUpstox()
    res = backfill_1m(
        ctx,
        ["AAA", "BBB", "NOKEY"],
        date(2026, 9, 1),
        end=date(2026, 10, 31),
        upstox=up,
        instruments=FakeMap(),
    )
    # 2 months x 2 symbols fetched, newest month first
    assert [c[1:] for c in up.calls[:2]] == [(date(2026, 10, 1), date(2026, 10, 31))] * 2
    assert len(up.calls) == 4 and sum(1 for r in res if r.status == "skipped") == 2
    df = read(tmp_data_root, "bars_1m").collect()
    # September 2026 has 22 weekdays, October 22 -> 44 sessions x 375 bars x 2 symbols
    assert df.height == 44 * 375 * 2
    assert set(df["source"].unique()) == {"upstox"} and df["volume"].dtype == pl.Int64
    assert ctx.state.job_status("bars_1m:AAA:2026-10", date(2026, 10, 1)) == "done"
    # second run: every symbol-month is done, no request
    up2 = FakeUpstox()
    backfill_1m(
        ctx, ["AAA", "BBB"], date(2026, 9, 1), end=date(2026, 10, 31), upstox=up2, instruments=FakeMap()
    )
    assert up2.calls == []
    assert read(tmp_data_root, "bars_1m").collect().height == df.height


def test_validate_month_counts_short_sessions_and_volume_mismatch():
    ts = [datetime(2026, 9, 1, 9, 15, tzinfo=IST) + timedelta(minutes=m) for m in range(300)]
    df = pl.DataFrame({"ts": ts, "volume": [2.0] * 300}).with_columns(
        pl.col("ts").dt.convert_time_zone("Asia/Kolkata")
    )
    bhav = pl.DataFrame({"symbol": ["A"], "date": [date(2026, 9, 1)], "volume": [1000]})
    # one session with 300 (< 375) bars; bar volume 600 vs bhavcopy 1000 (-40%)
    assert validate_month(df, bhav, "A") == {"sessions": 1, "short_sessions": 1, "volume_violations": 1}
    ok = pl.DataFrame({"symbol": ["A"], "date": [date(2026, 9, 1)], "volume": [610]})
    assert validate_month(df, ok, "A")["volume_violations"] == 0  # 600 vs 610 = -1.6%


def test_top_symbols_by_year_is_union_of_yearly_top_n(tmp_data_root):
    def eq(year, turn):
        return pl.DataFrame(
            [
                {
                    "symbol": s,
                    "series": "EQ",
                    "date": date(year, 6, 3),
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "prev_close": 1.0,
                    "last_price": 1.0,
                    "volume": 1,
                    "turnover_inr": t,
                    "n_trades": 1.0,
                    "era": "legacy",
                }
                for s, t in turn.items()
            ]
        )

    upsert(
        tmp_data_root,
        "bhavcopy_eq",
        pl.concat([eq(2022, {"A": 9, "B": 8, "C": 1}), eq(2023, {"A": 1, "B": 2, "C": 9})]),
    )
    assert top_symbols_by_year(tmp_data_root, 2022, 2023, n=1) == ["A", "C"]


# ---- G2-A3: the current month is re-fetched ---------------------------------------------------------------


def _ctx_at(settings, now):
    clock = FixedClock(now)
    state = State(settings, clock)
    state.migrate()
    return IngestContext(settings, clock, TradingCalendar.from_dates([date(2026, 11, 9)]), None, state)  # type: ignore[arg-type]


def test_backfill_1m_marks_current_month_partial_and_refetches_from_last_day(tmp_data_root):
    ctx = _ctx_at(tmp_data_root, datetime(2026, 11, 19, 20, 0, tzinfo=IST))
    up = FakeUpstox()
    backfill_1m(ctx, ["AAA"], date(2026, 10, 1), upstox=up, instruments=FakeMap())
    assert [c[1:] for c in up.calls] == [
        (date(2026, 11, 1), date(2026, 11, 18)),
        (date(2026, 10, 1), date(2026, 10, 31)),
    ]
    assert ctx.state.job_status("bars_1m:AAA:2026-10", date(2026, 10, 1)) == "done"
    assert ctx.state.job_status("bars_1m:AAA:2026-11", date(2026, 11, 1)) == "skipped"
    con = ctx.state.connect()
    try:
        (detail,) = con.execute(
            "SELECT detail FROM job_runs WHERE job='bars_1m:AAA:2026-11' ORDER BY attempt DESC LIMIT 1"
        ).fetchone()
    finally:
        con.close()
    assert detail == "partial:2026-11-18"
    # next day: only November, only from the day after the last stored bar (October is not requested)
    ctx2 = _ctx_at(tmp_data_root, datetime(2026, 11, 20, 20, 0, tzinfo=IST))
    up2 = FakeUpstox()
    backfill_1m(ctx2, ["AAA"], date(2026, 10, 1), upstox=up2, instruments=FakeMap())
    assert [c[1:] for c in up2.calls] == [(date(2026, 11, 19), date(2026, 11, 19))]
    df = read(tmp_data_root, "bars_1m").collect()
    assert df.select(pl.col("ts").dt.date().max()).item() == date(2026, 11, 19)


# ---- G2-A4: extend_5m ----------------------------------------------------------------------------------

CAL5 = TradingCalendar.from_dates([date(2026, 11, d) for d in (9, 10, 11, 12, 13, 16, 17, 18)])


@pytest.mark.parametrize(
    ("now", "want"),
    [
        (datetime(2026, 11, 18, 10, 7, tzinfo=IST), datetime(2026, 11, 18, 10, 0, tzinfo=IST)),
        (datetime(2026, 11, 18, 10, 10, tzinfo=IST), datetime(2026, 11, 18, 10, 5, tzinfo=IST)),
        (datetime(2026, 11, 18, 16, 0, tzinfo=IST), datetime(2026, 11, 18, 15, 25, tzinfo=IST)),
        (datetime(2026, 11, 18, 9, 16, tzinfo=IST), datetime(2026, 11, 17, 15, 25, tzinfo=IST)),
        (datetime(2026, 11, 21, 11, 0, tzinfo=IST), datetime(2026, 11, 20, 15, 25, tzinfo=IST)),
    ],
)
def test_last_complete_bar_start(now, want):
    assert last_complete_bar_start(now, CAL5) == want


class FakeAngel:
    def __init__(self, fail=False, clock=None, advance=None):
        self.calls: list[tuple[str, datetime, datetime]] = []
        self.fail, self.clock, self.advance = fail, clock, advance

    def candles(self, token, interval, start_dt, end_dt):
        self.calls.append((token, start_dt, end_dt))
        if self.clock is not None:
            self.clock.advance(self.advance)
        if self.fail:
            raise BrokerError("boom")
        ts = []
        t = start_dt
        while t <= end_dt + timedelta(minutes=5):  # a sloppy server that also returns the next bar
            ts.append(t)
            t += timedelta(minutes=5)
        return pl.DataFrame(
            {"ts": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}
        ).with_columns(pl.col("ts").dt.convert_time_zone("Asia/Kolkata").dt.cast_time_unit("us"))

    def close(self):
        pass


class FiveMap:
    def angel_token(self, s):
        return f"T{s}"

    def close(self):
        pass


def _ctx5(settings, clock):
    state = State(settings, clock)
    state.migrate()
    return IngestContext(settings, clock, CAL5, None, state)  # type: ignore[arg-type]


def _seed_5m(settings, symbols, last=datetime(2026, 11, 18, 9, 55, tzinfo=IST)):
    rows = [{**_bars(s, date(2026, 11, 18), [1.0]).to_dicts()[0], "ts": last} for s in symbols]
    upsert(settings, "intraday_bars", pl.DataFrame(rows))


def test_extend_5m_aborts_after_ten_consecutive_failures(tmp_data_root):
    syms = [f"S{i}" for i in range(15)]
    _seed_5m(tmp_data_root, syms)
    ctx = _ctx5(tmp_data_root, FixedClock(datetime(2026, 11, 18, 10, 7, tzinfo=IST)))
    ang = FakeAngel(fail=True)
    res = extend_5m(ctx, syms, angel=ang, instruments=FiveMap())
    assert len(ang.calls) == 10
    assert res[-1].status == "failed" and res[-1].detail == "aborted after 10 consecutive failures"
    assert sum(1 for r in res if r.status == "failed") == 11


def test_extend_5m_requests_only_complete_bars(tmp_data_root):
    _seed_5m(tmp_data_root, ["AAA"])
    ctx = _ctx5(tmp_data_root, FixedClock(datetime(2026, 11, 18, 10, 7, tzinfo=IST)))
    ang = FakeAngel()
    extend_5m(ctx, ["AAA"], angel=ang, instruments=FiveMap())
    assert ang.calls[0][1:] == (
        datetime(2026, 11, 18, 10, 0, tzinfo=IST),
        datetime(2026, 11, 18, 10, 0, tzinfo=IST),
    )
    ts = read(tmp_data_root, "intraday_bars").collect()["ts"].to_list()
    assert max(ts) == datetime(2026, 11, 18, 10, 0, tzinfo=IST)  # the 10:05 bar is not stored


def test_extend_5m_wall_budget(tmp_data_root):
    syms = [f"S{i}" for i in range(6)]
    _seed_5m(tmp_data_root, syms)
    clock = SteppingClock(datetime(2026, 11, 18, 10, 7, tzinfo=IST))
    ctx = _ctx5(tmp_data_root, clock)
    ang = FakeAngel(clock=clock, advance=timedelta(seconds=100))
    res = extend_5m(ctx, syms, angel=ang, instruments=FiveMap(), max_seconds=250)
    assert len(ang.calls) == 3
    assert res[-1].status == "skipped" and res[-1].detail == "wall budget"


def test_extend_5m_http_errors_count_as_failures(tmp_data_root):
    _seed_5m(tmp_data_root, ["AAA"])
    ctx = _ctx5(tmp_data_root, FixedClock(datetime(2026, 11, 18, 10, 7, tzinfo=IST)))

    class Boom(FakeAngel):
        def candles(self, *a):
            raise httpx.ConnectError("x")

    res = extend_5m(ctx, ["AAA"], angel=Boom(), instruments=FiveMap())
    assert res[0].status == "failed"
