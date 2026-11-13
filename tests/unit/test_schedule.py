from __future__ import annotations

from datetime import date, datetime, time

import pytest

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST
from stocksense.core.config import SpecialSession
from stocksense.core.errors import ConfigError
from stocksense.live.schedule import JobDef, due_times, load_schedule, runs_on

PAST = [date(2026, 11, 9), date(2026, 11, 10), date(2026, 11, 11), date(2026, 11, 12)]
# 2026-11-13 Fri session (projected), 14 Sat, 15 Sun, 16 Mon holiday, 17 Tue session
CAL = TradingCalendar.from_dates(
    PAST,
    holidays=[date(2026, 11, 16)],
    special_sessions=[SpecialSession(date=date(2026, 11, 21), tradable=False)],
)


@pytest.fixture
def jobs(tmp_data_root):
    return {j.key: j for j in load_schedule(tmp_data_root)}


def test_parse_every_job_of_the_plan_table(jobs):
    names = {j.name for j in jobs.values()}
    expected = {
        "preflight", "broker_login", "overnight_data", "capital_am", "plan_open", "preopen_capture",
        "s2_decide", "open_fills", "market_loop", "spread_capture", "capital_pm", "s1_entries",
        "s2_squareoff", "close_fills", "eod_mark", "eod_ingest", "eod_pipeline", "housekeeping",
        "weekly_research", "monthly_kronos", "collect_only",
    }  # fmt: skip
    assert expected <= names
    j = jobs["market_loop"]
    assert (j.at, j.every_s, j.until, j.days, j.needs_market) == (
        time(9, 16), 60, time(15, 30), "session", True,
    )  # fmt: skip
    assert jobs["capital_am"].retry_backoff_s == 120 and jobs["capital_am"].until == time(9, 5)
    assert jobs["open_fills"].at == time(9, 15, 10) and jobs["open_fills"].late_after == time(9, 25)
    assert jobs["eod_pipeline"].after == "eod_ingest"
    assert jobs["preflight"].fn == "preflight" and jobs["preflight_start"].fn == "preflight"


def test_multi_time_job_expands_to_keyed_slots(jobs):
    slots = sorted(k for k, j in jobs.items() if j.name == "spread_capture")
    assert slots == ["spread_capture@09:20", "spread_capture@11:00", "spread_capture@13:00",
                     "spread_capture@15:00"]  # fmt: skip


def test_due_times_for_one_shot_and_repeating_jobs(jobs):
    d = date(2026, 11, 13)
    assert due_times(jobs["s2_decide"], d) == [datetime(2026, 11, 13, 9, 7, tzinfo=IST)]
    pre = due_times(jobs["preopen_capture"], d)
    assert len(pre) == 9 and pre[0].time() == time(9, 0) and pre[-1].time() == time(9, 8)
    loop = due_times(jobs["market_loop"], d)
    assert len(loop) == 375 and loop[-1].time() == time(15, 30)


def test_days_filtering_with_calendar(jobs):
    sess, sat, sun, hol, muhurat = (date(2026, 11, 13), date(2026, 11, 14), date(2026, 11, 15),
                                    date(2026, 11, 16), date(2026, 11, 21))  # fmt: skip
    assert runs_on(jobs["market_loop"], sess, CAL)
    assert not runs_on(jobs["market_loop"], sat, CAL)
    assert not runs_on(jobs["market_loop"], hol, CAL)
    assert runs_on(jobs["preflight"], hol, CAL) and runs_on(jobs["housekeeping"], sun, CAL)
    assert runs_on(jobs["collect_only"], sat, CAL) and runs_on(jobs["collect_only"], hol, CAL)
    assert not runs_on(jobs["collect_only"], sess, CAL)
    assert runs_on(jobs["weekly_research"], sat, CAL) and not runs_on(jobs["weekly_research"], sun, CAL)
    assert runs_on(jobs["monthly_kronos"], date(2026, 11, 7), CAL)
    assert not runs_on(jobs["monthly_kronos"], sat, CAL)  # second Saturday
    # a non-tradable special session runs only the evening jobs
    assert runs_on(jobs["eod_ingest"], muhurat, CAL) and runs_on(jobs["eod_pipeline"], muhurat, CAL)
    assert not runs_on(jobs["market_loop"], muhurat, CAL)
    assert not runs_on(jobs["s1_entries"], muhurat, CAL)


def test_invalid_job_rows_raise_config_error(tmp_data_root):
    bad = tmp_data_root.schedule.model_copy(update={"job": [{"name": "x", "at": "9am", "days": "any"}]})
    with pytest.raises(ConfigError):
        load_schedule(tmp_data_root.model_copy(update={"schedule": bad}))
    bad = tmp_data_root.schedule.model_copy(update={"job": [{"name": "x", "at": "09:00", "days": "never"}]})
    with pytest.raises(ConfigError):
        load_schedule(tmp_data_root.model_copy(update={"schedule": bad}))


def test_jobdef_is_frozen():
    j = JobDef("a", "a", time(1), None, None, "any", 0, 60, False, None, None)
    with pytest.raises(AttributeError):
        j.name = "b"  # type: ignore[misc]
