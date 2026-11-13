from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytest

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, SteppingClock
from stocksense.core.errors import BrokerError, DataNotAvailable
from stocksense.core.state import State
from stocksense.live import jobs as J
from stocksense.live.daemon import Daemon
from stocksense.live.jobs import JobContext
from stocksense.live.schedule import JobDef

D = date(2026, 11, 13)  # a Friday session (projected)
CAL = TradingCalendar.from_dates([date(2026, 11, 11), date(2026, 11, 12)])


def at(h, m, s=0):
    return datetime(2026, 11, 13, h, m, s, tzinfo=IST)


class Spy:
    def __init__(self):
        self.msgs = []

    enabled = True

    def send(self, text):
        self.msgs.append(text)
        return True


def jobdef(name, h, m, **kw):
    base = dict(
        fn=name, every_s=None, until=None, days="session", max_retries=0, retry_backoff_s=60,
        needs_market=False, late_after=None, after=None,
    )  # fmt: skip
    base.update(kw)
    return JobDef(name=name, at=time(h, m), **base)


@pytest.fixture
def env(tmp_data_root, monkeypatch):
    clock = SteppingClock(at(8, 0))
    st = State(tmp_data_root, clock)
    st.migrate()
    spy = Spy()
    ctx = JobContext(tmp_data_root, clock, st, CAL, notifier=spy, steps=[], nse=object())
    calls: list[str] = []
    monkeypatch.setattr(J, "JOBS", dict(J.JOBS))
    return ctx, clock, st, spy, calls


def rows(st, job):
    con = st.connect()
    try:
        return con.execute(
            "SELECT attempt, status, detail FROM job_runs WHERE job=? ORDER BY attempt", (job,)
        ).fetchall()
    finally:
        con.close()


def raiser(exc):
    def f(c, d):
        raise exc

    return f


def test_due_jobs_run_in_time_order_once(env):
    ctx, clock, st, spy, calls = env
    J.JOBS["a"] = lambda c, d: calls.append("a") or "ok"
    J.JOBS["b"] = lambda c, d: calls.append("b") or "ok"
    dm = Daemon(ctx, [jobdef("b", 8, 30), jobdef("a", 8, 10), jobdef("late", 23, 0, fn="a")])
    assert dm.tick() == []
    clock.advance(timedelta(minutes=40))
    assert dm.tick() == ["a", "b"] and calls == ["a", "b"]
    assert dm.tick() == []  # done jobs do not run again
    assert Daemon(ctx, [jobdef("a", 8, 10)]).tick() == []  # a new daemon sees them done in job_runs
    assert rows(st, "a") == [(1, "done", "ok")]


def test_exception_is_recorded_notified_and_the_loop_continues(env):
    ctx, clock, st, spy, calls = env
    J.JOBS["boom"] = raiser(RuntimeError("kaboom"))
    J.JOBS["after"] = lambda c, d: calls.append("after") or "ok"
    dm = Daemon(ctx, [jobdef("boom", 8, 0), jobdef("after", 8, 1)])
    clock.advance(timedelta(minutes=5))
    dm.tick()
    assert calls == ["after"]
    (r,) = rows(st, "boom")
    assert r[1] == "failed" and "kaboom" in r[2]
    assert any("boom" in m for m in spy.msgs)


def test_data_not_available_is_retryable_not_an_error_and_respects_backoff_and_limit(env):
    ctx, clock, st, spy, calls = env

    def wait(c, d):
        calls.append("w")
        raise DataNotAvailable("not yet")

    J.JOBS["wait"] = wait
    dm = Daemon(ctx, [jobdef("wait", 8, 0, max_retries=2, retry_backoff_s=120)])
    dm.tick()
    clock.advance(timedelta(seconds=60))
    dm.tick()
    assert calls == ["w"]  # backoff not elapsed
    for _ in range(6):
        clock.advance(timedelta(seconds=120))
        dm.tick()
    assert calls == ["w", "w", "w"]  # 1 + max_retries attempts, then final
    rs = rows(st, "wait")
    assert [r[1] for r in rs] == ["failed"] * 3 and all(r[2].startswith("retry:") for r in rs)
    assert spy.msgs == []  # retryable waits are not notified


def test_retry_window_ends_at_until(env):
    ctx, clock, st, spy, calls = env

    def wait(c, d):
        calls.append("w")
        raise DataNotAvailable("x")

    J.JOBS["w"] = wait
    dm = Daemon(ctx, [jobdef("w", 8, 0, max_retries=50, retry_backoff_s=60, until=time(8, 3))])
    for _ in range(20):
        dm.tick()
        clock.advance(timedelta(seconds=60))
    assert len(calls) == 4  # 08:00 .. 08:03


def test_late_job_is_skipped_with_reason_and_never_runs(env):
    ctx, clock, st, spy, calls = env
    J.JOBS["s2"] = lambda c, d: calls.append("s2") or "ok"
    clock.advance(timedelta(hours=1, minutes=10))  # 09:10
    dm = Daemon(ctx, [jobdef("s2", 9, 7, late_after=time(9, 8))])
    assert dm.tick() == ["s2"] and calls == []
    assert rows(st, "s2") == [(1, "skipped", "skipped: late")]


def test_stub_job_records_skipped_not_implemented(env):
    ctx, clock, st, spy, calls = env
    clock.advance(timedelta(hours=1))
    Daemon(ctx, [jobdef("plan_open", 8, 50)]).tick()
    assert rows(st, "plan_open") == [(1, "skipped", "skipped: not implemented")]


def test_repeating_job_runs_on_its_grid_with_one_summary_row(env):
    ctx, clock, st, spy, calls = env
    J.JOBS["loop"] = lambda c, d: calls.append("x") or "ok"
    dm = Daemon(ctx, [jobdef("loop", 8, 0, every_s=60, until=time(8, 5))])
    for _ in range(40):
        dm.tick()
        clock.advance(timedelta(seconds=20))
    assert len(calls) == 6  # 08:00 .. 08:05
    (r,) = rows(st, "loop")
    assert r[1] == "done" and r[2].startswith("runs=6 errors=0")


def test_repeating_job_error_ticks_are_counted_and_do_not_stop_it(env):
    ctx, clock, st, spy, calls = env
    n = {"i": 0}

    def flaky(c, d):
        n["i"] += 1
        if n["i"] == 2:
            raise RuntimeError("tick blew up")
        return "ok"

    J.JOBS["loop"] = flaky
    dm = Daemon(ctx, [jobdef("loop", 8, 0, every_s=60, until=time(8, 3))])
    for _ in range(12):
        dm.tick()
        clock.advance(timedelta(seconds=20))
    assert n["i"] == 4
    (r,) = rows(st, "loop")
    assert "errors=1" in r[2] and r[1] == "done"  # the last tick was fine
    assert len(spy.msgs) == 1


def test_start_marks_interrupted_running_rows_failed(env):
    ctx, clock, st, spy, calls = env
    st.job_start("x", D)
    assert Daemon(ctx, []).recover() == 1
    assert rows(st, "x")[0][1:] == ("failed", "interrupted")


def test_job_waits_for_its_predecessor(env):
    ctx, clock, st, spy, calls = env
    state = {"ok": False}

    def first(c, d):
        if not state["ok"]:
            raise DataNotAvailable("no")
        return "ok"

    J.JOBS["first"] = first
    J.JOBS["second"] = lambda c, d: calls.append("second") or "ok"
    dm = Daemon(ctx, [jobdef("first", 8, 0, max_retries=5, retry_backoff_s=30),
                      jobdef("second", 8, 0, after="first")])  # fmt: skip
    dm.tick()
    assert calls == []
    state["ok"] = True
    clock.advance(timedelta(seconds=40))
    dm.tick()
    dm.tick()
    assert calls == ["second"]


def test_capital_am_falls_back_to_yesterday_only_in_the_last_minutes(env):
    ctx, clock, st, spy, calls = env

    class Down:
        def funds(self):
            raise BrokerError("login failed")

        def holdings(self):
            return []

    ctx._broker = Down()
    st.kv_set("capital_ref_inr", "123456.0")
    clock.advance(timedelta(minutes=45))  # 08:45
    with pytest.raises(BrokerError):
        J.capital_am(ctx, D)
    clock.advance(timedelta(minutes=19))  # 09:04
    assert J.capital_am(ctx, D) == "capital read stale=1"
    assert st.kv_get("capital_stale") == "1" and st.kv_get("capital_ref_inr") == "123456.0"


def test_capital_is_read_from_the_broker_never_a_constant(env):
    ctx, clock, st, spy, calls = env
    from stocksense.brokers.types import Funds, Holding

    class B:
        def funds(self):
            return Funds(available_cash=60_000.0, net=0.0, utilised=0.0)

        def holdings(self):
            return [Holding("AAA", 10, 100.0, 250.0)]

    ctx._broker = B()
    out = J.capital_am(ctx, D)
    assert "60000" not in out and "62500" not in out  # amounts stay out of job detail
    assert float(st.kv_get("capital_ref_inr")) == 62_500.0 and st.kv_get("capital_stale") == "0"


def test_secrets_never_reach_job_detail(env):
    ctx, clock, st, spy, calls = env
    from stocksense.core.config import Secrets

    ctx.settings = ctx.settings.model_copy(update={"secrets": Secrets(STOCKSENSE_ANGEL_PASSWORD="hunter2x")})
    J.JOBS["leak"] = raiser(RuntimeError("bad pin hunter2x"))
    J.execute_job(ctx, "leak", "leak", D)
    (r,) = rows(st, "leak")
    assert "hunter2x" not in r[2] and "***" in r[2]
    assert not any("hunter2x" in m for m in spy.msgs)


def test_every_job_fn_exists_and_listed_stubs_record_not_implemented(tmp_data_root):
    from stocksense.live.schedule import load_schedule

    assert {j.fn for j in load_schedule(tmp_data_root)} <= set(J.JOBS)
    for name in J.STUB_JOBS:
        assert J.JOBS[name](None, D) == J.NOT_IMPLEMENTED
