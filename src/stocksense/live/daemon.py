"""The all-day daemon (T10.5): a 5-second loop that runs whichever scheduled jobs are due.

State lives in SQLite (`job_runs`, `kv`), so a restart on the same data root picks up where the last
process stopped. Nothing here places an order: fills are paper fills inside the ledger.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import Clock, SystemClock
from stocksense.core.config import Settings
from stocksense.core.log import configure_logging, get_logger
from stocksense.core.paths import Paths
from stocksense.core.state import State
from stocksense.live import keepawake
from stocksense.live.jobs import LATE_HANDLERS, JobContext, execute_job, record_skip
from stocksense.live.preflight import daemon_lock, write_heartbeat
from stocksense.live.schedule import JobDef, load_schedule, runs_on

log = get_logger(__name__)

POLL_S = 5
HEARTBEAT_S = 30
_FINAL = ("done", "skipped")


class Daemon:
    """Evaluates the schedule against the clock; `tick()` runs every job that is due right now."""

    def __init__(self, ctx: JobContext, jobs: list[JobDef] | None = None) -> None:
        self.ctx = ctx
        self.jobs = jobs if jobs is not None else load_schedule(ctx.settings)
        self._day: date | None = None
        self._today: list[JobDef] = []
        self._final: set[str] = set()
        self._last_run: dict[str, datetime] = {}
        self._rep_attempt: dict[str, int] = {}
        self._rep_counts: dict[str, list[int]] = {}
        self._hb_at: datetime | None = None

    # ---- recovery ---------------------------------------------------------------------------

    def recover(self) -> int:
        """Mark attempts left `running` by a dead process as failed (we hold the single-instance lock)."""
        con = self.ctx.state.connect()
        try:
            cur = con.execute(
                "UPDATE job_runs SET status='failed', finished_at=?, detail='interrupted' "
                "WHERE status='running'",
                (self.ctx.clock.now().isoformat(timespec="seconds"),),
            )
            return cur.rowcount
        finally:
            con.close()

    # ---- schedule evaluation ----------------------------------------------------------------

    def _roll_day(self, d: date) -> None:
        if self._day == d:
            return
        self._day, self._final = d, set()
        self._last_run.clear()
        self._rep_attempt.clear()
        self._rep_counts.clear()
        cal = self.ctx.calendar
        self._today = sorted(
            (j for j in self.jobs if runs_on(j, d, cal)), key=lambda j: j.at
        )  # stable: file order breaks ties

    def _last_attempt(self, key: str, d: date) -> tuple[int, str, datetime | None] | None:
        con = self.ctx.state.connect()
        try:
            row = con.execute(
                "SELECT attempt, status, finished_at FROM job_runs WHERE job=? AND trade_date=? "
                "ORDER BY attempt DESC LIMIT 1",
                (key, d.isoformat()),
            ).fetchone()
            cnt = con.execute(
                "SELECT COUNT(*) FROM job_runs WHERE job=? AND trade_date=?", (key, d.isoformat())
            ).fetchone()[0]
        finally:
            con.close()
        if row is None:
            return None
        fin = datetime.fromisoformat(row[2]) if row[2] else None
        return int(cnt), row[1], fin

    def _market_open(self, d: date, now: datetime) -> bool:
        o, c = self.ctx.calendar.session_times(d)
        return o <= now <= c + timedelta(minutes=1)

    def _status(self, key: str, d: date) -> str | None:
        return self.ctx.state.job_status(key, d)

    def _eligible_one_shot(self, job: JobDef, now: datetime, d: date) -> str | None:
        """None if not runnable now, else 'run' or 'late'."""
        if now < datetime.combine(d, job.at, now.tzinfo):
            return None
        if job.after and self._status(job.after, d) not in _FINAL:
            return None
        last = self._last_attempt(job.key, d)
        if last is None:
            if job.late_after and now.time() > job.late_after:
                return "late"
            if job.needs_market and not self._market_open(d, now):
                self._final.add(job.key)  # the window is over; nothing to do
                return None
            return "run"
        n, status, fin = last
        if status in _FINAL:
            self._final.add(job.key)
            return None
        if n >= job.max_retries + 1 or (job.until is not None and now.time() > job.until):
            self._final.add(job.key)
            return None
        if job.needs_market and not self._market_open(d, now):
            return None
        if fin is not None and (now - fin).total_seconds() < job.retry_backoff_s:
            return None
        return "run"

    def _eligible_repeating(self, job: JobDef, now: datetime, d: date) -> bool:
        assert job.until is not None and job.every_s is not None
        if now < datetime.combine(d, job.at, now.tzinfo):
            return False
        if now.time() > job.until:
            self._final.add(job.key)
            return False
        if job.needs_market and not self._market_open(d, now):
            return False
        last = self._last_run.get(job.key)
        if last is None:
            raw = self.ctx.state.kv_get(f"rep:{job.key}:{d.isoformat()}")
            last = datetime.fromisoformat(raw) if raw else None
            if last is not None:
                self._last_run[job.key] = last
        return last is None or (now - last).total_seconds() >= job.every_s

    # ---- running ----------------------------------------------------------------------------

    def _run_repeating(self, job: JobDef, now: datetime, d: date) -> str:
        key = job.key
        if key not in self._rep_attempt:
            last = self._last_attempt(key, d)
            self._rep_attempt[key] = last[0] if last else self.ctx.state.job_start(key, d)
            self._rep_counts[key] = [0, 0]
        res = execute_job_inline(self.ctx, job, d)
        self._last_run[key] = now
        self.ctx.state.kv_set(f"rep:{key}:{d.isoformat()}", now.isoformat(timespec="seconds"))
        counts = self._rep_counts[key]
        counts[0] += 1
        counts[1] += res == "failed"
        self.ctx.state.job_finish(
            key, d, self._rep_attempt[key],
            "failed" if res == "failed" else "done", f"runs={counts[0]} errors={counts[1]} last={res}",
        )  # fmt: skip
        return res

    def tick(self) -> list[str]:
        """Run every job that is due now; returns the keys that ran (or were skipped as late)."""
        now = self.ctx.clock.now()
        d = now.date()
        self._roll_day(d)
        ran: list[str] = []
        for job in self._today:
            if job.key in self._final:
                continue
            now = self.ctx.clock.now()
            if job.every_s is not None:
                if self._eligible_repeating(job, now, d):
                    self._run_repeating(job, now, d)
                    ran.append(job.key)
                continue
            verdict = self._eligible_one_shot(job, now, d)
            if verdict is None:
                continue
            if verdict == "late":
                handler = LATE_HANDLERS.get(job.fn)
                detail = handler(self.ctx, d) if handler else "skipped: late"
                record_skip(self.ctx, job.key, d, detail)
                self._final.add(job.key)
            else:
                execute_job(self.ctx, job.key, job.fn, d)
                if self._status(job.key, d) in _FINAL:
                    self._final.add(job.key)
            ran.append(job.key)
        self._heartbeat(now)
        return ran

    def _heartbeat(self, now: datetime) -> None:
        if self._hb_at is not None and (now - self._hb_at).total_seconds() < HEARTBEAT_S:
            return
        self._hb_at = now
        write_heartbeat(
            Paths(self.ctx.settings).heartbeat, ts=now, pid=os.getpid(),
            current_job=self.ctx.current_job, last_ok_job=self.ctx.last_ok_job, degraded=self.ctx.degraded,
        )  # fmt: skip


def execute_job_inline(ctx: JobContext, job: JobDef, d: date) -> str:
    """One tick of a repeating job: run it, return done / skipped / failed (writes no job_runs row)."""
    from stocksense.core.errors import DataNotAvailable
    from stocksense.live.jobs import JOBS

    ctx.current_job = job.key
    try:
        detail = JOBS[job.fn](ctx, d)
        status = "skipped" if detail.startswith("skipped") else "done"
        ctx.last_ok_job = job.key
    except DataNotAvailable as e:
        log.info("job_tick_not_ready", job=job.key, reason=str(e)[:100])
        status = "done"  # not an error: the next tick tries again
    except Exception as e:  # noqa: BLE001 - job boundary
        log.exception("job_failed", job=job.key, date=str(d))
        ctx.notifier.send(f"job failed: {job.key} {d}: {ctx.scrub(repr(e))[:200]}")
        status = "failed"
    finally:
        ctx.current_job = None
    return status


def build_context(
    settings: Settings, clock: Clock, *, log_console: bool | None = True, **overrides: object
) -> JobContext:
    """The production wiring: real state, calendar from the store, lazy broker/quote objects."""
    state = State(settings, clock)
    state.migrate()
    cal = TradingCalendar.from_store(settings, clock)
    return JobContext(settings, clock, state, cal, log_console=log_console, **overrides)  # type: ignore[arg-type]


def run(
    settings: Settings,
    clock: Clock | None = None,
    once: bool = False,
    *,
    ctx: JobContext | None = None,
    poll_s: float = POLL_S,
    stop_at: datetime | None = None,
) -> int:
    """Run the daemon. Returns 0 (also when another instance holds the lock: prints "already running").

    `once`: process everything currently due and return. `stop_at`: return when the clock reaches it
    (tests drive a SteppingClock this way; a real run has neither).
    """
    clock = clock if clock is not None else SystemClock()
    lock = daemon_lock(settings)
    if not lock.acquire():
        print("already running")
        return 0
    try:
        if ctx is None or ctx.log_console is not None:
            configure_logging(settings, clock, console=True if ctx is None else ctx.log_console)
        ctx = ctx if ctx is not None else build_context(settings, clock)
        ctx.state.migrate()
        with keepawake.keep_awake():
            d = Daemon(ctx)
            d.recover()
            while True:
                d.tick()
                if once:
                    d.tick()  # a finished job can make another due (eod_pipeline after eod_ingest)
                    break
                if stop_at is not None and clock.now() >= stop_at:
                    break
                clock.sleep(poll_s)
    finally:
        lock.release()
    return 0
