"""The all-day job table (T10.4): parsing config/schedule.toml and answering "when is this job due".

Pure functions over a JobDef and a TradingCalendar; no clock, no state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST
from stocksense.core.config import Settings
from stocksense.core.errors import ConfigError

DAYS = ("session", "any", "nonsession", "saturday", "first_saturday")
#: jobs that still run on a non-tradable special session (Muhurat)
EVENING_JOBS = frozenset({"eod_ingest", "eod_pipeline"})
_KNOWN = {
    "name", "fn", "at", "every_s", "until", "days", "max_retries", "retry_backoff_s", "needs_market",
    "late_after", "after",
}  # fmt: skip


@dataclass(frozen=True)
class JobDef:
    """One scheduled slot of a job. `key` is what job_runs and kv use."""

    name: str
    fn: str
    at: time
    every_s: int | None
    until: time | None
    days: str
    max_retries: int
    retry_backoff_s: int
    needs_market: bool
    late_after: time | None
    after: str | None
    slot: bool = False  # True when the job has several `at` times (key carries @HH:MM)

    @property
    def key(self) -> str:
        return f"{self.name}@{self.at:%H:%M}" if self.slot else self.name


def _t(v: Any, field: str, job: str) -> time:
    try:
        return time.fromisoformat(str(v))
    except ValueError as e:
        raise ConfigError(f"schedule job {job!r}: bad time {v!r} in {field}") from e


def _parse(row: dict[str, Any]) -> list[JobDef]:
    name = row.get("name")
    if not name:
        raise ConfigError("schedule job without a name")
    extra = set(row) - _KNOWN
    if extra:
        raise ConfigError(f"schedule job {name!r}: unknown keys {sorted(extra)}")
    days = row.get("days", "session")
    if days not in DAYS:
        raise ConfigError(f"schedule job {name!r}: days must be one of {DAYS}, got {days!r}")
    if "at" not in row:
        raise ConfigError(f"schedule job {name!r}: missing at")
    ats = row["at"] if isinstance(row["at"], list) else [row["at"]]
    until = _t(row["until"], "until", name) if row.get("until") else None
    late = _t(row["late_after"], "late_after", name) if row.get("late_after") else None
    every = int(row["every_s"]) if row.get("every_s") else None
    if every is not None and until is None:
        raise ConfigError(f"schedule job {name!r}: every_s needs until")
    return [
        JobDef(
            name=name,
            fn=row.get("fn", name),
            at=_t(a, "at", name),
            every_s=every,
            until=until,
            days=days,
            max_retries=int(row.get("max_retries", 0)),
            retry_backoff_s=int(row.get("retry_backoff_s", 60)),
            needs_market=bool(row.get("needs_market", False)),
            late_after=late,
            after=row.get("after"),
            slot=len(ats) > 1,
        )
        for a in ats
    ]


def load_schedule(settings: Settings) -> list[JobDef]:
    """Every JobDef of config/schedule.toml in file order (multi-time jobs expanded to one per time)."""
    out: list[JobDef] = []
    for row in settings.schedule.job:
        out.extend(_parse(dict(row)))
    keys = [j.key for j in out]
    if len(set(keys)) != len(keys):
        raise ConfigError("schedule has duplicate job keys")
    return out


def due_times(job: JobDef, d: date) -> list[datetime]:
    """Every instant on day d at which the job is due: one for a one-shot, a grid for a repeating job."""
    start = datetime.combine(d, job.at, IST)
    if job.every_s is None or job.until is None:
        return [start]
    end = datetime.combine(d, job.until, IST)
    out, t = [], start
    while t <= end:
        out.append(t)
        t += timedelta(seconds=job.every_s)
    return out


def runs_on(job: JobDef, d: date, calendar: TradingCalendar) -> bool:
    """Whether the job belongs on day d (session / weekend rules; Muhurat runs only the evening jobs)."""
    if job.days == "any":
        return True
    if job.days == "saturday":
        return d.weekday() == 5
    if job.days == "first_saturday":
        return d.weekday() == 5 and d.day <= 7
    if job.days == "nonsession":
        return not calendar.is_session(d)
    if not calendar.is_session(d):
        return False
    return calendar.is_tradable(d) or job.name in EVENING_JOBS
