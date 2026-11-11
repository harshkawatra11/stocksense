"""Calendar features (PLAN T4.3h): day of week, position in the month, expiry / budget weeks, holidays.

Forward-looking counts use Monday-Friday arithmetic minus holidays, never the stored session list (a run
truncated at `as_of` does not know future sessions). A holiday is a weekday that is a known holiday
(`holidays` dataset, published in advance, read without an `as_of` cut) or a weekday that was not a
session inside the stored history.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import date, timedelta

import polars as pl

from stocksense.data import io
from stocksense.features.base import FeatureContext, feature

HOLIDAY_CAP = 10
_FUTURE_DAYS = 75  # weekdays are projected this far past the last stored session

# Budget presentation dates in years where it was not 1 February (best knowledge; interim budgets in
# election years are included). Years not listed use 1 February.
BUDGET_DATES: dict[int, tuple[date, ...]] = {
    2010: (date(2010, 2, 26),),
    2011: (date(2011, 2, 28),),
    2012: (date(2012, 3, 16),),
    2013: (date(2013, 2, 28),),
    2014: (date(2014, 2, 17), date(2014, 7, 10)),
    2015: (date(2015, 2, 28),),
    2016: (date(2016, 2, 29),),
    2019: (date(2019, 2, 1), date(2019, 7, 5)),
    2024: (date(2024, 2, 1), date(2024, 7, 23)),
}


def _weekday(d: date) -> bool:
    return d.weekday() < 5


def _week(d: date) -> tuple[int, int]:
    iso = d.isocalendar()
    return iso[0], iso[1]


def _holidays(ctx: FeatureContext, sessions: list[date]) -> list[date]:
    pub = io.read(ctx.settings, "holidays", columns=["date"]).collect()["date"].to_list()
    held = set(sessions)
    out = {d for d in pub if d is not None and _weekday(d)}
    if sessions:
        d, last = sessions[0], sessions[-1]
        while d <= last:
            if _weekday(d) and d not in held:
                out.add(d)
            d += timedelta(days=1)
    return sorted(out)


def _calendar(sessions: list[date], hol: set[date]) -> list[date]:
    """Past sessions plus projected weekdays (minus holidays) after the last one."""
    cal = set(sessions)
    if sessions:
        d = sessions[-1] + timedelta(days=1)
        for _ in range(_FUTURE_DAYS):
            if _weekday(d) and d not in hol:
                cal.add(d)
            d += timedelta(days=1)
    return sorted(cal)


def _base(ctx: FeatureContext) -> tuple[list[date], list[date], list[date]]:
    sessions = ctx.sessions().collect()["date"].to_list()
    hol = _holidays(ctx, sessions)
    return sessions, hol, _calendar(sessions, set(hol))


def _month_groups(cal: list[date]) -> dict[tuple[int, int], list[date]]:
    g: dict[tuple[int, int], list[date]] = defaultdict(list)
    for d in cal:
        g[(d.year, d.month)].append(d)
    return g


def _frame(sessions: list[date], name: str, values: list) -> pl.LazyFrame:
    return pl.DataFrame({"date": sessions, name: values}, schema={"date": pl.Date, name: pl.Int64}).lazy()


@feature(
    "dow",
    group="calendar",
    kind="mkt",
    window=1,
    label="Day of the week (0 = Monday ... 4 = Friday; 5-6 for weekend special sessions).",
)
def dow(ctx: FeatureContext) -> pl.LazyFrame:
    """Python weekday of the session date."""
    sessions = ctx.sessions().collect()["date"].to_list()
    return _frame(sessions, "dow", [d.weekday() for d in sessions])


@feature(
    "dom_bucket",
    group="calendar",
    kind="mkt",
    window=1,
    label="Where in the month the session falls: 0 = first 3 sessions, 2 = last 3, 1 = otherwise.",
)
def dom_bucket(ctx: FeatureContext) -> pl.LazyFrame:
    """0 for the first 3 sessions of the month, 2 for the last 3, else 1."""
    sessions, _, cal = _base(ctx)
    groups = _month_groups(cal)
    out = []
    for d in sessions:
        g = groups[(d.year, d.month)]
        pos = bisect_right(g, d)  # 1-based position
        left = len(g) - pos
        out.append(0 if pos <= 3 else 2 if left <= 2 else 1)
    return _frame(sessions, "dom_bucket", out)


@feature(
    "days_to_month_end",
    group="calendar",
    kind="mkt",
    window=1,
    label="Sessions left in the month (0 on the last session) - turn-of-month and window-dressing effects.",
)
def days_to_month_end(ctx: FeatureContext) -> pl.LazyFrame:
    """Number of sessions after the date within the same month (weekday calendar minus holidays)."""
    sessions, _, cal = _base(ctx)
    groups = _month_groups(cal)
    out = []
    for d in sessions:
        g = groups[(d.year, d.month)]
        out.append(len(g) - bisect_right(g, d))
    return _frame(sessions, "days_to_month_end", out)


def _expiry_weeks(ctx: FeatureContext, sessions: list[date], hol: set[date]) -> set[tuple[int, int]]:
    fo = (
        ctx.raw("fo_bhavcopy", ["instrument_type", "expiry_date"])
        .filter(pl.col("instrument_type").is_in(["IDF", "STF"]))
        .select("expiry_date")
        .unique()
        .collect()["expiry_date"]
        .drop_nulls()
        .to_list()
    )
    from_data: dict[tuple[int, int], date] = {}
    for e in fo:
        k = (e.year, e.month)
        if k not in from_data or e > from_data[k]:
            from_data[k] = e
    weeks: set[tuple[int, int]] = set()
    months = sorted({(d.year, d.month) for d in sessions})
    for y, m in months:
        if (y, m) in from_data:
            e = from_data[(y, m)]
        else:  # rule: last Thursday, moved back to the previous session if it is a holiday
            nxt = date(y + (m == 12), 1 if m == 12 else m + 1, 1)
            e = nxt - timedelta(days=1)
            while e.weekday() != 3:
                e -= timedelta(days=1)
            while e in hol or not _weekday(e):
                e -= timedelta(days=1)
        weeks.add(_week(e))
    return weeks


@feature(
    "is_expiry_week",
    group="calendar",
    kind="mkt",
    window=1,
    label="1 if this is the week of the monthly futures-and-options expiry.",
)
def is_expiry_week(ctx: FeatureContext) -> pl.LazyFrame:
    """1 if the date's Monday-Sunday week contains the month's F&O expiry (from data, else last Thursday)."""
    sessions, hol, _ = _base(ctx)
    weeks = _expiry_weeks(ctx, sessions, set(hol))
    return _frame(sessions, "is_expiry_week", [int(_week(d) in weeks) for d in sessions])


@feature(
    "sessions_to_next_holiday",
    group="calendar",
    kind="mkt",
    window=1,
    label="Sessions left before the next market holiday (0 = holiday tomorrow; 10 = none within 10).",
)
def sessions_to_next_holiday(ctx: FeatureContext) -> pl.LazyFrame:
    """Sessions strictly between the date and the next holiday, capped at 10."""
    sessions, hol, cal = _base(ctx)
    out = []
    for d in sessions:
        i = bisect_right(hol, d)
        if i >= len(hol):
            out.append(HOLIDAY_CAP)
            continue
        n = bisect_left(cal, hol[i]) - bisect_right(cal, d)
        out.append(min(n, HOLIDAY_CAP))
    return _frame(sessions, "sessions_to_next_holiday", out)


@feature(
    "is_budget_week",
    group="calendar",
    kind="mkt",
    window=1,
    label="1 if this is the week of the Union Budget presentation (a known high-volatility event).",
)
def is_budget_week(ctx: FeatureContext) -> pl.LazyFrame:
    """1 if the date's week contains the budget date (table for irregular years, else 1 February)."""
    sessions = ctx.sessions().collect()["date"].to_list()
    weeks: set[tuple[int, int]] = set()
    for y in {d.year for d in sessions}:
        for b in BUDGET_DATES.get(y, (date(y, 2, 1),)):
            weeks.add(_week(b))
    return _frame(sessions, "is_budget_week", [int(_week(d) in weeks) for d in sessions])
