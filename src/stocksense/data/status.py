from __future__ import annotations

import re
from datetime import date

import polars as pl
import pyarrow.parquet as pq

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.data.catalog import CATALOG, DatasetSpec
from stocksense.data.io import list_partitions, read
from stocksense.data.validate import validate_dataset

__all__ = ["status_table"]

# One row per trading session: completeness against the calendar is meaningful for these.
_DAILY = frozenset(
    {
        "bhavcopy_eq",
        "bhavcopy_delivery",
        "fo_bhavcopy",
        "india_vix",
        "intraday_bars",
        "index_daily",
        "price_bands",
        "fii_dii",
        "macro_daily",
        "panel_daily",
        "panel_universe",
        "panel_market",
    }
)
_DAILY_FRESHNESS_MAX = 5  # catalog freshness (calendar days) at or below this means "per session"
_RE_DAY_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.parquet$")

_SCHEMA = {
    "dataset": pl.Utf8,
    "first_date": pl.Date,
    "last_date": pl.Date,
    "rows": pl.Int64,
    "sessions_missing": pl.Int64,
    "fresh": pl.Boolean,
    "issues": pl.Int64,
}


def status_table(settings: Settings, calendar: TradingCalendar, clock: Clock) -> pl.DataFrame:
    """One row per stored dataset: date span, row count, missing sessions, freshness, issue count."""
    rows = []
    for spec in CATALOG.values():
        row = _dataset_row(settings, spec, calendar, clock)
        if row is not None:
            rows.append(row)
    return pl.DataFrame(rows, schema=_SCHEMA, orient="row")


def _dataset_row(
    settings: Settings, spec: DatasetSpec, calendar: TradingCalendar, clock: Clock
) -> tuple | None:
    base = settings.paths.data_root / spec.path
    files = sorted(base.glob("*/*.parquet")) if spec.nested else list_partitions(settings, spec.name)
    if not files:
        return None
    n_rows = sum(pq.read_metadata(f).num_rows for f in files)
    if spec.nested:  # nse_snapshots: no date column; the day files carry the dates
        days = sorted(m[1] for f in files if (m := _RE_DAY_FILE.match(f.name)))
        first = date.fromisoformat(days[0]) if days else None
        last = date.fromisoformat(days[-1]) if days else None
        return (spec.name, first, last, n_rows, None, _fresh(spec, last, calendar, clock), 0)
    if spec.date_col is None:
        return (spec.name, None, None, n_rows, None, None, len(validate_dataset(settings, spec.name)))
    col = pl.col(spec.date_col)
    if isinstance(spec.schema[spec.date_col], pl.Datetime):
        col = col.dt.date()
    dates = (
        read(settings, spec.name, columns=[spec.date_col])
        .select(col.alias("d"))
        .unique()
        .drop_nulls()
        .collect()["d"]
        .sort()
    )
    first, last = (dates[0], dates[-1]) if len(dates) else (None, None)
    missing = None
    if spec.name in _DAILY and first is not None:
        have = set(dates.to_list())
        missing = sum(1 for s in calendar.sessions(first, last) if s not in have)
    issues = len(validate_dataset(settings, spec.name))
    return (spec.name, first, last, n_rows, missing, _fresh(spec, last, calendar, clock), issues)


def _fresh(spec: DatasetSpec, last: date | None, calendar: TradingCalendar, clock: Clock) -> bool | None:
    """Per-session datasets: newest date is the last completed session. Others: age <= catalog days."""
    if spec.freshness is None:
        return None
    if last is None:
        return False
    if spec.name in _DAILY and spec.freshness <= _DAILY_FRESHNESS_MAX:
        return last >= calendar.last_completed_session(clock.now())
    return (clock.today() - last).days <= spec.freshness
