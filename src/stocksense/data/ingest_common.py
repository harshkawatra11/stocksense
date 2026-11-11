from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.core.errors import DataNotAvailable
from stocksense.core.state import State
from stocksense.data.nse_http import NseClient

NSE_ARCHIVES = "https://nsearchives.nseindia.com"
MARKER_AGE_DAYS = 3  # a 404 older than this is remembered and never fetched again


@dataclass(frozen=True)
class IngestResult:
    """Outcome of one dataset on one day; status is ok | not_published | no_session | skipped | failed."""

    dataset: str
    date: date
    rows: int
    status: str


@dataclass
class IngestContext:
    settings: Settings
    clock: Clock
    calendar: TradingCalendar
    nse: NseClient
    state: State


def daterange(a: date, b: date) -> list[date]:
    """Every calendar day in [a, b] inclusive (weekends included); empty if a > b."""
    return [a + timedelta(days=i) for i in range((b - a).days + 1)]


def marker(kind: str, d: date) -> str:
    """kv key remembering that `kind` has no file for `d`."""
    return f"no_session:{kind}:{d.isoformat()}"


def absent(ctx: IngestContext, dataset: str, kind: str, d: date) -> IngestResult:
    """Status for a 404: not_published if a session is expected and is at most a day old, else no_session."""
    age = (ctx.clock.today() - d).days
    if d.weekday() < 5 and ctx.calendar.is_session(d) and age <= 1:
        return IngestResult(dataset, d, 0, "not_published")
    if age > MARKER_AGE_DAYS:
        ctx.state.kv_set(marker(kind, d), "1")
    return IngestResult(dataset, d, 0, "no_session")


def fetch_cached(ctx: IngestContext, url: str, cache: Path) -> bytes | None:
    """Cached bytes if the file exists, else download (and cache) it; None on a 404."""
    if cache.is_file():
        return cache.read_bytes()
    raw = ctx.nse.get_archive(url)
    if raw is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
    return raw


def csv_from_zip(raw: bytes) -> bytes:
    """The single file inside a zip."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if len(names) != 1:
            raise DataNotAvailable(f"expected one file in the zip, found {names}")
        return z.read(names[0])


def read_strings(data: bytes) -> pl.DataFrame:
    """CSV bytes as an all-string frame: header and values stripped, empty/duplicated columns dropped."""
    df = pl.read_csv(io.BytesIO(data), infer_schema_length=0, truncate_ragged_lines=True)
    df = df.rename({c: c.strip() for c in df.columns})
    keep = [c for c in df.columns if c and not c.startswith(("_duplicated", "Unnamed"))]
    df = df.select(keep)
    return df.with_columns(pl.col(c).str.strip_chars() for c in df.columns)


def num(col: str) -> pl.Expr:
    """String column to Float64 with '-' and '' as null."""
    return pl.when(pl.col(col).is_in(["-", ""])).then(None).otherwise(pl.col(col)).cast(pl.Float64)
