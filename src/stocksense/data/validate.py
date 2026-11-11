from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl
import pyarrow.parquet as pq

from stocksense.core.config import Settings
from stocksense.data.catalog import DatasetSpec, get_spec
from stocksense.data.io import list_partitions, read

__all__ = ["Issue", "validate_dataset"]

_PRICE_EPS = 1e-6
_PREV_CLOSE_TOL = 0.0005  # |prev_close / prev_stored_close - 1| above this is a "changed" prev_close
_MISSING_SHARE = 0.5
_COUNT_BAND = 0.15
_COUNT_WINDOW = 20
_COUNT_MIN_SAMPLES = 5
_DEFAULT_MIN_SESSION_ROWS = 500


@dataclass(frozen=True)
class Issue:
    """One validation finding: `code` is machine-readable, `date` the session it is attached to."""

    dataset: str
    code: str
    detail: str = ""
    date: date | None = None


def validate_dataset(
    settings: Settings,
    name: str,
    start: date | None = None,
    end: date | None = None,
    *,
    min_session_rows: int = _DEFAULT_MIN_SESSION_ROWS,
) -> list[Issue]:
    """Validation issues of one dataset (empty list = clean); row counts are rows, prices are as stored."""
    spec = get_spec(name)
    issues: list[Issue] = _check_files(settings, spec)
    if spec.nested or not list_partitions(settings, name):
        return issues
    issues += _check_keys_and_dates(settings, spec, start, end)
    if {"open", "high", "low", "close"} <= set(spec.schema):
        issues += _check_prices(settings, spec, start, end)
    if name == "bhavcopy_eq":
        issues += _check_sessions(settings, start, end, min_session_rows)
    return sorted(issues, key=lambda i: (i.date or date.min, i.code))


# ---- generic checks ---------------------------------------------------------------------------


def _check_files(settings: Settings, spec: DatasetSpec) -> list[Issue]:
    """Raw files: key/date columns present; no columns outside the catalog (unless allowed)."""
    required = {*spec.key, *([spec.date_col] if spec.date_col else [])}
    bad_cols: set[str] = set()
    missing: set[str] = set()
    for f in list_partitions(settings, spec.name) if not spec.nested else []:
        cols = set(pq.read_schema(f).names)
        missing |= required - cols
        if not spec.keep_extra:
            bad_cols |= cols - set(spec.schema)
    out = [Issue(spec.name, f"missing_column:{c}") for c in sorted(missing)]
    out += [Issue(spec.name, f"unexpected_column:{c}") for c in sorted(bad_cols)]
    return out


def _check_keys_and_dates(
    settings: Settings, spec: DatasetSpec, start: date | None, end: date | None
) -> list[Issue]:
    out: list[Issue] = []
    key = list(spec.key)
    if spec.date_col:
        dc = pl.col(spec.date_col)
        if isinstance(spec.schema[spec.date_col], pl.Datetime):
            dc = dc.dt.date()
        n_null = (
            read(settings, spec.name, columns=[spec.date_col])
            .select(pl.col(spec.date_col).is_null().sum())
            .collect()
            .item()
        )
        if n_null:
            out.append(Issue(spec.name, "null_date", f"{n_null} rows"))
    if key:
        dups = (
            read(settings, spec.name, start, end, columns=key)
            .group_by(key)
            .len()
            .filter(pl.col("len") > 1)
            .select(pl.len().alias("groups"), (pl.col("len") - 1).sum().alias("extra"))
            .collect()
            .row(0)
        )
        if dups[0]:
            out.append(Issue(spec.name, "duplicate_key", f"{dups[0]} keys, {dups[1]} extra rows"))
    return out


def _check_prices(settings: Settings, spec: DatasetSpec, start: date | None, end: date | None) -> list[Issue]:
    cols = ["open", "high", "low", "close", *(["volume"] if "volume" in spec.schema else [])]
    o, h, lo, c = (pl.col(x) for x in ("open", "high", "low", "close"))
    # A row with open = high = low = 0 is an untraded instrument (only close/settle carries a value):
    # exempt from the OHLC checks (PLAN deviation, recorded in DECISIONS).
    lf = read(settings, spec.name, start, end, columns=cols).filter(~((o == 0) & (h == 0) & (lo == 0)))
    exprs = {
        "high_below_max_open_close": (h < pl.max_horizontal(o, c) - _PRICE_EPS).sum(),
        "low_above_min_open_close": (lo > pl.min_horizontal(o, c) + _PRICE_EPS).sum(),
        "nonpositive_price": ((o <= 0) | (h <= 0) | (lo <= 0) | (c <= 0)).sum(),
    }
    if "volume" in cols:
        exprs["negative_volume"] = (pl.col("volume") < 0).sum()
    row = lf.select(**exprs).collect().row(0, named=True)
    return [Issue(spec.name, code, f"{n} rows") for code, n in row.items() if n]


# ---- bhavcopy_eq session checks ---------------------------------------------------------------


def _check_sessions(settings: Settings, start: date | None, end: date | None, min_rows: int) -> list[Issue]:
    eq = (
        read(
            settings,
            "bhavcopy_eq",
            start,
            end,
            columns=["symbol", "series", "date", "close", "prev_close"],
        )
        .filter(pl.col("series") == "EQ")
        .unique(subset=["symbol", "date"], keep="last")
        .collect()
    )
    return _row_count_issues(eq, min_rows) + _missing_session_issues(eq)


def _row_count_issues(eq: pl.DataFrame, min_rows: int) -> list[Issue]:
    counts = eq.group_by("date").len().sort("date").rename({"len": "n"})
    counts = counts.with_columns(
        base=pl.col("n").shift(1).rolling_median(_COUNT_WINDOW, min_samples=_COUNT_MIN_SAMPLES)
    )
    out: list[Issue] = []
    for d, n, base in counts.iter_rows():
        if n < min_rows:
            out.append(Issue("bhavcopy_eq", "session_too_small", f"{n} EQ rows < {min_rows}", d))
        if base is not None and abs(n / base - 1) > _COUNT_BAND:
            out.append(
                Issue("bhavcopy_eq", "session_row_count_off", f"{n} EQ rows vs trailing median {base:g}", d)
            )
    return out


def _missing_session_issues(eq: pl.DataFrame) -> list[Issue]:
    """Finding 2: a session whose prev_close disagrees with the previous STORED close for most symbols."""
    idx = (
        eq.select("date")
        .unique()
        .sort("date")
        .with_row_index("idx")
        .with_columns(pl.col("idx").cast(pl.Int64))
    )
    df = (
        eq.join(idx, on="date")
        .sort("symbol", "idx")
        .with_columns(
            prev_idx=pl.col("idx").shift(1).over("symbol"),
            prev_stored_close=pl.col("close").shift(1).over("symbol"),
        )
        .filter((pl.col("idx") - pl.col("prev_idx") == 1) & (pl.col("prev_stored_close") > 0))
        .with_columns(
            changed=(pl.col("prev_close") / pl.col("prev_stored_close") - 1).abs() > _PREV_CLOSE_TOL
        )
    )
    per_day = df.group_by("date").agg(n=pl.len(), share=pl.col("changed").mean()).sort("date")
    return [
        Issue(
            "bhavcopy_eq",
            f"missing_session_before:{d}",
            f"{share:.1%} of {n} EQ symbols have prev_close != previous stored close",
            d,
        )
        for d, n, share in per_day.iter_rows()
        if share > _MISSING_SHARE
    ]
