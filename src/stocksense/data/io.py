from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path

import polars as pl

from stocksense.core.clock import IST, Clock, SystemClock
from stocksense.core.config import Settings
from stocksense.core.errors import DataValidationError
from stocksense.data.catalog import DatasetSpec, get_spec

IST_NAME = "Asia/Kolkata"
_RETRIES = 5
_RETRY_WAIT_S = 0.5

_RE_MONTH = re.compile(r"^part-(\d{4})-(\d{2})\.parquet$")
_RE_YEAR = re.compile(r"^part-(\d{4})\.parquet$")
_RE_DAY = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\.parquet$")
_KEY_FORMATS = {
    "month": re.compile(r"^\d{4}-\d{2}$"),
    "year": re.compile(r"^\d{4}$"),
    "day": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "all": re.compile(r"^all$"),
}

__all__ = ["IST", "list_partitions", "read", "upsert", "write_partition"]


# ---- locations --------------------------------------------------------------------------------


def _dir(settings: Settings, spec: DatasetSpec, sub: str | None) -> Path:
    if spec.nested and not sub:
        raise DataValidationError(
            f"{spec.name} is nested: pass sub=<folder> (e.g. a kind, version or model id)"
        )
    if sub and not spec.nested:
        raise DataValidationError(f"{spec.name} is not nested; `sub` is not allowed")
    base = Path(settings.paths.data_root) / spec.path
    return base / sub if sub else base


def _file_name(spec: DatasetSpec, key: str) -> str:
    fmt = _KEY_FORMATS[spec.partition]
    if not fmt.match(key):
        raise DataValidationError(
            f"{spec.name}: partition key {key!r} invalid for partition={spec.partition}"
        )
    if spec.partition == "month":
        return f"part-{key}.parquet"
    if spec.partition == "year":
        return f"part-{key}.parquet"
    if spec.partition == "day":
        return f"{key}.parquet"
    return spec.file or "part-all.parquet"


def _file_range(name: str) -> tuple[date, date] | None:
    """Inclusive date span a partition file can hold, from its name; None if unknown/unbounded."""
    if m := _RE_MONTH.match(name):
        y, mo = int(m[1]), int(m[2])
        nxt = date(y + (mo == 12), mo % 12 + 1, 1)
        return date(y, mo, 1), date.fromordinal(nxt.toordinal() - 1)
    if m := _RE_YEAR.match(name):
        y = int(m[1])
        return date(y, 1, 1), date(y, 12, 31)
    if m := _RE_DAY.match(name):
        d = date(int(m[1]), int(m[2]), int(m[3]))
        return d, d
    return None


def list_partitions(settings: Settings, dataset: str, sub: str | None = None) -> list[Path]:
    """Sorted partition files that exist for a dataset (empty list if the folder is missing)."""
    spec = get_spec(dataset)
    d = _dir(settings, spec, sub)
    if not d.is_dir():
        return []
    if spec.file:
        f = d / spec.file
        return [f] if f.is_file() else []
    return sorted(d.glob("*.parquet"))


# ---- schema conformance -----------------------------------------------------------------------


def _cast_expr(name: str, src: pl.DataType, tgt: pl.DataType) -> pl.Expr:
    c = pl.col(name)
    if src == tgt:
        return c
    if tgt == pl.Date:
        if isinstance(src, pl.Datetime):
            if src.time_zone:
                c = c.dt.convert_time_zone(IST_NAME)
            return c.dt.date().alias(name)
        return c.cast(pl.Date)
    if isinstance(tgt, pl.Datetime):
        tz = tgt.time_zone
        if isinstance(src, pl.Datetime):
            if tz:
                c = c.dt.convert_time_zone(tz) if src.time_zone else c.dt.replace_time_zone(tz)
            elif src.time_zone:
                c = c.dt.replace_time_zone(None)
            return c.dt.cast_time_unit(tgt.time_unit).alias(name)
        if src == pl.Date:
            c = c.cast(pl.Datetime(tgt.time_unit))
            return (c.dt.replace_time_zone(tz) if tz else c).alias(name)
    return c.cast(tgt)


def _conform_exprs(
    spec: DatasetSpec, have: pl.Schema, columns: list[str] | None, extra_needed: tuple[str, ...] = ()
) -> list[pl.Expr]:
    """Expressions producing the requested catalog columns: cast, typed nulls for absent ones."""
    if columns is None:
        wanted = list(spec.schema)
        if spec.keep_extra:
            wanted += [c for c in have.names() if c not in spec.schema]
    else:
        wanted = list(columns)
    names = list(dict.fromkeys([*wanted, *extra_needed]))
    out: list[pl.Expr] = []
    for n in names:
        tgt = spec.schema.get(n)
        if tgt is None:
            if n not in have:
                raise DataValidationError(f"{spec.name}: unknown column {n!r}")
            out.append(pl.col(n))  # keep_extra pass-through (or explicit request of an extra column)
        elif n in have:
            out.append(_cast_expr(n, have[n], tgt).alias(n))
        else:
            out.append(pl.lit(None, dtype=tgt).alias(n))
    return out


def _conform(lf: pl.LazyFrame, spec: DatasetSpec) -> pl.LazyFrame:
    return lf.select(_conform_exprs(spec, lf.collect_schema(), None))


def _empty(spec: DatasetSpec, columns: list[str] | None) -> pl.LazyFrame:
    cols = list(spec.schema) if columns is None else columns
    for c in cols:
        if c not in spec.schema:
            raise DataValidationError(f"{spec.name}: unknown column {c!r}")
    return pl.DataFrame(schema={c: spec.schema[c] for c in cols}).lazy()


# ---- read -------------------------------------------------------------------------------------


def read(
    settings: Settings,
    dataset: str,
    start: date | None = None,
    end: date | None = None,
    columns: list[str] | None = None,
    sub: str | None = None,
) -> pl.LazyFrame:
    """Lazy frame of a dataset conformed to its catalog schema, filtered to [start, end] inclusive.

    Each partition file is scanned separately and cast to the catalog schema (the adopted files mix
    Datetime units). Date-like columns come back as Date (session date) and timestamps as IST. `sub`
    selects the folder of nested datasets (nse_snapshots kind, features/labels version, preds model id).
    """
    spec = get_spec(dataset)
    files = list_partitions(settings, dataset, sub)
    use_range = spec.date_col is not None or spec.partition == "day"
    frames: list[pl.LazyFrame] = []
    for f in files:
        if use_range:
            rng = _file_range(f.name)
            if rng is not None and (
                (start is not None and rng[1] < start) or (end is not None and rng[0] > end)
            ):
                continue
        lf = pl.scan_parquet(f)
        have = lf.collect_schema()
        dc = spec.date_col if (start is not None or end is not None) else None
        exprs = _conform_exprs(spec, have, columns, extra_needed=(dc,) if dc else ())
        lf = lf.select(exprs)
        if dc:
            col = pl.col(dc)
            if isinstance(spec.schema[dc], pl.Datetime):
                col = col.dt.date()
            if start is not None:
                lf = lf.filter(col >= start)
            if end is not None:
                lf = lf.filter(col <= end)
            if columns is not None and dc not in columns:
                lf = lf.select(columns)
        frames.append(lf)
    if not frames:
        return _empty(spec, columns)
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="diagonal_relaxed" if spec.keep_extra else "vertical")


# ---- write ------------------------------------------------------------------------------------


def _validate_frame(spec: DatasetSpec, df: pl.DataFrame, *, check_nulls: bool) -> None:
    missing = [c for c in spec.key if c not in df.columns]
    if missing:
        raise DataValidationError(f"{spec.name}: missing key column(s) {missing}")
    if check_nulls:
        for c in (
            *(k for k in spec.key if k not in spec.nullable_key),
            *([spec.date_col] if spec.date_col else []),
        ):
            n = df[c].null_count()
            if n:
                raise DataValidationError(f"{spec.name}: {n} null value(s) in key column {c!r}")


def _atomic_write(path: Path, df: pl.DataFrame, clock: Clock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        df.write_parquet(tmp, compression="zstd")
        for attempt in range(_RETRIES + 1):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == _RETRIES:
                    raise
                clock.sleep(_RETRY_WAIT_S)
    finally:
        tmp.unlink(missing_ok=True)


def _prepare(spec: DatasetSpec, df: pl.DataFrame, *, check_nulls: bool) -> pl.DataFrame:
    _validate_frame(spec, df, check_nulls=False)
    out = df.lazy().select(_conform_exprs(spec, df.schema, None)).collect()
    _validate_frame(spec, out, check_nulls=check_nulls)
    return out.sort(list(spec.key), maintain_order=True) if spec.key else out


def write_partition(
    settings: Settings,
    dataset: str,
    key: str,
    df: pl.DataFrame,
    sub: str | None = None,
    clock: Clock | None = None,
) -> Path:
    """Atomically write `df` as the whole partition `key` (e.g. "2026-09", "all"); returns the path."""
    spec = get_spec(dataset)
    path = _dir(settings, spec, sub) / _file_name(spec, key)
    out = _prepare(spec, df, check_nulls=True)
    _atomic_write(path, out, clock if clock is not None else SystemClock())
    return path


def _partition_key_expr(spec: DatasetSpec) -> pl.Expr:
    if spec.partition == "all":
        return pl.lit("all")
    col = pl.col(spec.date_col)  # type: ignore[arg-type]
    fmt = {"month": "%Y-%m", "year": "%Y", "day": "%Y-%m-%d"}[spec.partition]
    return col.dt.strftime(fmt)


def upsert(
    settings: Settings,
    dataset: str,
    df: pl.DataFrame,
    sub: str | None = None,
    clock: Clock | None = None,
) -> list[Path]:
    """Merge rows into the dataset's partitions on its key (new rows win); returns paths written."""
    spec = get_spec(dataset)
    if not spec.key:
        raise DataValidationError(f"{spec.name}: no key; cannot upsert")
    if spec.partition != "all" and spec.date_col is None:
        raise DataValidationError(f"{spec.name}: partitioned dataset needs a date column")
    clk = clock if clock is not None else SystemClock()
    new = _prepare(spec, df, check_nulls=True)
    base = _dir(settings, spec, sub)
    written: list[Path] = []
    for part in new.with_columns(__pk=_partition_key_expr(spec)).partition_by("__pk", as_dict=True).values():
        pk = part["__pk"][0]
        part = part.drop("__pk")
        path = base / _file_name(spec, pk)
        if path.is_file():
            old = _conform(pl.scan_parquet(path), spec).collect()
            part = pl.concat([old, part], how="diagonal_relaxed")
        merged = part.unique(subset=list(spec.key), keep="last", maintain_order=True)
        merged = merged.sort(list(spec.key), maintain_order=True)
        _atomic_write(path, merged, clk)
        written.append(path)
    return sorted(written)
