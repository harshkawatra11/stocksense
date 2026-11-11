"""Weekday arithmetic shared by the calendar, event and auction features.

Forward-looking session counts cannot use the stored session list (a run truncated at `as_of` does not
know the future sessions), so they count Monday-Friday days, minus an optional set of known holidays.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import numpy as np
import polars as pl


def _arr(s: pl.Series) -> np.ndarray:
    return s.fill_null(date(1970, 1, 1)).to_numpy().astype("datetime64[D]")


def bday_count(a: pl.Series, b: pl.Series, holidays: Iterable[date] = ()) -> pl.Series:
    """Int64 series: weekdays in [a, b) (1 when b is the next weekday after a); null if either is null."""
    hol = np.array(sorted(holidays), dtype="datetime64[D]") if holidays else None
    out = np.busday_count(_arr(a), _arr(b), holidays=hol)
    mask = a.is_null() | b.is_null()
    return (
        pl.Series(out, dtype=pl.Int64)
        .to_frame("x")
        .select(pl.when(mask).then(None).otherwise(pl.col("x")).alias("x"))["x"]
    )


def bday_shift(a: pl.Series, n: int) -> pl.Series:
    """Date series moved by n weekdays (n < 0 backward; a weekend date rolls to the nearest weekday first)."""
    out = np.busday_offset(_arr(a), n, roll="forward" if n >= 0 else "backward")
    ser = pl.Series(out.astype("datetime64[D]")).cast(pl.Date)
    return pl.select(pl.when(a.is_null()).then(None).otherwise(ser).alias("x"))["x"]
