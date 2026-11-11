"""Feature framework: the decorator, FeatureSpec, FeatureContext and the rolling helpers (PLAN T4.1).

Every feature is a function `fn(ctx) -> LazyFrame[date, sid, <name>...]` (kind 'xs') or
`LazyFrame[date, <name>...]` (kind 'mkt'). All data reaches it through `FeatureContext`, whose
accessors cut everything at `as_of`: that single choke point is what the truncation test exercises.
Features are raw values; ranking and winsorising happen in `features.build`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time

import polars as pl

from stocksense.core.clock import IST
from stocksense.core.config import Settings
from stocksense.data import io
from stocksense.data.catalog import get_spec
from stocksense.features.registry import REGISTRY, register

KINDS = ("xs", "mkt")
CUTOFF = time(20, 0)  # features for t may use anything published by 20:00 IST on t
_DATE_GATE = "date"
_TS_GATES = ("ts", "available_ts", "announced_ts", "filing_ts", "ts_seen")

FeatureFn = Callable[["FeatureContext"], pl.LazyFrame]


@dataclass(frozen=True)
class FeatureSpec:
    """One registered feature (or family: `outputs` lists every column the function emits)."""

    name: str
    group: str
    kind: str
    fn: FeatureFn
    window: int
    version: int
    label: str
    outputs: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.outputs:
            object.__setattr__(self, "outputs", (self.name,))


def feature(
    name: str,
    *,
    group: str,
    kind: str = "xs",
    window: int = 1,
    version: int = 1,
    label: str = "",
    outputs: Sequence[str] | None = None,
    registry: dict[str, FeatureSpec] | None = None,
) -> Callable[[FeatureFn], FeatureFn]:
    """Decorator: register `fn` as feature `name` (kind 'xs' = per (date, sid), 'mkt' = per date)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    def deco(fn: FeatureFn) -> FeatureFn:
        spec = FeatureSpec(name, group, kind, fn, window, version, label, tuple(outputs or (name,)))
        register(spec, registry if registry is not None else REGISTRY)
        return fn

    return deco


# ---- context -----------------------------------------------------------------------------------


class FeatureContext:
    """Lazy, as_of-truncated access to the panel, market, universe and raw datasets."""

    def __init__(self, settings: Settings, as_of: date | None = None) -> None:
        self.settings = settings
        self.as_of = as_of

    @property
    def cutoff_ts(self) -> datetime | None:
        """20:00 IST on `as_of` (None when not truncating)."""
        return None if self.as_of is None else datetime.combine(self.as_of, CUTOFF, tzinfo=IST)

    def sessions(self) -> pl.LazyFrame:
        """(date, sx): every session of the calendar with its 0-based index (stable under truncation)."""
        return (
            io.read(self.settings, "bhavcopy_eq", end=self.as_of, columns=["date"])
            .select(pl.col("date").unique().sort())
            .with_row_index("sx")
            .with_columns(pl.col("sx").cast(pl.Int64))
            .select("date", "sx")
        )

    def panel(self) -> pl.LazyFrame:
        """Daily panel (all sids) sorted by sid, date, with `sx` the calendar session index."""
        return (
            io.read(self.settings, "panel_daily", end=self.as_of)
            .join(self.sessions(), on="date", how="left")
            .sort("sid", "date")
        )

    def market(self) -> pl.LazyFrame:
        """panel/market rows up to as_of."""
        return io.read(self.settings, "panel_market", end=self.as_of).sort("date")

    def universe(self) -> pl.LazyFrame:
        """panel/universe rows up to as_of (eligible rows only; u500/u200 flags)."""
        return io.read(self.settings, "panel_universe", end=self.as_of).sort("date", "sid")

    def raw(self, dataset: str, columns: list[str] | None = None, sub: str | None = None) -> pl.LazyFrame:
        """A raw dataset cut at as_of: `date <= as_of` and every timestamp column `<= as_of 20:00 IST`.

        Date-valued columns that look forward (corporate_actions.ex_date, board_meetings.meeting_date) are
        NOT filtered: a feature using them must gate on its own knowledge time (announced_ts, first seen).
        """
        names = list(get_spec(dataset).schema)
        gates = [c for c in (_DATE_GATE, *_TS_GATES) if c in names] if self.as_of is not None else []
        need = None if columns is None else list(dict.fromkeys([*columns, *gates]))
        lf = io.read(self.settings, dataset, end=self.as_of, columns=need, sub=sub)
        cut = self.cutoff_ts
        for c in gates:
            if c == _DATE_GATE:
                lf = lf.filter(pl.col(c) <= self.as_of)
            else:
                lf = lf.filter(pl.col(c).is_null() | (pl.col(c) <= cut))
        return lf if columns is None else lf.select(columns)


# ---- rolling helpers ---------------------------------------------------------------------------
# All run per sid over the sid's own rows (the frame must carry `sid` and `sx` and be sorted by sid,
# date). min_samples = w, and a window whose calendar span is too wide (it straddles a suspension)
# is null: sx - sx.shift(w-1) + 1 > ceil(1.25 w) + 2.


def _e(x: str | pl.Expr) -> pl.Expr:
    return pl.col(x) if isinstance(x, str) else x


def span_ok(w: int) -> pl.Expr:
    """True where the last `w` rows of the sid span at most ceil(1.25 w) + 2 sessions."""
    sx = pl.col("sx")
    return (sx - sx.shift(w - 1).over("sid") + 1) <= math.ceil(1.25 * w) + 2


def _guard(expr: pl.Expr, w: int) -> pl.Expr:
    return pl.when(span_ok(w)).then(expr).otherwise(None)


def roll_sum(col: str | pl.Expr, w: int) -> pl.Expr:
    """Rolling sum over the last w rows of each sid (null if fewer or too wide a span)."""
    return _guard(_e(col).rolling_sum(w, min_samples=w).over("sid"), w)


def roll_mean(col: str | pl.Expr, w: int) -> pl.Expr:
    """Rolling mean over the last w rows of each sid."""
    return _guard(_e(col).rolling_mean(w, min_samples=w).over("sid"), w)


def roll_std(col: str | pl.Expr, w: int) -> pl.Expr:
    """Rolling sample standard deviation (ddof=1)."""
    return _guard(_e(col).rolling_std(w, min_samples=w, ddof=1).over("sid"), w)


def roll_var(col: str | pl.Expr, w: int) -> pl.Expr:
    """Rolling sample variance (ddof=1)."""
    return _guard(_e(col).rolling_var(w, min_samples=w, ddof=1).over("sid"), w)


def roll_min(col: str | pl.Expr, w: int) -> pl.Expr:
    """Rolling minimum."""
    return _guard(_e(col).rolling_min(w, min_samples=w).over("sid"), w)


def roll_max(col: str | pl.Expr, w: int) -> pl.Expr:
    """Rolling maximum."""
    return _guard(_e(col).rolling_max(w, min_samples=w).over("sid"), w)


def roll_corr(a: str | pl.Expr, b: str | pl.Expr, w: int) -> pl.Expr:
    """Rolling Pearson correlation of two series."""
    return _guard(pl.rolling_corr(_e(a), _e(b), window_size=w, min_samples=w).over("sid"), w)


def roll_cov(a: str | pl.Expr, b: str | pl.Expr, w: int) -> pl.Expr:
    """Rolling sample covariance (ddof=1) of two series."""
    return _guard(pl.rolling_cov(_e(a), _e(b), window_size=w, min_samples=w, ddof=1).over("sid"), w)


def roll_slope_r2(col: str | pl.Expr, w: int) -> tuple[pl.Expr, pl.Expr]:
    """(slope per row, R^2) of an OLS line of `col` on the sid's row number over the last w rows.

    R^2 is unsigned (multiply by sign(slope) where a signed trend strength is wanted).
    """
    y = _e(col)
    rn = pl.int_range(pl.len()).over("sid").cast(pl.Float64)
    slope = roll_cov(y, rn, w) / roll_var(rn, w)
    corr = roll_corr(y, rn, w)
    return slope, corr * corr


def log_ret(col: str | pl.Expr = "ret") -> pl.Expr:
    """ln(1 + r) where r > -1, else null (the adjusted log return when given `ret`)."""
    r = _e(col)
    return pl.when(r > -1.0).then((1.0 + r).log()).otherwise(None)
