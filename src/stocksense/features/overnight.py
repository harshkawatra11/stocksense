"""Overnight / intraday split features (T4.3c; PLAN B.3 Findings 3-5).

lco = ln(1 + ret_co) (close-to-open), loc = ln(1 + ret_oc) (open-to-close), la = ln(1 + ret).
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from stocksense.features.base import (
    FeatureContext,
    feature,
    log_ret,
    roll_mean,
    roll_std,
    roll_sum,
    roll_var,
)


def _stage(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.sort("sid", "date").with_columns(
        log_ret("ret").alias("la"), log_ret("ret_co").alias("lco"), log_ret("ret_oc").alias("loc")
    )


def co_1() -> pl.Expr:
    return pl.col("lco")


def oc_1() -> pl.Expr:
    return pl.col("loc")


def co_sum(w: int) -> pl.Expr:
    return roll_sum("lco", w)


def oc_sum(w: int) -> pl.Expr:
    return roll_sum("loc", w)


def co_minus_oc(w: int) -> pl.Expr:
    return roll_sum("lco", w) - roll_sum("loc", w)


def co_vol_share(w: int) -> pl.Expr:
    vc, vo = roll_var("lco", w), roll_var("loc", w)
    return vc / (vc + vo)


def gap_abs_z(w: int) -> pl.Expr:
    """|overnight move| over the sid's previous-row std of the daily log return (not today's std)."""
    return pl.col("lco").abs() / roll_std("la", w).shift(1).over("sid")


def gap_fill(w: int) -> pl.Expr:
    """Mean over w rows of -sign(lco) * loc: positive when the day's move fades the opening gap."""
    return roll_mean(-pl.col("lco").sign() * pl.col("loc"), w)


def co_hit(w: int) -> pl.Expr:
    """Share of the last w rows with a positive overnight move."""
    up = pl.when(pl.col("lco").is_null()).then(None).otherwise((pl.col("lco") > 0).cast(pl.Float64))
    return roll_mean(up, w)


SPECS: dict[str, tuple[Callable[[], pl.Expr], int, str]] = {
    "co_1": (co_1, 1, "The opening gap: close-to-open move (log return)."),
    "co_sum_5": (lambda: co_sum(5), 5, "Sum of the last 5 overnight (close-to-open) moves."),
    "co_sum_21": (lambda: co_sum(21), 21, "Sum of the last 21 overnight (close-to-open) moves."),
    "oc_1": (oc_1, 1, "Today's intraday move: open-to-close (log return)."),
    "oc_sum_5": (lambda: oc_sum(5), 5, "Sum of the last 5 intraday (open-to-close) moves."),
    "oc_sum_21": (lambda: oc_sum(21), 21, "Sum of the last 21 intraday (open-to-close) moves."),
    "co_minus_oc_63": (
        lambda: co_minus_oc(63),
        63,
        "Three-month overnight drift minus three-month intraday drift.",
    ),
    "co_vol_share_63": (
        lambda: co_vol_share(63),
        63,
        "Share of the stock's variance that happens overnight rather than in the session (3 months).",
    ),
    "gap_abs_z": (
        lambda: gap_abs_z(21),
        21,
        "Size of today's opening gap relative to the stock's usual daily move.",
    ),
    "gap_fill_5": (
        lambda: gap_fill(5),
        5,
        "Average over 5 sessions of how much the day reversed the opening gap (positive = gaps fade).",
    ),
    "co_hit_21": (
        lambda: co_hit(21),
        21,
        "Share of the last 21 sessions that opened higher than the prior close.",
    ),
}


def compute(lf: pl.LazyFrame, names: list[str] | None = None) -> pl.LazyFrame:
    names = list(SPECS) if names is None else names
    return _stage(lf).select("date", "sid", *[SPECS[n][0]().alias(n) for n in names])


def _register(name: str) -> None:
    _, window, label = SPECS[name]

    def fn(ctx: FeatureContext, _n: str = name) -> pl.LazyFrame:
        return compute(ctx.panel(), [_n])

    feature(name, group="overnight", kind="xs", window=window, label=label)(fn)


for _name in SPECS:
    _register(_name)
