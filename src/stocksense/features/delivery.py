"""Delivery features (T4.3f). The delivery columns are null before 2019-08-23, so every feature is null
there (nothing is filled); a window that holds a null is null.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from stocksense.features.base import FeatureContext, feature, log_ret, roll_mean, roll_std


def _stage(lf: pl.LazyFrame) -> pl.LazyFrame:
    names = lf.collect_schema().names()
    lf = lf.sort("sid", "date")
    if "ret" in names:
        lf = lf.with_columns(log_ret("ret").alias("la"))
    return lf


def deliv_pct() -> pl.Expr:
    return pl.col("deliv_pct")


def deliv_z(w: int) -> pl.Expr:
    """Delivery share against the previous row's w-session mean and std (today is not in its baseline)."""
    mean = roll_mean("deliv_pct", w).shift(1).over("sid")
    std = roll_std("deliv_pct", w).shift(1).over("sid")
    return pl.when(std > 0).then((pl.col("deliv_pct") - mean) / std)


def deliv_val_shock(w: int) -> pl.Expr:
    val = pl.col("deliv_qty") * pl.col("close")
    return val / roll_mean(val, w).shift(1).over("sid")


def deliv_up(w: int) -> pl.Expr:
    return deliv_z(w) * pl.col("la").sign()


def deliv_trend(short: int, long: int) -> pl.Expr:
    return roll_mean("deliv_pct", short) / roll_mean("deliv_pct", long)


def deliv_z_mean(w: int, base: int) -> pl.Expr:
    """Mean over w rows of deliv_z (computed with a `base`-row baseline)."""
    return roll_mean(deliv_z(base), w)


SPECS: dict[str, tuple[Callable[[], pl.Expr], int, str]] = {
    "deliv_pct": (
        deliv_pct,
        1,
        "Share of the day's traded shares that were taken for delivery (not squared off).",
    ),
    "deliv_z": (
        lambda: deliv_z(60),
        61,
        "How unusual today's delivery share is against its previous 60-session norm (z-score).",
    ),
    "deliv_val_shock": (
        lambda: deliv_val_shock(60),
        61,
        "Rupee value delivered today relative to its previous 60-session average.",
    ),
    "deliv_up": (
        lambda: deliv_up(60),
        61,
        "Delivery-share z-score signed by the day's direction (positive = unusual delivery on an up day).",
    ),
    "deliv_trend": (
        lambda: deliv_trend(5, 60),
        60,
        "Delivery share of the last 5 sessions relative to its 60-session average.",
    ),
    "deliv_z_5": (
        lambda: deliv_z_mean(5, 60),
        65,
        "Average of the delivery-share z-score over the last 5 sessions.",
    ),
}


def compute(lf: pl.LazyFrame, names: list[str] | None = None) -> pl.LazyFrame:
    names = list(SPECS) if names is None else names
    return _stage(lf).select("date", "sid", *[SPECS[n][0]().alias(n) for n in names])


def _register(name: str) -> None:
    _, window, label = SPECS[name]

    def fn(ctx: FeatureContext, _n: str = name) -> pl.LazyFrame:
        return compute(ctx.panel(), [_n])

    feature(name, group="delivery", kind="xs", window=window, label=label)(fn)


for _name in SPECS:
    _register(_name)
