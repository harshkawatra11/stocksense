"""Price features: momentum and trend (T4.3a), reversal and price action (T4.3b).

Notation: la = ln(1 + ret) (adjusted log return), P = adj_close, lm = ln(1 + ew500_ret).
Every feature is an expression builder over a staged panel (`_stage` adds la, lp and, when the frame
carries `lm`, the one-row-lagged market beta and the residual return e). The registered functions only
join the market, stage the panel and select. Builders take the window so tests can use tiny windows.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from stocksense.features.base import (
    FeatureContext,
    feature,
    log_ret,
    roll_cov,
    roll_max,
    roll_min,
    roll_slope_r2,
    roll_std,
    roll_sum,
    roll_var,
    span_ok,
)

# ---- staging -----------------------------------------------------------------------------------


def _stage(lf: pl.LazyFrame, beta_w: int = 63) -> pl.LazyFrame:
    """Sort by sid, date and add the helper columns the builders use (those the frame supports)."""
    names = lf.collect_schema().names()
    lf = lf.sort("sid", "date")
    if "ret" in names:
        lf = lf.with_columns(log_ret("ret").alias("la"))
    if "adj_close" in names:
        lf = lf.with_columns(pl.when(pl.col("adj_close") > 0).then(pl.col("adj_close").log()).alias("lp"))
    if "ret" in names and "lm" in names:
        # beta estimated through the previous row: the beta used on day s never sees day s
        lf = lf.with_columns(
            (roll_cov("la", "lm", beta_w) / roll_var("lm", beta_w)).shift(1).over("sid").alias("beta")
        ).with_columns((pl.col("la") - pl.col("beta") * pl.col("lm")).alias("e"))
    return lf


# ---- T4.3a builders ----------------------------------------------------------------------------


def mom(w: int) -> pl.Expr:
    return roll_sum("la", w)


def mom_skip(long: int, skip: int) -> pl.Expr:
    return roll_sum("la", long) - roll_sum("la", skip)


def mom_accel(short: int, long: int) -> pl.Expr:
    return mom(short) - (mom(long) - mom(short))


def hi_prox(w: int) -> pl.Expr:
    return pl.col("adj_close") / roll_max("adj_high", w)


def lo_prox(w: int) -> pl.Expr:
    return pl.col("adj_close") / roll_min("adj_low", w)


def _rows_since_max(s: pl.Series, w: int) -> pl.Series:
    """Rows since the max of each trailing w-window (0 = today, latest tie wins); NaN if invalid."""
    a = s.cast(pl.Float64).to_numpy()
    out = np.full(len(a), np.nan)
    if len(a) >= w:
        win = np.lib.stride_tricks.sliding_window_view(a, w)
        pos = np.argmax(win[:, ::-1], axis=1).astype(np.float64)
        pos[np.isnan(win).any(axis=1)] = np.nan
        out[w - 1 :] = pos
    return pl.Series(out)


def days_since_hi(w: int) -> pl.Expr:
    rows = pl.col("adj_high").map_batches(lambda s: _rows_since_max(s, w), return_dtype=pl.Float64)
    rows = rows.over("sid").fill_nan(None)
    return pl.when(span_ok(w)).then(rows / w)


def ma_gap(w: int) -> pl.Expr:
    return pl.col("adj_close") / pl.col("adj_close").rolling_mean(w, min_samples=w).over("sid") - 1.0


def trend_r2(w: int) -> pl.Expr:
    slope, r2 = roll_slope_r2("lp", w)
    return r2 * slope.sign()


def resid_mom(w: int) -> pl.Expr:
    """Sum of the residual return over w rows divided by its std (needs `_stage` to have built `e`)."""
    return roll_sum("e", w) / roll_std("e", w)


_A: dict[str, tuple[Callable[[], pl.Expr], int, str]] = {
    "mom_5": (lambda: mom(5), 5, "Total price move over the last 5 sessions (log return)."),
    "mom_21": (lambda: mom(21), 21, "Total price move over the last month (21 sessions, log return)."),
    "mom_63": (lambda: mom(63), 63, "Total price move over the last 3 months (63 sessions, log return)."),
    "mom_126": (lambda: mom(126), 126, "Total price move over the last 6 months (126 sessions, log return)."),
    "mom_252_skip21": (
        lambda: mom_skip(252, 21),
        252,
        "Price move over the past year, leaving out the most recent month (12-1 momentum).",
    ),
    "mom_accel": (
        lambda: mom_accel(63, 126),
        126,
        "Whether the last 3 months moved more than the 3 months before them (momentum acceleration).",
    ),
    "hi52_prox": (lambda: hi_prox(252), 252, "Price as a fraction of its 52-week high (1 = at the high)."),
    "lo52_prox": (lambda: lo_prox(252), 252, "Price as a multiple of its 52-week low (1 = at the low)."),
    "days_since_hi52": (
        lambda: days_since_hi(252),
        252,
        "How long ago the 52-week high was set, as a share of the year (0 = today).",
    ),
    "ma_gap_20": (lambda: ma_gap(20), 20, "How far the price is above its 20-session average (fraction)."),
    "ma_gap_50": (lambda: ma_gap(50), 50, "How far the price is above its 50-session average (fraction)."),
    "ma_gap_200": (
        lambda: ma_gap(200),
        200,
        "How far the price is above its 200-session average (fraction).",
    ),
    "trend_r2_63": (
        lambda: trend_r2(63),
        63,
        "How straight the 3-month trend is (R-squared of log price, negative if falling).",
    ),
    "resid_mom_126": (
        lambda: resid_mom(126),
        126,
        "Six-month move beyond what the market explains, per unit of its own noise.",
    ),
}

# ---- T4.3b builders (filled in with T4.3b) -----------------------------------------------------

_B: dict[str, tuple[Callable[[], pl.Expr], int, str]] = {}

SPECS = {**_A, **_B}


def compute(lf: pl.LazyFrame, names: list[str] | None = None) -> pl.LazyFrame:
    """date, sid and the named price features computed from a panel frame (with `sx`, `lm` if needed)."""
    names = list(SPECS) if names is None else names
    st = _stage(lf)
    return st.select("date", "sid", *[SPECS[n][0]().alias(n) for n in names])


def _panel_with_market(ctx: FeatureContext) -> pl.LazyFrame:
    lm = ctx.market().select("date", log_ret("ew500_ret").alias("lm"))
    return ctx.panel().join(lm, on="date", how="left")


def _register(name: str) -> None:
    _, window, label = SPECS[name]

    def fn(ctx: FeatureContext, _n: str = name) -> pl.LazyFrame:
        return compute(_panel_with_market(ctx), [_n])

    feature(name, group="price", kind="xs", window=window, label=label)(fn)


for _name in SPECS:
    _register(_name)
