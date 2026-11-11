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
    roll_mean,
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

# ---- T4.3b builders -----------------------------------------------------------------------------


def ret_1() -> pl.Expr:
    return pl.col("la")


def wilder(x: str | pl.Expr, n: int) -> pl.Expr:
    """Wilder smoothing of x (first valid value at row 1 of each sid): the first average is the simple mean
    of the first n values (row n), then avg = (prev * (n - 1) + x) / n, via ewm_mean(alpha=1/n, adjust=False).
    Null before row n and where the last n + 1 rows span too many sessions."""
    x = pl.col(x) if isinstance(x, str) else x
    rn = pl.int_range(pl.len()).over("sid")
    seed = x.rolling_mean(n, min_samples=n).over("sid")
    y = pl.when(rn == n).then(seed).when(rn > n).then(x)
    avg = y.ewm_mean(alpha=1.0 / n, adjust=False, ignore_nulls=True, min_samples=1).over("sid")
    return pl.when(span_ok(n + 1)).then(avg)


def rsi(n: int) -> pl.Expr:
    delta = pl.col("adj_close") - pl.col("adj_close").shift(1).over("sid")
    ag = wilder(delta.clip(lower_bound=0.0), n)
    al = wilder((-delta).clip(lower_bound=0.0), n)
    return (
        pl.when(al == 0.0)
        .then(pl.when(ag == 0.0).then(50.0).otherwise(100.0))
        .otherwise(100.0 - 100.0 / (1.0 + ag / al))
    )


def bb_z(w: int) -> pl.Expr:
    p = pl.col("adj_close")
    return (p - p.rolling_mean(w, min_samples=w).over("sid")) / roll_std("adj_close", w)


def max_ret(w: int) -> pl.Expr:
    return roll_max("la", w)


def min_ret(w: int) -> pl.Expr:
    return roll_min("la", w)


def close_loc() -> pl.Expr:
    rng = pl.col("adj_high") - pl.col("adj_low")
    return pl.when(rng > 0).then((pl.col("adj_close") - pl.col("adj_low")) / rng)


def close_loc_mean(w: int) -> pl.Expr:
    return roll_mean(close_loc(), w)


def range_pct() -> pl.Expr:
    return (pl.col("adj_high") - pl.col("adj_low")) / pl.col("adj_close")


def nr7(w: int) -> pl.Expr:
    r = range_pct()
    m = roll_min(r, w)
    return pl.when(m.is_null()).then(None).otherwise((r == m).cast(pl.Float64))


def inside_day() -> pl.Expr:
    ph = pl.col("adj_high").shift(1).over("sid")
    pl_ = pl.col("adj_low").shift(1).over("sid")
    both = (pl.col("adj_high") <= ph) & (pl.col("adj_low") >= pl_)
    return pl.when(ph.is_null() | pl_.is_null()).then(None).otherwise(both.cast(pl.Float64))


def streak(up: bool, cap: int) -> pl.Expr:
    """Consecutive rows with la > 0 (up) or la < 0 (down) ending today, capped; 0 on any other day."""
    la = pl.col("la")
    cond = (la > 0) if up else (la < 0)
    hit = cond.fill_null(False)
    grp = (~hit).cast(pl.Int64).cum_sum().over("sid")
    run = hit.cast(pl.Int64).cum_sum().over(["sid", grp])
    return pl.when(la.is_null()).then(None).otherwise(run.clip(upper_bound=cap).cast(pl.Float64))


def dist_round() -> pl.Expr:
    c = pl.col("close")
    u = pl.lit(10.0).pow(c.log(10.0).floor())
    return (c - (c / u).round(0) * u).abs() / c


_B: dict[str, tuple[Callable[[], pl.Expr], int, str]] = {
    "ret_1": (ret_1, 1, "Yesterday's total price move (log return, adjusted for splits and bonuses)."),
    "rsi_14": (
        lambda: rsi(14),
        14,
        "Relative strength index over 14 sessions (0-100; high = recent gains dominate).",
    ),
    "bb_z_20": (
        lambda: bb_z(20),
        20,
        "Price in standard deviations from its 20-session average (Bollinger z-score).",
    ),
    "max_ret_21": (
        lambda: max_ret(21),
        21,
        "Best single-day return in the last 21 sessions (lottery-like jumps).",
    ),
    "min_ret_21": (lambda: min_ret(21), 21, "Worst single-day return in the last 21 sessions."),
    "close_loc": (close_loc, 1, "Where the day closed in its range (0 = at the low, 1 = at the high)."),
    "close_loc_5": (
        lambda: close_loc_mean(5),
        5,
        "Average of the closing position in the day's range over 5 sessions.",
    ),
    "range_pct": (range_pct, 1, "The day's high-low range as a fraction of the close."),
    "nr7": (lambda: nr7(7), 7, "1 if today's range is the narrowest of the last 7 sessions, else 0."),
    "inside_day": (inside_day, 2, "1 if today's high and low sit inside yesterday's range, else 0."),
    "up_streak": (lambda: streak(True, 10), 10, "Number of consecutive up days ending today (max 10)."),
    "down_streak": (lambda: streak(False, 10), 10, "Number of consecutive down days ending today (max 10)."),
    "dist_round": (dist_round, 1, "How far the close is from the nearest round price level, as a fraction."),
}

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
