"""Risk features (T4.3d): volatility, range estimators, tail shape, ATR and market exposure.

la = ln(1 + ret), lm = ln(1 + ew500_ret). Ranges use adjusted prices (ratios are unaffected by splits).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import polars as pl

from stocksense.features.base import (
    FeatureContext,
    feature,
    log_ret,
    roll_corr,
    roll_cov,
    roll_mean,
    roll_std,
    roll_var,
    span_ok,
)
from stocksense.features.price import wilder

LN2 = math.log(2.0)


def _stage(lf: pl.LazyFrame, beta_w: int = 126) -> pl.LazyFrame:
    """Sort, add la; when the frame carries `lm` also the current beta and residual on the LAGGED beta."""
    names = lf.collect_schema().names()
    lf = lf.sort("sid", "date")
    if "ret" in names:
        lf = lf.with_columns(log_ret("ret").alias("la"))
    if "ret" in names and "lm" in names:
        lf = lf.with_columns(
            (roll_cov("la", "lm", beta_w) / roll_var("lm", beta_w)).alias("beta")
        ).with_columns((pl.col("la") - pl.col("beta").shift(1).over("sid") * pl.col("lm")).alias("e_idio"))
    return lf


def vol(w: int) -> pl.Expr:
    return roll_std("la", w)


def vol_ratio(short: int, long: int) -> pl.Expr:
    return roll_std("la", short) / roll_std("la", long)


def _ln_hl() -> pl.Expr:
    ok = (pl.col("adj_low") > 0) & (pl.col("adj_high") >= pl.col("adj_low"))
    return pl.when(ok).then((pl.col("adj_high") / pl.col("adj_low")).log())


def _ln_co() -> pl.Expr:
    ok = (pl.col("adj_open") > 0) & (pl.col("adj_close") > 0)
    return pl.when(ok).then((pl.col("adj_close") / pl.col("adj_open")).log())


def parkinson(w: int) -> pl.Expr:
    return (roll_mean(_ln_hl() ** 2, w) / (4.0 * LN2)).sqrt()


def garman_klass(w: int) -> pl.Expr:
    term = 0.5 * _ln_hl() ** 2 - (2.0 * LN2 - 1.0) * _ln_co() ** 2
    return roll_mean(term, w).clip(lower_bound=0.0).sqrt()


def downside_vol(w: int) -> pl.Expr:
    return roll_mean(pl.col("la").clip(upper_bound=0.0) ** 2, w).sqrt()


def skew(w: int) -> pl.Expr:
    return pl.when(span_ok(w)).then(pl.col("la").rolling_skew(w, bias=False, min_samples=w).over("sid"))


def kurt(w: int) -> pl.Expr:
    kt = pl.col("la").rolling_kurtosis(w, fisher=True, bias=False, min_samples=w).over("sid")
    return pl.when(span_ok(w)).then(kt)


def atr_pct(n: int) -> pl.Expr:
    pc = pl.col("adj_close").shift(1).over("sid")
    h, lo = pl.col("adj_high"), pl.col("adj_low")
    tr = pl.max_horizontal(h - lo, (h - pc).abs(), (lo - pc).abs())
    return wilder(tr, n) / pl.col("adj_close")


def beta(w: int) -> pl.Expr:
    return roll_cov("la", "lm", w) / roll_var("lm", w)


def idio_vol(w: int) -> pl.Expr:
    return roll_std("e_idio", w)


def corr_mkt(w: int) -> pl.Expr:
    return roll_corr("la", "lm", w)


SPECS: dict[str, tuple[Callable[[], pl.Expr], int, str]] = {
    "vol_21": (
        lambda: vol(21),
        21,
        "How jumpy the stock has been over the last 21 sessions (std of daily log returns).",
    ),
    "vol_63": (
        lambda: vol(63),
        63,
        "How jumpy the stock has been over the last 3 months (std of daily returns).",
    ),
    "vol_ratio": (
        lambda: vol_ratio(21, 126),
        126,
        "Recent volatility (1 month) relative to its 6-month level; above 1 = getting jumpier.",
    ),
    "parkinson_21": (
        lambda: parkinson(21),
        21,
        "Volatility estimated from daily high-low ranges over 21 sessions (Parkinson).",
    ),
    "gk_21": (
        lambda: garman_klass(21),
        21,
        "Volatility estimated from open, high, low and close over 21 sessions (Garman-Klass).",
    ),
    "downside_vol_63": (
        lambda: downside_vol(63),
        63,
        "Volatility counting only down days over 3 months (downside deviation).",
    ),
    "skew_63": (
        lambda: skew(63),
        63,
        "Skewness of daily returns over 3 months (positive = occasional big gains).",
    ),
    "kurt_63": (lambda: kurt(63), 63, "Excess kurtosis of daily returns over 3 months (high = fat-tailed)."),
    "atr_14_pct": (
        lambda: atr_pct(14),
        15,
        "Average true range over 14 sessions (Wilder) as a fraction of the price.",
    ),
    "beta_126": (
        lambda: beta(126),
        126,
        "Sensitivity to the market over 6 months (1 = moves with the market).",
    ),
    "idio_vol_63": (
        lambda: idio_vol(63),
        63,
        "Volatility left after removing the market's influence, over 3 months.",
    ),
    "corr_mkt_63": (
        lambda: corr_mkt(63),
        63,
        "How closely the stock tracks the market over 3 months (correlation).",
    ),
}


def compute(lf: pl.LazyFrame, names: list[str] | None = None) -> pl.LazyFrame:
    names = list(SPECS) if names is None else names
    return _stage(lf).select("date", "sid", *[SPECS[n][0]().alias(n) for n in names])


def _panel_with_market(ctx: FeatureContext) -> pl.LazyFrame:
    lm = ctx.market().select("date", log_ret("ew500_ret").alias("lm"))
    return ctx.panel().join(lm, on="date", how="left")


def _register(name: str) -> None:
    _, window, label = SPECS[name]

    def fn(ctx: FeatureContext, _n: str = name) -> pl.LazyFrame:
        return compute(_panel_with_market(ctx), [_n])

    feature(name, group="risk", kind="xs", window=window, label=label)(fn)


for _name in SPECS:
    _register(_name)
