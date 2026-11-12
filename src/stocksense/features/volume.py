"""Volume and liquidity features (T4.3e). Turnover is in rupees; volume is split-adjusted."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from stocksense.features.base import FeatureContext, feature, log_ret, roll_corr, roll_mean


def _stage(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Sort and add la, ln(turnover) and the average trade size (null without trades)."""
    names = lf.collect_schema().names()
    lf = lf.sort("sid", "date")
    if "ret" in names:
        lf = lf.with_columns(log_ret("ret").alias("la"))
    if "turnover_inr" in names:
        t = pl.col("turnover_inr")
        lf = lf.with_columns(pl.when(t > 0).then(t.log()).alias("lt"))
        if "n_trades" in names:
            nt = pl.col("n_trades")
            lf = lf.with_columns(pl.when((nt > 0) & (t > 0)).then(t / nt).alias("ats"))
    return lf


def _log_pos(e: pl.Expr) -> pl.Expr:
    return pl.when(e > 0).then(e.log())


def log_turn(w: int) -> pl.Expr:
    """ln of the w-row mean turnover (null if the mean is not positive)."""
    return _log_pos(roll_mean("turnover_inr", w))


def amihud(w: int) -> pl.Expr:
    """Mean of |ret| per crore rupees traded (zero turnover days make the window null)."""
    t = pl.col("turnover_inr")
    illiq = pl.when(t > 0).then(pl.col("ret").abs() / (t / 1e7))
    return roll_mean(illiq, w)


def turn_shock(w: int) -> pl.Expr:
    return pl.col("turnover_inr") / roll_mean("turnover_inr", w).shift(1).over("sid")


def vol_shock(short: int, long: int) -> pl.Expr:
    return roll_mean("adj_volume", short) / roll_mean("adj_volume", long).shift(1).over("sid")


def turn_trend(short: int, long: int) -> pl.Expr:
    return roll_mean("turnover_inr", short) / roll_mean("turnover_inr", long)


def avg_trade_size_rel(w: int) -> pl.Expr:
    return pl.col("ats") / roll_mean("ats", w).shift(1).over("sid")


def zero_ret_frac(w: int) -> pl.Expr:
    flag = pl.when(pl.col("ret").is_null()).then(None).otherwise((pl.col("ret") == 0).cast(pl.Float64))
    return roll_mean(flag, w)


def pv_corr(w: int) -> pl.Expr:
    return roll_corr("la", "lt", w)


SPECS: dict[str, tuple[Callable[[], pl.Expr], int, str]] = {
    "log_turn_21": (
        lambda: log_turn(21),
        21,
        "Typical daily rupees traded over the last month (log of the mean).",
    ),
    "amihud_21": (
        lambda: amihud(21),
        21,
        "Price impact per crore rupees traded, averaged over a month (Amihud illiquidity).",
    ),
    "turn_shock": (
        lambda: turn_shock(60),
        60,
        "Today's rupee turnover relative to its previous 60-session average.",
    ),
    "vol_shock_5": (
        lambda: vol_shock(5, 60),
        60,
        "Share volume of the last 5 sessions relative to the previous 60-session average.",
    ),
    "turn_trend": (
        lambda: turn_trend(21, 126),
        126,
        "Whether trading is picking up: 1-month average turnover over the 6-month average.",
    ),
    "avg_trade_size_rel": (
        lambda: avg_trade_size_rel(60),
        60,
        "Average rupees per trade today relative to its previous 60-session average (big-ticket activity).",
    ),
    "zero_ret_frac_63": (
        lambda: zero_ret_frac(63),
        63,
        "Share of the last 63 sessions on which the price did not change (illiquidity sign).",
    ),
    "size_proxy": (
        lambda: log_turn(252),
        252,
        "Size of the stock by trading activity: log of average daily turnover over a year.",
    ),
    "pv_corr_21": (
        lambda: pv_corr(21),
        21,
        "Whether big moves come with heavy trading: correlation of returns and log turnover over a month.",
    ),
}


def compute(lf: pl.LazyFrame, names: list[str] | None = None) -> pl.LazyFrame:
    names = list(SPECS) if names is None else names
    return _stage(lf).select("date", "sid", *[SPECS[n][0]().alias(n) for n in names])


def _register(name: str) -> None:
    _, window, label = SPECS[name]

    def fn(ctx: FeatureContext, _n: str = name) -> pl.LazyFrame:
        return compute(ctx.panel(), [_n])

    feature(name, group="volume", kind="xs", window=window, label=label)(fn)


for _name in SPECS:
    _register(_name)
