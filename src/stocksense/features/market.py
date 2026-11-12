"""Market-state and industry-context features (PLAN T4.3g).

`mkt` features are one value per session (broadcast onto every stock by the builder). Every raw source
is optional: when its dataset is absent the feature is null for every date, never an error.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from stocksense.features.base import FeatureContext, feature, log_ret, roll_mean, roll_sum

# ---- helpers -----------------------------------------------------------------------------------


def pctile_last(x: np.ndarray, w: int) -> np.ndarray:
    """Percentile (0-1) of each value among the previous w-1 values: (#below + 0.5 #equal)/(w-1).

    NaN where fewer than w values exist or any value in the window is NaN.
    """
    n = len(x)
    out = np.full(n, np.nan)
    if n < w:
        return out
    win = sliding_window_view(x, w)  # (n-w+1, w)
    last = win[:, -1:]
    others = win[:, :-1]
    pct = ((others < last).sum(axis=1) + 0.5 * (others == last).sum(axis=1)) / (w - 1)
    bad = np.isnan(win).any(axis=1)
    pct = np.where(bad, np.nan, pct)
    out[w - 1 :] = pct
    return out


def _spine(ctx: FeatureContext) -> pl.LazyFrame:
    return ctx.market().select("date").sort("date")


def _lm() -> pl.Expr:
    return log_ret("ew500_ret")


def _mkt_roll(expr: pl.Expr, w: int, fn: str) -> pl.Expr:
    if fn == "sum":
        return expr.rolling_sum(w, min_samples=w)
    return expr.rolling_std(w, min_samples=w, ddof=1)


def _series_pctile(df: pl.DataFrame, col: str, w: int, out: str) -> pl.DataFrame:
    vals = df[col].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    p = pctile_last(vals, w)
    return df.with_columns(pl.Series(out, p).fill_nan(None))


def _on_spine(ctx: FeatureContext, series: pl.DataFrame, cols: list[str]) -> pl.LazyFrame:
    return _spine(ctx).join(series.lazy().select("date", *cols), on="date", how="left").sort("date")


# ---- market returns ----------------------------------------------------------------------------


@feature(
    "mkt_ret_1",
    group="market",
    kind="mkt",
    window=1,
    label="How the broad market (equal-weight top 500) moved on the day, log return.",
)
def mkt_ret_1(ctx: FeatureContext) -> pl.LazyFrame:
    """ln(1 + ew500_ret) per date."""
    return ctx.market().select("date", _lm().alias("mkt_ret_1"))


@feature(
    "mkt_ret_5",
    group="market",
    kind="mkt",
    window=5,
    label="How the broad market moved over the last 5 sessions (summed log return).",
)
def mkt_ret_5(ctx: FeatureContext) -> pl.LazyFrame:
    """Sum of ln(1 + ew500_ret) over 5 sessions."""
    return ctx.market().select("date", _mkt_roll(_lm(), 5, "sum").alias("mkt_ret_5"))


@feature(
    "mkt_ret_21",
    group="market",
    kind="mkt",
    window=21,
    label="How the broad market moved over the last month (21 sessions, summed log return).",
)
def mkt_ret_21(ctx: FeatureContext) -> pl.LazyFrame:
    """Sum of ln(1 + ew500_ret) over 21 sessions."""
    return ctx.market().select("date", _mkt_roll(_lm(), 21, "sum").alias("mkt_ret_21"))


@feature(
    "mkt_vol_21",
    group="market",
    kind="mkt",
    window=21,
    label="How jumpy the broad market has been over the last 21 sessions (std of daily log returns).",
)
def mkt_vol_21(ctx: FeatureContext) -> pl.LazyFrame:
    """Sample std of ln(1 + ew500_ret) over 21 sessions."""
    return ctx.market().select("date", _mkt_roll(_lm(), 21, "std").alias("mkt_vol_21"))


@feature(
    "mkt_co_5",
    group="market",
    kind="mkt",
    window=5,
    label="How much the market gapped overnight (close to next open) over the last 5 sessions.",
)
def mkt_co_5(ctx: FeatureContext) -> pl.LazyFrame:
    """Sum of ln(1 + ew500_co) over 5 sessions."""
    return ctx.market().select("date", _mkt_roll(log_ret("ew500_co"), 5, "sum").alias("mkt_co_5"))


@feature(
    "mkt_oc_5",
    group="market",
    kind="mkt",
    window=5,
    label="How much the market moved during trading hours (open to close) over the last 5 sessions.",
)
def mkt_oc_5(ctx: FeatureContext) -> pl.LazyFrame:
    """Sum of ln(1 + ew500_oc) over 5 sessions."""
    return ctx.market().select("date", _mkt_roll(log_ret("ew500_oc"), 5, "sum").alias("mkt_oc_5"))


# ---- breadth and dispersion --------------------------------------------------------------------


@feature(
    "breadth_200",
    group="market",
    kind="mkt",
    window=1,
    label="Share of top-200 stocks trading above their 200-day average (market breadth).",
)
def breadth_200(ctx: FeatureContext) -> pl.LazyFrame:
    """panel/market breadth_200 (fraction 0-1)."""
    return ctx.market().select("date", pl.col("breadth_200"))


@feature(
    "breadth_chg_10",
    group="market",
    kind="mkt",
    window=11,
    label="How much market breadth (share above the 200-day average) changed over 10 sessions.",
)
def breadth_chg_10(ctx: FeatureContext) -> pl.LazyFrame:
    """breadth_200 minus its value 10 sessions earlier."""
    return ctx.market().select(
        "date", (pl.col("breadth_200") - pl.col("breadth_200").shift(10)).alias("breadth_chg_10")
    )


@feature(
    "adv_dec_10",
    group="market",
    kind="mkt",
    window=10,
    label="Average of (advancers - decliners)/(advancers + decliners) among top-500 stocks over 10 sessions.",
)
def adv_dec_10(ctx: FeatureContext) -> pl.LazyFrame:
    """Mean over 10 sessions of the daily advance-decline ratio among u500 members, in [-1, 1]."""
    uni = ctx.universe().filter(pl.col("u500")).select("date", "sid")
    day = (
        uni.join(ctx.panel().select("date", "sid", "ret"), on=["date", "sid"], how="inner")
        .group_by("date")
        .agg(
            (pl.col("ret") > 0).sum().cast(pl.Int64).alias("_a"),
            (pl.col("ret") < 0).sum().cast(pl.Int64).alias("_d"),
        )
        .with_columns(
            pl.when(pl.col("_a") + pl.col("_d") > 0)
            .then((pl.col("_a") - pl.col("_d")) / (pl.col("_a") + pl.col("_d")))
            .alias("_x")
        )
    )
    return (
        _spine(ctx)
        .join(day.select("date", "_x"), on="date", how="left")
        .sort("date")
        .select("date", pl.col("_x").rolling_mean(10, min_samples=10).alias("adv_dec_10"))
    )


@feature(
    "xs_dispersion",
    group="market",
    kind="mkt",
    window=1,
    label="How spread out the stocks' daily returns were (std across the top 500).",
)
def xs_dispersion(ctx: FeatureContext) -> pl.LazyFrame:
    """panel/market cross-sectional return std (fraction)."""
    return ctx.market().select("date", "xs_dispersion")


@feature(
    "xs_disp_ratio",
    group="market",
    kind="mkt",
    window=60,
    label="Today's spread of stock returns relative to its 60-session average (dispersion regime).",
)
def xs_disp_ratio(ctx: FeatureContext) -> pl.LazyFrame:
    """xs_dispersion divided by the mean of its last 60 values (including today)."""
    x = pl.col("xs_dispersion")
    return ctx.market().select("date", (x / x.rolling_mean(60, min_samples=60)).alias("xs_disp_ratio"))


# ---- VIX and valuation -------------------------------------------------------------------------


def _vix(ctx: FeatureContext) -> pl.DataFrame:
    v = ctx.raw("india_vix", ["date", "close"]).sort("date").collect()
    v = v.with_columns(
        (pl.col("close") / pl.col("close").shift(5)).log().alias("chg5"),
    )
    return _series_pctile(v, "close", 252, "pct")


@feature(
    "vix",
    group="market",
    kind="mkt",
    window=1,
    label="India VIX: the market's expected 30-day volatility (fear gauge).",
)
def vix(ctx: FeatureContext) -> pl.LazyFrame:
    """India VIX close, in index points."""
    return _on_spine(ctx, _vix(ctx), ["close"]).select("date", pl.col("close").alias("vix"))


@feature(
    "vix_chg_5",
    group="market",
    kind="mkt",
    window=6,
    label="How much the fear gauge (India VIX) moved over the last 5 sessions (log change).",
)
def vix_chg_5(ctx: FeatureContext) -> pl.LazyFrame:
    """ln(vix / vix 5 observations earlier)."""
    return _on_spine(ctx, _vix(ctx), ["chg5"]).select("date", pl.col("chg5").alias("vix_chg_5"))


@feature(
    "vix_pctile_252",
    group="market",
    kind="mkt",
    window=252,
    label="Where India VIX stands within its own last year of values (0 = lowest, 1 = highest).",
)
def vix_pctile_252(ctx: FeatureContext) -> pl.LazyFrame:
    """Percentile of today's VIX among its last 252 values (0-1)."""
    return _on_spine(ctx, _vix(ctx), ["pct"]).select("date", pl.col("pct").alias("vix_pctile_252"))


@feature(
    "nifty_pe_pctile_756",
    group="market",
    kind="mkt",
    window=756,
    label="How expensive the Nifty 50 is versus its own last three years (percentile of its P/E).",
)
def nifty_pe_pctile_756(ctx: FeatureContext) -> pl.LazyFrame:
    """Percentile of the Nifty 50 P/E among its last 756 values (0-1); null before index data."""
    ix = (
        ctx.raw("index_daily", ["index_name", "date", "pe"])
        .filter(pl.col("index_name").str.to_uppercase() == "NIFTY 50")
        .select("date", "pe")
        .sort("date")
        .collect()
    )
    ix = _series_pctile(ix, "pe", 756, "pct")
    return _on_spine(ctx, ix, ["pct"]).select("date", pl.col("pct").alias("nifty_pe_pctile_756"))


# ---- macro (availability-timestamped) ----------------------------------------------------------

_MACRO = ("USDINR", "BRENT", "SPX", "NDX", "US10Y", "DXY", "GOLD")
_MACRO_LABEL = {
    "USDINR": "the rupee against the dollar",
    "BRENT": "Brent crude oil",
    "SPX": "the S&P 500",
    "NDX": "the Nasdaq 100",
    "US10Y": "the US 10-year Treasury yield (percentage points)",
    "DXY": "the US dollar index",
    "GOLD": "gold",
}


def _asof_values(ctx: FeatureContext, series: str) -> pl.DataFrame:
    """(date, v): the newest value of `series` with available_ts <= 20:00 IST on each session date."""
    sp = (
        _spine(ctx)
        .with_columns(
            (pl.col("date").cast(pl.Datetime("us")) + pl.duration(hours=20))
            .dt.replace_time_zone("Asia/Kolkata")
            .alias("_cut")
        )
        .sort("_cut")
        .collect()
    )
    m = (
        ctx.raw("macro_daily", ["series", "date", "close", "available_ts"])
        .filter(pl.col("series").str.to_uppercase() == series)
        .select(pl.col("available_ts"), pl.col("close").alias("v"))
        .drop_nulls("available_ts")
        .sort("available_ts")
        .collect()
    )
    out = sp.join_asof(m, left_on="_cut", right_on="available_ts", strategy="backward")
    return out.select("date", "v").sort("date")


def _make_macro(series: str) -> None:
    low = series.lower()
    diff = series == "US10Y"
    what = _MACRO_LABEL[series]

    def chg(k: int):
        def fn(ctx: FeatureContext) -> pl.LazyFrame:
            v = _asof_values(ctx, series)
            x = pl.col("v")
            c = (x - x.shift(k)) if diff else (x / x.shift(k)).log()
            return v.select("date", c.alias(f"{low}_chg_{k}")).lazy()

        return fn

    for k, word in ((1, "since the previous session"), (5, "over the last 5 sessions")):
        unit = "change" if diff else "log change"
        feature(
            f"{low}_chg_{k}",
            group="macro",
            kind="mkt",
            window=k + 1,
            label=f"How much {what} moved {word} ({unit}), using data published by 20:00 IST.",
        )(chg(k))


for _s in _MACRO:
    _make_macro(_s)


# ---- flows -------------------------------------------------------------------------------------


def _flow(ctx: FeatureContext, token: str, name: str) -> pl.LazyFrame:
    f = (
        ctx.raw("fii_dii", ["date", "category", "net_value"])
        .filter(pl.col("category").str.to_uppercase().str.contains(token))
        .group_by("date")
        .agg(pl.col("net_value").sum())
    )
    return (
        _spine(ctx)
        .join(f, on="date", how="left")
        .sort("date")
        .select("date", (pl.col("net_value").rolling_sum(5, min_samples=5) / 1e4).alias(name))
    )


@feature(
    "fii_net_5",
    group="flows",
    kind="mkt",
    window=5,
    label="Net buying by foreign institutions in cash equities over the last 5 sessions (Rs 10,000 cr).",
)
def fii_net_5(ctx: FeatureContext) -> pl.LazyFrame:
    """Sum of FII net cash-market flow over 5 sessions divided by 1e4 (net_value is in Rs crore)."""
    return _flow(ctx, "FII", "fii_net_5")


@feature(
    "dii_net_5",
    group="flows",
    kind="mkt",
    window=5,
    label="Net buying by domestic institutions in cash equities over the last 5 sessions (Rs 10,000 cr).",
)
def dii_net_5(ctx: FeatureContext) -> pl.LazyFrame:
    """Sum of DII net cash-market flow over 5 sessions divided by 1e4."""
    return _flow(ctx, "DII", "dii_net_5")


def _fii_fut(ctx: FeatureContext) -> pl.LazyFrame:
    p = (
        ctx.raw("participant_oi", ["date", "client_type", "fut_idx_long", "fut_idx_short"])
        .filter(pl.col("client_type").str.to_uppercase() == "FII")
        .select(
            "date",
            pl.when(pl.col("fut_idx_long") + pl.col("fut_idx_short") > 0)
            .then(
                (pl.col("fut_idx_long") - pl.col("fut_idx_short"))
                / (pl.col("fut_idx_long") + pl.col("fut_idx_short"))
            )
            .alias("net"),
        )
    )
    return _spine(ctx).join(p, on="date", how="left").sort("date")


@feature(
    "fii_idx_fut_net",
    group="flows",
    kind="mkt",
    window=1,
    label="Foreign institutions' stance in index futures, -1 (all short) to +1 (all long).",
)
def fii_idx_fut_net(ctx: FeatureContext) -> pl.LazyFrame:
    """FII (long - short)/(long + short) open interest in index futures."""
    return _fii_fut(ctx).select("date", pl.col("net").alias("fii_idx_fut_net"))


@feature(
    "fii_idx_fut_chg_5",
    group="flows",
    kind="mkt",
    window=6,
    label="How much foreign institutions' index-futures stance changed over the last 5 sessions.",
)
def fii_idx_fut_chg_5(ctx: FeatureContext) -> pl.LazyFrame:
    """fii_idx_fut_net minus its value 5 sessions earlier."""
    return _fii_fut(ctx).select("date", (pl.col("net") - pl.col("net").shift(5)).alias("fii_idx_fut_chg_5"))


# ---- industry context (cross-sectional) --------------------------------------------------------

_MIN_IND = 5


def _industry_base(ctx: FeatureContext) -> pl.LazyFrame:
    """u500 rows with mom_21, mom_126 (sums of adjusted log returns), a 50-row-mean flag and industry."""
    la = log_ret("ret")
    p = (
        ctx.panel()
        .with_columns(la.alias("_la"))
        .with_columns(
            roll_sum("_la", 21).alias("_m21"),
            roll_sum("_la", 126).alias("_m126"),
            (pl.col("adj_close") > roll_mean("adj_close", 50)).cast(pl.Float64).alias("_ab50"),
        )
        .select("date", "sid", "industry", "_m21", "_m126", "_ab50")
    )
    u = ctx.universe().filter(pl.col("u500")).select("date", "sid")
    return u.join(p, on=["date", "sid"], how="inner")


def _ind_mean_excl(col: str) -> list[pl.Expr]:
    """Mean of `col` over the industry's other members (null if the industry has < 5 members)."""
    grp = ["date", "industry"]
    s = pl.col(col).sum().over(grp)
    n = pl.col(col).count().over(grp)
    own = pl.col(col).fill_null(0.0)
    own_n = pl.col(col).is_not_null().cast(pl.Int64)
    members = pl.len().over(grp)
    ok = (members >= _MIN_IND) & (pl.col("industry") != "UNKNOWN") & ((n - own_n) > 0)
    return [pl.when(ok).then((s - own) / (n - own_n)).alias(f"{col}_ex")]


@feature(
    "ind_mom_21",
    group="industry",
    kind="xs",
    window=21,
    label="How the stock's industry peers (excluding itself) moved over the last month.",
)
def ind_mom_21(ctx: FeatureContext) -> pl.LazyFrame:
    """Mean mom_21 of the industry's u500 members other than the stock (null if < 5 members or UNKNOWN)."""
    return (
        _industry_base(ctx).select("date", "sid", *_ind_mean_excl("_m21")).rename({"_m21_ex": "ind_mom_21"})
    )


@feature(
    "ind_mom_126",
    group="industry",
    kind="xs",
    window=126,
    label="How the stock's industry peers (excluding itself) moved over six months.",
)
def ind_mom_126(ctx: FeatureContext) -> pl.LazyFrame:
    """Mean mom_126 of the industry's u500 members other than the stock."""
    return (
        _industry_base(ctx)
        .select("date", "sid", *_ind_mean_excl("_m126"))
        .rename({"_m126_ex": "ind_mom_126"})
    )


@feature(
    "rel_ind_mom_21",
    group="industry",
    kind="xs",
    window=21,
    label="How much the stock beat or lagged its industry peers over the last month.",
)
def rel_ind_mom_21(ctx: FeatureContext) -> pl.LazyFrame:
    """mom_21 minus ind_mom_21."""
    return (
        _industry_base(ctx)
        .with_columns(*_ind_mean_excl("_m21"))
        .select("date", "sid", (pl.col("_m21") - pl.col("_m21_ex")).alias("rel_ind_mom_21"))
    )


@feature(
    "ind_breadth_50",
    group="industry",
    kind="xs",
    window=50,
    label="Share of the stock's industry (top-500 members) trading above their 50-day average.",
)
def ind_breadth_50(ctx: FeatureContext) -> pl.LazyFrame:
    """Share of the industry's u500 members with adj_close above their 50-row mean (null if < 5 members)."""
    grp = ["date", "industry"]
    share = pl.col("_ab50").sum().over(grp) / pl.col("_ab50").count().over(grp)
    ok = (
        (pl.len().over(grp) >= _MIN_IND)
        & (pl.col("industry") != "UNKNOWN")
        & (pl.col("_ab50").count().over(grp) > 0)
    )
    return _industry_base(ctx).select("date", "sid", pl.when(ok).then(share).alias("ind_breadth_50"))
