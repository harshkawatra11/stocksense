"""Derivatives features from the F&O bhavcopy (PLAN T4.3i): futures basis and open interest, option
put-call ratio, implied volatility, skew and max pain. Null before 2024-07-08 and for non-F&O names.

The expensive part (implied volatility) is solved only for the at most four options per underlying-day
that a feature needs, and computed once per FeatureContext (the eleven features share it).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import polars as pl
from scipy.optimize import brentq

from stocksense.features.base import FeatureContext, feature, log_ret, roll_mean, roll_std, roll_sum, span_ok

R_FREE = 0.065
IV_BOUNDS = (0.01, 5.0)
MIN_PRICE = 0.5
MIN_DAYS = 3  # `near` = nearest expiry with at least this many calendar days left


# ---- Black-Scholes -----------------------------------------------------------------------------


def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bs_price(s: float, k: float, t: float, r: float, sigma: float, call: bool) -> float:
    """Black-Scholes price of a European option (t in years, sigma annualised, no dividends)."""
    if t <= 0 or sigma <= 0:
        return max(s - k, 0.0) if call else max(k - s, 0.0)
    v = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / v
    d2 = d1 - v
    disc = math.exp(-r * t)
    if call:
        return s * _ncdf(d1) - k * disc * _ncdf(d2)
    return k * disc * _ncdf(-d2) - s * _ncdf(-d1)


def implied_vol(price: float, s: float, k: float, t: float, call: bool, r: float = R_FREE) -> float | None:
    """Implied volatility by brentq on [0.01, 5.0]; None if price < 0.5, intrinsic > price, or no root."""
    if price is None or not (price >= MIN_PRICE) or t <= 0 or s <= 0 or k <= 0:
        return None
    intrinsic = max(s - k, 0.0) if call else max(k - s, 0.0)
    if intrinsic > price:
        return None

    def f(sig: float) -> float:
        return bs_price(s, k, t, r, sig, call) - price

    lo, hi = IV_BOUNDS
    if f(lo) * f(hi) > 0:
        return None
    try:
        return float(brentq(f, lo, hi, xtol=1e-10))
    except (ValueError, RuntimeError):
        return None


def max_pain(strikes: np.ndarray, call_oi: np.ndarray, put_oi: np.ndarray) -> float:
    """The strike at which the total expiry payoff to option holders is smallest (lowest on ties)."""
    x = strikes[:, None]
    pay = (call_oi[None, :] * np.maximum(x - strikes[None, :], 0.0)).sum(axis=1)
    pay += (put_oi[None, :] * np.maximum(strikes[None, :] - x, 0.0)).sum(axis=1)
    return float(strikes[int(np.argmin(pay))])


def oi_price_quad(price_chg: pl.Expr, oi_chg: pl.Expr) -> pl.Expr:
    """+1 price up & OI up, -1 price down & OI up, +0.5 price up & OI down, -0.5 both down; else null."""
    return (
        pl.when((price_chg > 0) & (oi_chg > 0))
        .then(1.0)
        .when((price_chg < 0) & (oi_chg > 0))
        .then(-1.0)
        .when((price_chg > 0) & (oi_chg < 0))
        .then(0.5)
        .when((price_chg < 0) & (oi_chg < 0))
        .then(-0.5)
    )


# ---- chain processing --------------------------------------------------------------------------


def _chain_stats(grp: pl.DataFrame) -> tuple[float | None, float | None, float | None]:
    """(atm_iv, skew_put_call, max_pain_dist) for one underlying-day's near-expiry option chain."""
    spot = grp["spot"][0]
    t = grp["dte"][0] / 365.0
    k = grp["strike_price"].to_numpy()
    price = grp["settle_price"].to_numpy()
    oi = grp["oi"].to_numpy().astype(float)
    is_call = grp["option_type"].str.to_uppercase().str.starts_with("C").to_numpy()
    if len(k) == 0 or not spot or spot <= 0:
        return None, None, None

    def pick(mask: np.ndarray, target: float) -> int | None:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return None
        return int(idx[np.argmin(np.abs(k[idx] - target))])

    def iv_of(i: int | None, call: bool) -> float | None:
        return None if i is None else implied_vol(float(price[i]), spot, float(k[i]), t, call)

    atm_k = k[np.argmin(np.abs(k - spot))]
    at_strike = k == atm_k
    ivs = [
        v
        for v in (
            iv_of(pick(at_strike & is_call, atm_k), True),
            iv_of(pick(at_strike & ~is_call, atm_k), False),
        )
        if v is not None
    ]
    atm = sum(ivs) / len(ivs) if ivs else None
    put_iv = iv_of(pick(~is_call, 0.95 * spot), False)
    call_iv = iv_of(pick(is_call, 1.05 * spot), True)
    skew = None if put_iv is None or call_iv is None else put_iv - call_iv
    strikes = np.unique(k)
    c_oi = np.array([oi[(k == x) & is_call].sum() for x in strikes])
    p_oi = np.array([oi[(k == x) & ~is_call].sum() for x in strikes])
    mp = max_pain(strikes, c_oi, p_oi)
    return atm, skew, (mp - spot) / spot


_SCHEMA = {
    "symbol": pl.Utf8,
    "date": pl.Date,
    "atm_iv": pl.Float64,
    "skew_put_call": pl.Float64,
    "max_pain_dist": pl.Float64,
}


def _derivs(ctx: FeatureContext) -> pl.DataFrame:
    """All derivatives features per (date, sid) panel row; memoised on the context."""
    cache = ctx.__dict__.setdefault("_derivs_cache", {})
    if "df" in cache:
        return cache["df"]
    panel = ctx.panel().select("date", "sid", "symbol", "sx", "close", "ret").collect()
    fo = ctx.raw(
        "fo_bhavcopy",
        [
            "symbol",
            "instrument_type",
            "expiry_date",
            "strike_price",
            "option_type",
            "date",
            "settle_price",
            "oi",
        ],
    ).filter(pl.col("instrument_type").is_in(["STF", "STO"]))
    fo = fo.with_columns((pl.col("expiry_date") - pl.col("date")).dt.total_days().alias("dte")).collect()
    fut = fo.filter(pl.col("instrument_type") == "STF")
    opt = fo.filter(pl.col("instrument_type") == "STO")
    keys = ["symbol", "date"]

    near_f = (
        fut.filter(pl.col("dte") >= MIN_DAYS)
        .sort(*keys, "expiry_date")
        .group_by(keys)
        .agg(pl.col("settle_price").first().alias("f_settle"), pl.col("dte").first().alias("f_dte"))
    )
    oi_sum = fut.group_by(keys).agg(pl.col("oi").sum().cast(pl.Float64).alias("f_oi"))
    chain = (
        opt.filter(pl.col("dte") >= MIN_DAYS)
        .with_columns(pl.col("dte").min().over(keys).alias("_min"))
        .filter(pl.col("dte") == pl.col("_min"))
        .join(panel.select(*keys, pl.col("close").alias("spot")), on=keys, how="inner")
    )
    is_put = pl.col("option_type").str.to_uppercase().str.starts_with("P")
    pcr = (
        chain.group_by(keys)
        .agg(
            pl.col("oi").filter(is_put).sum().cast(pl.Float64).alias("_p"),
            pl.col("oi").filter(~is_put).sum().cast(pl.Float64).alias("_c"),
        )
        .select(*keys, pl.when(pl.col("_c") > 0).then(pl.col("_p") / pl.col("_c")).alias("pcr_oi"))
    )
    rows = []
    for (sym, d), grp in chain.sort(*keys, "strike_price").group_by(keys, maintain_order=True):
        rows.append((sym, d, *_chain_stats(grp)))
    stats = pl.DataFrame(rows, schema=_SCHEMA, orient="row")

    df = (
        panel.join(near_f, on=keys, how="left")
        .join(oi_sum, on=keys, how="left")
        .join(pcr, on=keys, how="left")
        .join(stats, on=keys, how="left")
        .sort("sid", "date")
        .with_columns(log_ret("ret").alias("_la"))
    )
    ok5 = span_ok(6)
    df = df.with_columns(
        pl.when(ok5).then(pl.col("f_oi").shift(5).over("sid")).alias("_oi5"),
        pl.when(ok5).then(pl.col("atm_iv").shift(5).over("sid")).alias("_iv5"),
        roll_sum("_la", 5).alias("_p5"),
        roll_std("_la", 21).alias("_vol21"),
        roll_mean("pcr_oi", 20).shift(1).over("sid").alias("_pm"),
        roll_std("pcr_oi", 20).shift(1).over("sid").alias("_ps"),
        roll_mean("f_oi", 20).alias("_oim"),
    )
    out = df.select(
        "date",
        "sid",
        ((pl.col("f_settle") - pl.col("close")) / pl.col("close") * 365.0 / pl.col("f_dte")).alias(
            "fut_basis"
        ),
        ((pl.col("f_oi") - pl.col("_oi5")) / pl.col("_oim")).alias("fut_oi_chg_5"),
        oi_price_quad(pl.col("_p5"), pl.col("f_oi") - pl.col("_oi5")).alias("oi_price_quad"),
        pl.col("pcr_oi"),
        pl.when(pl.col("_ps") > 0)
        .then((pl.col("pcr_oi") - pl.col("_pm")) / pl.col("_ps"))
        .alias("pcr_oi_z_20"),
        pl.col("atm_iv"),
        (pl.col("atm_iv") - pl.col("_vol21") * math.sqrt(252.0)).alias("iv_minus_rv"),
        (pl.col("atm_iv") - pl.col("_iv5")).alias("iv_chg_5"),
        pl.col("skew_put_call"),
        pl.col("max_pain_dist"),
        pl.col("f_dte").cast(pl.Float64).alias("days_to_expiry"),
    )
    cache["df"] = out
    return out


# ---- registration ------------------------------------------------------------------------------

_SPECS: tuple[tuple[str, int, str], ...] = (
    (
        "fut_basis",
        1,
        "Annualised premium of the nearest stock future over the cash price (cost of carry / bullishness).",
    ),
    (
        "fut_oi_chg_5",
        25,
        "Change in total futures open interest over 5 sessions, relative to its 20-session average.",
    ),
    (
        "oi_price_quad",
        6,
        "Price and open-interest direction over 5 sessions: +1 long build-up, -1 short build-up, "
        "+0.5 short covering, -0.5 long unwinding.",
    ),
    (
        "pcr_oi",
        1,
        "Put open interest divided by call open interest for the nearest expiry (hedging / bearishness).",
    ),
    (
        "pcr_oi_z_20",
        21,
        "Today's put-call open-interest ratio versus its previous 20-session average (z-score).",
    ),
    (
        "atm_iv",
        1,
        "Implied volatility of the at-the-money options (market's expected jumpiness, annualised).",
    ),
    (
        "iv_minus_rv",
        22,
        "Implied volatility minus the stock's realised 21-session volatility (annualised): fear premium.",
    ),
    ("iv_chg_5", 6, "Change in at-the-money implied volatility over 5 sessions."),
    (
        "skew_put_call",
        1,
        "Implied volatility of a put 5% below the price minus a call 5% above it (crash fear).",
    ),
    (
        "max_pain_dist",
        1,
        "Distance of the options 'max pain' strike from the current price, as a fraction of price.",
    ),
    ("days_to_expiry", 1, "Calendar days to the nearest futures expiry (at least 3 days away)."),
)


def _make(name: str) -> Callable[[FeatureContext], pl.LazyFrame]:
    def fn(ctx: FeatureContext) -> pl.LazyFrame:
        return _derivs(ctx).select("date", "sid", name).lazy()

    fn.__doc__ = f"{name} from the shared derivatives frame (null for non-F&O names and before 2024-07-08)."
    return fn


for _name, _w, _label in _SPECS:
    feature(_name, group="derivs", kind="xs", window=_w, label=_label)(_make(_name))
