"""Mean reversion as an Ornstein-Uhlenbeck process (G2-E5): s-score, half-life, pair z and residual reversal.

Residual return (Avellaneda & Lee 2010, with the industry mean standing in for their sector ETF):
    e_t = la_t - beta_{t-1} * lm_t - mean over the industry on that date of (la - beta_{t-1} * lm)
with `la`, `lm` the stock's and the market's adjusted log returns and `beta` the 63-row rolling regression
slope, shifted one row so that day t's beta uses returns up to t-1. The industry term is dropped (set to 0)
for the UNKNOWN industry and for industries with fewer than 5 members that day.

OU fit. X is the cumulative residual over the last 60 rows (X_0 = e_0, ...). AR(1) by OLS:
X_{k+1} = a + b X_k + eps. Then kappa = -ln b * 252, m = a / (1 - b), sigma_eq = sqrt(var(eps) / (1 - b^2)),
half-life ln 2 / -ln b sessions. `ou_sscore = (X_t - m) / sigma_eq` when 0 < b < 1 and the half-life is at
most 30 sessions, else null; `ou_halflife` is the half-life whenever 0 < b < 1. A 60-point AR(1) fit is
biased (b is under-estimated by about (1 + 3b) / n), so the half-life of a single window is short on
average; the bias is shared by all names and ranks are unaffected.

Pairs. Each January, on the last 252 sessions up to the previous 31 December, within each industry the 50
most liquid names (median turnover) with complete history are tested pairwise (Engle-Granger: OLS of one
log price on the other, ADF without lags on the residuals, MacKinnon p-value for two variables). Pairs with
p < 0.05 and a spread half-life of 5 to 30 sessions are kept and each stock keeps its most stationary pair
(lowest p). `pair_z` is the stock's z-score of the pair's spread (fixed alpha, beta) against its own last 60
rows: positive when the stock is rich against its partner (sign +1 for the dependent leg, -1 for the other).
`resid_rev_5 = -sum of the last 5 residuals` (Blitz, Huij, Lansdorp & Verbeek 2013).

Everything is causal: a row on day t uses rows <= t and pair parameters estimated on data before its year.
The module registers its features when imported but is NOT in `features.registry.FEATURE_MODULES` (E9 adds
it once, for feature version 2), so the nightly build and the live system are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from stocksense.features.base import (
    FeatureContext,
    FeatureSpec,
    feature,
    log_ret,
    roll_cov,
    roll_sum,
    roll_var,
)
from stocksense.features.registry import REGISTRY

LN2 = math.log(2.0)
OU_WINDOW = 60
BETA_WINDOW = 63
MAX_HALFLIFE = 30.0  # sessions: the s-score is null above this
MIN_INDUSTRY_RESID = 5  # members needed for the industry term of the residual
PAIR_WINDOW = 252
PAIR_TOP = 50
PAIR_ADF_P = 0.05
PAIR_HL_MIN = 5.0
PAIR_HL_MAX = 30.0
PAIR_Z_WINDOW = 60
MIN_INDUSTRY_PAIR_MEMBERS = 2
OUTPUTS = ("ou_sscore", "ou_halflife", "resid_rev_5", "pair_z")


# ---- the AR(1) / OU fit --------------------------------------------------------------------------------


def halflife_of(b: float) -> float | None:
    """ln 2 / -ln b sessions for 0 < b < 1, else None."""
    return LN2 / -math.log(b) if 0.0 < b < 1.0 else None


@dataclass(frozen=True)
class Ar1Fit:
    a: float
    b: float
    var_eps: float
    m: float
    sigma_eq: float
    kappa: float  # per year
    halflife: float  # sessions


def fit_ar1(x: np.ndarray) -> Ar1Fit | None:
    """OLS X_{k+1} = a + b X_k + eps on the path `x`; None unless 0 < b < 1 (a mean-reverting fit)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 5 or not np.isfinite(x).all():
        return None
    x0, y1 = x[:-1], x[1:]
    mx, my = x0.mean(), y1.mean()
    sxx = float(((x0 - mx) ** 2).sum())
    if sxx <= 0:
        return None
    b = float(((x0 - mx) * (y1 - my)).sum() / sxx)
    if not 0.0 < b < 1.0:
        return None
    a = float(my - b * mx)
    res = y1 - a - b * x0
    var = float((res**2).sum() / max(len(x0) - 2, 1))
    if var <= 0:
        return None
    return Ar1Fit(
        a, b, var, a / (1.0 - b), math.sqrt(var / (1.0 - b * b)), -math.log(b) * 252.0, LN2 / -math.log(b)
    )


def sscore(x_last: float, a: float, b: float, var_eps: float) -> float | None:
    """(X_t - m) / sigma_eq from AR(1) coefficients; None when 0 < b < 1 fails or the half-life exceeds 30."""
    hl = halflife_of(b)
    if hl is None or hl > MAX_HALFLIFE or var_eps <= 0:
        return None
    m = a / (1.0 - b)
    sig = math.sqrt(var_eps / (1.0 - b * b))
    return (x_last - m) / sig


def rolling_ou(
    e: np.ndarray, window: int = OU_WINDOW, sx: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(ou_sscore, ou_halflife) at every row of one stock's residual series `e` (NaN before `window` rows,
    where the window holds a NaN, or where `sx` shows the window straddles a long gap)."""
    e = np.asarray(e, dtype=float)
    n = len(e)
    score = np.full(n, np.nan)
    hl = np.full(n, np.nan)
    if n < window:
        return score, hl
    win = sliding_window_view(e, window)
    ok = np.isfinite(win).all(axis=1)
    if sx is not None:
        sxa = np.asarray(sx, dtype=float)
        span = sxa[window - 1 :] - sxa[: n - window + 1] + 1
        ok &= span <= math.ceil(1.25 * window) + 2
    idx = np.flatnonzero(ok)
    if not len(idx):
        return score, hl
    cs = np.cumsum(win[idx], axis=1)
    x0, y1 = cs[:, :-1], cs[:, 1:]
    mx, my = x0.mean(axis=1, keepdims=True), y1.mean(axis=1, keepdims=True)
    sxx = ((x0 - mx) ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = ((x0 - mx) * (y1 - my)).sum(axis=1) / sxx
        a = my[:, 0] - b * mx[:, 0]
        res = y1 - a[:, None] - b[:, None] * x0
        var = (res**2).sum(axis=1) / max(window - 3, 1)
        valid = (b > 0) & (b < 1) & (sxx > 0) & (var > 0)
        h = LN2 / -np.log(b)
        sig = np.sqrt(var / (1.0 - b * b))
        s = (cs[:, -1] - a / (1.0 - b)) / sig
    pos = idx + window - 1
    hl[pos[valid]] = h[valid]
    good = valid & (h <= MAX_HALFLIFE)
    score[pos[good]] = s[good]
    return score, hl


def resid_rev_5_expr() -> pl.Expr:
    """-(sum of the last 5 residuals) per sid (the frame needs sid, sx and e)."""
    return (-roll_sum("e", 5)).alias("resid_rev_5")


# ---- Engle-Granger pairs -------------------------------------------------------------------------------


@dataclass(frozen=True)
class PairFit:
    i: int  # column of the dependent log price
    j: int
    alpha: float
    beta: float
    tstat: float
    pval: float
    halflife: float


def _mackinnon_p(t: np.ndarray) -> np.ndarray:
    from statsmodels.tsa.adfvalues import mackinnonp

    out = np.ones(len(t))
    for k, v in enumerate(t):
        if np.isfinite(v) and v < -1.5:
            out[k] = float(mackinnonp(v, regression="c", N=2))
    return out


def eg_pairs(
    logp: np.ndarray,
    *,
    adf_p: float | None = None,
    hl_min: float | None = None,
    hl_max: float | None = None,
) -> list[PairFit]:
    """Engle-Granger tests of every column pair of `logp` [W, m] (each unordered pair once, the lower column
    index regressed on the higher). Keeps p < adf_p and a spread half-life in [hl_min, hl_max]; most
    stationary first."""
    p_max = PAIR_ADF_P if adf_p is None else adf_p
    lo = PAIR_HL_MIN if hl_min is None else hl_min
    hi = PAIR_HL_MAX if hl_max is None else hl_max
    w, m = logp.shape
    out: list[PairFit] = []
    for i in range(m - 1):
        y = logp[:, i : i + 1]
        x = logp[:, i + 1 :]
        xm = x - x.mean(axis=0)
        ym = y - y.mean()
        sxx = (xm**2).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            beta = (xm * ym).sum(axis=0) / sxx
            alpha = y.mean() - beta * x.mean(axis=0)
            u = y - alpha - beta * x
            u0, du = u[:-1], u[1:] - u[:-1]
            s00 = (u0**2).sum(axis=0)
            gam = (u0 * du).sum(axis=0) / s00
            rss = ((du - gam * u0) ** 2).sum(axis=0)
            se = np.sqrt(rss / max(w - 2, 1) / s00)
            t = gam / se
            b = 1.0 + gam
            hl = LN2 / -np.log(np.where((b > 0) & (b < 1), b, np.nan))
        pv = _mackinnon_p(t)
        for k in np.flatnonzero((pv < p_max) & np.isfinite(hl) & (hl >= lo) & (hl <= hi)):
            out.append(
                PairFit(
                    i,
                    i + 1 + int(k),
                    float(alpha[k]),
                    float(beta[k]),
                    float(t[k]),
                    float(pv[k]),
                    float(hl[k]),
                )
            )
    out.sort(key=lambda p: (p.pval, p.i, p.j))
    return out


_PAIR_SCHEMA = {
    "sid": pl.Int64,
    "sign": pl.Int64,
    "dep_sid": pl.Int64,
    "ind_sid": pl.Int64,
    "alpha": pl.Float64,
    "beta": pl.Float64,
    "pval": pl.Float64,
}


def select_pairs(
    panel: pl.DataFrame,
    cutoff: date,
    *,
    window: int | None = None,
    top: int | None = None,
    min_members: int | None = None,
) -> pl.DataFrame:
    """The January pair table from rows dated <= `cutoff` of a panel with sid, date, sx, industry, adj_close,
    turnover_inr: one row per stock that has a pair (its lowest-p pair): sid, sign (+1 when the stock is the
    dependent leg), dep_sid, ind_sid, alpha, beta, pval. Only the last `window` sessions up to the cutoff are
    read and only names with a complete history in it."""
    w = PAIR_WINDOW if window is None else window
    n_top = PAIR_TOP if top is None else top
    need = MIN_INDUSTRY_PAIR_MEMBERS if min_members is None else min_members
    empty = pl.DataFrame(schema=_PAIR_SCHEMA)
    sub = panel.filter((pl.col("date") <= cutoff) & (pl.col("adj_close") > 0))
    if sub.is_empty():
        return empty
    smax = sub["sx"].max()
    sub = sub.filter(pl.col("sx") > smax - w)
    stats = sub.group_by("sid").agg(
        pl.len().alias("n"),
        pl.col("industry").last().alias("industry"),
        pl.col("turnover_inr").median().alias("liq"),
    )
    stats = stats.filter(
        (pl.col("n") == w) & (pl.col("industry") != "UNKNOWN") & pl.col("industry").is_not_null()
    )
    best: dict[int, tuple[float, int, int, int, float, float]] = {}
    for _, g in stats.group_by("industry", maintain_order=True):
        top_ids = g.sort(["liq", "sid"], descending=[True, False]).head(n_top)["sid"].to_list()
        if len(top_ids) < max(need, 2):
            continue
        sids = sorted(int(s) for s in top_ids)
        d = sub.filter(pl.col("sid").is_in(sids)).sort("sid", "date")
        lp = np.log(d["adj_close"].to_numpy()).reshape(len(sids), w).T
        for p in eg_pairs(lp):
            dep, ind = sids[p.i], sids[p.j]
            for sid, sign in ((dep, 1), (ind, -1)):
                if sid not in best or p.pval < best[sid][0]:
                    best[sid] = (p.pval, sign, dep, ind, p.alpha, p.beta)
    if not best:
        return empty
    rows = sorted((sid, v[1], v[2], v[3], v[4], v[5], v[0]) for sid, v in best.items())
    return pl.DataFrame(rows, schema=_PAIR_SCHEMA, orient="row")


def _pair_z_year(panel: pl.DataFrame, table: pl.DataFrame, year: int) -> pl.DataFrame:
    """(date, sid, pair_z) for the year's rows of the stocks in `table` (spreads at fixed alpha, beta)."""
    rows = panel.filter(pl.col("date").dt.year() == year)
    if rows.is_empty() or table.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "sid": pl.Int64, "pair_z": pl.Float64})
    sx_lo = int(rows["sx"].min()) - (PAIR_Z_WINDOW - 1)
    sx_hi = int(rows["sx"].max())
    need = sorted({int(s) for s in table["dep_sid"].to_list()} | {int(s) for s in table["ind_sid"].to_list()})
    cols = {s: k for k, s in enumerate(need)}
    px = panel.filter(
        pl.col("sid").is_in(need)
        & (pl.col("sx") >= sx_lo)
        & (pl.col("sx") <= sx_hi)
        & (pl.col("adj_close") > 0)
    )
    lp = np.full((sx_hi - sx_lo + 1, len(need)), np.nan)
    ri = (px["sx"].to_numpy() - sx_lo).astype(int)
    ci = np.array([cols[int(s)] for s in px["sid"].to_list()], dtype=int)
    lp[ri, ci] = np.log(px["adj_close"].to_numpy())
    dep = np.array([cols[int(s)] for s in table["dep_sid"].to_list()])
    ind = np.array([cols[int(s)] for s in table["ind_sid"].to_list()])
    spread = lp[:, dep] - table["alpha"].to_numpy()[None, :] - table["beta"].to_numpy()[None, :] * lp[:, ind]
    fin = np.isfinite(spread)
    v = np.where(fin, spread, 0.0)
    c1 = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)])
    c2 = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v * v, axis=0)])
    cn = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(fin, axis=0)])
    k = PAIR_Z_WINDOW
    s1 = c1[k:] - c1[:-k]
    s2 = c2[k:] - c2[:-k]
    cnt = cn[k:] - cn[:-k]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / k
        var = (s2 - s1 * s1 / k) / (k - 1)
        z = np.full(spread.shape, np.nan)
        z[k - 1 :] = (spread[k - 1 :] - mean) / np.sqrt(var)
        z[k - 1 :][(cnt < k) | (var <= 0)] = np.nan
    sign = table["sign"].to_numpy()
    sids = table["sid"].to_numpy()
    pos = {int(s): j for j, s in enumerate(sids)}
    r = rows.filter(pl.col("sid").is_in(sids.tolist()))
    rr = (r["sx"].to_numpy() - sx_lo).astype(int)
    cc = np.array([pos[int(s)] for s in r["sid"].to_list()], dtype=int)
    vals = z[rr, cc] * sign[cc]
    return pl.DataFrame({"date": r["date"], "sid": r["sid"].cast(pl.Int64), "pair_z": vals}).with_columns(
        pl.col("pair_z").fill_nan(None)
    )


def pair_z_series(panel: pl.DataFrame) -> pl.DataFrame:
    """(date, sid, pair_z) for a panel with sid, date, sx, industry, adj_close, turnover_inr. Rows of year y
    use the pair table selected on data <= 31 December of y-1."""
    parts = []
    for y in sorted({d.year for d in panel["date"].unique().to_list()}):
        table = select_pairs(panel, date(y - 1, 12, 31))
        if not table.is_empty():
            parts.append(_pair_z_year(panel, table, y))
    if not parts:
        return pl.DataFrame(schema={"date": pl.Date, "sid": pl.Int64, "pair_z": pl.Float64})
    return pl.concat(parts)


# ---- features ------------------------------------------------------------------------------------------


def _resid_frame(ctx: FeatureContext) -> pl.DataFrame:
    """date, sid, sx, e (the market- and industry-neutral residual log return), memoised on the context."""
    cache = ctx.__dict__.setdefault("_ou_resid", {})
    if "df" in cache:
        return cache["df"]
    mk = ctx.market().select("date", "ew500_ret").collect()
    p = (
        ctx.panel()
        .select("date", "sid", "sx", "industry", "ret")
        .collect()
        .join(mk, on="date", how="left")
        .sort("sid", "date")
        .with_columns(log_ret("ret").alias("_la"), log_ret("ew500_ret").alias("_lm"))
        .with_columns((roll_cov("_la", "_lm", BETA_WINDOW) / roll_var("_lm", BETA_WINDOW)).alias("_beta"))
        .with_columns(pl.col("_beta").shift(1).over("sid").alias("_bp"))
        .with_columns((pl.col("_la") - pl.col("_bp") * pl.col("_lm")).alias("_r"))
    )
    grp = ["date", "industry"]
    ok = (pl.col("industry") != "UNKNOWN") & (pl.col("_r").count().over(grp) >= MIN_INDUSTRY_RESID)
    ind = pl.when(ok).then(pl.col("_r").mean().over(grp)).otherwise(0.0)
    out = p.select("date", "sid", "sx", (pl.col("_r") - ind).alias("e")).sort("sid", "date")
    cache["df"] = out
    return out


def _ou_resid_feature(ctx: FeatureContext) -> pl.LazyFrame:
    f = _resid_frame(ctx)
    sid = f["sid"].to_numpy()
    e = f["e"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    sx = f["sx"].to_numpy()
    score = np.full(len(f), np.nan)
    hl = np.full(len(f), np.nan)
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(sid)) + 1, [len(sid)]])
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        score[a:b], hl[a:b] = rolling_ou(e[a:b], OU_WINDOW, sx[a:b])
    out = f.with_columns(
        pl.Series("ou_sscore", score).fill_nan(None),
        pl.Series("ou_halflife", hl).fill_nan(None),
        resid_rev_5_expr(),
    )
    return out.select("date", "sid", "ou_sscore", "ou_halflife", "resid_rev_5").lazy()


def _pair_z_feature(ctx: FeatureContext) -> pl.LazyFrame:
    panel = (
        ctx.panel()
        .select("date", "sid", "sx", "industry", "adj_close", "turnover_inr")
        .collect()
        .with_columns(pl.col("sid").cast(pl.Int64))
    )
    z = pair_z_series(panel)
    keys = panel.select("date", "sid")
    return keys.join(z, on=["date", "sid"], how="left").with_columns(pl.col("sid").cast(pl.Int32)).lazy()


feature(
    "ou_resid",
    group="reversal",
    kind="xs",
    window=123,
    version=2,
    label="How far the stock's market- and industry-neutral drift is from its own mean (OU s-score).",
    outputs=("ou_sscore", "ou_halflife", "resid_rev_5"),
)(_ou_resid_feature)
feature(
    "pair_z",
    group="reversal",
    kind="xs",
    window=312,
    version=2,
    label="How stretched the stock is against its best cointegrated industry peer (z-score of the spread).",
)(_pair_z_feature)

SPECS: tuple[FeatureSpec, ...] = (REGISTRY["ou_resid"], REGISTRY["pair_z"])
