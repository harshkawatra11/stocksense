"""Market regime features (G2-E4): a filtered 3-state Gaussian HMM and Bayesian online change-point detection.

HMM. Daily observations [ew500_ret, 5-day ew500 log return, 21-day market vol, breadth_200, xs_dispersion,
5-day VIX change], each standardised with EXPANDING statistics (mean and std of the values up to and
including that day). A diagonal-covariance Gaussian HMM with three states is fitted by Baum-Welch in numpy
(hmmlearn is not installed and would add a compiled dependency for a 200-line algorithm) on all rows dated
on or before the previous 31 December, once per calendar year. The probabilities used as features are the
FORWARD (filtering) recursion only: the state distribution on day t given observations up to t. A smoother
(the `predict_proba` of the usual libraries) would condition on the future and is not offered by
`filter_probs`; `smooth_probs` exists only so that a test can show the two differ. States are ordered by
their mean of the first column (the market return): state 0 = bear, 1 = neutral, 2 = bull. Features
`regime_p_bear`, `regime_p_bull` (kind mkt).

BOCPD (Adams & MacKay 2007). Run-length posterior with a constant hazard 1/250 on ew500_ret under a
Normal-Gamma prior (mu0 = 0, kappa0 = 1, alpha0 = 1, beta0 = alpha0 * scale^2 with scale = 1 %), Student-t
predictives, run lengths capped at 500. Features `cp_prob` (probability that the run length is below 5
after observing day t) and `run_length_mean`. The recursion is online, so it is causal by construction.

The module registers its features when imported but is NOT in `features.registry.FEATURE_MODULES` (E9 adds
it once, for feature version 2), so the nightly build and the live system are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
from scipy.special import gammaln

from stocksense.features.base import FeatureContext, FeatureSpec, feature, log_ret
from stocksense.features.registry import REGISTRY

N_STATES = 3
MIN_FIT_ROWS = 750  # a January fit needs this many rows (about three years)
STD_MIN_ROWS = 60  # expanding standardisation is undefined before this many values
VAR_FLOOR = 0.02
HAZARD = 1.0 / 250.0
BOCPD_SCALE = 0.01  # prior standard deviation of a daily ew500 return
BOCPD_MAX_RUN = 500
CP_SHORT = 5
BEAR_P = 0.6  # the S1 regime overlay halves the gross above this bear probability
OUTPUTS = ("regime_p_bear", "regime_p_bull", "cp_prob", "run_length_mean")


# ---- the Gaussian HMM ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class HmmParams:
    pi: np.ndarray  # [K] initial state distribution
    trans: np.ndarray  # [K, K] row-stochastic
    mean: np.ndarray  # [K, D]
    var: np.ndarray  # [K, D] diagonal covariance


def expanding_standardise(x: np.ndarray, min_rows: int = STD_MIN_ROWS) -> np.ndarray:
    """Column-wise (x_t - mean_t) / std_t with mean_t, std_t over the finite values of rows <= t only.
    NaN where fewer than `min_rows` finite values exist so far, where x_t is NaN, or where std_t is 0."""
    x = np.asarray(x, dtype=float)
    fin = np.isfinite(x)
    v = np.where(fin, x, 0.0)
    n = np.cumsum(fin, axis=0)
    s1 = np.cumsum(v, axis=0)
    s2 = np.cumsum(v * v, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        var = (s2 - n * mean * mean) / np.maximum(n - 1, 1)
        z = (x - mean) / np.sqrt(var)
    bad = (n < min_rows) | ~fin | ~np.isfinite(z) | (var <= 0)
    return np.where(bad, np.nan, z)


def _loglik(x: np.ndarray, p: HmmParams) -> np.ndarray:
    """[T, K] log N(x_t | state k) over the finite dimensions of each row (0 where none is finite)."""
    x = np.asarray(x, dtype=float)
    fin = np.isfinite(x)
    xv = np.where(fin, x, 0.0)
    out = np.zeros((len(x), len(p.pi)))
    for k in range(len(p.pi)):
        d = (xv - p.mean[k]) ** 2 / p.var[k] + np.log(2.0 * np.pi * p.var[k])
        out[:, k] = -0.5 * np.where(fin, d, 0.0).sum(axis=1)
    return out


def _lik(ll: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = ll.max(axis=1)
    return np.exp(ll - m[:, None]), m


def forward_filter(pi: np.ndarray, trans: np.ndarray, lik: np.ndarray) -> np.ndarray:
    """Filtered state distributions P(s_t | obs_1..t) from the initial distribution `pi`, the transition
    matrix and the observation likelihoods lik[t, k] = P(obs_t | s_t = k) (any positive row scale)."""
    t_n, k_n = lik.shape
    out = np.empty((t_n, k_n))
    a = np.asarray(pi, dtype=float) * lik[0]
    tot = a.sum()
    out[0] = a / tot if tot > 0 else pi
    for t in range(1, t_n):
        a = (out[t - 1] @ trans) * lik[t]
        tot = a.sum()
        out[t] = a / tot if tot > 0 else out[t - 1] @ trans
    return out


def filter_probs(x: np.ndarray, p: HmmParams) -> np.ndarray:
    """Forward-filtered state probabilities of the observations `x` under `p` (row t uses rows <= t only)."""
    lik, _ = _lik(_loglik(x, p))
    return forward_filter(p.pi, p.trans, lik)


def _forward_backward(p: HmmParams, lik: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_n, k_n = lik.shape
    alpha, c = np.empty((t_n, k_n)), np.empty(t_n)
    a = p.pi * lik[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, t_n):
        a = (alpha[t - 1] @ p.trans) * lik[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    beta = np.ones((t_n, k_n))
    for t in range(t_n - 2, -1, -1):
        beta[t] = (p.trans @ (lik[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.zeros((k_n, k_n))
    for t in range(t_n - 1):
        xi += alpha[t][:, None] * p.trans * (lik[t + 1] * beta[t + 1])[None, :] / c[t + 1]
    return alpha, gamma, xi, c


def smooth_probs(x: np.ndarray, p: HmmParams) -> np.ndarray:
    """Forward-backward (smoothed) probabilities: conditions on the FUTURE. For tests only."""
    lik, _ = _lik(_loglik(x, p))
    return _forward_backward(p, lik)[1]


def _order(p: HmmParams) -> HmmParams:
    o = np.argsort(p.mean[:, 0], kind="stable")
    return HmmParams(p.pi[o], p.trans[np.ix_(o, o)], p.mean[o], p.var[o])


def fit_hmm(
    x: np.ndarray, n_states: int = N_STATES, seed: int = 0, n_iter: int = 80, tol: float = 1e-5
) -> HmmParams:
    """Baum-Welch (diagonal Gaussians) on rows of `x` (NaN entries are missing at random). Initialised from
    quantile groups of the first column plus a small seeded jitter; states are returned ordered by the
    mean of the first column (lowest first)."""
    x = np.asarray(x, dtype=float)
    fin = np.isfinite(x)
    xv = np.where(fin, x, 0.0)
    rng = np.random.default_rng(seed)
    first = np.where(fin[:, 0], x[:, 0], np.nan)
    ok = np.isfinite(first)
    edges = np.nanquantile(first[ok], np.linspace(0, 1, n_states + 1)[1:-1]) if ok.any() else []
    grp = np.digitize(np.where(ok, first, np.nanmedian(first[ok]) if ok.any() else 0.0), edges)
    d_n = x.shape[1]
    mean = np.zeros((n_states, d_n))
    var = np.ones((n_states, d_n))
    for k in range(n_states):
        sel = (grp == k)[:, None] & fin
        cnt = sel.sum(axis=0)
        mean[k] = np.where(cnt > 0, (xv * sel).sum(axis=0) / np.maximum(cnt, 1), 0.0)
        var[k] = np.where(cnt > 1, (((xv - mean[k]) ** 2) * sel).sum(axis=0) / np.maximum(cnt, 1), 1.0)
    mean = mean + rng.normal(0.0, 0.01, mean.shape)
    var = np.maximum(var, VAR_FLOOR)
    trans = np.full((n_states, n_states), 0.1 / max(n_states - 1, 1))
    np.fill_diagonal(trans, 0.9)
    p = HmmParams(np.full(n_states, 1.0 / n_states), trans, mean, var)
    prev = -np.inf
    for _ in range(n_iter):
        ll = _loglik(x, p)
        lik, m = _lik(ll)
        _, gamma, xi, c = _forward_backward(p, lik)
        total = float(np.log(c).sum() + m.sum())
        w = gamma[:, :, None] * fin[:, None, :]  # [T, K, D]
        den = np.maximum(w.sum(axis=0), 1e-9)
        mean = (w * xv[:, None, :]).sum(axis=0) / den
        var = np.maximum((w * (xv[:, None, :] - mean[None]) ** 2).sum(axis=0) / den, VAR_FLOOR)
        trans = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-12)
        pi = gamma[0]
        p = HmmParams(pi, trans, mean, var)
        if abs(total - prev) < tol * max(1.0, abs(total)):
            break
        prev = total
    return _order(p)


def regime_series(
    dates: list[date],
    x_raw: np.ndarray,
    *,
    min_fit_rows: int | None = None,
    std_min_rows: int | None = None,
    n_states: int = N_STATES,
    seed: int = 0,
) -> np.ndarray:
    """[T, K] filtered state probabilities per date (NaN before the first January with enough earlier rows).

    Each calendar year y uses parameters estimated on the rows dated <= 31 December of y-1, then filters the
    whole history up to the end of y from its first row; only the rows of year y are kept. A row therefore
    depends on observations up to its own date and on parameters from before its year."""
    need = MIN_FIT_ROWS if min_fit_rows is None else min_fit_rows
    z = expanding_standardise(x_raw, STD_MIN_ROWS if std_min_rows is None else std_min_rows)
    yrs = np.array([d.year for d in dates])
    out = np.full((len(dates), n_states), np.nan)
    for y in sorted(set(yrs.tolist())):
        before = yrs < y
        if int(np.isfinite(z[before]).any(axis=1).sum()) < need:
            continue
        params = fit_hmm(z[before], n_states=n_states, seed=seed)
        upto = yrs <= y
        probs = filter_probs(z[upto], params)
        out[yrs == y] = probs[yrs[upto] == y]
    return out


# ---- Bayesian online change-point detection ----------------------------------------------------------


def bocpd(
    x: np.ndarray,
    *,
    hazard: float = HAZARD,
    scale: float = BOCPD_SCALE,
    max_run: int = BOCPD_MAX_RUN,
    short: int = CP_SHORT,
) -> tuple[np.ndarray, np.ndarray]:
    """(cp_prob, run_length_mean) per observation: P(run length < `short`) and E[run length] after seeing
    x_t. Constant hazard, Normal-Gamma prior (mu0 0, kappa0 1, alpha0 1, beta0 = scale^2), Student-t
    predictive. A NaN observation repeats the previous outputs and leaves the posterior untouched."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    cp = np.full(n, np.nan)
    rl = np.full(n, np.nan)
    mu0, k0, a0, b0 = 0.0, 1.0, 1.0, scale * scale
    r = np.array([1.0])
    mu, kap, alp, bet = np.array([mu0]), np.array([k0]), np.array([a0]), np.array([b0])
    last: tuple[float, float] | None = None
    for t in range(n):
        xt = x[t]
        if not math.isfinite(xt):
            if last is not None:
                cp[t], rl[t] = last
            continue
        nu = 2.0 * alp
        s2 = bet * (kap + 1.0) / (alp * kap)
        logp = (
            gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi * s2)
            - (nu + 1.0) / 2.0 * np.log1p((xt - mu) ** 2 / (nu * s2))
        )  # fmt: skip
        pred = np.exp(logp)
        growth = r * pred * (1.0 - hazard)
        change = float((r * pred).sum() * hazard)
        new_r = np.concatenate([[change], growth])
        tot = new_r.sum()
        if not math.isfinite(tot) or tot <= 0:  # an impossible observation: restart from the prior
            new_r = np.zeros(len(r) + 1)
            new_r[0] = 1.0
            tot = 1.0
        new_r = new_r / tot
        new_mu = np.concatenate([[mu0], (kap * mu + xt) / (kap + 1.0)])
        new_bet = np.concatenate([[b0], bet + kap * (xt - mu) ** 2 / (2.0 * (kap + 1.0))])
        new_kap = np.concatenate([[k0], kap + 1.0])
        new_alp = np.concatenate([[a0], alp + 0.5])
        if len(new_r) > max_run:  # fold the longest runs together
            new_r = np.concatenate([new_r[: max_run - 1], [new_r[max_run - 1 :].sum()]])
            new_mu, new_bet, new_kap, new_alp = (v[:max_run] for v in (new_mu, new_bet, new_kap, new_alp))
        r, mu, kap, alp, bet = new_r, new_mu, new_kap, new_alp, new_bet
        cp[t] = float(r[:short].sum())
        rl[t] = float((r * np.arange(len(r))).sum())
        last = (cp[t], rl[t])
    return cp, rl


# ---- features ------------------------------------------------------------------------------------------


def _market_obs(ctx: FeatureContext) -> tuple[list[date], np.ndarray]:
    """(dates, [T, 6] raw observations) from panel_market, cut at as_of by the context."""
    from stocksense.core.errors import DataNotAvailable
    from stocksense.features.market import _vix

    m = ctx.market().select("date", "ew500_ret", "breadth_200", "xs_dispersion").collect().sort("date")
    la = log_ret("ew500_ret")
    m = m.with_columns(
        la.rolling_sum(5, min_samples=5).alias("ret5"),
        la.rolling_std(21, min_samples=21, ddof=1).alias("vol21"),
    )
    try:
        v = _vix(ctx).select("date", pl.col("chg5").alias("vix_chg_5"))
        m = m.join(v, on="date", how="left").sort("date")
    except (DataNotAvailable, FileNotFoundError, KeyError, pl.exceptions.PolarsError):
        m = m.with_columns(pl.lit(None, dtype=pl.Float64).alias("vix_chg_5"))
    cols = ["ew500_ret", "ret5", "vol21", "breadth_200", "xs_dispersion", "vix_chg_5"]
    x = np.column_stack([m[c].cast(pl.Float64).fill_null(float("nan")).to_numpy() for c in cols])
    return m["date"].to_list(), x


def _regime_feature(ctx: FeatureContext) -> pl.LazyFrame:
    dates, x = _market_obs(ctx)
    p = regime_series(dates, x)
    return (
        pl.DataFrame(
            {"date": dates, "regime_p_bear": p[:, 0], "regime_p_bull": p[:, -1]},
            schema={"date": pl.Date, "regime_p_bear": pl.Float64, "regime_p_bull": pl.Float64},
        )
        .with_columns(pl.col("regime_p_bear", "regime_p_bull").fill_nan(None))
        .lazy()
    )


def _bocpd_feature(ctx: FeatureContext) -> pl.LazyFrame:
    m = ctx.market().select("date", "ew500_ret").collect().sort("date")
    cp, rl = bocpd(m["ew500_ret"].cast(pl.Float64).fill_null(float("nan")).to_numpy())
    return (
        pl.DataFrame({"date": m["date"], "cp_prob": cp, "run_length_mean": rl})
        .with_columns(pl.col("cp_prob", "run_length_mean").fill_nan(None))
        .lazy()
    )


feature(
    "regime_state",
    group="market",
    kind="mkt",
    window=750,
    version=2,
    label="Probability that the market is in a bear or a bull regime (3-state HMM, filtered, no look-ahead).",
    outputs=("regime_p_bear", "regime_p_bull"),
)(_regime_feature)
feature(
    "bocpd",
    group="market",
    kind="mkt",
    window=500,
    version=2,
    label="Chance that the market's return process just changed, and the age of the current regime.",
    outputs=("cp_prob", "run_length_mean"),
)(_bocpd_feature)

SPECS: tuple[FeatureSpec, ...] = (REGISTRY["regime_state"], REGISTRY["bocpd"])


# ---- the backtest loader (research worlds) -------------------------------------------------------------


def p_bear_by_date(settings, end: date) -> dict[date, float]:
    """date -> filtered bear probability for every session up to `end` (causal), for the S1 overlay."""
    ctx = FeatureContext(settings, as_of=end)
    out = _regime_feature(ctx).collect()
    return {
        d: float(p)
        for d, p in zip(out["date"].to_list(), out["regime_p_bear"].to_list(), strict=True)
        if p is not None
    }
