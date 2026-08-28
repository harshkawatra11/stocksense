"""
Phase K0.1: the metrics an alpha-research loop actually optimizes toward.

Everything here is a PURE ADDITION consuming existing outputs -- `ScoredFold`
already stores `scores_by_date`/`rel_actual_by_date` as per-date dicts, and
`FoldResult.net_returns` already holds the per-rebalance net series. Nothing in
evaluation/backtest.py, evaluation/gate.py or evaluation/walkforward.py is
touched (all three are protected paths, foreman/policy.py).

WHY THIS EXISTS. `FoldResult.information_coefficient` (backtest.py:173-180) is a
POOLED PEARSON correlation over every (symbol, date) pair in a fold, concatenated
into one flat vector before a single `np.corrcoef`. The standard quant definition
is a per-date CROSS-SECTIONAL RANK IC -- one correlation per rebalance date,
then summarized. The pooled version mixes cross-sectional skill with time-series
level effects, and reports a single number where the whole point is the
DISTRIBUTION across dates. The legacy metric is deliberately left exactly as it
is so Phase 0's committed numbers stay reproducible; this module computes the
standard one alongside it.

DECISION AUTHORITY -- fixed here so it cannot drift later:

    | metric                        | role                                    |
    |-------------------------------|-----------------------------------------|
    | ICIR, half_life               | loop objective + screen                 |
    | alpha_net via gate.py         | promotion gate -- UNCHANGED, untouched  |
    | DSR >= 0.95, PBO <= 0.5       | final gate on the sealed vault          |
    | IC, Sharpe, Sortino, Calmar,  | diagnostics only -- never a gate alone  |
    | max_drawdown                  |                                         |

Calibration, so a result gets read correctly: documented equity factors run
IC ~ 0.02-0.05; a stable 0.05 is strong. IC above ~0.15 is a red flag for
overfitting or leakage, NOT a triumph. Grinold: IR ~= IC * sqrt(breadth).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stocksense.evaluation.backtest import ScoredFold
from stocksense.labels.forward_return import add_forward_return_labels, add_relative_forward_return

MIN_PAIRS_PER_DATE = 10
"""A cross-section smaller than this makes a per-date correlation meaningless.
Matches the spirit of backtest.py's own `valid.sum() > 10` guard."""

MIN_DATES_FOR_ICIR = 3
"""ICIR is mean/std across dates; below 3 dates the std is not a number worth
dividing by."""


def cross_sectional_ic(scored: ScoredFold, method: str = "spearman") -> pd.Series:
    """One IC per rebalance date -- the standard cross-sectional definition.

    `method="spearman"` (the default) gives RANK IC, which is what quant desks
    mean by "IC" and is robust to the fat tails of cross-sectional returns.
    `method="pearson"` is offered only so the legacy pooled metric's flavour can
    be compared like-for-like; it is not the default for a reason.

    Returns a Series indexed by rebalance date. Dates with fewer than
    MIN_PAIRS_PER_DATE valid (score, return) pairs are DROPPED rather than
    returned as NaN -- a date that could not be measured is absent from the
    sample, not a measurement of zero.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")

    rows: dict[pd.Timestamp, float] = {}
    for rdate in scored.rebalance_dates:
        scores = scored.scores_by_date.get(rdate)
        actual = scored.rel_actual_by_date.get(rdate)
        if scores is None or actual is None:
            continue
        aligned = pd.DataFrame({"score": scores, "actual": actual}).dropna()
        if len(aligned) < MIN_PAIRS_PER_DATE:
            continue
        # A constant column has no rank order and .corr() returns NaN; skip
        # rather than propagate a NaN into the mean.
        if aligned["score"].nunique() < 2 or aligned["actual"].nunique() < 2:
            continue
        rows[rdate] = float(aligned["score"].corr(aligned["actual"], method=method))

    return pd.Series(rows, dtype="float64").sort_index()


def icir(ic_series: pd.Series) -> float:
    """mean(IC) / std(IC, ddof=1) -- THE LOOP'S OBJECTIVE FUNCTION.

    This is the metric that distinguishes "makes a little every week" from
    "made everything in one week and gave it back" -- consistency, not size.
    Returns nan below MIN_DATES_FOR_ICIR dates, or when std is 0 (a perfectly
    constant IC is a synthetic artifact, not an infinitely good signal).
    """
    clean = pd.Series(ic_series, dtype="float64").dropna()
    if len(clean) < MIN_DATES_FOR_ICIR:
        return float("nan")
    std = float(clean.std(ddof=1))
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return float(clean.mean()) / std


def build_labeled_by_horizon(candles: pd.DataFrame, horizons: tuple[int, ...]) -> dict[int, pd.DataFrame]:
    """Builds decay_curve's `labeled_by_horizon` input. For each horizon h the
    relative-return column is named f"fwd_ret_{h}b_rel", matching
    labels/forward_return.py's own naming."""
    out: dict[int, pd.DataFrame] = {}
    for h in horizons:
        lab = add_forward_return_labels(candles, horizon_bars=h)
        lab = add_relative_forward_return(lab, horizon_bars=h)
        out[h] = lab
    return out


def decay_curve(
    ranker,
    feats: pd.DataFrame,
    labeled_by_horizon: dict[int, pd.DataFrame],
    scoring_dates: list[pd.Timestamp],
    feature_cols: list[str],
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 40),
) -> pd.DataFrame:
    """How fast the signal's predictive power dies as the forward horizon grows.

    CRITICAL -- why this cannot reuse `train_and_score_fold`: that function uses
    `horizon_bars` for BOTH the label AND the rebalance-date subsampling
    (`test_dates[::horizon_bars]`, backtest.py:87). So h=1 and h=20 get
    different folds, different scoring dates, and different sample sizes. Those
    are unrelated experiments, not a curve. A real decay curve needs ONE fitted
    model scored ONCE on ONE fixed set of dates, then compared against several
    forward horizons -- which is exactly what this does.

    `ranker` must already be fitted. `scoring_dates` must all be present in
    `feats`. Returns one row per horizon with columns:
        horizon, mean_ic, std_ic, icir, n_dates
    `n_dates` is asserted-by-construction to be identical across horizons for
    any date where every horizon has a label; where a longer horizon's label has
    not matured, that date drops out for that horizon only, which is real and is
    reported rather than hidden.
    """
    feats_by_date = {d: g for d, g in feats.groupby("date")}

    # Score ONCE per date -- the whole point. The scores do not depend on the
    # forward horizon at all; only the label being compared against does.
    scores_by_date: dict[pd.Timestamp, pd.Series] = {}
    for d in scoring_dates:
        day = feats_by_date.get(d)
        if day is None:
            continue
        day = day.dropna(subset=feature_cols, how="any")
        if len(day) < MIN_PAIRS_PER_DATE:
            continue
        # np.asarray, not `.values`: CrossSectionalRanker.predict returns a
        # pandas Series today (backtest.py:103 relies on that), but a bare
        # ndarray is an equally valid predictor contract and this module has no
        # reason to break on one. asarray accepts both.
        scores_by_date[d] = pd.Series(
            np.asarray(ranker.predict(day[feature_cols])), index=day["symbol"].values
        )

    rows = []
    for h in horizons:
        labeled = labeled_by_horizon.get(h)
        if labeled is None:
            raise KeyError(f"labeled_by_horizon is missing horizon {h}; build it with build_labeled_by_horizon")
        rel_col = f"fwd_ret_{h}b_rel"
        labeled_by_date = {d: g for d, g in labeled.groupby("date")}

        ics: list[float] = []
        for d, scores in scores_by_date.items():
            lab_day = labeled_by_date.get(d)
            if lab_day is None:
                continue
            actual = pd.Series(lab_day[rel_col].values, index=lab_day["symbol"].values)
            aligned = pd.DataFrame({"score": scores, "actual": actual}).dropna()
            if len(aligned) < MIN_PAIRS_PER_DATE:
                continue
            if aligned["score"].nunique() < 2 or aligned["actual"].nunique() < 2:
                continue
            ics.append(float(aligned["score"].corr(aligned["actual"], method="spearman")))

        ic_series = pd.Series(ics, dtype="float64")
        rows.append({
            "horizon": h,
            "mean_ic": float(ic_series.mean()) if len(ic_series) else float("nan"),
            "std_ic": float(ic_series.std(ddof=1)) if len(ic_series) > 1 else float("nan"),
            "icir": icir(ic_series),
            "n_dates": int(len(ic_series)),
        })

    return pd.DataFrame(rows)


def half_life(curve: pd.DataFrame) -> float:
    """The horizon at which mean IC has fallen to half its peak -- the video's
    "reject a signal that decays in two days" filter, made numeric.

    Linearly interpolated between the two bracketing horizons. Returns nan when
    the peak IC is <= 0 (no signal to decay) or when IC never halves within the
    tested range (report it as "longer than the range tested", not as a number
    the data does not support).
    """
    if curve.empty or "mean_ic" not in curve.columns:
        return float("nan")
    c = curve.dropna(subset=["mean_ic"]).sort_values("horizon")
    if c.empty:
        return float("nan")

    peak_idx = c["mean_ic"].idxmax()
    peak_ic = float(c.loc[peak_idx, "mean_ic"])
    peak_h = float(c.loc[peak_idx, "horizon"])
    if peak_ic <= 0:
        return float("nan")

    target = peak_ic / 2.0
    after = c[c["horizon"] > peak_h]
    prev_h, prev_ic = peak_h, peak_ic
    for _, row in after.iterrows():
        h, ic = float(row["horizon"]), float(row["mean_ic"])
        if ic <= target:
            if prev_ic == ic:
                return h
            frac = (prev_ic - target) / (prev_ic - ic)
            return prev_h + frac * (h - prev_h)
        prev_h, prev_ic = h, ic
    return float("nan")


# ---- risk-adjusted return diagnostics (never a gate on their own) ----


def _as_array(returns) -> np.ndarray:
    arr = np.asarray(pd.Series(returns, dtype="float64").dropna(), dtype="float64")
    return arr


def sharpe(returns, periods_per_year: int) -> float:
    """Annualised Sharpe of a per-period return series. Risk-free is taken as 0
    -- this measures the strategy's own excess-over-nothing, and every
    comparison in this codebase is against a cross-sectional benchmark that is
    subtracted upstream (alpha_net), not against cash."""
    arr = _as_array(returns)
    if len(arr) < 2:
        return float("nan")
    std = float(arr.std(ddof=1))
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return float(arr.mean()) / std * float(np.sqrt(periods_per_year))


def sortino(returns, periods_per_year: int, target: float = 0.0) -> float:
    """Like Sharpe but penalising only DOWNSIDE deviation -- upside volatility
    is not a risk anyone needs protecting from.

    Uses the STANDARD downside deviation:

        DD = sqrt( mean( min(r - target, 0)^2 ) )     over ALL periods

    NOT the sample std of the negative subset. That distinction is a real bug
    this module had on first write and its test caught: taking `std(ddof=1)` of
    only the losing periods returns 0 (and so nan) for a series whose losses are
    all the SAME size -- e.g. [0.03, 0.03, 0.03, -0.01, -0.01]. But consistent
    small losses are the best possible downside profile, so that case must
    produce a high Sortino, not an undefined one. The formula above handles it
    correctly and is also what every standard reference specifies.
    """
    arr = _as_array(returns)
    if len(arr) < 2:
        return float("nan")
    shortfall = np.minimum(arr - target, 0.0)
    dd = float(np.sqrt(np.mean(shortfall**2)))
    if dd == 0 or not np.isfinite(dd):
        # No period ever fell below target -- there is no downside risk to
        # divide by, so the ratio is genuinely undefined rather than infinite.
        return float("nan")
    return float(arr.mean() - target) / dd * float(np.sqrt(periods_per_year))


def max_drawdown(returns) -> float:
    """Worst peak-to-trough decline of the compounded equity curve. Always <= 0
    by construction (0.0 for a series that never draws down)."""
    arr = _as_array(returns)
    if len(arr) == 0:
        return float("nan")
    equity = np.cumprod(1.0 + arr)
    running_max = np.maximum.accumulate(equity)
    dd = equity / running_max - 1.0
    return float(min(0.0, dd.min()))


def calmar(returns, periods_per_year: int) -> float:
    """Annualised return divided by the absolute max drawdown."""
    arr = _as_array(returns)
    if len(arr) < 2:
        return float("nan")
    mdd = max_drawdown(arr)
    if not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    annualised = float(arr.mean()) * periods_per_year
    return annualised / abs(mdd)
