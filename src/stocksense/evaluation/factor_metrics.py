"""Factor quality metrics -- the search's objective function and first screen.

This is the cheap end of the cascade. Stages 0-3 of the search evaluate
thousands of configurations here in milliseconds each, and only the survivors
earn the expensive purged walk-forward. So everything in this module is
vectorised and allocation-light by intent.

DECISION AUTHORITY (also stated in evaluation/robustness.py):

    ICIR, decay half-life       loop objective + first screen   <- THIS MODULE
    alpha_net via gate.py       promotion gate -- unchanged, untouched
    DSR >= 0.95, PBO <= 0.5     final gate, on the sealed vault only
    Sharpe/Sortino/Calmar/maxDD diagnostics -- never a gate on their own

CALIBRATION, so a result can be read correctly:

    Real equity factors run IC 0.02-0.05.
    **IC > 0.15 is an overfitting RED FLAG, not a win.**
    Grinold's fundamental law: IR ~= IC * sqrt(breadth).

Two design choices worth stating, because both were wrong in the previous build:

1. IC is computed **per rebalance date and then aggregated**, not as one pooled
   correlation over every (symbol, date) pair. The pooled version silently mixes
   cross-sectional signal with time-series drift, and it is not what "IC" means
   anywhere else in the industry.

2. `decay_curve` scores ONE fitted model over ONE fixed set of dates and then
   looks up each horizon's label over those SAME dates. The previous
   implementation re-subsampled the rebalance dates per horizon, so every
   horizon saw different dates and different sample sizes -- which is not a
   decay curve, it is four unrelated numbers on one axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Below this many names, a cross-sectional correlation on one date is noise.
MIN_NAMES_PER_DATE = 10

# Above this, treat the result as suspicious rather than excellent.
IC_OVERFIT_FLAG = 0.15


def cross_sectional_ic(
    scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "spearman",
    min_names: int = MIN_NAMES_PER_DATE,
) -> pd.Series:
    """One information coefficient per rebalance date.

    Args:
        scores: long frame with columns [date, symbol, score].
        forward_returns: long frame with columns [date, symbol, fwd_ret].
        method: "spearman" for rank IC (the standard, and the default) or
            "pearson" for the legacy pooled metric's flavour. Rank IC is
            preferred because it is robust to the fat tails that make raw
            return correlations unstable.
        min_names: dates with fewer non-null pairs than this are dropped.

    Returns:
        Series indexed by date, valued IC. Empty if nothing qualifies.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError("method must be 'spearman' or 'pearson'")

    merged = scores.merge(forward_returns, on=["date", "symbol"], how="inner")
    merged = merged.dropna(subset=["score", "fwd_ret"])
    if merged.empty:
        return pd.Series(dtype=float, name="ic")

    out: dict[pd.Timestamp, float] = {}
    for d, grp in merged.groupby("date", sort=True):
        if len(grp) < min_names:
            continue
        # A constant score on a date has undefined correlation; skip rather than
        # emit a NaN that quietly poisons the mean.
        if grp["score"].nunique() < 2 or grp["fwd_ret"].nunique() < 2:
            continue
        out[d] = float(grp["score"].corr(grp["fwd_ret"], method=method))

    return pd.Series(out, name="ic").dropna()


def icir(ic_series: pd.Series, min_dates: int = 3) -> float:
    """mean(IC) / std(IC, ddof=1) -- THE SEARCH'S OBJECTIVE FUNCTION.

    Prefers a small consistent edge over a large erratic one, which is the right
    preference for something that must survive costs and be sized safely.

    Returns nan when there is too little to judge (fewer than `min_dates`
    observations, or zero dispersion) rather than a flattering number.
    """
    s = ic_series.dropna()
    if len(s) < min_dates:
        return float("nan")
    sd = s.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(s.mean() / sd)


def ic_summary(ic_series: pd.Series) -> dict[str, float]:
    """Everything worth knowing about an IC series, including the red flag."""
    s = ic_series.dropna()
    if s.empty:
        return {
            "mean_ic": float("nan"), "std_ic": float("nan"), "icir": float("nan"),
            "n_dates": 0, "hit_rate": float("nan"), "overfit_flag": False,
        }
    mean_ic = float(s.mean())
    return {
        "mean_ic": mean_ic,
        "std_ic": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
        "icir": icir(s),
        "n_dates": int(len(s)),
        "hit_rate": float((s > 0).mean()),
        # Not "this is great" -- real equity factors live at 0.02-0.05.
        "overfit_flag": bool(abs(mean_ic) > IC_OVERFIT_FLAG),
    }


def decay_curve(
    scores: pd.DataFrame,
    forward_returns_by_horizon: dict[int, pd.DataFrame],
    method: str = "spearman",
    min_names: int = MIN_NAMES_PER_DATE,
) -> pd.DataFrame:
    """How fast the signal's predictive power decays with horizon.

    CRITICAL, and the reason this does not reuse the fold-scoring path: ONE set
    of scores is evaluated against EVERY horizon's labels over the SAME dates.
    Re-subsampling dates per horizon (which the previous build did, because the
    horizon drove both the label and the rebalance spacing) gives each horizon a
    different sample and a different size, which is not a curve.

    Args:
        scores: [date, symbol, score] -- one score set, scored once.
        forward_returns_by_horizon: {horizon_in_bars: [date, symbol, fwd_ret]}.

    Returns:
        Frame with columns [horizon, mean_ic, std_ic, icir, n_dates, hit_rate],
        sorted by horizon.
    """
    rows = []
    for h in sorted(forward_returns_by_horizon):
        ic = cross_sectional_ic(scores, forward_returns_by_horizon[h], method, min_names)
        summary = ic_summary(ic)
        rows.append({"horizon": h, **{k: summary[k] for k in
                                      ("mean_ic", "std_ic", "icir", "n_dates", "hit_rate")}})
    return pd.DataFrame(rows)


def half_life(curve: pd.DataFrame) -> float:
    """Horizon at which mean IC falls to half its peak, linearly interpolated.

    This is the video's "it decays in two days" filter, made numeric. The search
    rejects anything below ~3 bars: a signal that is gone before it can be
    traded net of costs is not tradeable, however high its peak IC.

    Returns nan if the peak is non-positive or the IC never halves within the
    tested range -- both of which mean "this question does not apply", not "this
    is good".
    """
    if curve.empty or "mean_ic" not in curve:
        return float("nan")

    c = curve.dropna(subset=["mean_ic"]).sort_values("horizon").reset_index(drop=True)
    if c.empty:
        return float("nan")

    peak_idx = int(c["mean_ic"].idxmax())
    peak = float(c.loc[peak_idx, "mean_ic"])
    if peak <= 0:
        return float("nan")

    target = 0.5 * peak
    # Walk forward from the peak looking for the first crossing.
    for i in range(peak_idx + 1, len(c)):
        prev_ic, cur_ic = float(c.loc[i - 1, "mean_ic"]), float(c.loc[i, "mean_ic"])
        if cur_ic <= target <= prev_ic:
            h_prev, h_cur = float(c.loc[i - 1, "horizon"]), float(c.loc[i, "horizon"])
            if prev_ic == cur_ic:
                return h_cur
            frac = (prev_ic - target) / (prev_ic - cur_ic)
            return float(h_prev + frac * (h_cur - h_prev))
    return float("nan")


# ----------------------------------------------------------------- diagnostics
# Never a gate on their own -- see the decision-authority table above.
def sharpe(returns: np.ndarray | pd.Series, periods_per_year: int) -> float:
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def sortino(returns: np.ndarray | pd.Series, periods_per_year: int) -> float:
    """Like Sharpe but penalising only downside deviation.

    Upside volatility is not risk, and a strategy with occasional large wins is
    unfairly punished by Sharpe.
    """
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if r.size < 2:
        return float("nan")
    downside = r[r < 0]
    if downside.size < 2:
        return float("nan")
    dd = downside.std(ddof=1)
    if dd == 0:
        return float("nan")
    return float(r.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray | pd.Series) -> float:
    """Worst peak-to-trough decline on the compounded curve. Always <= 0."""
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if r.size == 0:
        return 0.0
    curve = np.cumprod(1.0 + r)
    running_peak = np.maximum.accumulate(curve)
    return float(np.min(curve / running_peak - 1.0))


def calmar(returns: np.ndarray | pd.Series, periods_per_year: int) -> float:
    """Annualised return divided by the absolute max drawdown."""
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if r.size < 2:
        return float("nan")
    mdd = max_drawdown(r)
    if mdd == 0:
        return float("nan")
    annualised = (1.0 + r.mean()) ** periods_per_year - 1.0
    return float(annualised / abs(mdd))


def grinold_ir(ic: float, breadth: int) -> float:
    """The fundamental law of active management: IR ~= IC * sqrt(breadth).

    Useful as a sanity check on a claimed result: a small IC across many
    independent bets can be a real business, while a large IC on two bets a day
    is mostly luck. At 1-2 positions a day, breadth is tiny -- which is exactly
    why this project's edge has to come from selectivity, not from IC magnitude.
    """
    if breadth < 1:
        raise ValueError("breadth must be >= 1")
    return float(ic * np.sqrt(breadth))
