"""Multiple-testing guards. Build these BEFORE the strategy generator.

This module exists because of one number:

    After 1,000 independent backtests the expected best Sharpe ratio is 3.26 --
    even when the true edge is exactly zero.
    (Bailey & Lopez de Prado, "The Deflated Sharpe Ratio")

This project's plan is to sweep 2,000-5,000 configurations per night. At that
scale a beautiful equity curve is not evidence of anything; it is the arithmetic
of maximum order statistics. A search that reports its best result without
correcting for how many results it looked at is not doing research, it is
generating press releases.

So the search's throughput fix and its false-positive fix are DIFFERENT fixes and
this build needs both:

    expected_max_sharpe            what a pure-noise search would have produced
    deflated_sharpe_ratio          P(the observed Sharpe is real), given n_trials
    probability_of_backtest_overfitting
                                   does the in-sample winner beat the OOS median?

DECISION AUTHORITY (also stated in evaluation/factor_metrics.py):

    ICIR, decay half-life       loop objective + first screen
    alpha_net via gate.py       promotion gate -- unchanged, untouched
    DSR >= 0.95, PBO <= 0.5     final gate, on the sealed vault only
    Sharpe/Sortino/Calmar/maxDD diagnostics -- never a gate on their own

Nothing here reads market data or holds state; every function is pure, so the
tests can pin exact published values.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm

# Euler-Mascheroni constant, used in the expected-maximum-of-N-Gaussians formula.
EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(
    n_trials: int,
    trial_sharpe_std: float,
    trial_sharpe_mean: float = 0.0,
) -> float:
    """The Sharpe a pure-noise search of `n_trials` configs would be expected to
    produce as its BEST result.

        SR0 = mean + std * [ (1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N*e)) ]

    with g = Euler-Mascheroni, Z^-1 = the inverse standard normal CDF.

    This is the bar a real edge has to clear. Pre-verified against the published
    worked example: n_trials=1000, trial_sharpe_std=1.0, mean=0.0 -> 3.2551
    (published 3.26). Other reference values, asserted in the tests:
    N=10 -> 1.5746, N=100 -> 2.5306, N=10000 -> 3.8607.

    Args:
        n_trials: how many configurations were ACTUALLY evaluated -- read from
            the attempt registry, never guessed. Every candidate counts, not
            just the survivors.
        trial_sharpe_std: cross-sectional dispersion of Sharpe across trials.
        trial_sharpe_mean: expected Sharpe of a trial. Leave at 0.0 unless you
            genuinely believe the average candidate has edge.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if trial_sharpe_std < 0:
        raise ValueError("trial_sharpe_std must be >= 0")
    if n_trials == 1:
        return trial_sharpe_mean

    g = EULER_MASCHERONI
    term_1 = norm.ppf(1.0 - 1.0 / n_trials)
    term_2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(trial_sharpe_mean + trial_sharpe_std * ((1.0 - g) * term_1 + g * term_2))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    trial_sharpe_std: float,
    sample_length: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    trial_sharpe_mean: float = 0.0,
) -> float:
    """Probability that `observed_sharpe` reflects a real edge rather than the
    best of `n_trials` draws from noise.

        DSR = Z[ (SR - SR0) * sqrt(T - 1)
                 / sqrt(1 - skew*SR + ((kurtosis - 1) / 4) * SR^2) ]

    where Z is the standard normal CDF and SR0 comes from `expected_max_sharpe`.

    Returns a probability in [0, 1]. **>= 0.95 is the bar.**

    Args:
        observed_sharpe: the candidate's Sharpe, on the SAME periodicity as
            `sample_length` (both per-rebalance, or both daily -- not mixed).
        n_trials: from the attempt registry. See `expected_max_sharpe`.
        sample_length: number of return observations, T.
        skew: skewness of the return series.
        kurtosis: **RAW fourth moment (normal = 3.0), NOT excess kurtosis.**
            Passing excess kurtosis silently shifts the answer, so it is spelled
            out here and asserted in the tests.
        trial_sharpe_mean: as in `expected_max_sharpe`.
    """
    if sample_length < 2:
        raise ValueError("sample_length must be >= 2")
    if kurtosis < 1.0:
        raise ValueError("kurtosis is the RAW fourth moment (normal = 3.0), so it must be >= 1")

    sr0 = expected_max_sharpe(n_trials, trial_sharpe_std, trial_sharpe_mean)
    variance = 1.0 - skew * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2
    if variance <= 0:
        # Degenerate higher moments; refuse to report a confident answer.
        return float("nan")

    z = (observed_sharpe - sr0) * math.sqrt(sample_length - 1) / math.sqrt(variance)
    return float(norm.cdf(z))


def _slice_perf(block: np.ndarray, metric: str) -> np.ndarray:
    """Per-config performance within one time slice."""
    if metric == "mean":
        return block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    sd = np.where(sd == 0, np.nan, sd)  # a constant config has undefined Sharpe
    return block.mean(axis=0) / sd


def probability_of_backtest_overfitting(
    performance: pd.DataFrame,
    s: int = 16,
    metric: str = "sharpe",
) -> dict[str, object]:
    """Combinatorially Symmetric Cross-Validation (CSCV).

    Asks the question that matters about a search: **does the configuration that
    wins in-sample actually beat the median out-of-sample?** For a search driven
    by noise it does not, and this reports that as a probability.

    Algorithm:
      1. Split the rows (time slices) into `s` even contiguous subsets.
      2. For every combination of s/2 subsets as IS, the complement is OOS.
      3. n* = the column with the best IS mean performance.
      4. w = relative rank of n* among all columns in OOS, in (0, 1),
         ascending so that 1.0 is best.
      5. logit = ln(w / (1 - w)).
      6. PBO = the fraction of combinations whose logit <= 0, i.e. the winner
         landed at or below the OOS median.

    **PBO <= 0.5 is the bar.** PBO near 1.0 means the selection is pure
    overfitting: the in-sample winner is systematically a below-median performer
    out of sample.

    Args:
        performance: rows = time slices, columns = strategy configurations,
            values = that config's performance in that slice.
        s: number of subsets. Must be even and >= 4. C(s, s/2) grows fast --
            s=16 gives 12,870 combinations, which is the usual choice.
        metric: how a slice's performance is scored. "sharpe" (mean/std, the
            standard) or "mean".

    Measured behaviour worth knowing before reading a result (iid noise columns,
    320 rows, s=8):

        50 noise configs   -> PBO ~ 0.25
        200 noise configs  -> PBO ~ 0.53
        one persistent edge among noise -> PBO ~ 0.00

    PBO rises toward 0.5 as the number of CONFIGS grows, because selection gets
    more fragile the more things you try. With only a handful of candidates the
    in-sample winner really is usually the best one; at 5,000 it usually is not.
    That is the property this metric exists to expose, and it means a PBO from a
    small sweep is not comparable to one from a large sweep.
    """
    if s % 2 != 0 or s < 4:
        raise ValueError("s must be even and >= 4")
    if metric not in ("sharpe", "mean"):
        raise ValueError("metric must be 'sharpe' or 'mean'")
    if performance.shape[1] < 2:
        raise ValueError("need at least 2 configurations to compare")
    if len(performance) < s:
        raise ValueError(f"need at least s={s} rows, got {len(performance)}")

    values = performance.to_numpy(dtype=float)
    n_cols = values.shape[1]
    # Even contiguous subsets; any remainder rows at the end are dropped so that
    # every subset is the same size (an uneven split biases the IS/OOS means).
    rows_per = len(values) // s
    subsets = [values[i * rows_per : (i + 1) * rows_per] for i in range(s)]

    logits: list[float] = []
    for is_idx in combinations(range(s), s // 2):
        oos_idx = [i for i in range(s) if i not in is_idx]
        is_perf = _slice_perf(np.concatenate([subsets[i] for i in is_idx]), metric)
        oos_perf = _slice_perf(np.concatenate([subsets[i] for i in oos_idx]), metric)

        n_star = int(np.nanargmax(is_perf))
        # Rank ascending: 1 = worst, n_cols = best. Then map into (0, 1).
        rank = float((oos_perf <= oos_perf[n_star]).sum())
        w = rank / (n_cols + 1)
        w = min(max(w, 1e-12), 1 - 1e-12)  # keep the logit finite
        logits.append(math.log(w / (1.0 - w)))

    arr = np.asarray(logits)
    return {
        "pbo": float((arr <= 0).mean()),
        "logits": logits,
        "n_combinations": len(logits),
        "median_logit": float(np.median(arr)),
        "metric": metric,
        "n_configs": n_cols,
    }


def haircut_sharpe(observed_sharpe: float, n_trials: int, trial_sharpe_std: float) -> float:
    """How much of the observed Sharpe survives the multiplicity correction.

    A convenience diagnostic for the search leaderboard: `observed - SR0`, i.e.
    the excess over what a pure-noise search of this size would have produced.
    Negative means the result is *worse* than noise would be expected to deliver.
    """
    return float(observed_sharpe - expected_max_sharpe(n_trials, trial_sharpe_std))
