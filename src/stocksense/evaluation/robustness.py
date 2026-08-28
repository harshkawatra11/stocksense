"""
Phase K0.2: the two guards that make an automated search loop honest.

THE NUMBER THIS MODULE EXISTS FOR:

    After 1,000 independent backtests, the expected BEST Sharpe ratio is 3.26 --
    even when the true edge of every single one is exactly zero.
    (Bailey & Lopez de Prado, "The Deflated Sharpe Ratio", 2014.)

Verified directly with `expected_max_sharpe(1000, 1.0)` below: 3.2551. So a
loop that generates and tests hundreds of candidates will hand back a
spectacular-looking Sharpe ratio whether or not anything real is there. Reporting
that number without deflating it is not research, it is a lottery result printed
on letterhead.

This project has already run ~40 distinct gate configurations across Phase 0
Runs 1-4, the 16 bhavcopy cap-band combinations, the intraday sweep and the F&O
ablation -- with no multiplicity correction anywhere.
`research/gate_criteria_preregistration.md` names this exact gap in its own text:
pre-registering the THRESHOLDS stops one failure mode, but "if this exact gate is
run repeatedly against re-tunings of the model until one clears it, the gate
becomes an overfitting instrument again -- just one level removed."

TWO COMPLEMENTARY GUARDS, because they catch different things:

  * DEFLATED SHARPE RATIO answers "given I ran N trials on returns this skewed
    and fat-tailed over this sample length, how surprised should I be by the
    best one?" It is strictly stronger than the Bonferroni correction already in
    evaluation/attempts.py because it also handles NON-NORMALITY -- and intraday
    return distributions are badly non-normal.

  * PROBABILITY OF BACKTEST OVERFITTING answers a different question entirely:
    "when I pick the in-sample winner, does it actually stay good out of sample,
    or does it land below median?" It is model-free and non-parametric, and it
    catches selection pathologies that a distributional correction cannot see.

Neither replaces evaluation/gate.py, which is untouched and remains the
promotion gate on `alpha_net`. These are the FINAL gate applied on the sealed
vault (Phase K1), after the loop has already produced a survivor.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329

DSR_SIGNIFICANCE_THRESHOLD = 0.95
"""A DSR is a probability that the true Sharpe exceeds the multiple-testing-
adjusted benchmark. 0.95 is the conventional deployment bar."""

PBO_SIGNIFICANCE_THRESHOLD = 0.5
"""PBO above 0.5 means the in-sample winner lands below the out-of-sample median
MORE often than not -- i.e. the selection procedure is worse than useless."""


def expected_max_sharpe(
    n_trials: int, trial_sharpe_std: float, trial_sharpe_mean: float = 0.0
) -> float:
    """E[max SR] over N independent trials whose Sharpe ratios are Normal.

        SR0 = mean + std * [ (1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N*e)) ]

    with g = Euler-Mascheroni (0.5772...), Z^-1 = the standard-Normal quantile.

    This is the benchmark a reported Sharpe must BEAT, not zero. Reference
    values (mean=0, std=1), reproduced by the tests:
        N=10    -> 1.5746
        N=100   -> 2.5306
        N=1000  -> 3.2551   (the published 3.26)
        N=10000 -> 3.8607
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if trial_sharpe_std < 0:
        raise ValueError(f"trial_sharpe_std must be >= 0, got {trial_sharpe_std}")
    if n_trials == 1:
        return float(trial_sharpe_mean)

    g = EULER_MASCHERONI
    term = (1.0 - g) * norm.ppf(1.0 - 1.0 / n_trials) + g * norm.ppf(
        1.0 - 1.0 / (n_trials * math.e)
    )
    return float(trial_sharpe_mean + trial_sharpe_std * term)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    trial_sharpe_std: float,
    sample_length: int,
    skew: float,
    kurtosis: float,
) -> float:
    """The probability that `observed_sharpe` reflects real skill rather than
    the best of N tries on a non-Normal return series.

        DSR = Z[ (SR - SR0) * sqrt(T - 1)
                 / sqrt(1 - skew*SR + ((kurtosis - 1)/4) * SR^2) ]

    `kurtosis` is the RAW fourth standardised moment -- a Normal distribution
    has kurtosis 3.0, NOT 0.0. Passing EXCESS kurtosis silently shifts the answer
    in the optimistic direction, so this is stated loudly here and asserted in
    the tests. `scipy.stats.kurtosis(x, fisher=False)` or
    `pandas.Series.kurt() + 3.0` both give the right thing.

    All Sharpe inputs must be on the SAME periodicity as `sample_length`: if you
    pass a per-rebalance Sharpe, `sample_length` is the number of rebalances.
    Do not mix an annualised Sharpe with a daily sample length.

    Returns a probability in [0, 1]. >= DSR_SIGNIFICANCE_THRESHOLD is the bar.
    """
    if sample_length < 2:
        return float("nan")

    sr0 = expected_max_sharpe(n_trials, trial_sharpe_std)
    denom_sq = 1.0 - skew * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2
    if denom_sq <= 0 or not np.isfinite(denom_sq):
        # A degenerate variance estimate cannot be turned into a probability;
        # returning nan is honest, returning 1.0 would be a silent free pass.
        return float("nan")

    z = (observed_sharpe - sr0) * math.sqrt(sample_length - 1) / math.sqrt(denom_sq)
    return float(norm.cdf(z))


def probability_of_backtest_overfitting(performance: pd.DataFrame, s: int = 16) -> dict:
    """CSCV -- Combinatorially Symmetric Cross-Validation (Bailey, Borwein,
    Lopez de Prado & Zhu).

    `performance`: rows are time slices in chronological order, columns are the
    candidate configurations, values are that configuration's performance in
    that slice (any consistent measure -- per-period return, Sharpe, alpha).

    The algorithm, exactly:
      1. Split the rows into `s` even contiguous subsets.
      2. For every way of choosing s/2 subsets as IN-SAMPLE, the complement is
         OUT-OF-SAMPLE. That is C(s, s/2) combinations -- symmetric by
         construction, and IS/OOS are always the same size, which is what makes
         the two performance figures directly comparable.
      3. Find n*, the column with the best IS mean performance.
      4. Compute w, n*'s RELATIVE RANK among all columns out of sample, mapped
         into the open interval (0, 1) as rank / (n_cols + 1), ranked ascending
         so w near 1.0 means n* was among the best OOS.
      5. logit = ln(w / (1 - w)).
      6. PBO = the fraction of combinations with logit <= 0, i.e. the share of
         splits where the in-sample winner landed at or below the OOS median.

    Returns {"pbo", "logits", "n_combinations", "n_configs", "n_slices"}.
    PBO <= 0.5 is the bar; PBO near 1.0 means the selection procedure is pure
    overfitting and the "winner" carries no information.
    """
    if s % 2 != 0:
        raise ValueError(f"s must be even (IS and OOS halves must match), got {s}")
    if s < 4:
        raise ValueError(f"s must be >= 4 to give a usable number of combinations, got {s}")

    perf = performance.dropna(axis=1, how="all")
    n_rows, n_cols = perf.shape
    if n_cols < 2:
        raise ValueError(f"need at least 2 candidate configurations to detect selection bias, got {n_cols}")
    if n_rows < s:
        raise ValueError(f"need at least s={s} time slices, got {n_rows}")

    # Contiguous, even subsets. np.array_split would make uneven ones; the
    # method's symmetry argument depends on them being the same size, so the
    # remainder rows at the end are dropped rather than silently unbalancing it.
    per_subset = n_rows // s
    subsets = [perf.iloc[i * per_subset : (i + 1) * per_subset] for i in range(s)]

    logits: list[float] = []
    for is_idx in combinations(range(s), s // 2):
        oos_idx = [i for i in range(s) if i not in is_idx]
        is_perf = pd.concat([subsets[i] for i in is_idx]).mean()
        oos_perf = pd.concat([subsets[i] for i in oos_idx]).mean()

        n_star = is_perf.idxmax()
        # Ascending rank: 1 = worst, n_cols = best. So a high w is a GOOD OOS
        # outcome for the in-sample winner, which is what we hope to see.
        oos_ranks = oos_perf.rank(ascending=True, method="average")
        w = float(oos_ranks[n_star]) / (n_cols + 1.0)
        w = min(max(w, 1e-12), 1.0 - 1e-12)  # keep the logit finite at the extremes
        logits.append(math.log(w / (1.0 - w)))

    pbo = float(np.mean([lg <= 0 for lg in logits])) if logits else float("nan")
    return {
        "pbo": pbo,
        "logits": logits,
        "n_combinations": len(logits),
        "n_configs": int(n_cols),
        "n_slices": int(n_rows),
    }


def summarize_trial_sharpes(trial_sharpes) -> dict:
    """Convenience: the two inputs the DSR needs about the SEARCH itself, as
    opposed to about the winning strategy. `trial_sharpes` is the Sharpe of
    every candidate the loop evaluated -- which is exactly what makes N a
    counted fact rather than a guess (see evaluation/attempts.py)."""
    arr = np.asarray(pd.Series(trial_sharpes, dtype="float64").dropna(), dtype="float64")
    if len(arr) < 2:
        return {"n_trials": int(len(arr)), "trial_sharpe_std": float("nan"), "trial_sharpe_mean": float("nan")}
    return {
        "n_trials": int(len(arr)),
        "trial_sharpe_std": float(arr.std(ddof=1)),
        "trial_sharpe_mean": float(arr.mean()),
    }
