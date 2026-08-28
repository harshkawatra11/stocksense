"""Phase K0.2: the multiplicity guards.

The headline assertion here is `test_expected_max_sharpe_matches_published_example`
-- it pins the module against Bailey & Lopez de Prado's own worked example
(1,000 trials, true Sharpe 0, expected best Sharpe 3.26). If that drifts, the
whole deflation is wrong and every DSR this project ever reports is wrong with
it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocksense.evaluation.robustness import (
    DSR_SIGNIFICANCE_THRESHOLD,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
    summarize_trial_sharpes,
)


# ---- expected_max_sharpe ----


def test_expected_max_sharpe_matches_published_example() -> None:
    """Bailey & Lopez de Prado's own worked example: with 1,000 independent
    trials whose true Sharpe is all zero, the expected BEST observed Sharpe is
    3.26. This is the number the whole module exists to respect."""
    assert expected_max_sharpe(1000, 1.0, 0.0) == pytest.approx(3.26, abs=0.01)


@pytest.mark.parametrize(
    "n_trials,expected",
    [(10, 1.5746), (100, 2.5306), (1000, 3.2551), (10000, 3.8607)],
)
def test_expected_max_sharpe_reference_values(n_trials, expected) -> None:
    assert expected_max_sharpe(n_trials, 1.0, 0.0) == pytest.approx(expected, abs=1e-3)


def test_expected_max_sharpe_grows_with_trials() -> None:
    """The core intuition: more tries, higher the bar a winner must clear."""
    values = [expected_max_sharpe(n, 1.0) for n in (10, 100, 1000, 10000)]
    assert values == sorted(values)


def test_expected_max_sharpe_single_trial_is_the_mean() -> None:
    """One trial is not a selection problem at all -- no deflation to apply."""
    assert expected_max_sharpe(1, 1.0, trial_sharpe_mean=0.4) == pytest.approx(0.4)


def test_expected_max_sharpe_scales_with_trial_dispersion() -> None:
    assert expected_max_sharpe(100, 2.0) == pytest.approx(2 * expected_max_sharpe(100, 1.0))


def test_expected_max_sharpe_rejects_invalid_trials() -> None:
    with pytest.raises(ValueError):
        expected_max_sharpe(0, 1.0)


# ---- deflated_sharpe_ratio ----


def _normal_moments() -> dict:
    """Skew/kurtosis of a Normal distribution. kurtosis is RAW (3.0), not
    excess -- passing 0.0 here is the classic silent error this module warns
    about."""
    return {"skew": 0.0, "kurtosis": 3.0}


def test_dsr_rejects_a_sharpe_that_only_looks_good_because_of_many_trials() -> None:
    """A Sharpe of 2.0 sounds excellent -- but if it is the best of 1,000 tries
    it sits well BELOW the 3.26 expected maximum under a true edge of zero, so
    the DSR must be low."""
    dsr = deflated_sharpe_ratio(
        observed_sharpe=2.0, n_trials=1000, trial_sharpe_std=1.0,
        sample_length=1000, **_normal_moments(),
    )
    assert dsr < 0.5
    assert dsr < DSR_SIGNIFICANCE_THRESHOLD


def test_dsr_accepts_the_same_sharpe_when_it_was_the_only_trial() -> None:
    """Identical observed Sharpe, but honestly arrived at -- must pass."""
    dsr = deflated_sharpe_ratio(
        observed_sharpe=2.0, n_trials=1, trial_sharpe_std=1.0,
        sample_length=1000, **_normal_moments(),
    )
    assert dsr > DSR_SIGNIFICANCE_THRESHOLD


def test_dsr_is_monotonically_punished_by_more_trials() -> None:
    dsrs = [
        deflated_sharpe_ratio(
            observed_sharpe=2.5, n_trials=n, trial_sharpe_std=1.0,
            sample_length=1000, **_normal_moments(),
        )
        for n in (1, 10, 100, 1000)
    ]
    assert dsrs == sorted(dsrs, reverse=True)


def test_dsr_punishes_negative_skew_and_fat_tails() -> None:
    """Same Sharpe, uglier return distribution -> lower confidence. This is the
    part a plain Bonferroni correction cannot see, and it is exactly the shape
    of an intraday return series."""
    normal = deflated_sharpe_ratio(
        observed_sharpe=2.0, n_trials=10, trial_sharpe_std=1.0,
        sample_length=1000, skew=0.0, kurtosis=3.0,
    )
    ugly = deflated_sharpe_ratio(
        observed_sharpe=2.0, n_trials=10, trial_sharpe_std=1.0,
        sample_length=1000, skew=-1.5, kurtosis=12.0,
    )
    assert ugly < normal


def test_dsr_returns_nan_on_degenerate_sample() -> None:
    assert np.isnan(
        deflated_sharpe_ratio(
            observed_sharpe=2.0, n_trials=10, trial_sharpe_std=1.0,
            sample_length=1, **_normal_moments(),
        )
    )


def test_dsr_is_a_probability() -> None:
    for n in (1, 5, 50, 500):
        dsr = deflated_sharpe_ratio(
            observed_sharpe=1.5, n_trials=n, trial_sharpe_std=0.8,
            sample_length=500, **_normal_moments(),
        )
        assert 0.0 <= dsr <= 1.0


# ---- probability_of_backtest_overfitting ----


def test_pbo_is_high_for_random_noise_configs() -> None:
    """50 columns of pure noise: whichever one wins in-sample has no reason to
    keep winning out of sample, so it lands below median about half the time or
    worse. PBO must be high."""
    rng = np.random.default_rng(42)
    perf = pd.DataFrame(rng.normal(0, 1, size=(320, 50)))
    result = probability_of_backtest_overfitting(perf, s=8)
    assert result["pbo"] > 0.4
    assert result["n_configs"] == 50


def test_pbo_is_low_for_one_genuinely_dominant_config() -> None:
    """One column with a real, constant edge -- the in-sample winner is the same
    column every time and it stays best out of sample. PBO must be ~0."""
    rng = np.random.default_rng(7)
    perf = pd.DataFrame(rng.normal(0, 1, size=(320, 20)))
    perf[0] = perf[0] + 5.0  # unmistakably dominant
    result = probability_of_backtest_overfitting(perf, s=8)
    assert result["pbo"] < 0.2


def test_pbo_combination_count_is_s_choose_half() -> None:
    import math as _math

    perf = pd.DataFrame(np.random.default_rng(0).normal(0, 1, size=(320, 5)))
    result = probability_of_backtest_overfitting(perf, s=8)
    assert result["n_combinations"] == _math.comb(8, 4)


def test_pbo_rejects_odd_s() -> None:
    perf = pd.DataFrame(np.random.default_rng(0).normal(0, 1, size=(100, 5)))
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(perf, s=7)


def test_pbo_rejects_single_config() -> None:
    """Selection bias needs something to select BETWEEN."""
    perf = pd.DataFrame(np.random.default_rng(0).normal(0, 1, size=(100, 1)))
    with pytest.raises(ValueError, match="at least 2"):
        probability_of_backtest_overfitting(perf, s=8)


def test_pbo_rejects_too_few_slices() -> None:
    perf = pd.DataFrame(np.random.default_rng(0).normal(0, 1, size=(4, 5)))
    with pytest.raises(ValueError, match="time slices"):
        probability_of_backtest_overfitting(perf, s=8)


def test_pbo_is_a_probability() -> None:
    perf = pd.DataFrame(np.random.default_rng(1).normal(0, 1, size=(320, 10)))
    result = probability_of_backtest_overfitting(perf, s=8)
    assert 0.0 <= result["pbo"] <= 1.0
    assert len(result["logits"]) == result["n_combinations"]


# ---- summarize_trial_sharpes ----


def test_summarize_trial_sharpes_counts_trials() -> None:
    out = summarize_trial_sharpes([0.1, 0.5, -0.2, 1.1])
    assert out["n_trials"] == 4
    assert out["trial_sharpe_std"] == pytest.approx(float(np.std([0.1, 0.5, -0.2, 1.1], ddof=1)))


def test_summarize_trial_sharpes_handles_too_few() -> None:
    out = summarize_trial_sharpes([0.3])
    assert out["n_trials"] == 1
    assert np.isnan(out["trial_sharpe_std"])
