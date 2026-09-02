"""Tests for the multiple-testing guards.

These pin PUBLISHED values, not our own outputs. That matters: the whole point of
this module is to be an independent referee for a search that will otherwise
happily report noise, so "it returns what our implementation returns" would be
worthless. If a refactor moves these numbers, the refactor is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocksense.evaluation.robustness import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    haircut_sharpe,
    probability_of_backtest_overfitting,
)


# ------------------------------------------------------- expected max Sharpe
def test_matches_the_published_worked_example():
    """THE number this whole module exists for.

    After 1,000 independent backtests the expected best Sharpe is 3.26 even when
    the true edge is exactly zero (Bailey & Lopez de Prado). Our formula
    reproduces 3.2551.
    """
    assert expected_max_sharpe(1000, 1.0, 0.0) == pytest.approx(3.26, abs=0.01)


@pytest.mark.parametrize(
    ("n", "expected"),
    [(10, 1.5746), (100, 2.5306), (1000, 3.2551), (10000, 3.8607)],
)
def test_reference_values_across_trial_counts(n, expected):
    assert expected_max_sharpe(n, 1.0, 0.0) == pytest.approx(expected, abs=0.001)


def test_the_bar_rises_with_the_size_of_the_search():
    """The property that makes a bigger sweep self-policing: searching 5,000
    configs instead of 40 automatically raises the Sharpe a result must clear."""
    small = expected_max_sharpe(40, 1.0)
    large = expected_max_sharpe(5000, 1.0)
    assert large > small
    assert large == pytest.approx(3.688, abs=0.01)


def test_a_single_trial_needs_no_correction():
    assert expected_max_sharpe(1, 1.0, 0.3) == pytest.approx(0.3)


def test_rejects_nonsense_inputs():
    with pytest.raises(ValueError):
        expected_max_sharpe(0, 1.0)
    with pytest.raises(ValueError):
        expected_max_sharpe(100, -1.0)


# ---------------------------------------------------------- deflated Sharpe
def test_dsr_rejects_a_sharpe_that_only_looks_good_because_of_many_trials():
    """A Sharpe of 2.0 is impressive from one test and unremarkable as the best
    of 1,000 -- where pure noise is expected to deliver 3.26."""
    from_one_trial = deflated_sharpe_ratio(2.0, n_trials=1, trial_sharpe_std=1.0, sample_length=500)
    from_many = deflated_sharpe_ratio(2.0, n_trials=1000, trial_sharpe_std=1.0, sample_length=500)

    assert from_one_trial > 0.95, "a clean single-test result should pass"
    assert from_many < 0.05, "the same Sharpe as best-of-1000 should be rejected"


def test_dsr_is_monotone_in_the_observed_sharpe():
    kw = dict(n_trials=500, trial_sharpe_std=1.0, sample_length=1000)
    assert (
        deflated_sharpe_ratio(1.0, **kw)
        < deflated_sharpe_ratio(3.0, **kw)
        < deflated_sharpe_ratio(6.0, **kw)
    )


def test_dsr_rises_with_a_longer_sample():
    """More observations means more confidence in the same Sharpe."""
    kw = dict(observed_sharpe=4.0, n_trials=1000, trial_sharpe_std=1.0)
    assert deflated_sharpe_ratio(sample_length=100, **kw) < deflated_sharpe_ratio(
        sample_length=5000, **kw
    )


def test_negative_skew_and_fat_tails_reduce_confidence():
    """Two strategies with identical Sharpe are not equally trustworthy: the one
    that earns it via a fat left tail deserves less confidence."""
    kw = dict(observed_sharpe=3.5, n_trials=200, trial_sharpe_std=1.0, sample_length=1000)
    clean = deflated_sharpe_ratio(skew=0.0, kurtosis=3.0, **kw)
    ugly = deflated_sharpe_ratio(skew=-1.5, kurtosis=9.0, **kw)
    assert ugly < clean


def test_kurtosis_is_the_raw_fourth_moment_not_excess():
    """Documented explicitly because getting it wrong silently shifts the answer.
    Normal = 3.0 is the no-op; passing 0.0 (excess) must be rejected as invalid
    rather than quietly interpreted."""
    ok = deflated_sharpe_ratio(2.0, 100, 1.0, 500, kurtosis=3.0)
    assert 0.0 <= ok <= 1.0
    with pytest.raises(ValueError, match="RAW fourth moment"):
        deflated_sharpe_ratio(2.0, 100, 1.0, 500, kurtosis=0.0)


# -------------------------------------------------------------------- PBO
def _noise_configs(n_rows=320, n_cols=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 1, (n_rows, n_cols)))


def test_pbo_is_about_one_half_for_a_realistically_large_noise_search():
    """200 noise configs -- the realistic scale for this project's sweeps.

    With nothing but noise, the in-sample winner is a coin flip out of sample,
    so PBO lands at ~0.5. Measured: 0.50.
    """
    res = probability_of_backtest_overfitting(_noise_configs(), s=8)
    assert res["pbo"] == pytest.approx(0.5, abs=0.12)
    assert res["n_combinations"] == 70  # C(8, 4)
    assert res["n_configs"] == 200


def test_pbo_is_low_when_one_config_genuinely_dominates():
    """A real, persistent edge must be recognised as real."""
    perf = _noise_configs(seed=1)
    perf[0] = perf[0] + 3.0  # column 0 has a large constant edge in every slice
    res = probability_of_backtest_overfitting(perf, s=8)
    assert res["pbo"] < 0.2


def test_pbo_separates_a_real_edge_from_noise():
    noise = probability_of_backtest_overfitting(_noise_configs(seed=2), s=8)["pbo"]
    real = _noise_configs(seed=2)
    real[0] = real[0] + 3.0
    assert probability_of_backtest_overfitting(real, s=8)["pbo"] < noise


def test_pbo_rises_as_the_search_widens():
    """THE property that makes PBO the right metric for a 5,000-config sweep.

    Selection fragility grows with the number of things tried. With a handful of
    candidates the in-sample winner really is usually the best one; at scale it
    usually is not. Measured on identical noise: 50 configs -> ~0.27,
    200 configs -> ~0.50.

    Consequence worth remembering when reading results: a PBO from a small sweep
    is NOT comparable to one from a large sweep.
    """
    narrow = probability_of_backtest_overfitting(_noise_configs(n_cols=50, seed=0), s=8)["pbo"]
    wide = probability_of_backtest_overfitting(_noise_configs(n_cols=200, seed=0), s=8)["pbo"]
    assert wide > narrow


def test_metric_choice_is_validated_and_both_options_work():
    perf = _noise_configs(seed=3)
    for metric in ("sharpe", "mean"):
        res = probability_of_backtest_overfitting(perf, s=8, metric=metric)
        assert 0.0 <= res["pbo"] <= 1.0
        assert res["metric"] == metric
    with pytest.raises(ValueError, match="metric must be"):
        probability_of_backtest_overfitting(perf, s=8, metric="calmar")


def test_pbo_validates_its_arguments():
    perf = _noise_configs()
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(perf, s=7)
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(perf, s=2)
    with pytest.raises(ValueError, match="at least 2 configurations"):
        probability_of_backtest_overfitting(perf.iloc[:, :1], s=8)
    with pytest.raises(ValueError, match="at least s="):
        probability_of_backtest_overfitting(perf.iloc[:4], s=8)


# ---------------------------------------------------------------- haircut
def test_haircut_is_negative_when_a_result_is_worse_than_noise():
    """A Sharpe of 2.0 as best-of-1000 is BELOW what noise alone would produce."""
    assert haircut_sharpe(2.0, 1000, 1.0) < 0
    assert haircut_sharpe(5.0, 1000, 1.0) > 0
