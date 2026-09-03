"""Tests for the promotion gate.

The property that matters most: INCONCLUSIVE is a distinct answer from FAIL,
and it must win whenever there is not enough usable evidence to rule at all --
"we don't know" must never be silently reported as "no."
"""

from __future__ import annotations

import pytest

from stocksense.evaluation.gate import GATE, evaluate_gate


def _folds(n_positive: int, n_negative: int, value_pos=0.01, value_neg=-0.01) -> list[float]:
    return [value_pos] * n_positive + [value_neg] * n_negative


# ------------------------------------------------------------------------ PASS
def test_clearly_positive_folds_pass():
    folds = _folds(18, 2)  # 20 folds, 18 positive
    result = evaluate_gate(folds)
    assert result.verdict == "PASS"
    assert result.n_positive == 18
    assert result.mean_alpha_net > 0


def test_pass_requires_both_mean_alpha_and_significance():
    """A high positive-fold count with a negative MEAN alpha (e.g. a few huge
    losses swamping many small wins) must not pass -- the gate checks both."""
    folds = [0.001] * 15 + [-10.0] * 5  # 15/20 positive folds, but mean is deeply negative
    result = evaluate_gate(folds)
    assert result.mean_alpha_net < 0
    assert result.verdict == "FAIL"


# ------------------------------------------------------------------------ FAIL
def test_mostly_negative_folds_fail():
    folds = _folds(2, 18)
    result = evaluate_gate(folds)
    assert result.verdict == "FAIL"


def test_a_coin_flip_split_fails_on_significance():
    """10/20 positive is exactly chance -- binomial p for 'greater' at k=n/2
    is ~0.5, far above the 0.05 bar, so this must FAIL, not PASS."""
    folds = _folds(10, 10)
    result = evaluate_gate(folds)
    assert result.binomial_p > GATE["max_binomial_p"]
    assert result.verdict == "FAIL"


# ------------------------------------------------------------------ INCONCLUSIVE
def test_too_few_folds_is_inconclusive_not_a_pass_or_fail():
    """Even 100% positive folds must not PASS with fewer than
    min_folds_required -- 'not enough evidence' is a different answer from
    'the evidence says yes.'"""
    folds = _folds(5, 0)  # all positive, but only 5 folds < min_folds_required=10
    result = evaluate_gate(folds)
    assert result.verdict == "INCONCLUSIVE"


def test_excessive_dropped_folds_is_inconclusive():
    """>15% of folds unusable (None) must be INCONCLUSIVE regardless of how
    the remaining folds look."""
    folds = [0.01] * 20 + [None] * 5  # 5/25 = 20% dropped > 15% ceiling
    result = evaluate_gate(folds)
    assert result.verdict == "INCONCLUSIVE"
    assert result.drop_fraction == pytest.approx(0.20)


def test_dropped_folds_within_tolerance_do_not_block_a_result():
    folds = [0.01] * 19 + [None] * 1  # 1/20 = 5% dropped, within 15%
    result = evaluate_gate(folds)
    assert result.verdict == "PASS"
    assert result.n_folds_used == 19


def test_none_folds_are_dropped_not_treated_as_zero():
    """A None fold is 'could not be scored', not 'scored at exactly zero' --
    treating it as zero would silently bias mean_alpha_net."""
    all_none = [None] * 20
    result = evaluate_gate(all_none)
    assert result.n_folds_used == 0
    assert result.verdict == "INCONCLUSIVE"


def test_empty_fold_list_is_inconclusive():
    result = evaluate_gate([])
    assert result.verdict == "INCONCLUSIVE"
    assert result.n_folds_attempted == 0


# ------------------------------------------------------------------------ result
def test_result_reports_every_field_honestly():
    folds = [0.02, 0.01, -0.01, None, 0.03] * 4  # 20 attempted, 4 dropped, rest scored
    result = evaluate_gate(folds)
    assert result.n_folds_attempted == 20
    assert result.n_folds_dropped == 4
    assert result.n_folds_used == 16
    assert result.drop_fraction == pytest.approx(0.20)


def test_gate_thresholds_are_the_frozen_defaults_unless_overridden():
    assert GATE["min_folds_required"] == 10
    assert GATE["min_mean_alpha_net"] == 0.0
    assert GATE["max_binomial_p"] == 0.05
    assert GATE["max_drop_fraction"] == 0.15


def test_a_softer_gate_can_only_be_passed_explicitly_for_testing():
    """Confirms evaluate_gate is parameterised on `gate`, not hardwired to the
    module constant -- required so a test can probe the function's logic
    without ever touching the frozen GATE dict itself."""
    folds = _folds(6, 4)  # 6/10 positive: passes a looser bar, not the real one
    loose = evaluate_gate(folds, gate=dict(GATE, min_mean_alpha_net=-1.0, max_binomial_p=0.5))
    real = evaluate_gate(folds)
    assert loose.verdict == "PASS"
    assert real.verdict != "PASS"
