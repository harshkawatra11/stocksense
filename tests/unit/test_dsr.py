from __future__ import annotations

import math

import pytest

from stocksense.validation.dsr import deflated_sharpe


def test_hand_example():
    # sr = 0.08, T = 2500, skew = -0.3, kurt = 5, N = 50, V = 0.0004
    # Phi^-1(0.98) = 2.0537, Phi^-1(1 - 1/(50e)) = Phi^-1(0.99264) = 2.4398
    # SR0 = 0.02 * (0.4228 * 2.0537 + 0.5772 * 2.4398) = 0.02 * 2.2766 = 0.045532
    # denominator = sqrt(1 + 0.024 + 0.0064) = 1.01509; z = (0.08 - 0.045532) * 49.99 / 1.01509 = 1.6975
    # DSR = Phi(1.6975) = 0.9552
    d = deflated_sharpe(0.08, 2500, -0.3, 5.0, 50, 0.0004)
    assert d == pytest.approx(0.9552, abs=0.002)


def test_decreases_with_more_trials():
    vals = [deflated_sharpe(0.08, 2500, -0.3, 5.0, n, 0.0004) for n in (2, 10, 50, 500, 5000)]
    assert vals == sorted(vals, reverse=True) and len(set(vals)) == 5


def test_n_one_behaves_as_two():
    a = deflated_sharpe(0.08, 2500, -0.3, 5.0, 1, 0.0004)
    b = deflated_sharpe(0.08, 2500, -0.3, 5.0, 2, 0.0004)
    assert a == b


def test_invalid_variance_falls_back_to_one_over_t():
    # V = 1/T = 0.0004 for T = 2500, so nan / 0 / negative variance equal the explicit 0.0004
    ref = deflated_sharpe(0.08, 2500, -0.3, 5.0, 50, 1 / 2500)
    for bad in (float("nan"), 0.0, -1.0, float("inf")):
        assert deflated_sharpe(0.08, 2500, -0.3, 5.0, 50, bad) == pytest.approx(ref)


def test_zero_edge_is_below_half_and_probability_range():
    d = deflated_sharpe(0.0, 2500, 0.0, 3.0, 50, 0.0004)
    assert 0.0 <= d < 0.5
    assert not math.isnan(d)
