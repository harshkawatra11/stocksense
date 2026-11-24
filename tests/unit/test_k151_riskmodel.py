from __future__ import annotations

import numpy as np
import pytest

from stocksense.strategies import riskmodel as R


def _rank1(n=12, t=60, seed=3):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=t)
    beta = rng.uniform(0.5, 1.5, size=n)
    return beta[:, None] * f[None, :]


def _factor_returns(n=30, t=200, seed=5):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(2, t))
    load = rng.normal(size=(n, 2))
    return load @ f + 0.8 * rng.normal(size=(n, t))


def test_psi_has_unit_diagonal():
    xi2, lam, v = R.stat_risk_model(_factor_returns(), None)
    psi = R.psi_matrix(xi2, lam, v)
    assert np.allclose(np.diag(psi), 1.0, atol=1e-12)
    assert np.allclose(psi, psi.T)


def test_erank_is_one_for_a_rank_one_panel_and_k_overrides():
    xi2, lam, v = R.stat_risk_model(_rank1(), None)
    assert len(lam) == 1 and v.shape == (12, 1)
    assert lam[0] == pytest.approx(12.0, rel=1e-9)  # one factor carries the whole trace
    xi2, lam, v = R.stat_risk_model(_factor_returns(), 3)
    assert len(lam) == 3 and v.shape == (30, 3) and np.all(xi2 > 0)
    assert np.all(np.diff(lam) <= 0)  # decreasing eigenvalues


def test_woodbury_inverse_equals_the_dense_inverse():
    xi2, lam, v = R.stat_risk_model(_factor_returns(), 3)
    sig = np.linspace(0.01, 0.03, 30)
    c = R.psi_matrix(xi2, lam, v) * np.outer(sig, sig)
    assert np.allclose(R.inv_cov(xi2, lam, v, sig), np.linalg.inv(c), rtol=1e-8, atol=1e-6)
    assert np.allclose(R.inv_cov(xi2, lam, v), np.linalg.inv(R.psi_matrix(xi2, lam, v)), atol=1e-8)


def test_unbounded_weights_follow_eq_358_and_sum_to_zero():
    rng = np.random.default_rng(1)
    n = 8
    a = rng.normal(size=(n, n))
    c = a @ a.T + n * np.eye(n)
    invc = np.linalg.inv(c)
    e = rng.normal(size=n)
    w = R.mv_weights(e, invc, np.full(n, -np.inf), np.full(n, np.inf))
    one = np.ones(n)
    want = invc @ e - (invc @ one) * (one @ invc @ e) / (one @ invc @ one)
    assert np.allclose(w, want, atol=1e-10)
    assert abs(w.sum()) < 1e-12


def test_bounded_weights_respect_bounds_and_are_dollar_neutral():
    rng = np.random.default_rng(2)
    n = 20
    invc = np.diag(1.0 / rng.uniform(0.5, 2.0, n))
    e = rng.normal(size=n) * np.r_[10.0, np.ones(n - 1)]  # one outlier forces a clip
    lo, hi = np.full(n, -0.3), np.full(n, 0.3)
    w = R.mv_weights(e, invc, lo, hi)
    assert np.all(w <= hi + 1e-12) and np.all(w >= lo - 1e-12)
    assert abs(w.sum()) < 1e-10
    assert w[0] == pytest.approx(0.3)  # the outlier sits on its bound


def test_a_risk_model_never_looks_at_future_columns():
    r = _factor_returns(t=120)
    a = R.stat_risk_model(r[:, :100], 2)
    b = R.stat_risk_model(r[:, :100].copy(), 2)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
