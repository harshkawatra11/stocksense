from __future__ import annotations

import numpy as np
import pytest

from stocksense.models.base import ModelSpec, build_model, mean_daily_rank_ic
from stocksense.models.gbm import GRID, GbmModel, lambdarank_groups, make_mean_ic_feval, per_date_slices


def _panel(n_dates=60, n=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_dates * n, 4)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan
    dates = np.repeat(np.arange(n_dates), n)
    y = 0.5 * np.nan_to_num(X[:, 0]) + rng.normal(size=n_dates * n)
    return X, y, dates


def test_grid_has_twelve_configs():
    assert len(GRID) == 12
    assert {g["num_leaves"] for g in GRID} == {15, 31, 63}


def test_custom_metric_equals_direct_per_date_spearman():
    rng = np.random.default_rng(1)
    d = np.repeat(np.arange(5), 20)
    y = rng.normal(size=100)
    p = y * 0.3 + rng.normal(size=100)
    name, val, hib = make_mean_ic_feval(d, y)(y * 0 + 1.0, p)
    assert name == "mean_ic" and hib is True
    assert val == pytest.approx(mean_daily_rank_ic(p, y, d))
    assert len(per_date_slices(d)) == 5


def test_lambdarank_groups_sum_to_rows():
    X, y, d = _panel()
    groups = lambdarank_groups(d)
    assert groups.sum() == len(d) and len(groups) == 60
    m = GbmModel(objective="lambdarank", n_estimators=30, num_leaves=7, min_child_samples=20)
    info = m.fit(X[:2000], y[:2000], np.ones(2000), d[:2000], X[2000:], y[2000:], d[2000:], seed=1)
    assert info.best_iteration >= 1
    assert np.isfinite(m.predict(X)).all()


def test_regression_two_step_fit_and_explain_additivity():
    X, y, d = _panel()
    m = GbmModel(n_estimators=200, num_leaves=7, min_child_samples=20)
    info = m.fit(X[:2000], y[:2000], np.ones(2000), d[:2000], X[2000:], y[2000:], d[2000:], seed=3)
    assert info.val_ic > 0.1
    m.refit_full(X, y, np.ones(len(y)), d, info, seed=3)
    assert m._est.n_estimators == int(np.ceil(1.1 * info.best_iteration))
    c = m.explain(X[:200], list("abcd"))
    assert c.shape == (200, 4)
    bias = m._booster.predict(X[:200], pred_contrib=True)[:, -1]
    assert np.abs(c.sum(1) + bias - m.predict(X[:200])).max() < 1e-5


def test_determinism_and_save_load(tmp_path):
    X, y, d = _panel()
    outs = []
    for _ in range(2):
        m = GbmModel(n_estimators=40, num_leaves=7, min_child_samples=20)
        info = m.fit(X[:2000], y[:2000], np.ones(2000), d[:2000], X[2000:], y[2000:], d[2000:], seed=5)
        m.refit_full(X, y, np.ones(len(y)), d, info, seed=5)
        outs.append(m.predict(X))
    assert np.array_equal(outs[0], outs[1])
    m.save(tmp_path)
    m2 = GbmModel()
    m2.load(tmp_path)
    assert np.array_equal(m2.predict(X), outs[0])


def test_registered_under_family_key():
    m = build_model(ModelSpec("s1-gbm", "y_c2o_5", params={"objective": "lambdarank", "num_leaves": 15}))
    assert isinstance(m, GbmModel) and m.objective == "lambdarank" and m.num_leaves == 15
