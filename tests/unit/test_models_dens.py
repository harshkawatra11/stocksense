from __future__ import annotations

import numpy as np
import pytest

from stocksense.models.base import ModelSpec, build_model, mean_daily_rank_ic
from stocksense.models.dens import DensModel, loss_curve_slope_weights

N_DATES, N, F = 220, 120, 8


def _panel(seed=0, planted_ic=0.08):
    rng = np.random.default_rng(seed)
    n = N_DATES * N
    X = rng.normal(size=(n, F)).astype(np.float32)
    dates = np.repeat(np.arange(N_DATES), N)
    # per-date rank IC of x0 with y is about planted_ic / sqrt(1 + planted_ic^2) ~ planted_ic
    y = planted_ic * X[:, 0] + rng.normal(size=n)
    X[rng.random(X.shape) < 0.03] = np.nan
    return X, y, dates


def _small(**kw):
    base = dict(
        k=3, n_estimators=120, num_leaves=7, min_child_samples=50, n_snapshots=4, importance_rows=4000
    )
    base.update(kw)
    return DensModel(**base)


def _fit(m, X, y, d, seed=1, cut=140):
    tr, va = d < 100, (d >= 100) & (d < cut)
    info = m.fit(X[tr], y[tr], np.ones(tr.sum()), d[tr], X[va], y[va], d[va], seed=seed)
    return info, tr | va


def test_out_of_fold_ic_on_planted_signal():
    X, y, d = _panel()
    m = _small()
    info, used = _fit(m, X, y, d)
    assert info.best_iteration is not None and info.best_iteration >= 1
    m.refit_full(X[used], y[used], np.ones(used.sum()), d[used], info, seed=1)
    oof = d >= 140
    assert mean_daily_rank_ic(m.predict(X[oof]), y[oof], d[oof]) > 0.04


def test_deterministic_with_same_seed():
    X, y, d = _panel()
    outs = []
    for _ in range(2):
        m = _small()
        info, used = _fit(m, X, y, d, seed=7)
        m.refit_full(X[used], y[used], np.ones(used.sum()), d[used], info, seed=7)
        outs.append(m.predict(X))
    assert np.array_equal(outs[0], outs[1])


def test_save_load_round_trip(tmp_path):
    X, y, d = _panel()
    m = _small()
    info, used = _fit(m, X, y, d, seed=2)
    m.refit_full(X[used], y[used], np.ones(used.sum()), d[used], info, seed=2)
    p = m.predict(X)
    m.save(tmp_path)
    m2 = DensModel()
    m2.load(tmp_path)
    assert np.array_equal(m2.predict(X), p)
    assert m.explain(X[:5], list("abcdefgh")) is None


def test_sub_models_use_shrinking_feature_sets():
    X, y, d = _panel()
    m = _small(k=4, ratios=(0.8, 0.5, 0.25))
    info, _ = _fit(m, X, y, d)
    sizes = [len(f) for f in info.params["features"]]
    assert sizes[0] == F and sizes == sorted(sizes, reverse=True) and sizes[-1] < F


def test_registered_under_family_key():
    m = build_model(ModelSpec("s1-dens", "y_c2o_10", params={"k": 4, "decay": 0.3, "num_leaves": 15}))
    assert isinstance(m, DensModel) and m.k == 4 and m.decay == 0.3 and m.num_leaves == 15


def test_reweighting_favors_low_loss_curve_slope_samples():
    # four samples, three snapshots. Loss ranks over snapshots (higher = worse):
    #  0: stays low and easy, 1: improves a lot, 2: stays high, 3: gets worse.
    curve = np.array(
        [
            [0.1, 0.1, 0.1],
            [4.0, 2.0, 0.5],
            [5.0, 5.0, 5.0],
            [1.0, 3.0, 9.0],
        ]
    )
    final_loss = curve[:, -1]
    w = loss_curve_slope_weights(curve, final_loss, k=1, decay=0.5, bins=4)
    assert w.shape == (4,) and np.all(w > 0) and w.mean() == pytest.approx(1.0)
    # a falling curve (sample 1) is favored over a rising one (sample 3) and over a stuck-high one (2)
    assert w[1] > w[3] and w[1] > w[2]
    # the sample whose loss rank rises and ends worst gets the smallest weight
    assert w[3] == w.min()


def test_first_submodel_weights_are_uniform_when_k_is_zero():
    curve = np.random.default_rng(0).random((50, 5))
    w = loss_curve_slope_weights(curve, curve[:, -1], k=0, decay=0.5, bins=10)
    assert np.allclose(w, 1.0)
