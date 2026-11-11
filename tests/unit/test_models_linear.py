from __future__ import annotations

import numpy as np
import pytest

from stocksense.models.base import FitInfo, ModelSpec, build_model, mean_daily_rank_ic
from stocksense.models.linear import ALPHAS, RidgeModel, SingleFactor


def _panel(n_dates=40, n=60, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-0.5, 0.5, size=(n_dates * n, 3)).astype(np.float32)
    dates = np.repeat(np.arange(n_dates), n)
    y = 0.8 * X[:, 0] + noise * 0.3 * rng.normal(size=n_dates * n)
    return X, y, dates


def test_ridge_matches_the_closed_form_weighted_solution():
    X, y, d = _panel()
    w = np.linspace(0.5, 1.0, len(y))
    m = RidgeModel()
    m.configure(["a", "b", "c"], ["xs"] * 3)
    m.refit_full(X, y, w, d, FitInfo(params={"alpha": 10.0}), 0)
    Xd = X.astype(float)
    expected = np.linalg.solve(Xd.T @ (Xd * w[:, None]) + 10.0 * np.eye(3), Xd.T @ (w * y))
    assert np.allclose(m._coef, expected)  # type: ignore[arg-type]
    assert np.allclose(m.predict(X), Xd @ expected)


def test_nan_counts_as_zero_and_mkt_features_are_excluded():
    X, y, d = _panel()
    m = RidgeModel()
    m.configure(["a", "b", "mkt"], ["xs", "xs", "mkt"])
    m.refit_full(X, y, np.ones(len(y)), d, FitInfo(params={"alpha": 1.0}), 0)
    assert m._coef[2] == 0.0  # type: ignore[index]
    X2 = X.copy()
    X2[:, 2] = 99.0  # the market column cannot move a score
    assert np.array_equal(m.predict(X), m.predict(X2))
    X3 = X.copy()
    X3[::2, 0] = np.nan
    X4 = X.copy()
    X4[::2, 0] = 0.0
    assert np.array_equal(m.predict(X3), m.predict(X4))
    assert np.all(np.isfinite(m.predict(X3)))


def test_alpha_is_chosen_by_inner_validation_ic():
    X, y, d = _panel(n_dates=60, noise=0.5)
    cut = 40 * 60
    m = RidgeModel()
    m.configure(["a", "b", "c"], ["xs"] * 3)
    info = m.fit(X[:cut], y[:cut], np.ones(cut), d[:cut], X[cut:], y[cut:], d[cut:], seed=0)
    by_alpha = info.params["ic_by_alpha"]
    assert set(by_alpha) == set(ALPHAS)
    assert info.params["alpha"] == max(by_alpha, key=lambda a: by_alpha[a])
    assert info.val_ic == pytest.approx(max(by_alpha.values()))
    assert info.val_ic > 0.3


def test_heavy_shrinkage_is_picked_when_the_signal_is_pure_noise():
    rng = np.random.default_rng(3)
    X = rng.uniform(-0.5, 0.5, size=(3000, 3)).astype(np.float32)
    y = rng.normal(size=3000)
    d = np.repeat(np.arange(50), 60)
    m = RidgeModel()
    m.configure(["a", "b", "c"], ["xs"] * 3)
    info = m.fit(X[:2000], y[:2000], np.ones(2000), d[:2000], X[2000:], y[2000:], d[2000:])
    assert abs(info.val_ic or 0.0) < 0.1


def test_explain_is_coef_times_x_and_sums_to_the_score():
    X, y, d = _panel()
    m = RidgeModel()
    m.configure(["a", "b", "c"], ["xs", "xs", "mkt"])
    m.refit_full(X, y, np.ones(len(y)), d, FitInfo(params={"alpha": 1.0}), 0)
    e = m.explain(X[:5], ["a", "b", "c"])
    assert e is not None and e.shape == (5, 3)
    assert np.allclose(e, np.nan_to_num(X[:5].astype(float)) * m._coef * np.array([1, 1, 0]))  # type: ignore[operator]
    assert np.allclose(e.sum(axis=1), m.predict(X[:5]))


def test_ridge_save_load_round_trip(tmp_path):
    X, y, d = _panel()
    m = RidgeModel()
    m.configure(["a", "b", "c"], ["xs", "mkt", "xs"])
    m.refit_full(X, y, np.ones(len(y)), d, FitInfo(params={"alpha": 100.0}), 0)
    m.save(tmp_path)
    m2 = RidgeModel()
    m2.load(tmp_path)
    assert np.array_equal(m.predict(X), m2.predict(X)) and m2.alpha == 100.0


def test_single_factor_is_sign_times_feature():
    X, y, d = _panel()
    f = SingleFactor("b", -1.0)
    f.configure(["a", "b", "c"], ["xs"] * 3)
    info = f.fit(X, y, np.ones(len(y)), d)
    f.refit_full(X, y, np.ones(len(y)), d, info, 0)
    assert np.array_equal(f.predict(X), -X[:, 1].astype(float))
    assert np.array_equal(f.explain(X[:3], ["a", "b", "c"])[:, 1], f.predict(X[:3]))  # type: ignore[index]
    with pytest.raises(KeyError):
        SingleFactor("zzz").configure(["a"], ["xs"])


def test_single_factor_save_load(tmp_path):
    f = SingleFactor("vol_21", -1.0)
    f.configure(["ret_1", "vol_21"], ["xs", "xs"])
    f.save(tmp_path)
    g = SingleFactor("x")
    g.load(tmp_path)
    X = np.array([[1.0, 2.0]], dtype=np.float32)
    assert g.predict(X)[0] == -2.0


def test_families_resolve_from_specs():
    assert isinstance(build_model(ModelSpec("s1-ridge", "y_c2o_5")), RidgeModel)
    assert isinstance(build_model(ModelSpec("ridge", "y_oc")), RidgeModel)
    sf = build_model(ModelSpec("s1-single", "y_oc", params={"feature": "vol_21", "sign": -1}))
    assert isinstance(sf, SingleFactor) and sf.sign == -1.0
    with pytest.raises(KeyError):
        build_model(ModelSpec("s1-nothing", "y_oc"))


def test_mean_daily_rank_ic_hand_example():
    # day 0: perfectly ordered (+1); day 1: perfectly reversed (-1); day 2 has too few pairs and is skipped
    pred = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2.0])
    y = np.array([10, 20, 30, 40, 4, 3, 2, 1, 5, 6.0])
    d = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    assert mean_daily_rank_ic(pred, y, d, min_pairs=3) == pytest.approx(0.0)
    assert mean_daily_rank_ic(pred[:4], y[:4], d[:4], min_pairs=3) == pytest.approx(1.0)
    assert np.isnan(mean_daily_rank_ic(pred, y, d, min_pairs=10))


def test_floor_specs_cover_the_card():
    from stocksense.models.linear import floor_specs

    specs = floor_specs()
    assert len(specs) == 7 * 4 + 4
    singles = [s for s in specs if s.family == "floor-single"]
    assert {s.params["feature"] for s in singles} == {
        "vol_21", "co_sum_5", "deliv_z", "ret_1", "mom_5", "mom_252_skip21", "mom_126",
    }  # fmt: skip
    assert {s.params["feature"]: s.params["sign"] for s in singles}["vol_21"] == -1.0
    assert {s.target for s in specs} == {"y_c2o_5", "y_c2o_10", "y_c2o_20", "y_oc"}
    assert all(s.entry == ("open" if s.target == "y_oc" else "close") for s in specs)
    assert all(isinstance(build_model(s), (RidgeModel, SingleFactor)) for s in specs)
    assert len({(s.family, s.target, str(s.params)) for s in specs}) == len(specs)
