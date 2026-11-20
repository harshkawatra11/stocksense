"""G2-E4: filtered (causal) regime probabilities and Bayesian online change-point detection."""

from __future__ import annotations

import numpy as np
import pytest

from stocksense.features import regime as rg
from stocksense.features.registry import FEATURE_MODULES
from stocksense.portfolio.construct import PanelDay, S1Config, plan_s1
from tests.unit.test_feature_truncation import (  # noqa: F401  (the session fixture builds the synthetic market)
    _compute,
    truncation_env,
    violations,
)

# ---- the forward (filtering) recursion -----------------------------------------------------------------


def test_forward_filter_matches_a_hand_computation():
    pi = np.array([0.6, 0.4])
    a = np.array([[0.7, 0.3], [0.4, 0.6]])
    lik = np.array([[0.9, 0.2], [0.1, 0.8], [0.5, 0.5], [0.3, 0.7]])  # P(obs_t | state)
    got = rg.forward_filter(pi, a, lik)
    # t0: alpha0 = pi * lik0 = [0.54, 0.08] -> [0.870967.., 0.129032..]
    a0 = np.array([0.54, 0.08])
    a0 /= a0.sum()
    # t1: predict a0 @ A, times lik1
    a1 = (a0 @ a) * lik[1]
    a1 /= a1.sum()
    a2 = (a1 @ a) * lik[2]
    a2 /= a2.sum()
    a3 = (a2 @ a) * lik[3]
    a3 /= a3.sum()
    assert a0 == pytest.approx([0.54 / 0.62, 0.08 / 0.62])
    assert got.shape == (4, 2)
    for row, want in zip(got, (a0, a1, a2, a3), strict=True):
        assert row == pytest.approx(want)
    # explicit numbers for t1: predicted = [0.870968*0.7+0.129032*0.4, 0.870968*0.3+0.129032*0.6]
    pred = np.array([0.870968 * 0.7 + 0.129032 * 0.4, 0.870968 * 0.3 + 0.129032 * 0.6])
    post = pred * np.array([0.1, 0.8])
    assert got[1] == pytest.approx(post / post.sum(), abs=1e-5)


def test_appending_later_observations_never_changes_earlier_filtered_probabilities():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(-1, 0.5, (80, 2)), rng.normal(1, 0.5, (80, 2))])
    params = rg.fit_hmm(x, n_states=2, seed=1)
    short = rg.filter_probs(x[:100], params)
    longer = rg.filter_probs(x, params)
    assert short.shape == (100, 2)
    assert np.array_equal(short, longer[:100])  # bit-identical, not just close


def test_the_filter_differs_from_the_smoother_on_the_same_parameters():
    """Guard that the module really filters: a smoother would use the future and change early rows."""
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(-1, 0.5, (60, 1)), rng.normal(1, 0.5, (60, 1))])
    params = rg.fit_hmm(x, n_states=2, seed=2)
    filt = rg.filter_probs(x, params)
    smooth = rg.smooth_probs(x, params)
    assert not np.allclose(filt, smooth)


def test_nan_observations_carry_the_state_forward_without_an_update():
    pi = np.array([0.5, 0.5])
    a = np.array([[0.9, 0.1], [0.2, 0.8]])
    lik = np.array([[0.8, 0.1], [1.0, 1.0], [0.8, 0.1]])  # row 1: missing observation (likelihood 1)
    p = rg.forward_filter(pi, a, lik)
    assert p[1] == pytest.approx(p[0] @ a)


def test_fit_hmm_recovers_two_separated_regimes_and_orders_states_by_mean_return():
    rng = np.random.default_rng(5)
    seg = []
    for _ in range(6):
        seg.append(rng.normal(1.0, 0.4, (60, 2)))
        seg.append(rng.normal(-1.0, 0.4, (60, 2)))
    x = np.vstack(seg)
    p = rg.fit_hmm(x, n_states=2, seed=1)
    assert p.mean[0, 0] < p.mean[1, 0]  # state 0 = lowest mean of the first column (the return)
    assert p.mean[0, 0] == pytest.approx(-1.0, abs=0.2) and p.mean[1, 0] == pytest.approx(1.0, abs=0.2)
    assert np.allclose(p.trans.sum(1), 1.0)
    assert p.trans[0, 0] > 0.8 and p.trans[1, 1] > 0.8  # regimes are persistent


def test_expanding_standardisation_is_causal():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(200, 3))
    full = rg.expanding_standardise(x, min_rows=30)
    cut = rg.expanding_standardise(x[:120], min_rows=30)
    assert np.array_equal(full[:120], cut, equal_nan=True)
    assert np.isnan(full[:29]).all() and np.isfinite(full[30:]).all()


# ---- BOCPD ---------------------------------------------------------------------------------------------


def test_bocpd_detects_a_simulated_mean_shift_within_ten_observations():
    rng = np.random.default_rng(7)
    n0 = 250
    x = np.concatenate([rng.normal(0.0, 1.0, n0), rng.normal(4.0, 1.0, 60)])
    cp, rl = rg.bocpd(x, hazard=1 / 250, scale=1.0)
    quiet = cp[60:n0]
    assert quiet.max() < 0.35  # no false alarm after the warm-up
    assert cp[n0 : n0 + 10].max() > 0.5  # detected within 10 observations of the shift
    assert rl[n0 + 9] < rl[n0 - 1] / 3  # and the run-length mean collapsed
    assert rl[-1] > 30  # then it grows again as the new regime persists


def test_bocpd_is_causal():
    rng = np.random.default_rng(8)
    x = rng.normal(0.0, 1.0, 300)
    cp1, rl1 = rg.bocpd(x[:200], hazard=1 / 250, scale=1.0)
    cp2, rl2 = rg.bocpd(x, hazard=1 / 250, scale=1.0)
    assert np.array_equal(cp1, cp2[:200]) and np.array_equal(rl1, rl2[:200])


# ---- registry and truncation -----------------------------------------------------------------------------


def test_the_features_are_not_in_the_default_build():
    assert "regime" not in FEATURE_MODULES
    assert {"regime_state", "bocpd"} <= {s.name for s in rg.SPECS}
    assert rg.OUTPUTS == ("regime_p_bear", "regime_p_bull", "cp_prob", "run_length_mean")


@pytest.fixture
def small_thresholds(monkeypatch):
    monkeypatch.setattr(rg, "MIN_FIT_ROWS", 120)
    monkeypatch.setattr(rg, "STD_MIN_ROWS", 40)


@pytest.mark.parametrize("name", ["regime_state", "bocpd"])
def test_feature_does_not_look_ahead(name, truncation_env, small_thresholds):  # noqa: F811
    s, sessions, _ = truncation_env
    spec = next(x for x in rg.SPECS if x.name == name)
    full = _compute(spec, s, None)
    assert full.height > 0
    non_null = {o: full[o].drop_nulls().len() for o in spec.outputs}
    assert all(v > 0 for v in non_null.values()), f"the guard would be vacuous: {non_null}"
    for idx in (300, 380, 450, 520, 590):
        cut = sessions[idx]
        bad = violations(spec, full, _compute(spec, s, cut), cut)
        assert not bad, f"{name} @ {idx} ({cut}): " + "; ".join(bad)


def test_regime_params_are_refit_each_january_on_data_before_it(truncation_env, small_thresholds):  # noqa: F811
    """The first probability is dated in the first January that has enough earlier rows."""
    s, sessions, _ = truncation_env
    spec = next(x for x in rg.SPECS if x.name == "regime_state")
    full = _compute(spec, s, None)
    first = full.filter(full["regime_p_bear"].is_not_null())["date"].min()
    assert first.month == 1 and first.day <= 7
    assert first > sessions[0]


# ---- the S1 overlay ------------------------------------------------------------------------------------


def _gross(p_bear, **opts) -> float:
    from tests.unit import test_s1_policy_v2 as v2

    sc, er, rows = v2.world()
    pd_ = PanelDay(v2.T, v2.D1, rows, p_bear=p_bear)
    c = v2.cfg(**opts)
    return plan_s1(v2.T, sc, er, v2.book(), v2.EQUITY, pd_, c).gross_target


def test_regime_overlay_halves_gross_only_above_the_threshold():
    base = _gross(None)
    assert _gross(0.3, regime_overlay=True) == pytest.approx(base)
    assert _gross(0.6, regime_overlay=True) == pytest.approx(base)  # strictly above 0.6
    assert _gross(0.7, regime_overlay=True) == pytest.approx(0.5 * base)
    assert _gross(0.9) == pytest.approx(base)  # option off (the default): p_bear is ignored
    assert _gross(None, regime_overlay=True) == pytest.approx(base)  # no estimate: no overlay
    assert S1Config.__dataclass_fields__["regime_overlay"].default is False
