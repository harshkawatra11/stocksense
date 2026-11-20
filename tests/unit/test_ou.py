"""G2-E5: Ornstein-Uhlenbeck mean reversion features (s-score, half-life, pair z, residual reversal)."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.features import ou
from stocksense.features.registry import FEATURE_MODULES
from tests.unit.test_feature_truncation import (  # noqa: F401  (the session fixture builds the synthetic market)
    _compute,
    truncation_env,
    violations,
)


def _simulate_ou(kappa: float, sigma: float, n: int, seed: int, mean: float = 0.0) -> np.ndarray:
    dt = 1.0 / 252.0
    b = math.exp(-kappa * dt)
    sd = sigma * math.sqrt((1.0 - b * b) / (2.0 * kappa))
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = mean
    for k in range(n - 1):
        x[k + 1] = mean + b * (x[k] - mean) + sd * rng.normal()
    return x


# ---- the AR(1) / OU fit --------------------------------------------------------------------------------


def test_kappa_is_recovered_from_a_simulated_ou_path():
    kappa, sigma, mean = 20.0, 0.3, 0.05
    x = _simulate_ou(kappa, sigma, 6000, seed=3, mean=mean)
    fit = ou.fit_ar1(x)
    assert fit is not None
    assert fit.kappa == pytest.approx(kappa, rel=0.25)
    assert fit.m == pytest.approx(mean, abs=0.03)
    assert fit.sigma_eq == pytest.approx(sigma / math.sqrt(2 * kappa), rel=0.1)
    assert fit.halflife == pytest.approx(math.log(2) / (kappa / 252.0), rel=0.25)  # about 8.7 sessions


def test_a_random_walk_has_no_ou_fit_or_a_long_half_life():
    x = np.cumsum(np.random.default_rng(1).normal(size=3000))
    fit = ou.fit_ar1(x)
    assert fit is None or fit.halflife > 100  # b >= 1 gives None; b just under 1 a very long half-life


def test_sscore_and_halflife_by_hand():
    # X_{k+1} = 0.9 X_k + eps: b = 0.9, half-life ln 2 / -ln 0.9 = 6.58 sessions
    assert ou.halflife_of(0.9) == pytest.approx(math.log(2) / -math.log(0.9))
    assert ou.halflife_of(1.0) is None and ou.halflife_of(-0.2) is None
    # a = 0.02, b = 0.8 -> m = 0.1; eps variance 0.0036 -> sigma_eq = sqrt(0.0036 / (1 - 0.64)) = 0.1
    s = ou.sscore(x_last=0.3, a=0.02, b=0.8, var_eps=0.0036)
    assert s == pytest.approx((0.3 - 0.1) / 0.1)
    # half-life filter: b = 0.98 has half-life 34 sessions > 30 -> null
    assert ou.sscore(x_last=0.3, a=0.02, b=0.98, var_eps=0.0036) is None
    assert ou.sscore(x_last=0.3, a=0.02, b=1.01, var_eps=0.0036) is None  # explosive
    assert ou.sscore(x_last=0.3, a=0.02, b=-0.5, var_eps=0.0036) is None


def test_rolling_window_fit_equals_the_scalar_fit_on_each_window():
    rng = np.random.default_rng(2)
    e = rng.normal(0.0, 0.02, 200)
    s, hl = ou.rolling_ou(e, window=60)
    assert np.isnan(s[:59]).all() and np.isnan(hl[:59]).all()
    for t in (59, 100, 150, 199):
        x = np.cumsum(e[t - 59 : t + 1])
        fit = ou.fit_ar1(x)
        if fit is None or not (0 < fit.b < 1):
            assert np.isnan(hl[t])
            continue
        assert hl[t] == pytest.approx(fit.halflife)
        want = (x[-1] - fit.m) / fit.sigma_eq if fit.halflife <= ou.MAX_HALFLIFE else math.nan
        if math.isnan(want):
            assert np.isnan(s[t])
        else:
            assert s[t] == pytest.approx(want)


def test_a_nan_in_the_window_nulls_the_row():
    e = np.random.default_rng(5).normal(0.0, 0.02, 120)
    e[70] = np.nan
    s, hl = ou.rolling_ou(e, window=60)
    assert np.isnan(s[70:130]).all() and np.isnan(hl[70:130]).all()


# ---- residual reversal ---------------------------------------------------------------------------------


def test_resid_rev_5_is_minus_the_sum_of_the_last_five_residuals():
    df = pl.DataFrame(
        {
            "sid": [1] * 7,
            "date": [date(2020, 1, 1) + timedelta(days=i) for i in range(7)],
            "sx": list(range(7)),
            "e": [0.01, 0.02, -0.01, 0.03, 0.0, 0.02, -0.04],
        }
    )
    out = df.with_columns(ou.resid_rev_5_expr())
    assert out["resid_rev_5"][3] is None  # fewer than five rows
    assert out["resid_rev_5"][4] == pytest.approx(-(0.01 + 0.02 - 0.01 + 0.03 + 0.0))
    assert out["resid_rev_5"][6] == pytest.approx(-(-0.01 + 0.03 + 0.0 + 0.02 - 0.04))


# ---- Engle-Granger pairs -------------------------------------------------------------------------------


def _pair_world(n_days=500, seed=4):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 0.01, n_days)) + 4.0
    spread = np.zeros(n_days)
    for k in range(1, n_days):  # AR(1) b = 0.93: half-life 9.6 sessions
        spread[k] = 0.93 * spread[k - 1] + rng.normal(0, 0.01)
    y = 0.3 + 1.5 * x + spread
    z = np.cumsum(rng.normal(0, 0.01, n_days)) + 4.0  # unrelated random walk
    return np.column_stack([y, x, z])


def test_eg_pairs_finds_a_cointegrated_pair_and_not_an_unrelated_walk():
    lp = _pair_world()
    pairs = ou.eg_pairs(lp)
    ids = {(p.i, p.j) for p in pairs}
    assert (0, 1) in ids
    assert (0, 2) not in ids and (1, 2) not in ids
    p = next(p for p in pairs if (p.i, p.j) == (0, 1))
    assert p.beta == pytest.approx(1.5, abs=0.1) and p.pval < 0.05
    assert 5 <= p.halflife <= 30


def test_eg_pairs_applies_the_half_life_window():
    lp = _pair_world()
    assert ou.eg_pairs(lp, hl_min=20.0) == []  # the pair's half-life (about 8) is below the window


def test_pair_selection_ignores_data_after_the_cutoff():
    """The January selection reads only rows dated <= the previous 31 December."""
    rng = np.random.default_rng(9)
    lp = _pair_world(n_days=700)
    dates = [date(2018, 1, 1) + timedelta(days=i) for i in range(700)]
    rows = []
    for k, sid in enumerate((1, 2, 3)):
        for t, d in enumerate(dates):
            rows.append((sid, d, t, "IND", float(np.exp(lp[t, k])), 1e8 * (k + 1)))
    panel = pl.DataFrame(
        rows, schema=["sid", "date", "sx", "industry", "adj_close", "turnover_inr"], orient="row"
    )
    cutoff = dates[599]
    clean = ou.select_pairs(panel, cutoff, window=500, min_members=2)
    poisoned = panel.with_columns(
        pl.when(pl.col("date") > cutoff)
        .then(pl.lit(rng.uniform(1, 100)))
        .otherwise(pl.col("adj_close"))
        .alias("adj_close")
    )
    other = ou.select_pairs(poisoned, cutoff, window=500, min_members=2)
    assert clean.height > 0
    assert clean.equals(other)


# ---- registry and truncation -----------------------------------------------------------------------------


def test_the_features_are_not_in_the_default_build():
    assert "ou" not in FEATURE_MODULES
    assert {"ou_resid", "pair_z"} <= {s.name for s in ou.SPECS}
    assert ou.OUTPUTS == ("ou_sscore", "ou_halflife", "resid_rev_5", "pair_z")


@pytest.fixture
def small_thresholds(monkeypatch):
    monkeypatch.setattr(ou, "PAIR_WINDOW", 120)
    monkeypatch.setattr(ou, "PAIR_ADF_P", 0.5)  # a synthetic market of random walks has few real pairs
    monkeypatch.setattr(ou, "PAIR_HL_MIN", 1.0)
    monkeypatch.setattr(ou, "PAIR_HL_MAX", 200.0)
    monkeypatch.setattr(ou, "MIN_INDUSTRY_PAIR_MEMBERS", 3)


@pytest.mark.parametrize("name", ["ou_resid", "pair_z"])
def test_feature_does_not_look_ahead(name, truncation_env, small_thresholds):  # noqa: F811
    s, sessions, _ = truncation_env
    spec = next(x for x in ou.SPECS if x.name == name)
    full = _compute(spec, s, None)
    assert full.height > 0
    non_null = {o: full[o].drop_nulls().len() for o in spec.outputs}
    assert all(v > 0 for v in non_null.values()), f"the guard would be vacuous: {non_null}"
    for idx in (300, 380, 450, 520, 590):
        cut = sessions[idx]
        bad = violations(spec, full, _compute(spec, s, cut), cut)
        assert not bad, f"{name} @ {idx} ({cut}): " + "; ".join(bad)
