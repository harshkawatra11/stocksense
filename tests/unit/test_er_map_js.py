"""G2-E2: James-Stein shrinkage of the decile expected-return map (`shrink="james_stein"`, `decile_js`)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.models.ensemble import expected_return_map_decile, james_stein_shrink
from stocksense.validation.g1 import G1Config, _spec_s1, blend_models

D0 = date(2020, 1, 1)
TARGET = "y_c2o_5"
LAG = 7


# ---- the shrinkage formula (hand numbers) -------------------------------------------------------------


def test_tau2_zero_collapses_every_bin_to_the_grand_mean():
    m = np.array([0.01, 0.02, 0.03])  # var (ddof 1) = 1e-4
    v = np.array([1e-4, 1e-4, 1e-4])  # mean sampling variance 1e-4 -> tau^2 = max(0, 1e-4 - 1e-4) = 0
    out = james_stein_shrink(m, v)
    assert out == pytest.approx([0.02, 0.02, 0.02])  # mu = mean(m) = 0.02
    # noise larger than the spread: still tau^2 = 0 -> mu
    assert james_stein_shrink(m, v * 5) == pytest.approx([0.02, 0.02, 0.02])


def test_vanishing_sampling_variance_keeps_the_raw_means():
    m = np.array([0.01, 0.02, 0.03])
    out = james_stein_shrink(m, np.full(3, 1e-14))
    assert out == pytest.approx(m, abs=1e-9)
    assert james_stein_shrink(m, np.zeros(3)) == pytest.approx(m)  # exactly zero noise: nothing to shrink


def test_intermediate_noise_shrinks_by_the_stated_factor():
    m = np.array([0.01, 0.02, 0.03])
    v = np.full(3, 0.5e-4)  # tau^2 = 1e-4 - 0.5e-4 = 0.5e-4; B = v / (v + tau^2) = 0.5
    out = james_stein_shrink(m, v)
    assert out == pytest.approx([0.015, 0.02, 0.025])  # mu + (1 - 0.5) (m - mu)
    # unequal sampling variances shrink the noisy bin more
    v2 = np.array([1e-6, 0.5e-4, 1.5e-4])  # mean 0.67e-4 -> tau^2 = 0.33e-4
    tau2 = 1e-4 - float(v2.mean())
    mu = 0.02
    expect = mu + (1 - v2 / (v2 + tau2)) * (m - mu)
    assert james_stein_shrink(m, v2) == pytest.approx(expect)


def test_bins_without_data_are_ignored():
    m = np.array([0.01, np.nan, 0.03])
    out = james_stein_shrink(m, np.array([1e-14, np.nan, 1e-14]))
    assert out[0] == pytest.approx(0.01, abs=1e-9) and out[2] == pytest.approx(0.03, abs=1e-9)
    assert np.isnan(out[1])


# ---- the map ------------------------------------------------------------------------------------------


def _frames(n_dates: int, n_sid: int = 100, noise: float = 0.0, slope: float = 0.001, seed: int = 0):
    ds = [D0 + timedelta(days=i) for i in range(n_dates)]
    rng = np.random.default_rng(seed)
    rows, lab = [], []
    for d in ds:
        for s in range(n_sid):
            rows.append((d, s, s / n_sid - 0.5))
            lab.append((d, s, slope * (s // (n_sid // 10)) + noise * rng.normal(), d + timedelta(days=LAG)))
    blend = pl.DataFrame(rows, schema=["date", "sid", "score"], orient="row")
    labels = pl.DataFrame(lab, schema=["date", "sid", TARGET, "end_5"], orient="row")
    return blend, labels, ds


def _deciles(out: pl.DataFrame, d: date) -> list[float]:
    g = out.filter(pl.col("date") == d).sort("sid")
    return [g["exp_ret"][j * 10] for j in range(10)]


def test_default_shrink_is_unchanged():
    blend, labels, ds = _frames(40, noise=0.01)
    a = expected_return_map_decile(blend, labels, TARGET)
    b = expected_return_map_decile(blend, labels, TARGET, shrink="k")
    assert a.sort("date", "sid")["exp_ret"].to_list() == b.sort("date", "sid")["exp_ret"].to_list()
    with pytest.raises(ValueError):
        expected_return_map_decile(blend, labels, TARGET, shrink="bayes")


def test_noiseless_planted_slope_passes_through_james_stein():
    blend, labels, ds = _frames(60)
    out = expected_return_map_decile(blend, labels, TARGET, shrink="james_stein")
    assert _deciles(out, ds[-1]) == pytest.approx([0.001 * j for j in range(10)], abs=1e-9)
    assert set(out.filter(pl.col("date") == ds[0])["exp_ret"].to_list()) == {0.0}  # nothing known yet


def test_pure_noise_collapses_to_the_grand_mean_and_signal_is_kept_with_noise():
    # no slope, big noise: the decile means differ only by sampling noise -> tau^2 ~ 0 -> (near) flat map
    blend, labels, ds = _frames(60, noise=0.05, slope=0.0, seed=1)
    js = _deciles(expected_return_map_decile(blend, labels, TARGET, shrink="james_stein"), ds[-1])
    assert max(js) - min(js) < 1e-3
    # a strong planted slope with the same noise keeps its order and most of its spread
    blend, labels, ds = _frames(60, noise=0.05, slope=0.004, seed=1)
    js = _deciles(expected_return_map_decile(blend, labels, TARGET, shrink="james_stein"), ds[-1])
    assert all(b >= a - 1e-12 for a, b in zip(js, js[1:], strict=False))
    assert js[9] - js[0] > 0.03  # raw spread 0.036


def test_never_uses_a_label_that_ends_on_or_after_t():
    blend, labels, ds = _frames(30)
    t = ds[12]
    poisoned = labels.with_columns(
        pl.when(pl.col("date").is_between(ds[5], ds[11])).then(1.0).otherwise(pl.col(TARGET)).alias(TARGET)
    )
    a = expected_return_map_decile(blend, labels, TARGET, shrink="james_stein").filter(pl.col("date") == t)
    b = expected_return_map_decile(blend, poisoned, TARGET, shrink="james_stein").filter(pl.col("date") == t)
    assert a.sort("sid")["exp_ret"].to_list() == pytest.approx(b.sort("sid")["exp_ret"].to_list())


# ---- G1 wiring ----------------------------------------------------------------------------------------


def test_spec_key_only_when_non_default():
    cfg = G1Config(capitals=(5e5,), reference_capital=5e5)
    base = _spec_s1(cfg, "y_c2o_10", 10, False, [])
    assert "er_map" not in base
    spec = _spec_s1(replace(cfg, er_map="decile_js"), "y_c2o_10", 10, False, [])
    assert spec["er_map"] == "decile_js"
    assert _spec_s1(replace(cfg, er_map="decile"), "y_c2o_10", 10, False, [])["er_map"] == "decile"


def test_blend_models_accepts_decile_js():
    blend, labels, ds = _frames(80, noise=0.02)
    out = blend_models({"m": blend}, labels, TARGET, er_map="decile_js")
    last = out.frame.filter(pl.col("date") == ds[-1]).sort("score")
    assert last["exp_ret"].to_list() == sorted(last["exp_ret"].to_list())
    with pytest.raises(ValueError):
        blend_models({"m": blend}, labels, TARGET, er_map="nope")
