"""G2-E1: HAR-RV and GARCH(1,1) volatility forecasts (features/volforecast.py)."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.features import volforecast as vf
from stocksense.features.registry import FEATURE_MODULES
from tests.unit.test_feature_truncation import (  # noqa: F401  (the session fixture builds the synthetic market)
    _compute,
    truncation_env,
    violations,
)

D0 = date(2010, 1, 4)


def _har_panel(n_sid=30, n_days=2500, b=(-0.42, 0.35, 0.30, 0.25), noise=0.3, seed=1) -> pl.DataFrame:
    """Simulated log-RV panel obeying the HAR recursion exactly; la = rv * N(0,1) (so scale = 1)."""
    rng = np.random.default_rng(seed)
    lrv = np.full((n_sid, n_days), -4.2)
    lrv[:, :22] += rng.normal(0, 0.3, (n_sid, 22))
    for t in range(21, n_days - 1):
        x1 = lrv[:, t]
        x2 = lrv[:, t - 4 : t + 1].mean(1)
        x3 = lrv[:, t - 21 : t + 1].mean(1)
        lrv[:, t + 1] = b[0] + b[1] * x1 + b[2] * x2 + b[3] * x3 + rng.normal(0, noise, n_sid)
    rv = np.exp(lrv)
    la = rv * rng.normal(0, 1, (n_sid, n_days))
    rows = {
        "sid": np.repeat(np.arange(n_sid), n_days),
        "sx": np.tile(np.arange(n_days), n_sid),
        "lrv": lrv.ravel(),
        "rv": rv.ravel(),
        "la": la.ravel(),
    }
    df = pl.DataFrame(rows).with_columns(
        pl.col("sid").cast(pl.Int32), (pl.lit(D0) + pl.duration(days=pl.col("sx"))).alias("date")
    )
    return df.select("sid", "date", "sx", "lrv", "rv", "la")


# ---- HAR ----------------------------------------------------------------------------------------------


def test_har_coefficients_are_recovered_from_a_simulated_har_process():
    df = _har_panel()
    fit = vf.fit_har(vf.har_design(df), cutoff=df["date"].max())
    assert fit is not None
    assert fit.b == pytest.approx((-0.42, 0.35, 0.30, 0.25), abs=0.05)
    assert fit.sigma2 == pytest.approx(0.3**2, rel=0.1)
    assert fit.scale == pytest.approx(1.0, abs=0.1)


def test_har_fit_uses_only_rows_whose_target_is_known_by_the_cutoff():
    df = _har_panel(n_sid=10, n_days=1500)
    cut = D0 + timedelta(days=999)
    clean = vf.fit_har(vf.har_design(df), cutoff=cut, min_rows=2000)
    poisoned = df.with_columns(pl.when(pl.col("date") > cut).then(150.0).otherwise(pl.col("rv")).alias("rv"))
    other = vf.fit_har(vf.har_design(poisoned), cutoff=cut, min_rows=2000)
    assert clean is not None and other is not None
    assert clean.b == pytest.approx(other.b) and clean.n == other.n


def test_har_fit_needs_enough_rows():
    df = _har_panel(n_sid=2, n_days=300)
    assert vf.fit_har(vf.har_design(df), cutoff=df["date"].max()) is None  # fewer than MIN_HAR_ROWS


def test_har_iterated_forecast_hand_numbers():
    # b0 = 0 and b1 + b2 + b3 = 1: a flat window stays flat, so every iterated forecast equals the window
    fit = vf.HarFit(b=(0.0, 0.5, 0.3, 0.2), sigma2=0.0, scale=1.0, n=10_000)
    w = np.full((1, 22), math.log(0.02))
    v1, v10 = vf.har_forecast(w, fit, horizon=10)
    assert v1[0] == pytest.approx(0.02)
    assert v10[0] == pytest.approx(0.02 * math.sqrt(10))
    # the lognormal level correction exp(sigma2 / 2) and the scale multiply the 1-day vol
    fit2 = vf.HarFit(b=(0.0, 0.5, 0.3, 0.2), sigma2=0.2, scale=1.1, n=10_000)
    v1b, _ = vf.har_forecast(w, fit2, horizon=10)
    assert v1b[0] == pytest.approx(0.02 * 1.1 * math.exp(0.1))
    # a one-step recursion by hand: window [ln .02 (latest), then ln .01 x 21]: x1 = ln .02,
    # x2 = (ln .02 + 4 ln .01) / 5, x3 = (ln .02 + 21 ln .01) / 22
    w2 = np.concatenate([[math.log(0.02)], np.full(21, math.log(0.01))])[None, :]
    a, c = math.log(0.02), math.log(0.01)
    f = 0.5 * a + 0.3 * (a + 4 * c) / 5 + 0.2 * (a + 21 * c) / 22
    assert vf.har_forecast(w2, fit, horizon=1)[0][0] == pytest.approx(math.exp(f))


def test_har_features_are_null_before_the_first_january_fit_and_fixed_within_a_year():
    df = _har_panel(n_sid=10, n_days=1500)  # ~ 2010 .. 2014 as calendar days
    out = vf.har_series(df, min_rows=500)
    first_fit_year = D0.year + 1
    early = out.filter(pl.col("date") < date(first_fit_year, 1, 1))
    assert early["har_vol_1d"].null_count() == early.height
    late = out.filter(pl.col("date") >= date(first_fit_year + 1, 1, 1))
    assert late["har_vol_1d"].drop_nulls().len() > 0
    assert (late["har_vol_10d"].drop_nulls() > 0).all()


# ---- GARCH --------------------------------------------------------------------------------------------

OMEGA, ALPHA, BETA = 1e-6, 0.08, 0.90


def _garch_path(n: int, seed: int) -> tuple[np.ndarray, float]:
    """Simulate GARCH(1,1) and return (returns, true 10-day vol forecast made after the last return)."""
    rng = np.random.default_rng(seed)
    uncond = OMEGA / (1 - ALPHA - BETA)
    r = np.empty(n + 200)
    s2 = uncond
    for t in range(n + 200):
        r[t] = math.sqrt(s2) * rng.normal()
        s2 = OMEGA + ALPHA * r[t] ** 2 + BETA * s2
    r = r[200:]  # burn-in; s2 now holds the true variance of the next return
    path = [uncond + (ALPHA + BETA) ** j * (s2 - uncond) for j in range(10)]
    return r, math.sqrt(sum(path))


@pytest.mark.parametrize("seed", [11, 12, 13, 14, 15])
def test_garch_forecast_tracks_the_true_conditional_vol(seed):
    r, truth = _garch_path(1000, seed)
    est = vf.garch_vol_h(r, horizon=10)
    assert est == pytest.approx(truth, rel=0.15)


def test_garch_returns_nan_on_degenerate_input():
    assert math.isnan(vf.garch_vol_h(np.zeros(600), horizon=10))
    assert math.isnan(vf.garch_vol_h(np.full(600, np.nan), horizon=10))


def test_garch_series_holds_the_month_end_forecast_for_the_next_month(monkeypatch):
    monkeypatch.setattr(vf, "MIN_GARCH_OBS", 150)
    r, _ = _garch_path(400, 3)
    ds = [D0 + timedelta(days=i) for i in range(len(r))]
    df = pl.DataFrame({"sid": np.int32(1), "date": ds, "la": r})
    out = vf.garch_series(df, n_jobs=1)
    v = out.filter(pl.col("garch_vol_10d").is_not_null())
    assert v.height > 0
    # constant inside every calendar month
    per_month = (
        v.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("m"))
        .group_by("m")
        .agg(pl.col("garch_vol_10d").n_unique().alias("n"))
    )
    assert (per_month["n"] == 1).all()
    # the value of month M was fitted on data ending at M-1's last row: perturbing M's returns changes nothing
    first = v["date"].min()
    month_end = date(first.year, first.month, 1) + timedelta(days=40)
    month_end = date(month_end.year, month_end.month, 1) - timedelta(days=1)
    r2 = r.copy()
    idx = [i for i, d in enumerate(ds) if d >= date(first.year, first.month, 1)]
    r2[idx] = r2[idx] * 3.0
    out2 = vf.garch_series(pl.DataFrame({"sid": np.int32(1), "date": ds, "la": r2}), n_jobs=1)
    a = out.filter(pl.col("date").is_between(first, month_end))["garch_vol_10d"].to_list()
    b = out2.filter(pl.col("date").is_between(first, month_end))["garch_vol_10d"].to_list()
    assert a == pytest.approx(b)


# ---- registry and truncation --------------------------------------------------------------------------


def test_the_features_are_not_in_the_default_build():
    """A feature build must not change until E9 adds the module (live builds features nightly)."""
    assert "volforecast" not in FEATURE_MODULES
    assert {"har_vol", "garch_vol"} <= {s.name for s in vf.SPECS}
    assert vf.OUTPUTS == ("har_vol_1d", "har_vol_10d", "garch_vol_10d")


@pytest.fixture
def small_thresholds(monkeypatch):
    monkeypatch.setattr(vf, "MIN_HAR_ROWS", 2000)
    monkeypatch.setattr(vf, "MIN_GARCH_OBS", 120)


@pytest.mark.parametrize("name", ["har_vol", "garch_vol"])
def test_feature_does_not_look_ahead(name, truncation_env, small_thresholds):  # noqa: F811
    s, sessions, _ = truncation_env
    spec = next(x for x in vf.SPECS if x.name == name)
    full = _compute(spec, s, None)
    assert full.height > 0
    non_null = {o: full[o].drop_nulls().len() for o in spec.outputs}
    assert all(v > 0 for v in non_null.values()), f"the guard would be vacuous: {non_null}"
    for idx in (300, 380, 450, 520, 590):
        cut = sessions[idx]
        bad = violations(spec, full, _compute(spec, s, cut), cut)
        assert not bad, f"{name} @ {idx} ({cut}): " + "; ".join(bad)


# ---- the world loader's alignment ---------------------------------------------------------------------


def test_align_to_panel_rows():
    sid = np.array([1, 1, 2, 2], dtype=np.int64)
    dts = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 1), date(2020, 1, 2)]
    frame = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 1)],
            "sid": np.array([1, 2], dtype=np.int32),
            "har_vol_1d": [0.02, 0.03],
            "har_vol_10d": [0.06, 0.09],
            "garch_vol_10d": [0.07, None],
        }
    )
    har, garch = vf.align_daily_vol(sid, dts, frame, horizon=10)
    assert har[1] == pytest.approx(0.06 / math.sqrt(10)) and garch[1] == pytest.approx(0.07 / math.sqrt(10))
    assert har[2] == pytest.approx(0.09 / math.sqrt(10)) and math.isnan(garch[2])
    assert math.isnan(har[0]) and math.isnan(har[3])  # no forecast on those rows
