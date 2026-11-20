"""G2-E6: extreme-value (GPD peaks-over-threshold) expected-shortfall stops."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from scipy import stats

from stocksense.portfolio import evt
from stocksense.portfolio.construct import Holding, PanelDay, S1Config, plan_s1
from tests.unit import test_s1_policy_v2 as v2

# ---- the GPD fit and the expected shortfall --------------------------------------------------------------


def test_gpd_fit_recovers_the_shape_of_a_simulated_gpd():
    x = stats.genpareto.rvs(c=0.2, scale=1.0, size=20_000, random_state=np.random.default_rng(1))
    xi, beta = evt.fit_gpd(x)
    assert xi == pytest.approx(0.2, abs=0.1)
    assert beta == pytest.approx(1.0, rel=0.15)


def test_gpd_es_matches_the_closed_form_and_a_monte_carlo():
    # exponential tail (xi = 0), threshold 0, every point an excess: VaR99 = ln 100, ES99 = ln 100 + beta
    assert evt.gpd_es(0.0, 0.0, 1.0, 1.0, 0.99) == pytest.approx(math.log(100.0) + 1.0)
    # xi = 0.2, beta = 1: ES = VaR/(1 - xi) + beta/(1 - xi) with VaR = beta/xi ((1-q)^-xi - 1)
    var = (0.01**-0.2 - 1.0) / 0.2
    assert evt.gpd_es(0.0, 0.2, 1.0, 1.0, 0.99) == pytest.approx((var + 1.0) / 0.8)
    draws = stats.genpareto.rvs(c=0.2, scale=1.0, size=400_000, random_state=np.random.default_rng(2))
    q = np.quantile(draws, 0.99)
    assert evt.gpd_es(0.0, 0.2, 1.0, 1.0, 0.99) == pytest.approx(draws[draws > q].mean(), rel=0.05)
    # a threshold shifts everything by u, and a tail probability p_u < 1 pushes the 99% point out less
    assert evt.gpd_es(2.0, 0.0, 1.0, 0.05, 0.99) == pytest.approx(2.0 + math.log(5.0) + 1.0)
    assert math.isinf(evt.gpd_es(0.0, 1.2, 1.0, 1.0, 0.99))  # xi >= 1: the tail mean is infinite


def test_pooled_es_of_student_t_losses_is_close_to_the_truth():
    nu = 4.0
    losses = stats.t.rvs(nu, size=400_000, random_state=np.random.default_rng(3))
    var99 = stats.t.ppf(0.99, nu)
    true_es = stats.t.pdf(var99, nu) / 0.01 * (nu + var99**2) / (nu - 1.0)
    got = evt.pooled_es(-losses)  # a loss is minus the vol-scaled return
    assert got is not None
    assert got.es == pytest.approx(true_es, rel=0.15)
    assert 0.0 < got.xi < 0.6  # t4 has a tail index of 1/4


def test_pooled_es_needs_enough_observations():
    assert evt.pooled_es(np.random.default_rng(0).normal(size=100)) is None
    assert evt.pooled_es(np.full(5000, np.nan)) is None


# ---- the monthly table ----------------------------------------------------------------------------------


def _market(n_sid=60, n_days=520, seed=5, tail_by_tercile=(None, None, 3.0)):
    """(dates, sid, ret, vol, rank) arrays: ranks 1..n_sid, terciles of 20; the last tercile has fat tails."""
    rng = np.random.default_rng(seed)
    d0 = date(2015, 1, 5)
    dates = [d0 + timedelta(days=i) for i in range(n_days)]
    sid, dts, ret, vol, rank = [], [], [], [], []
    for s in range(n_sid):
        t = s * 3 // n_sid
        df = tail_by_tercile[t]
        r = (
            rng.normal(0, 0.01, n_days)
            if df is None
            else rng.standard_t(df, n_days) * 0.01 / math.sqrt(df / (df - 2))
        )
        for k in range(n_days):
            sid.append(s)
            dts.append(dates[k])
            ret.append(r[k])
            vol.append(0.01)
            rank.append(s + 1)
    return dts, np.array(sid), np.array(ret), np.array(vol), np.array(rank)


def test_fat_tailed_terciles_get_a_larger_es_multiplier():
    dts, sid, ret, vol, rank = _market()
    tab = evt.es_mult_table(
        dts, sid, ret, vol, rank, from_date=date(2016, 1, 1), universe_rank=60, window=500, min_obs=3000
    )
    assert not tab.is_empty()
    m = tab.filter(pl.col("ym") == 2016 * 12).sort("sid")
    thin = m.filter(pl.col("sid") < 20)["es_mult"].mean()
    fat = m.filter(pl.col("sid") >= 40)["es_mult"].mean()
    assert fat > thin * 1.15
    assert 2.0 < thin < 3.6  # the normal 99% expected shortfall is 2.67 vol units


def test_the_table_of_a_month_does_not_depend_on_data_from_that_month():
    dts, sid, ret, vol, rank = _market()
    base = evt.es_mult_table(
        dts, sid, ret, vol, rank, from_date=date(2016, 1, 1), universe_rank=60, window=500, min_obs=3000
    )
    ret2 = ret.copy()
    for i, d in enumerate(dts):
        if d >= date(2016, 3, 1):
            ret2[i] = ret2[i] * 9.0  # an enormous crash from March on
    pois = evt.es_mult_table(
        dts, sid, ret2, vol, rank, from_date=date(2016, 1, 1), universe_rank=60, window=500, min_obs=3000
    )
    for ym in (2016 * 12, 2016 * 12 + 1, 2016 * 12 + 2):  # January, February, March (0-based months)
        a = base.filter(pl.col("ym") == ym).sort("sid")
        b = pois.filter(pl.col("ym") == ym).sort("sid")
        assert a.height > 0 and a.equals(b), ym
    a = base.filter(pl.col("ym") == 2016 * 12 + 3).sort("sid")  # April does see March
    b = pois.filter(pl.col("ym") == 2016 * 12 + 3).sort("sid")
    assert not a.equals(b)


# ---- the S1 stop ----------------------------------------------------------------------------------------


def _stop_plan(close: float, stop_model: str, es_mult: float | None):
    sc, er, rows = v2.world()
    rows = dict(rows)
    rows[3] = replace(rows[3], close=close)
    hs = [Holding(3, 500, 100.0, 2, 0.02, es_mult)]
    cfg = v2.cfg(stop_model=stop_model)
    return plan_s1(v2.T, sc, er, v2.book(hs), v2.EQUITY, PanelDay(v2.T, v2.D1, rows), cfg)


def test_evt_stop_is_es_times_vol_times_sqrt_h_and_the_fixed_stop_is_unchanged():
    # vol_at_entry 0.02, h = 5: fixed distance = max(0.08, 3 * 0.02 * sqrt 5) = 0.1342
    assert _stop_plan(86.0, "fixed", 4.0).exits.get(3) == "stop"  # -14 % breaches the fixed stop
    # evt with ES multiplier 4: 4 * 0.02 * sqrt 5 = 0.1789: -14 % holds, -18 % stops
    assert _stop_plan(86.0, "evt", 4.0).exits.get(3) != "stop"
    assert _stop_plan(81.0, "evt", 4.0).exits.get(3) == "stop"
    # a small ES keeps the stop_min floor: 1.0 * 0.02 * sqrt 5 = 0.0447 < 0.08
    assert _stop_plan(93.0, "evt", 1.0).exits.get(3) != "stop"
    assert _stop_plan(91.0, "evt", 1.0).exits.get(3) == "stop"


def test_evt_without_a_multiplier_falls_back_to_the_fixed_stop():
    assert _stop_plan(86.0, "evt", None).exits.get(3) == "stop"
    assert _stop_plan(86.0, "evt", float("nan")).exits.get(3) == "stop"


def test_default_stop_model_is_fixed_and_unknown_values_are_rejected():
    assert S1Config.__dataclass_fields__["stop_model"].default == "fixed"
    with pytest.raises(ValueError):
        v2.cfg(stop_model="garch")


def test_entry_log_carries_the_multiplier_only_under_evt():
    sc, er, rows = v2.world()
    rows = {s: replace(r, es_mult=3.3) for s, r in rows.items()}
    pd_ = PanelDay(v2.T, v2.D1, rows)
    fixed = plan_s1(v2.T, sc, er, v2.book(), v2.EQUITY, pd_, v2.cfg())
    evt_p = plan_s1(v2.T, sc, er, v2.book(), v2.EQUITY, pd_, v2.cfg(stop_model="evt"))
    assert fixed.entries and all("es_mult" not in e for e in fixed.entries)
    assert evt_p.entries and all(e["es_mult"] == pytest.approx(3.3) for e in evt_p.entries)
