"""Factor-metric tests.

Constructed signals with KNOWN answers, so a refactor that changes a number
fails loudly. The decay-curve test in particular pins the property the previous
build got wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocksense.evaluation.factor_metrics import (
    calmar,
    cross_sectional_ic,
    decay_curve,
    grinold_ir,
    half_life,
    ic_summary,
    icir,
    max_drawdown,
    sharpe,
    sortino,
)


def _panel(n_dates=20, n_symbols=30, seed=0):
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"v": rng.normal(size=len(idx))}, index=idx).reset_index()


# ------------------------------------------------------------------- rank IC
def test_perfectly_monotone_signal_has_ic_of_one():
    """scores == forward returns => rank IC is exactly 1.0 on every date."""
    p = _panel()
    scores = p.rename(columns={"v": "score"})
    fwd = p.rename(columns={"v": "fwd_ret"})
    ic = cross_sectional_ic(scores, fwd)
    assert len(ic) == 20
    assert np.allclose(ic.values, 1.0)


def test_perfectly_inverted_signal_has_ic_of_minus_one():
    p = _panel()
    scores = p.assign(score=-p["v"]).drop(columns="v")
    fwd = p.rename(columns={"v": "fwd_ret"})
    assert np.allclose(cross_sectional_ic(scores, fwd).values, -1.0)


def test_ic_is_per_date_not_pooled():
    """The previous build computed ONE pooled correlation across every
    (symbol, date) pair, which mixes cross-sectional signal with time-series
    drift and is not what IC means anywhere else in the industry.

    Constructed so the two differ: the signal is perfect WITHIN each date, but
    the dates have opposing level shifts that a pooled correlation would see.
    """
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    symbols = [f"S{i}" for i in range(12)]
    rows_s, rows_f = [], []
    for k, d in enumerate(dates):
        for j, s in enumerate(symbols):
            level = 100.0 if k == 0 else -100.0
            rows_s.append({"date": d, "symbol": s, "score": j + level})
            rows_f.append({"date": d, "symbol": s, "fwd_ret": j - level})
    ic = cross_sectional_ic(pd.DataFrame(rows_s), pd.DataFrame(rows_f))

    assert np.allclose(ic.values, 1.0), "within each date the signal is perfect"
    pooled = (
        pd.DataFrame(rows_s)
        .merge(pd.DataFrame(rows_f), on=["date", "symbol"])[["score", "fwd_ret"]]
        .corr(method="spearman")
        .iloc[0, 1]
    )
    assert pooled < 0, "pooling inverts the sign -- which is the bug"


def test_thin_dates_are_dropped_not_averaged_in():
    """A cross-sectional correlation over 3 names is noise, not information."""
    p = _panel(n_dates=5, n_symbols=30)
    thin = p[p.date == p.date.max()].head(3)
    p = pd.concat([p[p.date != p.date.max()], thin])
    ic = cross_sectional_ic(p.rename(columns={"v": "score"}),
                            p.rename(columns={"v": "fwd_ret"}))
    assert len(ic) == 4


def test_a_constant_score_yields_no_ic_rather_than_nan():
    """A config that scores everything identically has undefined IC. It must be
    skipped, not emitted as NaN that quietly poisons the mean."""
    p = _panel(n_dates=3, n_symbols=20)
    ic = cross_sectional_ic(p.assign(score=1.0).drop(columns="v"),
                            p.rename(columns={"v": "fwd_ret"}))
    assert ic.empty


# --------------------------------------------------------------------- ICIR
def test_icir_prefers_consistency_over_magnitude():
    """THE objective function's defining property: a small steady edge beats a
    large erratic one, because that is what survives costs and can be sized."""
    steady = pd.Series([0.03, 0.032, 0.028, 0.031, 0.029, 0.030])
    erratic = pd.Series([0.20, -0.15, 0.25, -0.18, 0.22, -0.14])
    assert erratic.mean() > steady.mean()
    assert icir(steady) > icir(erratic)


def test_icir_is_nan_below_three_dates():
    assert np.isnan(icir(pd.Series([0.1, 0.2])))
    assert not np.isnan(icir(pd.Series([0.1, 0.2, 0.15])))


def test_icir_is_nan_with_zero_dispersion():
    assert np.isnan(icir(pd.Series([0.05, 0.05, 0.05, 0.05])))


def test_summary_flags_an_implausibly_high_ic():
    """Real equity factors run IC 0.02-0.05. A mean IC of 0.40 is a red flag --
    almost always leakage or a bug -- and must not read as a triumph."""
    assert ic_summary(pd.Series([0.40, 0.42, 0.38, 0.41]))["overfit_flag"] is True
    assert ic_summary(pd.Series([0.03, 0.04, 0.02, 0.035]))["overfit_flag"] is False


# ---------------------------------------------------------------- decay curve
def _decaying_panel(ic_by_h: dict[int, float], n_dates=40, n_symbols=60, seed=3):
    """Build labels whose correlation with a fixed score is known per horizon."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    symbols = [f"S{i:03d}" for i in range(n_symbols)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    score = pd.Series(rng.normal(size=len(idx)), index=idx)
    scores = score.rename("score").reset_index()

    fwd_by_h = {}
    for h, rho in ic_by_h.items():
        noise = pd.Series(rng.normal(size=len(idx)), index=idx)
        blended = rho * score + np.sqrt(max(0.0, 1 - rho**2)) * noise
        fwd_by_h[h] = blended.rename("fwd_ret").reset_index()
    return scores, fwd_by_h


def test_decay_curve_uses_identical_dates_across_horizons():
    """THE property the previous build broke. It drove both the label AND the
    rebalance spacing from the horizon, so every horizon saw a different sample
    of a different size -- four unrelated numbers on one axis, not a curve."""
    scores, fwd = _decaying_panel({1: 0.30, 2: 0.20, 5: 0.10, 10: 0.02})
    curve = decay_curve(scores, fwd)
    assert curve["n_dates"].nunique() == 1, "horizons must share one date set"
    assert list(curve["horizon"]) == [1, 2, 5, 10]


def test_decay_curve_recovers_the_built_in_ordering():
    scores, fwd = _decaying_panel({1: 0.35, 2: 0.25, 5: 0.12, 10: 0.01})
    curve = decay_curve(scores, fwd).set_index("horizon")
    assert curve.loc[1, "mean_ic"] > curve.loc[2, "mean_ic"] > curve.loc[5, "mean_ic"]
    assert curve.loc[10, "mean_ic"] < curve.loc[5, "mean_ic"]


def test_half_life_finds_the_crossing():
    """Peak IC 0.40 at h=1; half is 0.20, which the curve crosses between h=2
    (0.30) and h=4 (0.10), i.e. at h=3."""
    curve = pd.DataFrame(
        {"horizon": [1, 2, 4, 8], "mean_ic": [0.40, 0.30, 0.10, 0.02]}
    )
    assert half_life(curve) == pytest.approx(3.0, abs=0.01)


def test_half_life_is_nan_when_the_signal_never_halves():
    """"Does not apply" must not be reported as a good result."""
    flat = pd.DataFrame({"horizon": [1, 2, 4], "mean_ic": [0.10, 0.099, 0.098]})
    assert np.isnan(half_life(flat))


def test_half_life_is_nan_for_a_signal_with_no_positive_peak():
    dud = pd.DataFrame({"horizon": [1, 2, 4], "mean_ic": [-0.01, -0.03, -0.05]})
    assert np.isnan(half_life(dud))


def test_fast_decay_is_detectable_as_the_search_screen():
    """The screen the search actually applies: reject half_life < 3 bars. A
    signal gone before it can be traded net of costs is not tradeable, however
    high its peak."""
    fast = pd.DataFrame({"horizon": [1, 2, 3, 6], "mean_ic": [0.40, 0.12, 0.04, 0.01]})
    slow = pd.DataFrame({"horizon": [1, 2, 3, 6], "mean_ic": [0.10, 0.095, 0.09, 0.06]})
    assert half_life(fast) < 3.0
    assert np.isnan(half_life(slow)) or half_life(slow) >= 3.0


# --------------------------------------------------------------- diagnostics
def test_max_drawdown_is_never_positive():
    rng = np.random.default_rng(0)
    assert max_drawdown(rng.normal(0.001, 0.02, 500)) <= 0
    assert max_drawdown(np.array([0.01, 0.01, 0.01])) == pytest.approx(0.0)


def test_max_drawdown_matches_a_hand_computed_case():
    # 1.0 -> 1.5 -> 0.75 : a 50% drawdown from the peak.
    assert max_drawdown(np.array([0.5, -0.5])) == pytest.approx(-0.5)


def test_sortino_exceeds_sharpe_when_upside_dominates():
    """Upside volatility is not risk; Sortino should not punish it."""
    r = np.array([0.05, 0.001, 0.002, -0.005, 0.06, -0.004, 0.001, -0.003])
    assert sortino(r, 252) > sharpe(r, 252)


def test_metrics_return_nan_rather_than_lying_on_thin_input():
    assert np.isnan(sharpe(np.array([0.01]), 252))
    assert np.isnan(sortino(np.array([0.01, 0.02]), 252))
    assert np.isnan(calmar(np.array([0.01]), 252))


def test_grinold_law_shows_why_breadth_matters_here():
    """A small IC across many bets beats a large IC across two. At 1-2 positions
    a day breadth is tiny, which is why this project's edge must come from
    selectivity rather than IC magnitude."""
    assert grinold_ir(0.03, 500) > grinold_ir(0.15, 2)
