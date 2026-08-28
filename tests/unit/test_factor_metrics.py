"""Phase K0.1: the loop's objective function and its decay screen.

The load-bearing test here is
`test_decay_curve_uses_identical_scoring_dates_across_horizons` -- the entire
reason decay_curve exists as a separate function (rather than looping
train_and_score_fold over horizons) is that the existing path gives each
horizon DIFFERENT folds and DIFFERENT sample sizes, which is not a curve. If
that property breaks, the half-life numbers become meaningless comparisons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocksense.evaluation.backtest import ScoredFold
from stocksense.evaluation.factor_metrics import (
    build_labeled_by_horizon,
    calmar,
    cross_sectional_ic,
    decay_curve,
    half_life,
    icir,
    max_drawdown,
    sharpe,
    sortino,
)


def _scored(scores_by_date: dict, actual_by_date: dict) -> ScoredFold:
    dates = sorted(scores_by_date)
    return ScoredFold(
        fold_id=0,
        horizon_bars=10,
        n_train_rows=1000,
        rebalance_dates=list(dates),
        scores_by_date=scores_by_date,
        raw_actual_by_date=actual_by_date,
        rel_actual_by_date=actual_by_date,
    )


def _symbols(n: int) -> list[str]:
    return [f"S{i:03d}" for i in range(n)]


# ---- cross_sectional_ic ----


def test_rank_ic_of_perfectly_monotone_signal_is_one() -> None:
    syms = _symbols(30)
    d = pd.Timestamp("2024-01-02")
    scores = pd.Series(np.arange(30, dtype=float), index=syms)
    actual = pd.Series(np.arange(30, dtype=float) * 3.0, index=syms)  # monotone, different scale
    ic = cross_sectional_ic(_scored({d: scores}, {d: actual}))
    assert ic.loc[d] == pytest.approx(1.0)


def test_rank_ic_of_reversed_signal_is_minus_one() -> None:
    syms = _symbols(30)
    d = pd.Timestamp("2024-01-02")
    scores = pd.Series(np.arange(30, dtype=float), index=syms)
    actual = pd.Series(np.arange(30, dtype=float)[::-1], index=syms)
    ic = cross_sectional_ic(_scored({d: scores}, {d: actual}))
    assert ic.loc[d] == pytest.approx(-1.0)


def test_ic_is_one_value_per_date_not_a_pooled_scalar() -> None:
    """The whole point of this module vs the legacy pooled metric."""
    syms = _symbols(20)
    dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
    scores = {d: pd.Series(np.arange(20, dtype=float), index=syms) for d in dates}
    actual = {
        dates[0]: pd.Series(np.arange(20, dtype=float), index=syms),        # IC +1
        dates[1]: pd.Series(np.arange(20, dtype=float)[::-1], index=syms),  # IC -1
        dates[2]: pd.Series(np.arange(20, dtype=float), index=syms),        # IC +1
    }
    ic = cross_sectional_ic(_scored(scores, actual))
    assert len(ic) == 3
    assert ic.iloc[0] == pytest.approx(1.0)
    assert ic.iloc[1] == pytest.approx(-1.0)


def test_dates_with_too_few_names_are_dropped_not_returned_as_nan() -> None:
    """A date that could not be measured is absent from the sample, not a
    measurement of zero -- averaging in a NaN-as-zero would bias ICIR."""
    syms = _symbols(5)  # below MIN_PAIRS_PER_DATE
    d = pd.Timestamp("2024-01-02")
    scores = pd.Series(np.arange(5, dtype=float), index=syms)
    ic = cross_sectional_ic(_scored({d: scores}, {d: scores}))
    assert ic.empty


def test_constant_score_column_is_skipped() -> None:
    syms = _symbols(20)
    d = pd.Timestamp("2024-01-02")
    flat = pd.Series(np.ones(20), index=syms)
    actual = pd.Series(np.arange(20, dtype=float), index=syms)
    ic = cross_sectional_ic(_scored({d: flat}, {d: actual}))
    assert ic.empty


def test_ic_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        cross_sectional_ic(_scored({}, {}), method="kendall")


# ---- icir ----


def test_icir_is_mean_over_std() -> None:
    series = pd.Series([0.02, 0.04, 0.03, 0.05])
    assert icir(series) == pytest.approx(series.mean() / series.std(ddof=1))


def test_icir_prefers_consistency_over_magnitude() -> None:
    """The video's core claim, as a test: a small steady IC beats a big erratic
    one."""
    steady = pd.Series([0.03, 0.031, 0.029, 0.030, 0.031])
    erratic = pd.Series([0.20, -0.15, 0.18, -0.12, 0.19])
    assert erratic.mean() > steady.mean()      # bigger average IC
    assert icir(steady) > icir(erratic)         # but worse consistency


def test_icir_nan_when_fewer_than_three_dates() -> None:
    assert np.isnan(icir(pd.Series([0.02, 0.03])))


def test_icir_nan_on_zero_variance() -> None:
    assert np.isnan(icir(pd.Series([0.03, 0.03, 0.03, 0.03])))


# ---- decay_curve / half_life ----


class _DecayRanker:
    """A stand-in whose score correlates with a hidden 'true' value; the test
    data is then built so that longer horizons are progressively noisier."""

    def __init__(self, feature_cols):
        self._cols = feature_cols

    def predict(self, X):
        return X[self._cols[0]].to_numpy()


def _decay_fixture(n_symbols: int = 60, n_dates: int = 40, seed: int = 3):
    rng = np.random.default_rng(seed)
    syms = _symbols(n_symbols)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)

    signal = {d: pd.Series(rng.normal(size=n_symbols), index=syms) for d in dates}
    feats = pd.DataFrame(
        [
            {"symbol": s, "date": d, "f0": float(signal[d][s])}
            for d in dates
            for s in syms
        ]
    )

    # Horizon h's realized return = signal * strength(h) + noise*NOISE_SD.
    #
    # NOTE the subtlety this fixture originally got wrong: IC is a CORRELATION,
    # so it does NOT scale linearly with signal strength -- it follows
    #     IC = s / sqrt(s^2 + noise^2)
    # Halving the signal strength does NOT halve the IC. The strengths below are
    # solved backwards from the IC we want, with NOISE_SD = 0.75:
    #     h=1  -> s=1.000 -> IC ~ 0.80
    #     h=4  -> s=0.327 -> IC ~ 0.40   (half of the peak, by construction)
    # so the IC half-life of this fixture is genuinely h=4.
    NOISE_SD = 0.75
    STRENGTH = {1: 1.0, 2: 0.60, 4: 0.327, 8: 0.15, 16: 0.07}
    labeled_by_horizon = {}
    for h in (1, 2, 4, 8, 16):
        strength = STRENGTH[h]
        rows = []
        for d in dates:
            noise = rng.normal(size=n_symbols) * NOISE_SD
            realized = signal[d].to_numpy() * strength + noise
            for i, s in enumerate(syms):
                rows.append({"symbol": s, "date": d, f"fwd_ret_{h}b_rel": float(realized[i])})
        labeled_by_horizon[h] = pd.DataFrame(rows)

    return feats, labeled_by_horizon, list(dates)


def test_decay_curve_ic_falls_as_horizon_grows() -> None:
    feats, labeled, dates = _decay_fixture()
    curve = decay_curve(
        _DecayRanker(["f0"]), feats, labeled, dates, ["f0"], horizons=(1, 2, 4, 8, 16)
    )
    ics = curve.sort_values("horizon")["mean_ic"].tolist()
    assert ics[0] > ics[-1]
    assert ics[0] > 0.3  # the shortest horizon carries real signal


def test_decay_curve_uses_identical_scoring_dates_across_horizons() -> None:
    """THE load-bearing property. train_and_score_fold cannot give this because
    it subsamples rebalance dates by horizon_bars, so each horizon would be
    measured over a different (and smaller) set of dates."""
    feats, labeled, dates = _decay_fixture()
    curve = decay_curve(
        _DecayRanker(["f0"]), feats, labeled, dates, ["f0"], horizons=(1, 2, 4, 8, 16)
    )
    assert curve["n_dates"].nunique() == 1
    assert int(curve["n_dates"].iloc[0]) == len(dates)


def test_decay_curve_raises_on_missing_horizon_label() -> None:
    feats, labeled, dates = _decay_fixture()
    with pytest.raises(KeyError):
        decay_curve(_DecayRanker(["f0"]), feats, labeled, dates, ["f0"], horizons=(1, 999))


def test_half_life_recovers_a_known_decay() -> None:
    """The fixture is solved backwards so that IC ITSELF halves at h=4 (see the
    note in _decay_fixture about IC being a correlation, which does not scale
    linearly with signal strength)."""
    feats, labeled, dates = _decay_fixture()
    curve = decay_curve(
        _DecayRanker(["f0"]), feats, labeled, dates, ["f0"], horizons=(1, 2, 4, 8, 16)
    )
    hl = half_life(curve)
    assert 2.0 <= hl <= 6.0  # brackets the constructed h=4; sampling noise moves it a little


def test_half_life_is_nan_when_ic_never_halves() -> None:
    curve = pd.DataFrame({"horizon": [1, 2, 5, 10], "mean_ic": [0.05, 0.049, 0.048, 0.047]})
    assert np.isnan(half_life(curve))


def test_half_life_is_nan_when_there_is_no_signal_to_decay() -> None:
    curve = pd.DataFrame({"horizon": [1, 2, 5], "mean_ic": [-0.01, -0.02, -0.03]})
    assert np.isnan(half_life(curve))


def test_half_life_interpolates_between_horizons() -> None:
    curve = pd.DataFrame({"horizon": [1, 2, 3], "mean_ic": [0.10, 0.08, 0.04]})
    # peak 0.10 at h=1, target 0.05, crossed between h=2 (0.08) and h=3 (0.04)
    assert half_life(curve) == pytest.approx(2.0 + (0.08 - 0.05) / (0.08 - 0.04))


# ---- risk-adjusted diagnostics ----


def test_sharpe_annualises() -> None:
    r = pd.Series([0.01] * 10 + [-0.005] * 10)
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe(r, 252) == pytest.approx(expected)


def test_sortino_ignores_upside_volatility() -> None:
    """Two series with IDENTICAL mean and IDENTICAL downside legs, differing
    only in how lumpy the upside is. Sortino must score them the same; Sharpe
    must penalise the lumpy one. That is precisely the difference between the
    two measures."""
    steady_up = pd.Series([0.03, 0.03, 0.03, -0.01, -0.01])
    spiky_up = pd.Series([0.09, 0.00, 0.00, -0.01, -0.01])
    assert steady_up.mean() == pytest.approx(spiky_up.mean())

    assert sortino(steady_up, 252) == pytest.approx(sortino(spiky_up, 252))
    assert sharpe(steady_up, 252) > sharpe(spiky_up, 252)


def test_sortino_penalises_deeper_downside() -> None:
    shallow = pd.Series([0.02, 0.02, -0.01, -0.01, 0.02])
    deep = pd.Series([0.02, 0.02, -0.05, -0.05, 0.02])
    assert sortino(shallow, 252) > sortino(deep, 252)


def test_max_drawdown_is_never_positive() -> None:
    assert max_drawdown(pd.Series([0.01, 0.02, 0.03])) <= 0.0
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) < 0.0


def test_max_drawdown_matches_hand_computation() -> None:
    # equity: 1.0 -> 1.5 -> 0.75 ; worst dd = 0.75/1.5 - 1 = -0.5
    assert max_drawdown(pd.Series([0.5, -0.5])) == pytest.approx(-0.5)


def test_calmar_is_return_over_drawdown() -> None:
    r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    assert calmar(r, 252) == pytest.approx(r.mean() * 252 / abs(max_drawdown(r)))


def test_risk_metrics_nan_on_degenerate_input() -> None:
    assert np.isnan(sharpe(pd.Series([0.01]), 252))
    assert np.isnan(sortino(pd.Series([0.01]), 252))
    assert np.isnan(calmar(pd.Series([0.01]), 252))


# ---- build_labeled_by_horizon ----


def test_build_labeled_by_horizon_names_columns_correctly() -> None:
    candles = pd.DataFrame(
        [
            {"symbol": s, "date": d, "adj_close": 100.0 + i, "close": 100.0 + i}
            for s in ("AAA", "BBB")
            for i, d in enumerate(pd.bdate_range("2024-01-01", periods=30))
        ]
    )
    out = build_labeled_by_horizon(candles, (1, 5))
    assert set(out) == {1, 5}
    assert "fwd_ret_1b_rel" in out[1].columns
    assert "fwd_ret_5b_rel" in out[5].columns
