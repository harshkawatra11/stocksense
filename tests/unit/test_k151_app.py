"""PLAN-M M5.4a: Appendix A (sign conventions, linear cost, bounds, past-only, planted reversal)."""

from __future__ import annotations

import numpy as np
import pytest

from stocksense.strategies import base
from stocksense.strategies import k151 as K
from stocksense.strategies import k151_app as A
from tests.unit.test_k151_stock import bdays


def test_expected_return_and_the_sign_of_the_holding_for_both_delays():
    eye = np.eye(3)
    # delay 0: day 1 opens +2% / -1% / -1% from yesterday's close; mean reversion shorts the gap-up name
    close = np.array([[100.0, 100.0, 100.0], [101.0, 101.0, 101.0]])
    open_ = np.array([[100.0, 100.0, 100.0], [102.0, 99.0, 99.0]])
    e0 = A.expected_return(0, open_, close, 1)
    assert np.allclose(e0, np.log([1.02, 0.99, 0.99]))
    h0 = A.holdings(e0, eye, 1.0)
    assert h0[0] < 0 < h0[1] and h0[2] > 0
    assert abs(h0.sum()) < 1e-12 and np.abs(h0).sum() == pytest.approx(1.0)
    # delay 1: yesterday (row 0) the first name rose 2%; E = -ln(C/O) and hold = -optimum -> long the winner
    close1 = np.array([[102.0, 99.0, 99.0], [100.0, 100.0, 100.0]])
    open1 = np.array([[100.0, 100.0, 100.0], [100.0, 100.0, 100.0]])
    e1 = A.expected_return(1, open1, close1, 1)
    assert np.allclose(e1, -np.log([1.02, 0.99, 0.99]))
    h1 = A.holdings(e1, eye, 1.0)
    assert h1[0] > 0 > h1[1] and h1[2] < 0


def test_linear_cost_has_the_requested_mean_and_cuts_small_expected_returns_to_zero():
    sig = np.array([0.01, 0.02, 0.03])
    addv = np.array([1e8, 1e8, 3e8])
    tau = A.linear_cost(sig, addv)
    assert tau.mean() == pytest.approx(1e-3) and tau[1] == pytest.approx(2 * tau[0])
    out = A.effective_return(np.array([0.002, -0.0005, 0.0003, -0.003]), np.full(4, 0.001))
    assert np.allclose(out, [0.001, 0.0, 0.0, -0.002])


def test_bounds_spread_the_book_and_the_gross_is_one():
    rng = np.random.default_rng(1)
    n = 40
    e = rng.normal(0, 0.01, n)
    e[0] = 0.2  # one dominant signal
    free = A.holdings(e, np.eye(n), np.full(n, 10.0))
    capped = A.holdings(e, np.eye(n), np.full(n, 0.05))
    assert np.abs(free).max() > 0.3 and np.abs(capped).max() < np.abs(free).max()
    assert np.abs(capped).sum() == pytest.approx(1.0) and abs(capped.sum()) < 1e-9
    assert (A.holdings(np.zeros(n), np.eye(n), 1.0) == 0).all()


def planted(t=360, n=60, reversal=0.7, seed=4):
    rng = np.random.default_rng(seed)
    g = rng.normal(0, 0.01, (t, n))
    close = np.empty((t, n))
    open_ = np.empty((t, n))
    close[0] = 100.0
    open_[0] = 100.0
    for i in range(1, t):
        open_[i] = close[i - 1] * np.exp(g[i])
        close[i] = open_[i] * np.exp(-reversal * g[i] + rng.normal(0, 0.003, n))
    return K.Wide.make(
        close,
        dates=bdays(t),
        open_=open_,
        volume=rng.uniform(1e5, 3e5, (t, n)),
        med_turn=np.full((t, n), 1e7),
    )


def test_delay0_earns_on_a_planted_gap_reversal_and_delay1_does_not(tmp_data_root):
    w = planted()
    d0 = A.appa_run(w, 0, 1_000_000, tmp_data_root)
    d1 = A.appa_run(w, 1, 1_000_000, tmp_data_root)
    assert d0.height > 200 and np.allclose(d0["gross"].to_numpy(), 1.0)
    assert d0["ret_gross"].mean() > 0.002 and d0["ret_gross"].mean() / d0["ret_gross"].std() > 0.5
    assert abs(d1["ret_gross"].mean()) < d0["ret_gross"].mean() / 3
    assert (d0["net"].to_numpy() < d0["ret_gross"].to_numpy()).all()
    assert (d0["net_2x"].to_numpy() < d0["net"].to_numpy()).all()


def test_a_bigger_book_meets_its_position_bounds_sooner(tmp_data_root):
    w = planted(t=300)
    small = A.appa_run(w, 0, 1_000, tmp_data_root)
    big = A.appa_run(w, 0, 5_000_000_000, tmp_data_root)  # bound = 1% ADDV / capital is tiny
    assert big["ret_gross"].abs().mean() != pytest.approx(small["ret_gross"].abs().mean(), rel=1e-3)


def test_the_series_at_t_is_unchanged_when_later_sessions_are_removed(tmp_data_root):
    w = planted(t=330)
    full = A.appa_run(w, 0, 1_000_000, tmp_data_root)
    cut = A.appa_run(w.truncate(280), 0, 1_000_000, tmp_data_root)
    j = cut.join(full, on="date", suffix="_f")
    assert j.height == cut.height > 100
    for col in ("gross", "ret_gross", "net"):
        assert np.allclose(j[col].to_numpy(), j[col + "_f"].to_numpy(), rtol=0, atol=1e-12)


def test_appendix_a_keys_are_registered_as_intraday_timings():
    reg = base.load_library()
    for key in ("k151_appA_d0_mr", "k151_appA_d0_mr_5m", "k151_appA_d1_mom"):
        s = reg[key]
        assert s.family == "m-intra" and s.timing is not None and "sec A" in s.reference
