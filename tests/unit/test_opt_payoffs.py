"""M5.4c: option combo payoffs against the closed-form equations of the paper, and the registry."""

from __future__ import annotations

import numpy as np
import pytest

from stocksense.strategies.base import REGISTRY, load_library
from stocksense.strategies.k151_opt import COMBOS, L
from stocksense.strategies.options_bt import Leg, intrinsic, resolve_strikes, terminal_pnl

NAN = float("nan")


def pp(x: float) -> float:
    return max(x, 0.0)


def test_covered_call_eq1():
    s0, k, c = 100.0, 102.0, 1.5
    legs = (Leg("FUT", 1), Leg("CE", -1, 1))
    for st in (80.0, 100.0, 102.0, 110.0):
        got = terminal_pnl(legs, (NAN, k), (s0, c), st)
        assert got == pytest.approx(st - s0 - pp(st - k) + c)


def test_iron_condor_eq226():
    k1, k2, k3, k4 = 90.0, 95.0, 105.0, 110.0
    p1, p2, c3, c4 = 0.5, 1.5, 1.4, 0.4
    legs = (Leg("PE", 1), Leg("PE", -1), Leg("CE", -1), Leg("CE", 1))
    ks, ent = (k1, k2, k3, k4), (p1, p2, c3, c4)
    for st in (80.0, 92.0, 100.0, 107.0, 120.0):
        want = (p2 - p1 + c3 - c4) + pp(k1 - st) - pp(k2 - st) - pp(st - k3) + pp(st - k4)
        assert terminal_pnl(legs, ks, ent, st) == pytest.approx(want)
    assert terminal_pnl(legs, ks, ent, 100.0) == pytest.approx(2.0)
    assert terminal_pnl(legs, ks, ent, 70.0) == pytest.approx(2.0 - 5.0)


def test_long_straddle_payoff_40():
    legs = (Leg("CE", 1), Leg("PE", 1))
    # premium paid 150 in total, ST - K = 190 on the call: net 190 - 150 = 40
    assert terminal_pnl(legs, (100.0, 100.0), (100.0, 50.0), 290.0) == pytest.approx(40.0)


def test_call_ratio_backspread_eq148():
    k1, k2, c1, c2 = 100.0, 105.0, 4.0, 1.5
    legs = (Leg("CE", -1), Leg("CE", 1, 1, 2.0))
    st = k2 + 10
    h = 2 * c2 - c1  # net debit
    assert terminal_pnl(legs, (k1, k2), (c1, c2), st) == pytest.approx(2 * 10 - (st - k1) - h)
    for st in (90.0, 100.0, 105.0, 130.0):
        w = 2 * pp(st - k2) - pp(st - k1) + c1 - 2 * c2
        assert terminal_pnl(legs, (k1, k2), (c1, c2), st) == pytest.approx(w)


def test_butterfly_and_collar_formulas():
    k1, k2, k3 = 95.0, 100.0, 105.0
    c1, c2, c3 = 6.0, 3.0, 1.2
    legs = (Leg("CE", 1), Leg("CE", -1, qty=2.0), Leg("CE", 1))
    for st in (80.0, 97.0, 100.0, 104.0, 120.0):
        w = pp(st - k1) - 2 * pp(st - k2) + pp(st - k3) - (c1 - 2 * c2 + c3)
        assert terminal_pnl(legs, (k1, k2, k3), (c1, c2, c3), st) == pytest.approx(w)
    s0, kp, kc, p, c = 100.0, 95.0, 105.0, 1.0, 1.2
    col = (Leg("FUT", 1), Leg("PE", 1), Leg("CE", -1))
    for st in (80.0, 100.0, 120.0):
        w = st - s0 + pp(kp - st) - pp(st - kc) - p + c
        assert terminal_pnl(col, (NAN, kp, kc), (s0, p, c), st) == pytest.approx(w)


def test_calendar_near_exit_override():
    legs = (Leg("CE", -1), Leg("CE", 1, expiry="next"))
    got = terminal_pnl(legs, (100.0, 100.0), (3.0, 5.0), 104.0, near_exit=(None, 6.5))
    assert got == pytest.approx(-(4.0 - 3.0) + (6.5 - 5.0))


def test_resolve_strikes_equidistant_steps():
    listed = np.arange(90.0, 111.0, 1.0)
    legs = (L("CE", 1, -2), L("CE", 1), L("CE", 1, 1), L("PE", 1, x=-0.05), L("FUT", 1))
    k = resolve_strikes(legs, listed, 100.2)
    assert k[:3] == [96.0, 100.0, 102.0]  # atm 100, step = nearest(102.2) - 100 = 2
    assert k[3] == 95.0 and np.isnan(k[4])
    assert intrinsic("PE", 100.0, 90.0) == 10.0


def test_registry_counts_and_keys():
    reg = load_library()
    opt = [s for s in reg.values() if s.family == "m-opt"]
    assert len(COMBOS) == 57
    assert len(opt) == 2 * 57 + 3
    assert "k151_opt_nifty_covered_call" in REGISTRY and "k151_opt_banknifty_long_iron_condor" in REGISTRY
    assert all(s.kind == "options" and s.timing is not None for s in opt)
    assert {"k151_vrp_straddle", "k151_vrp_straddle_dh", "k151_risk_reversal"} <= set(REGISTRY)
