"""G2-E1 (vol model) and G2-E3 (Kelly sizing): S1Config options that default to the pre-E behaviour."""

from __future__ import annotations

from datetime import date

import pytest

from stocksense.portfolio.construct import (
    Holding,
    NameRow,
    PanelDay,
    S1Book,
    S1Config,
    _raw_weight,
    _sel_vol,
    plan_s1,
)
from tests.fixtures.engine_panel import settings, zero_slippage

S = settings()
T, D1 = date(2020, 3, 2), date(2020, 3, 3)
H = 5


def cfg(k_max: int = 5, **opts) -> S1Config:
    return S1Config.from_settings(S, H, zero_slippage(), k_max=k_max, **opts)


def row(vol=0.005, har=None, garch=None, ind="I") -> NameRow:
    return NameRow(100.0, vol, 10, 1e10, ind, vol_har=har, vol_garch=garch)


# ---- E1: vol_model ------------------------------------------------------------------------------------


def test_defaults_are_the_pre_e_behaviour():
    c = cfg()
    assert c.vol_model == "vol21" and c.sizing == "score_over_vol"
    r = row(0.01, har=0.02, garch=0.03)
    assert _sel_vol(r, c) == 0.01  # forecasts present but not selected


def test_vol_model_selects_the_forecast_and_falls_back_to_vol21():
    r = row(0.01, har=0.02, garch=0.03)
    assert _sel_vol(r, cfg(vol_model="har")) == 0.02
    assert _sel_vol(r, cfg(vol_model="garch")) == 0.03
    none = row(0.01)  # no forecast for this name/date: vol_21
    assert _sel_vol(none, cfg(vol_model="har")) == 0.01
    assert _sel_vol(row(0.01, har=float("nan")), cfg(vol_model="har")) == 0.01
    assert _sel_vol(row(0.01, har=0.0), cfg(vol_model="har")) == 0.01  # non-positive: fall back


def test_unknown_vol_model_or_sizing_is_rejected():
    with pytest.raises(ValueError):
        cfg(vol_model="ewma")
    with pytest.raises(ValueError):
        cfg(sizing="martingale")


def _entries(c: S1Config, rows, equity=250_000.0):
    n = len(rows)
    sc = {i: 0.45 - 0.03 * i for i in range(n)}
    er = {i: 0.05 for i in range(n)}
    return plan_s1(T, sc, er, S1Book({}, 1e9), equity, PanelDay(T, D1, rows), c), sc


def test_raw_weights_use_the_selected_vol_and_default_ignores_forecasts():
    # four names, ranks 1..4; sid 0 has a har vol twice its vol_21, the others equal
    # (vol 3 % a day keeps every weight under the per-name cap, so the raw ratios survive allocation)
    rows = {i: row(0.03, har=0.06 if i == 0 else 0.03, ind=f"I{i}") for i in range(4)}
    base, sc = _entries(cfg(k_max=4), rows, equity=1_000_000.0)
    har, _ = _entries(cfg(k_max=4, vol_model="har"), rows, equity=1_000_000.0)
    # default: raw_i = (0.5 + score_i) / 0.03
    assert _raw_weight(cfg(k_max=4), sc[0], 0.05, rows[0], 1.0) == pytest.approx((0.5 + sc[0]) / 0.03)
    assert _raw_weight(cfg(k_max=4, vol_model="har"), sc[0], 0.05, rows[0], 1.0) == pytest.approx(
        (0.5 + sc[0]) / 0.06
    )
    # the plan: sid 0 loses weight relative to sid 1 under har (ratio halves) and not under vol21
    r_base = base.weights[0] / base.weights[1]
    r_har = har.weights[0] / har.weights[1]
    assert r_base == pytest.approx((0.5 + sc[0]) / (0.5 + sc[1]))
    assert r_har == pytest.approx(r_base / 2.0)


def test_entry_log_carries_the_selected_vol_only_for_a_forecast_model():
    rows = {i: row(0.01, har=0.02, ind=f"I{i}") for i in range(4)}
    base, _ = _entries(cfg(k_max=4), rows)
    har, _ = _entries(cfg(k_max=4, vol_model="har"), rows)
    assert all("vol_sel" not in e for e in base.entries)
    assert har.entries and all(e["vol_sel"] == 0.02 and e["vol_21"] == 0.01 for e in har.entries)


def test_the_stop_uses_the_vol_at_entry_it_is_given():
    """vol_at_entry (set from vol_sel by S1Policy) widens the stop: 3 * 0.03 * sqrt(5) = 0.201 > 0.08."""
    rows = {0: NameRow(85.0, 0.005, 10, 1e10)}  # -15 % from entry 100
    sc, er = {0: 0.4}, {0: 0.05}
    narrow = plan_s1(
        T, sc, er, S1Book({0: Holding(0, 500, 100.0, 3, 0.005)}, 1e9), 250_000.0, PanelDay(T, D1, rows), cfg()
    )
    wide = plan_s1(
        T, sc, er, S1Book({0: Holding(0, 500, 100.0, 3, 0.03)}, 1e9), 250_000.0, PanelDay(T, D1, rows), cfg()
    )
    assert narrow.exits == {0: "stop"} and wide.exits == {}


# ---- E3: Kelly sizing ---------------------------------------------------------------------------------


def test_kelly_raw_weights_are_er_over_variance():
    c = cfg(sizing="kelly", kelly_c=0.25)
    a = _raw_weight(c, 0.4, 0.01, row(0.02), 1.0)
    b = _raw_weight(c, 0.1, 0.02, row(0.02), 1.0)
    assert b / a == pytest.approx(2.0)  # er 0.01 : 0.02 at equal vol -> weights 1 : 2 (score is ignored)
    assert a == pytest.approx(0.25 * 0.01 / (0.02**2 * H))  # c * er / sigma_h^2, sigma_h^2 = vol^2 * h
    assert _raw_weight(c, 0.4, -0.001, row(0.02), 1.0) == 0.0
    assert _raw_weight(c, 0.4, 0.0, row(0.02), 1.0) == 0.0
    # the meta multiplier still applies
    assert _raw_weight(c, 0.4, 0.01, row(0.02), 0.5) == pytest.approx(0.5 * a)


def test_kelly_uses_the_selected_vol_model():
    c = cfg(sizing="kelly", kelly_c=0.5, vol_model="garch")
    r = row(0.02, garch=0.04)
    assert _raw_weight(c, 0.0, 0.01, r, 1.0) == pytest.approx(0.5 * 0.01 / (0.04**2 * H))


def test_default_sizing_is_score_over_vol():
    assert _raw_weight(cfg(), 0.3, 0.05, row(0.01), 2.0) == pytest.approx((0.5 + 0.3) / 0.01 * 2.0)


def _kelly_plan(c: float):
    rows = {0: row(0.08, ind="A"), 1: row(0.08, ind="B")}
    er = {0: 0.006, 1: 0.006}
    return plan_s1(
        T, {0: 0.4, 1: 0.3}, er, S1Book({}, 1e9), 2_000_000.0, PanelDay(T, D1, rows),
        cfg(k_max=2, sizing="kelly", kelly_c=c),
    )  # fmt: skip


def test_fractional_kelly_deploys_c_er_over_variance_not_the_whole_gross_target():
    small = _kelly_plan(0.25)
    each = 0.25 * 0.006 / (0.08**2 * H)  # 0.046875 per name, below the gross target (~0.12) and the cap
    assert small.weights[0] == pytest.approx(each) and small.weights[1] == pytest.approx(each)
    assert sum(small.weights.values()) < small.gross_target
    big = _kelly_plan(0.5)  # twice the Kelly weight would exceed the room: capped at the gross target
    assert sum(big.weights.values()) == pytest.approx(big.gross_target, rel=1e-6)
    assert sum(big.weights.values()) > sum(small.weights.values())
