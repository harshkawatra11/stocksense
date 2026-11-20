"""G2-E1: the S1 vol model (vol21 | har | garch); every option defaults to the pre-E behaviour."""

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
    assert c.vol_model == "vol21"
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


def test_unknown_vol_model_is_rejected():
    with pytest.raises(ValueError):
        cfg(vol_model="ewma")


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
