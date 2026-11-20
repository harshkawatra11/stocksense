"""G2-E7: Hawkes-style announcement intensities, category counts and the result-drift proxy."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.features import events_v2 as ev2
from stocksense.features.registry import FEATURE_MODULES
from tests.unit.test_feature_truncation import (  # noqa: F401  (the session fixture builds the synthetic market)
    _compute,
    truncation_env,
    violations,
)

# ---- the exponential kernel ----------------------------------------------------------------------------


def test_intensity_by_hand_for_events_at_delta_zero_and_two():
    counts = np.array([[1.0, 0.0, 1.0, 0.0, 0.0]])  # events in sessions 0 and 2
    lam2 = ev2.exp_intensity(counts, tau=2.0)
    # at session 2: the new event counts 1, the one two sessions ago exp(-2/2) = e^-1
    assert lam2[0, 2] == pytest.approx(1.0 + math.exp(-1.0))
    assert lam2[0, 0] == pytest.approx(1.0)
    assert lam2[0, 1] == pytest.approx(math.exp(-0.5))
    assert lam2[0, 4] == pytest.approx(math.exp(-2.0) + math.exp(-1.0))  # deltas 4 and 2
    lam10 = ev2.exp_intensity(counts, tau=10.0)
    assert lam10[0, 2] == pytest.approx(1.0 + math.exp(-0.2))
    assert lam10[0, 4] == pytest.approx(math.exp(-0.4) + math.exp(-0.2))


def test_several_events_in_one_session_add():
    lam = ev2.exp_intensity(np.array([[3.0, 0.0, 2.0]]), tau=2.0)
    assert lam[0, 2] == pytest.approx(2.0 + 3.0 * math.exp(-1.0))


def test_row_intensity_uses_sessions_not_rows():
    """A stock with a trading gap: the events' age is counted in calendar sessions."""
    events = pl.DataFrame({"sid": [7, 7], "sx": [0, 4], "n": [1.0, 1.0]})
    rows = pl.DataFrame({"sid": [7, 7, 7], "sx": [4, 5, 9]})
    out = ev2.row_intensity(events, rows, tau=2.0, n_sessions=10)
    assert out.to_list()[0] == pytest.approx(1.0 + math.exp(-2.0))  # Delta 4 -> e^{-4/2}
    assert out.to_list()[1] == pytest.approx(math.exp(-0.5) + math.exp(-2.5))
    assert out.to_list()[2] == pytest.approx(math.exp(-2.5) + math.exp(-4.5))


# ---- categories ----------------------------------------------------------------------------------------


def test_the_eight_categories_and_their_matching():
    assert len(ev2.CATEGORIES) == 8
    assert len(set(ev2.CATEGORIES.values())) == 8
    assert ev2.category_name("Press Release") == "ann_cat_press_21"
    assert ev2.category_name("  press release ") == "ann_cat_press_21"  # case and padding are ignored
    assert ev2.category_name("Outcome of Board Meeting") == "ann_cat_outcome_21"
    assert ev2.category_name("Financial Result Updates") == "ann_cat_result_21"
    assert ev2.category_name("Analysts/Institutional Investor Meet/Con. Call Updates") == "ann_cat_analyst_21"
    assert ev2.category_name("Loss of Share Certificates") is None  # not among the eight


def test_category_counts_over_21_sessions_by_hand():
    d0 = date(2020, 1, 1)
    panel = pl.DataFrame(
        {
            "date": [d0 + timedelta(days=i) for i in range(30)],
            "sid": [1] * 30,
            "sx": list(range(30)),
        }
    )
    events = pl.DataFrame(
        {
            "sid": [1, 1, 1, 1],
            "date": [d0, d0 + timedelta(days=2), d0 + timedelta(days=2), d0 + timedelta(days=25)],
            "col": ["ann_cat_press_21", "ann_cat_press_21", "ann_cat_updates_21", "ann_cat_press_21"],
        }
    )
    out = ev2.category_counts(events, panel, window=21).sort("date")
    press = out["ann_cat_press_21"].to_list()
    assert press[19] is None  # fewer than 21 rows
    assert press[20] == 2.0  # events on rows 0 and 2 are inside rows 0..20
    assert press[21] == 1.0  # row 0 dropped out
    assert press[23] == 0.0  # row 2 dropped out
    assert press[29] == 1.0  # the event on row 25
    assert out["ann_cat_updates_21"].to_list()[20] == 1.0
    assert out["ann_cat_outcome_21"].to_list()[20] == 0.0  # no such event: zero, not null


# ---- result drift --------------------------------------------------------------------------------------


def test_result_drift_proxy_decays_with_sessions_since_the_result():
    df = pl.DataFrame(
        {
            "result_abret": [0.04, 0.04, -0.03, None],
            "days_since_result": [0, 20, 40, 5],
        }
    )
    out = df.select(ev2.result_drift_expr()).to_series().to_list()
    assert out[0] == pytest.approx(0.04)
    assert out[1] == pytest.approx(0.04 * math.exp(-1.0))
    assert out[2] == pytest.approx(-0.03 * math.exp(-2.0))
    assert out[3] is None


# ---- registry and truncation -----------------------------------------------------------------------------


def test_the_features_are_not_in_the_default_build():
    assert "events_v2" not in FEATURE_MODULES
    assert {"ann_intensity", "ann_categories", "result_drift"} <= {s.name for s in ev2.SPECS}
    assert ev2.OUTPUTS[:2] == ("ann_intensity_2", "ann_intensity_10")
    assert ev2.OUTPUTS[-1] == "result_drift_proxy"


@pytest.mark.parametrize("name", ["ann_intensity", "ann_categories", "result_drift"])
def test_feature_does_not_look_ahead(name, truncation_env):  # noqa: F811
    s, sessions, _ = truncation_env
    spec = next(x for x in ev2.SPECS if x.name == name)
    full = _compute(spec, s, None)
    assert full.height > 0
    non_null = {o: full[o].drop_nulls().len() for o in spec.outputs}
    if name != "ann_categories":
        assert all(v > 0 for v in non_null.values()), f"the guard would be vacuous: {non_null}"
    else:  # the synthetic announcements have two categories: at least press release must be populated
        assert non_null["ann_cat_press_21"] > 0
    for idx in (300, 380, 450, 520, 590):
        cut = sessions[idx]
        bad = violations(spec, full, _compute(spec, s, cut), cut)
        assert not bad, f"{name} @ {idx} ({cut}): " + "; ".join(bad)
