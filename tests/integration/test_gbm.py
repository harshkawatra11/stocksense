from __future__ import annotations

import numpy as np
import pytest

from stocksense.core.state import State
from stocksense.models import gbm as gbm_mod
from stocksense.models.base import ModelSpec
from stocksense.validation.trials import n_trials
from stocksense.validation.walkforward import run_walk_forward
from tests.fixtures.wf_world import WFWorld, build_world

pytestmark = pytest.mark.integration

PARAMS = {"n_estimators": 150, "num_leaves": 7, "min_child_samples": 50}


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> WFWorld:
    return build_world(tmp_path_factory.mktemp("gbm_world"), planted_ic=0.08)


def _state(w):
    st = State(w.settings)
    st.migrate()
    return st


def test_oof_ic_recovers_planted_signal(world):
    spec = ModelSpec("s2-gbm", "y_oc", features="all", params=PARAMS, entry="open")
    res = run_walk_forward(
        spec, world.settings, _state(world), folds=world.folds, calendar=world.calendar,
        registry=world.registry, seeds=(1, 2), inner_months=2,
    )  # fmt: skip
    print(f"\ngbm synthetic OOF IC {res.ic['mean_ic']:.4f} t {res.ic['ic_tstat']:.1f}")
    assert res.ic["mean_ic"] > 0.04


def test_selection_reads_no_test_rows_and_registers_no_trial(world, monkeypatch):
    st = _state(world)
    before = n_trials(st)
    anchors = [world.folds[2], world.folds[4]]
    seen = {"loads": [], "fit_dates": []}
    orig_lf = gbm_mod.load_features

    def spy_lf(settings, purpose, start=None, end=None, columns=None):
        seen["loads"].append(end)
        return orig_lf(settings, purpose, start, end, columns)

    monkeypatch.setattr(gbm_mod, "load_features", spy_lf)
    orig_fit = gbm_mod.GbmModel.fit

    def spy_fit(self, X, y, w, dates, X_val=None, y_val=None, dates_val=None, seed=0):
        seen["fit_dates"].append(
            (dates.max().astype("datetime64[D]"), dates_val.max().astype("datetime64[D]"))
        )
        return orig_fit(self, X, y, w, dates, X_val, y_val, dates_val, seed)

    monkeypatch.setattr(gbm_mod.GbmModel, "fit", spy_fit)
    grid = gbm_mod.GRID[:2]
    out = gbm_mod.select_hyperparams(
        "y_oc", "regression", world.settings, st, anchors=anchors, date_frac=0.5, seed=1,
        folds=anchors, calendar=world.calendar, inner_months=2, grid=grid, base_params=PARAMS,
        universe="u500", entry="open",
    )  # fmt: skip
    first_test = np.datetime64(min(a.test_start for a in anchors), "D")
    last_test = max(a.test_start for a in anchors)
    last_np = np.datetime64(last_test, "D")
    assert all(e is not None and e < last_test for e in seen["loads"])
    assert len(seen["fit_dates"]) == len(grid) * len(anchors)
    assert all(max(a, b) < last_np for a, b in seen["fit_dates"])
    assert min(max(a, b) for a, b in seen["fit_dates"]) < first_test
    assert out["best"] in [dict(g) for g in grid]
    assert n_trials(st) == before
