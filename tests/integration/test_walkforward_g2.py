from __future__ import annotations

import numpy as np
import pytest

from stocksense.core.state import State
from stocksense.models.base import FitInfo, Model, ModelSpec, build_model
from stocksense.validation.trials import n_trials
from stocksense.validation.walkforward import run_walk_forward
from tests.fixtures.wf_world import WFWorld, build_world

pytestmark = pytest.mark.integration

SPEC = ModelSpec(family="s2-ridge", target="y_oc", features="all", entry="open")


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> WFWorld:
    return build_world(tmp_path_factory.mktemp("wf_g2"), planted_ic=0.06, n=60)


def _state(w: WFWorld) -> State:
    st = State(w.settings)
    st.migrate()
    return st


def _run(w: WFWorld, spec=SPEC, **kw):
    return run_walk_forward(
        spec, w.settings, _state(w), folds=w.folds, calendar=w.calendar, registry=w.registry,
        seeds=(1,), inner_months=2, **kw,
    )  # fmt: skip


def _counting(counter):
    def factory(spec):
        m = build_model(spec)
        orig = m.fit

        def fit(*a, **k):
            counter["n"] += 1
            return orig(*a, **k)

        m.fit = fit  # type: ignore[method-assign]
        return m

    return factory


def test_trial_family_override_changes_registered_family(world):
    st = _state(world)
    before_g2 = n_trials(st, "g2-ridge")
    res = _run(world, trial_family="g2-ridge", fold_names=["f1"])
    assert res.trial_id.startswith("g2-ridge-")
    assert n_trials(st, "g2-ridge") == before_g2 + 1


def test_default_trial_id_unchanged(world):
    a = _run(world, fold_names=["f1"])
    b = _run(world, trial_family=None, fold_names=["f1"])
    assert a.trial_id == b.trial_id and a.trial_id.startswith("s2-ridge-")


def test_fold_names_restrict(world):
    res = _run(world, spec=ModelSpec(family="s2-ridge", target="y_oc", features=["alpha_t"], entry="open"),
               fold_names=["f2", "f4"])  # fmt: skip
    assert [m["fold"] for m in res.fold_metrics] == ["f2", "f4"]
    assert set(res.preds["fold"].unique()) == {"f2", "f4"}


def test_refit_every_2_fits_half_the_times(world):
    c = {"n": 0}
    spec = ModelSpec(family="s2-ridge", target="y_oc", features=["alpha_t", "ret_1"], entry="open")
    res = _run(world, spec=spec, model_factory=_counting(c), trial_family="g2-ridge-r2", refit_every=2)
    assert c["n"] == 3  # 6 folds, seeds=1
    assert set(res.preds["fold"].unique()) == {f.name for f in world.folds}
    for f in world.folds:  # every fold's preds stay inside its own window
        d = res.preds.filter(res.preds["fold"] == f.name)["date"]
        assert d.min() >= f.test_start and d.max() <= f.test_end
    c2 = {"n": 0}
    _run(world, spec=spec, model_factory=_counting(c2), trial_family="g2-ridge-r2", refit_every=2)
    assert c2["n"] == 0  # resumable


def test_refit_every_resumes_a_half_done_group(world):
    c = {"n": 0}
    spec = ModelSpec(family="s2-ridge", target="y_oc", features=["alpha_t", "ret_1", "vol_21"], entry="open")
    _run(world, spec=spec, model_factory=_counting(c), trial_family="g2-ridge-r3", refit_every=2,
         fold_names=["f1", "f2"])  # fmt: skip
    assert c["n"] == 1
    c["n"] = 0
    _run(world, spec=spec, model_factory=_counting(c), trial_family="g2-ridge-r3", refit_every=2,
         fold_names=["f1", "f2", "f3"])  # fmt: skip
    assert c["n"] == 1  # only the group (f3, f4-not-requested) anchor fits


def test_should_stop_halts_between_groups(world):
    c = {"n": 0}
    spec = ModelSpec(family="s2-ridge", target="y_oc", features=["alpha_t", "mkt_ret_1"], entry="open")
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    res = _run(world, spec=spec, model_factory=_counting(c), trial_family="g2-ridge-stop", should_stop=stop)
    assert c["n"] == 2
    assert set(res.preds["fold"].unique()) == {"f1", "f2"}


class _KeyProbe(Model):
    needs_keys = True
    target_prefix = "r_"
    seen: list[np.ndarray] = []

    def fit(self, X, y, w, dates, X_val=None, y_val=None, dates_val=None, seed=0):
        _KeyProbe.seen.append(X)
        return FitInfo(best_iteration=1)

    def refit_full(self, X, y, w, dates, fit_info, seed):
        pass

    def predict(self, X):
        _KeyProbe.seen.append(X)
        return np.asarray(X[:, 0], dtype=float)

    def save(self, dir):
        pass

    def load(self, dir):
        pass


def test_needs_keys_appends_sid_and_days(world):
    from stocksense.models.seqnet import date_to_days

    _KeyProbe.seen = []
    spec = ModelSpec(family="s2-probe", target="y_oc", features=["alpha_t", "ret_1"], entry="open")
    res = _run(
        world, spec=spec, model_factory=lambda s: _KeyProbe(), trial_family="g2-probe", fold_names=["f1"]
    )
    X = _KeyProbe.seen[-1]
    assert X.shape[1] == 2 + 2
    assert set(np.unique(X[:, -2]).astype(int)) <= set(res.preds["sid"].unique().to_list())
    d = res.preds["date"].to_numpy()
    assert set(np.unique(X[:, -1]).astype(int)) == set(np.unique(date_to_days(d)).astype(int))
