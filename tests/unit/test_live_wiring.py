"""Live wiring (G2-G1): live ids, the predict stage and the live score source on the synthetic market."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from stocksense.core.clock import IST, SteppingClock
from stocksense.core.errors import DataNotAvailable
from stocksense.core.state import State
from stocksense.data import io
from stocksense.features.build import load_features
from stocksense.learn.nightly import fit_production
from stocksense.live import wiring
from stocksense.models.base import ModelSpec
from stocksense.validation.walkforward import run_walk_forward
from tests.fixtures.g1_world import LOCKBOX_START, build_g1_world

SPEC = ModelSpec(family="s1-ridge", target="y_c2o_10", universe="u500", entry="close")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    w = build_g1_world(tmp_path_factory.mktemp("wiring"), planted_ic=0.08)
    clock = SteppingClock(datetime(2020, 3, 20, 18, 30, tzinfo=IST))
    st = State(w.settings, clock)
    st.migrate()
    res = run_walk_forward(
        SPEC,
        w.settings,
        st,
        folds=w.folds,
        calendar=w.calendar,
        registry=w.registry,
        seeds=(1, 2),
        inner_months=2,
        trial_family="g2-test",
    )
    as_of = w.calendar.prev_session(LOCKBOX_START)
    mid = fit_production(
        SPEC,
        w.settings,
        st,
        as_of,
        seeds=(1, 2),
        calendar=w.calendar,
        inner_months=2,
        allow_beyond_research=True,
    )
    st.kv_set("live_model:s1", mid)
    st.kv_set("live_wf_trial:s1", res.trial_id)
    feats = load_features(w.settings, "production", columns=["planted_c2o_10"]).collect()
    last = feats["date"].max()
    ctx = SimpleNamespace(settings=w.settings, state=st, clock=clock, calendar=w.calendar)
    return SimpleNamespace(w=w, st=st, ctx=ctx, mid=mid, last=last, trial=res.trial_id)


def test_live_ids_initialises_missing_keys_with_defaults(tmp_path):
    from tests.fixtures.synth import _settings_for

    st = State(_settings_for(tmp_path))
    st.migrate()
    ids = wiring.live_ids(st)
    assert ids["live_model:s1"] == "s1-gbm-y_c2o_10-120055de-20250930"
    assert ids["live_model:s2"] == "s2-gbm-y_oc-4fd8ff9b-20250930"
    assert ids["live_wf_trial:s1"] is None  # no finished walk-forward trial in a fresh state
    assert st.kv_get("live_model:s1") == ids["live_model:s1"]
    st.kv_set("live_model:s1", "other")
    assert wiring.live_ids(st)["live_model:s1"] == "other"  # existing values are kept


def test_predict_stage_writes_live_preds_for_every_u500_name(env):
    out = wiring.stage_predict(env.ctx, env.last)
    uni = io.read(env.w.settings, "panel_universe", start=env.last, end=env.last).collect()
    n = int(uni["u500"].sum())
    assert out == f"s1:{n} names"
    p = io.read(env.w.settings, "preds", start=env.last, end=env.last, sub=env.mid).collect()
    assert p.height == n and set(p["mode"]) == {"live"}


def test_live_score_source_ranks_and_gives_nonnegative_er(env):
    wiring.stage_predict(env.ctx, env.last)
    sc = wiring.LiveScoreSource(env.w.settings, env.st).scores(env.last)
    assert sc is not None and len(sc.score) > 50
    v = np.array(list(sc.score.values()))
    assert v.min() >= -0.5 - 1e-9 and v.max() <= 0.5 + 1e-9
    assert min(sc.er.values()) >= 0.0 and max(sc.er.values()) > 0.0
    assert (
        wiring.LiveScoreSource(env.w.settings, env.st).scores(env.w.calendar.prev_session(env.last)) is None
    )


def test_decile_er_map_option(env):
    sc = wiring.LiveScoreSource(env.w.settings, env.st, er_map="decile").scores(env.last)
    assert sc is not None and all(np.isfinite(x) and x >= 0.0 for x in sc.er.values())


def test_missing_features_raise_data_not_available(env):
    with pytest.raises(DataNotAvailable):
        wiring.predict_frame(env.w.settings, env.mid, env.last.replace(year=2031))


def test_stage_in_jobs_uses_the_hook_and_skips_without_one(env):
    from stocksense.live import jobs

    assert jobs._stage_predict(SimpleNamespace(hook=lambda n: None), env.last).startswith("skipped")
    hooks = {"predict": wiring.stage_predict}
    ctx = SimpleNamespace(**vars(env.ctx), hook=hooks.get)
    assert jobs._stage_predict(ctx, env.last).startswith("s1:")
    assert "predict" not in jobs.STUB_STAGES


