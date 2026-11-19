"""G1 trial grid (PLAN T9.5 step 1-2): structure, registration, survivors, score source. No backtests run."""

from __future__ import annotations

import json
from datetime import date

import polars as pl
import pytest

from stocksense.core.calendar import TradingCalendar
from stocksense.core.state import State
from stocksense.validation import g1
from stocksense.validation.g1 import G1Config, ModelTrial
from stocksense.validation.trials import n_trials


@pytest.fixture
def state(tmp_data_root):
    st = State(tmp_data_root)
    st.migrate()
    return st


def _mt(tid: str, target: str) -> ModelTrial:
    return ModelTrial(tid, "s1-gbm", target, {}, 0.03, 4.0)


def _models():
    s1 = {t: [_mt(f"m-{t}", t)] for t in g1.S1_TARGETS}
    return s1, [_mt("m-oc", "y_oc")]


CFG = G1Config(
    start=date(2014, 1, 1),
    end=date(2025, 9, 30),
    capitals=(25000.0, 100000.0, 500000.0, 2500000.0),
    reference_capital=500000.0,
)


def test_grid_is_exactly_the_cards(tmp_data_root):
    s1, s2 = _models()
    plans = g1.plan_grid(CFG.resolved(tmp_data_root), s1, s2)
    kinds = [p.kind for p in plans]
    assert kinds.count("s1") == 12 and kinds.count("s2") == 2 and len(plans) == 14
    s1_cells = {(p.target, p.spec["k_max"], p.spec["overlay"]) for p in plans if p.kind == "s1"}
    assert s1_cells == {(t, k, o) for t in g1.S1_TARGETS for k in (10, 20) for o in (False, True)}
    assert {p.spec["k_max"] for p in plans if p.kind == "s2"} == {5, 10}
    assert {p.family for p in plans if p.kind == "s1"} == {"g1-s1"}
    assert {p.family for p in plans if p.kind == "s2"} == {"g1-s2"}
    assert all(p.spec["capitals"] == [25000.0, 100000.0, 500000.0, 2500000.0] for p in plans)
    assert all(p.spec["stress"] == 2.0 for p in plans)


def test_target_without_survivors_contributes_nothing(tmp_data_root):
    s1, s2 = _models()
    s1["y_c2o_10"] = []
    plans = g1.plan_grid(CFG.resolved(tmp_data_root), s1, s2)
    assert len(plans) == 8 + 2 and "y_c2o_10" not in {p.target for p in plans}


def test_config_defaults_come_from_the_settings(tmp_data_root):
    c = G1Config().resolved(tmp_data_root)
    assert c.start == date(2014, 1, 1) and c.end == date(2025, 9, 30)
    assert c.capitals == (25000.0, 100000.0, 500000.0, 2500000.0) and c.reference_capital == 500000.0


def test_registration_is_idempotent_and_counts_every_trial(tmp_data_root, state):
    s1, s2 = _models()
    plans = g1.plan_grid(CFG.resolved(tmp_data_root), s1, s2)
    before = n_trials(state)
    a = g1.register_plans(state, plans)
    assert n_trials(state) == before + 14 and all(p.trial_id for p in a)
    b = g1.register_plans(state, plans)
    assert [p.trial_id for p in a] == [p.trial_id for p in b] and n_trials(state) == before + 14
    assert len({p.trial_id for p in a}) == 14


def _finish(state, tid, family, target, ic, status="done"):
    spec = {"family": family, "target": target}
    con = state.connect()
    try:
        con.execute(
            "INSERT INTO trials(trial_id, created_at, family, target, spec_json, spec_hash, status, "
            "metrics_json) VALUES(?,?,?,?,?,?,?,?)",
            (tid, "2026-11-12", family, target, json.dumps(spec), tid, status,
             json.dumps({"mean_ic": ic, "ic_tstat": 3.0}) if ic is not None else None),
        )  # fmt: skip
    finally:
        con.close()


def test_survivors_are_finished_positive_ic_trials_of_the_strategy(state):
    _finish(state, "a", "s1-gbm", "y_c2o_5", 0.03)
    _finish(state, "b", "s1-ridge", "y_c2o_5", -0.01)  # negative IC: out
    _finish(state, "c", "s1-seq", "y_c2o_5", 0.05, status="registered")  # not finished: out
    _finish(state, "d", "floor-ridge", "y_c2o_5", 0.04)  # a floor, not a strategy model: out
    _finish(state, "e", "s2-gbm", "y_c2o_5", 0.04)  # another strategy: out
    _finish(state, "f", "s1-lgbm", "y_c2o_10", 0.04)  # another target: out
    got = g1.surviving_models(state, "s1", "y_c2o_5")
    assert [m.trial_id for m in got] == ["a"]
    assert g1.surviving_models(state, "s1", "y_c2o_10")[0].trial_id == "f"
    assert g1.surviving_models(state, "s1", "y_c2o_20") == []


def _blend_frame():
    rows = []
    for d in (date(2020, 1, 6), date(2020, 1, 7)):
        for sid, sc in ((1, 0.4), (2, 0.0), (3, -0.4)):
            rows.append((d, sid, sc, 0.01 * sc, 0.1))
    return pl.DataFrame(rows, schema=["date", "sid", "score", "exp_ret", "disagreement"], orient="row")


def test_score_source_returns_day_scores_and_none_off_the_stored_dates():
    src = g1.StoredScoreSource(_blend_frame())
    ds = src.scores(date(2020, 1, 6))
    assert ds is not None and ds.score == {1: 0.4, 2: 0.0, 3: -0.4}
    assert ds.er[1] == pytest.approx(0.004) and ds.disagreement[3] == pytest.approx(0.1)
    assert src.scores(date(2020, 1, 8)) is None


def test_s2_score_source_asks_for_the_previous_session():
    cal = TradingCalendar.from_dates([date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8)])
    src = g1.StoredScoreSource(_blend_frame(), calendar=cal, trade_date=True)
    assert src.scores(date(2020, 1, 7)).score[1] == 0.4  # scores stored for 01-06 serve trade date 01-07
    assert src.scores(date(2020, 1, 6)) is None  # no previous session
    with pytest.raises(ValueError):
        g1.StoredScoreSource(_blend_frame(), trade_date=True)


def test_walk_forward_meta_is_neutral_outside_folds_and_picks_the_fold_by_date():
    from stocksense.validation.splits import Fold

    f = Fold("a", date(2020, 1, 1), date(2020, 1, 2), date(2020, 6, 30))
    m = g1.WalkForwardMeta(pl.DataFrame(), [f], factory=lambda: _Stub())
    rows = [{"score": 0.1}]
    m.set_date(date(2019, 12, 1))
    assert m.predict_rows(rows).tolist() != [0.9] and str(m.predict_rows(rows)[0]) == "nan"
    m.set_date(date(2020, 3, 1))
    assert m.predict_rows(rows).tolist() == [0.9]
    assert m.size_multiplier(0.9) == pytest.approx(0.8) and m.veto(0.4) and not m.veto(0.6)


class _Stub:
    def fit(self, samples, test_start, seed=0):
        return None

    def predict_rows(self, rows):
        import numpy as np

        return np.full(len(rows), 0.9)
