from __future__ import annotations

import json

import pytest

from stocksense.core.state import State
from stocksense.validation import g2_eval as ev
from stocksense.validation.g1 import G1Config, G1Runner
from stocksense.validation.trials import n_trials
from tests.fixtures.g1_world import build_g1_world, run_model_trials

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_g1_world(tmp_path_factory.mktemp("g2_eval"))


@pytest.fixture(scope="module")
def state(world):
    st = State(world.settings)
    st.migrate()
    return st


@pytest.fixture(scope="module")
def trials(world, state):
    return run_model_trials(world, state)


def _runner(world, state):
    cfg = G1Config(start=world.folds[0].test_start, capitals=(500_000.0,), reference_capital=500_000.0)
    return G1Runner(world.settings, state, world.calendar, cfg)


def test_find_base_trial_needs_a_gbm_trial(world, state, trials):
    with pytest.raises(LookupError):
        ev.find_base_trial(state, "y_c2o_10")  # only ridge trials exist in the synthetic world


def test_evaluate_model_against_itself_writes_eval_json(world, state, trials):
    tid = trials["y_c2o_10"]
    out = ev.evaluate_model(
        tid, world.settings, state, world.calendar, base_id=tid, backtest=False, min_names=20
    )
    assert out["trial_id"] == tid and out["base_trial"] == tid
    assert abs(out["orth_ic"]) < 1e-9 and abs(out["blend_gain"]) < 1e-9 and out["n_dates"] > 20
    assert out["mean_ic"] > 0.02 and out["admitted"] is False and out["net_sharpe_s1"] is None
    p = ev.eval_dir(world.settings, tid) / "eval.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["mean_ic"] == pytest.approx(out["mean_ic"])
    assert tid in [r["trial_id"] for r in ev.load_evals(world.settings)]


def test_evaluate_model_against_a_neighbouring_horizon(world, state, trials):
    a, b = trials["y_c2o_5"], trials["y_c2o_10"]
    out = ev.evaluate_model(a, world.settings, state, world.calendar, base_id=b, backtest=False, min_names=20)
    assert -1.0 <= out["corr_with_base"] <= 1.0 and out["horizon"] == 5


def test_s1_backtest_registers_a_g2_s1_trial_and_is_idempotent(world, state, trials):
    tid = trials["y_c2o_10"]
    run = _runner(world, state)
    before = n_trials(state, "g2-s1")
    r1 = ev.s1_backtest(tid, world.settings, state, world.calendar, runner=run)
    assert r1 is not None and r1["s1_trial"].startswith("g2-s1-y_c2o_10-")
    assert n_trials(state, "g2-s1") == before + 1
    assert r1["auto_resume"] == 20 and r1["sharpe"] == r1["sharpe"]  # finite
    con = state.connect()
    try:
        spec = json.loads(
            con.execute("SELECT spec_json FROM trials WHERE trial_id=?", (r1["s1_trial"],)).fetchone()[0]
        )
    finally:
        con.close()
    assert spec["model_eval_of"] == tid and spec["models"] == [tid] and spec["auto_resume_sessions"] == 20
    r2 = ev.s1_backtest(tid, world.settings, state, world.calendar, runner=run)
    assert r2["s1_trial"] == r1["s1_trial"] and n_trials(state, "g2-s1") == before + 1


def test_s1_backtest_is_none_for_the_open_target(world, state, trials):
    assert ev.s1_backtest(trials["y_oc"], world.settings, state, world.calendar) is None


def test_eval_with_backtest_fills_net_sharpe(world, state, trials):
    tid = trials["y_c2o_20"]
    out = ev.evaluate_model(
        tid, world.settings, state, world.calendar, base_id=trials["y_c2o_10"], min_names=20,
        runner=_runner(world, state),
    )  # fmt: skip
    s = out["net_sharpe_s1"]
    assert s is not None and {"sharpe", "net_2x", "mean_turnover", "s1_trial"} <= set(s)
