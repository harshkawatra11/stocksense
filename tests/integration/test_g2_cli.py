from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from stocksense import cli_research
from stocksense.cli import app
from stocksense.core.gpu import gpu_lock
from stocksense.core.paths import Paths
from stocksense.core.state import State
from stocksense.models.base import ModelSpec
from stocksense.validation import g2_eval, g2_select
from stocksense.validation.trials import n_trials
from tests.fixtures.g1_world import build_g1_world, run_model_trials
from tests.fixtures.wf_world import WFWorld, build_world

pytestmark = pytest.mark.integration
runner = CliRunner()
FOLDS = "f1:2019-05-01:2019-06-30,f2:2019-07-01:2019-08-31,f3:2019-09-01:2019-10-31,f4:2019-11-01:2019-12-31"


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> WFWorld:
    return build_world(tmp_path_factory.mktemp("g2_cli"), planted_ic=0.08, n=80)


def _run(w, *args):
    return runner.invoke(app, list(args), obj=w.settings)


def _spec_file(tmp_path, **kw):
    spec = ModelSpec(family="s2-ridge", target="y_oc", features=["alpha_t", "ret_1"], entry="open", **kw)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    return p


def test_wf_run_family_folds_refit_and_resume(world, tmp_path):
    p = _spec_file(tmp_path)
    args = (
        "research", "wf-run", "--spec-json", str(p), "--trial-family", "g2-ridge", "--folds", "f1,f2,f3,f4",
        "--refit-every", "2", "--seeds", "1", "--inner-months", "2", "--custom-folds", FOLDS,
    )  # fmt: skip
    res = _run(world, *args)
    assert res.exit_code == 0, res.output
    assert "trial g2-ridge-y_oc-" in res.output and "fold f4: IC" in res.output
    again = _run(world, *args)
    assert again.exit_code == 0 and "trial g2-ridge-y_oc-" in again.output


def test_wf_run_wall_cap_stops_before_any_fold(world, tmp_path):
    p = _spec_file(tmp_path, params={"x": 1})
    res = _run(
        world, "research", "wf-run", "--spec-json", str(p), "--trial-family", "g2-cap", "--folds", "f1,f2",
        "--max-hours", "0", "--seeds", "1", "--inner-months", "2", "--custom-folds", FOLDS,
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert "stopped: 0/2 folds done" in res.output


def test_wf_run_and_select_take_the_gpu_lock(world, tmp_path):
    p = _spec_file(tmp_path, params={"y": 2})
    with gpu_lock(world.settings):
        res = _run(
            world, "research", "wf-run", "--spec-json", str(p), "--folds", "f1", "--custom-folds", FOLDS
        )
        assert res.exit_code == 3 and "gpu busy" in res.output
        cfg = tmp_path / "c.json"
        cfg.write_text(json.dumps([{"alphas": [1.0]}]), encoding="utf-8")
        res = _run(
            world, "research", "select-nn", "--family", "ridge", "--target", "y_oc", "--configs", str(cfg),
            "--fold", "f2", "--custom-folds", FOLDS,
        )  # fmt: skip
        assert res.exit_code == 3 and "gpu busy" in res.output


def test_select_nn_scores_configs_on_inner_validation_and_registers_nothing(world, tmp_path):
    st = State(world.settings)
    st.migrate()
    before = n_trials(st)
    cfgs = [{"alphas": [1e-6]}, {"alphas": [1e3]}, {"alphas": [1.0, 100.0]}]
    out = g2_select.select_nn(
        "ridge", "y_oc", cfgs, world.settings, fold_name="f3", folds=world.folds, calendar=world.calendar,
        inner_months=2, universe="u500",
    )  # fmt: skip
    assert [r["config"] for r in out["table"]] == cfgs
    assert all(r["val_ic"] is not None for r in out["table"])
    assert out["best"] in cfgs and out["best_val_ic"] == max(r["val_ic"] for r in out["table"])
    assert out["family"] == "g2-ridge" and out["fold"] == "f3"
    saved = json.loads(g2_select.selection_path(world.settings, "ridge").read_text(encoding="utf-8"))
    assert saved["best"] == out["best"]
    assert (
        g2_select.selection_path(world.settings, "ridge").parent
        == Paths(world.settings).models / "_selection"
    )
    assert n_trials(st) == before


def test_select_nn_limits_configs_and_stop(world):
    with pytest.raises(ValueError):
        g2_select.select_nn(
            "ridge", "y_oc", [{}] * 7, world.settings, folds=world.folds, calendar=world.calendar
        )
    with pytest.raises(ValueError):
        g2_select.select_nn("ridge", "y_oc", [], world.settings, folds=world.folds, calendar=world.calendar)
    out = g2_select.select_nn(
        "ridge", "y_oc", [{"alphas": [1.0]}, {"alphas": [2.0]}], world.settings, fold_name="f3",
        folds=world.folds, calendar=world.calendar, inner_months=2, should_stop=lambda: True, write=False,
    )  # fmt: skip
    assert all(r.get("skipped") == "wall cap" for r in out["table"]) and out["best"] is None


def test_select_nn_cli(world, tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps([{"alphas": [1.0]}, {"alphas": [100.0]}]), encoding="utf-8")
    res = _run(
        world, "research", "select-nn", "--family", "ridge", "--target", "y_oc", "--configs", str(cfg),
        "--fold", "f3", "--inner-months", "2", "--custom-folds", FOLDS,
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert "selected" in res.output and "g2-ridge.json" in res.output
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{}] * 7), encoding="utf-8")
    res = _run(
        world, "research", "select-nn", "--family", "ridge", "--target", "y_oc", "--configs", str(bad),
        "--fold", "f3", "--custom-folds", FOLDS,
    )  # fmt: skip
    assert res.exit_code == 2


# ---- model-eval and g2-table on the Gate-G1 world ------------------------------------------------------


@pytest.fixture(scope="module")
def g1world(tmp_path_factory):
    w = build_g1_world(tmp_path_factory.mktemp("g2_cli_g1"))
    st = State(w.settings)
    st.migrate()
    return w, run_model_trials(w, st)


def test_model_eval_and_g2_table(g1world, monkeypatch):
    w, trials = g1world
    monkeypatch.setattr(g2_eval, "MIN_NAMES", 20)
    tid = trials["y_c2o_10"]
    res = _run(w, "research", "model-eval", "--trial", tid, "--base", tid, "--no-backtest")
    assert res.exit_code == 0, res.output
    assert tid in res.output and "no incremental value established" in res.output
    assert (g2_eval.eval_dir(w.settings, tid) / "eval.json").is_file()
    tab = _run(w, "research", "g2-table")
    assert tab.exit_code == 0 and tid in tab.output
    bad = _run(w, "research", "model-eval", "--trial", "nope", "--no-backtest")
    assert bad.exit_code == 1
    miss = _run(w, "research", "model-eval", "--trial", tid, "--no-backtest")  # no GBM base in this world
    assert miss.exit_code == 1 and "no finished" in miss.output


def test_parse_fold_names():
    assert cli_research.parse_fold_names("2016..2018") == ["2016", "2017", "2018"]
    assert cli_research.parse_fold_names("a, b") == ["a", "b"]


def test_cli_registers_the_commands():
    names = {c.name for c in cli_research.research_app.registered_commands}
    assert {"model-eval", "g2-table", "wf-run", "select-nn"} <= names
