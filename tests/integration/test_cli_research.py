from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from stocksense import cli_research
from stocksense.cli import app
from stocksense.core.paths import Paths
from stocksense.models.base import ModelSpec
from tests.fixtures.wf_world import WFWorld, build_world

pytestmark = pytest.mark.integration
runner = CliRunner()
FOLDS = "f1:2019-05-01:2019-06-30,f2:2019-07-01:2019-08-31"


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> WFWorld:
    return build_world(tmp_path_factory.mktemp("cli_world"), planted_ic=0.08, n=80)


@pytest.fixture(autouse=True)
def tiny_specs(monkeypatch):
    spec = ModelSpec("floor-ridge", "y_oc", "u500", "all", {}, "open")
    monkeypatch.setattr(cli_research, "named_specs", lambda settings: {"t-ridge-y_oc": spec})


def _run(world, *args):
    return runner.invoke(app, list(args), obj=world.settings)


def test_train_list_and_unknown_spec(world):
    res = _run(world, "train", "--list")
    assert res.exit_code == 0 and "t-ridge-y_oc" in res.output
    bad = _run(world, "train", "--spec", "nope")
    assert bad.exit_code != 0 and "unknown spec" in bad.output


def test_train_then_status_and_ic_table(world):
    res = _run(
        world, "train", "--spec", "t-ridge-y_oc", "--seeds", "1", "--folds", FOLDS, "--inner-months", "2"
    )
    assert res.exit_code == 0, res.output
    assert "floor-ridge-y_oc-" in res.output and "mean IC" in res.output
    st = _run(world, "research", "wf-status", "--expected", "2")
    assert st.exit_code == 0 and "floor-ridge-y_oc-" in st.output and "2/2" in st.output
    tab = _run(world, "research", "ic-table", "--family", "floor-ridge")
    assert tab.exit_code == 0, tab.output
    assert "floor-ridge-y_oc-" in tab.output and "2019" in tab.output


def test_train_is_resumable_and_validate_unknown_candidate(world):
    again = _run(
        world, "train", "--spec", "t-ridge-y_oc", "--seeds", "1", "--folds", FOLDS, "--inner-months", "2"
    )
    assert again.exit_code == 0
    v = _run(world, "validate", "no-such-candidate")
    assert v.exit_code != 0


def test_parse_folds_uses_prior_session(world):
    from stocksense.core.calendar import TradingCalendar

    cal = TradingCalendar.from_dates(world.market.sessions)
    folds = cli_research.parse_folds(FOLDS, cal)
    assert [f.name for f in folds] == ["f1", "f2"]
    assert folds[0].train_end == cal.prev_session(folds[0].test_start)


def test_named_specs_cover_floors_and_need_selection_for_gbm(tmp_data_root, monkeypatch):
    monkeypatch.undo()
    specs = cli_research.named_specs(tmp_data_root)
    assert "floor-ridge-y_oc" in specs and "floor-single-vol_21-y_c2o_5" in specs
    assert sum(n.startswith("floor-") for n in specs) == 32
    assert not any(n.startswith("gbm-") for n in specs)  # no selection written yet
    d = Paths(tmp_data_root).models / "_selection"
    d.mkdir(parents=True)
    (d / "y_oc_regression.json").write_text(
        json.dumps({"best": {"num_leaves": 15, "min_child_samples": 1000, "colsample_bytree": 0.4}})
    )
    s = cli_research.named_specs(tmp_data_root)["gbm-y_oc-regression"]
    assert s.universe == "u200" and s.entry == "open" and s.params["auction"] is True
    assert s.params["num_leaves"] == 15 and s.params["objective"] == "regression" and s.family == "s2-gbm"
