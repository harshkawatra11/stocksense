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
SPEC = ModelSpec(
    "s2-gbm",
    "y_oc",
    "u500",
    "all",
    {"objective": "regression", "num_leaves": 7, "min_child_samples": 20, "n_estimators": 60},
    "open",
)


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> WFWorld:
    return build_world(tmp_path_factory.mktemp("g1diag"), planted_ic=0.08, n=80)


def test_diagnostics_group_gain_and_masking(world, monkeypatch):
    monkeypatch.setattr(cli_research, "named_specs", lambda settings: {"t-gbm": SPEC})
    res = runner.invoke(
        app,
        ["train", "--spec", "t-gbm", "--seeds", "1", "--folds", FOLDS, "--inner-months", "2"],
        obj=world.settings,
    )
    assert res.exit_code == 0, res.output
    tid = next(ln.split()[1] for ln in res.output.splitlines() if ln.startswith("trial "))
    d = runner.invoke(
        app,
        ["research", "diagnostics", "--trial", tid, "--folds", FOLDS, "--groups", "test,price"],
        obj=world.settings,
    )
    assert d.exit_code == 0, d.output
    out = json.loads((Paths(world.settings).models / "_wf" / tid / "diagnostics.json").read_text())
    # the planted alpha lives in the 'test' group: it must hold most of the gain
    assert out["group_gain"]["test"] > 0.5
    assert abs(sum(out["group_gain"].values()) - 1.0) < 1e-6
    abl = out["ablation"]["test"]
    assert abl["n_dates"] > 20 and abl["ic_with"] > abl["ic_without"] + 0.02
    assert "price" in out["ablation"]
