from __future__ import annotations

import tomllib

from typer.testing import CliRunner

from stocksense.cli import app
from stocksense.cli_lab import registry_toml_text
from stocksense.strategies import base
from tests.unit.test_explore_scan import ASOF_IDX, sessions, write_panel

runner = CliRunner()


def test_registry_toml_parses_and_lists_every_strategy():
    doc = tomllib.loads(registry_toml_text())
    reg = base.load_library()
    assert set(doc["strategy"]) == set(reg)
    vcp = doc["strategy"]["ml_swing_vcp"]
    assert vcp["name"] == "VCP tightening break" and vcp["default_exit"] == "rr3_40"
    assert "mr5" not in vcp["exits"] and vcp["params"] == {} and vcp["timeframe"] == "1d"
    cli = runner.invoke(app, ["lab", "registry-toml"])
    assert cli.exit_code == 0 and "[strategy.ml_swing_vcp]" in cli.output


def test_lab_scan_command_writes_the_files(tmp_data_root):
    write_panel(tmp_data_root)
    day = sessions()[ASOF_IDX].isoformat()
    res = runner.invoke(app, ["lab", "scan", "--asof", day, "--workers", "1"], obj=tmp_data_root)
    assert res.exit_code == 0, res.output
    assert (tmp_data_root.paths.data_root / "lab" / "scan" / f"{day}.parquet").is_file()


def test_lab_explore_rejects_an_unknown_strategy(tmp_data_root):
    res = runner.invoke(
        app, ["lab", "explore", "--sym", "S00", "--strat", "no such strategy"], obj=tmp_data_root
    )
    assert res.exit_code == 2 and "error" in res.output
