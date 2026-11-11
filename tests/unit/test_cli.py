from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from typer.testing import CliRunner

from stocksense.cli import app
from stocksense.data import io

runner = CliRunner()

NOT_IMPL = {
    "data-status": 2,
    "ingest": 2,
    "build-panel": 3,
    "build-features": 4,
    "build-labels": 4,
    "train": 7,
    "validate": 6,
    "backtest": 5,
    "run": 10,
    "job": 10,
    "report": 12,
    "resume": 10,
}


def test_doctor_empty_root_exits_0_and_prints_data_root(tmp_data_root):
    res = runner.invoke(app, ["doctor"], obj=tmp_data_root)
    assert res.exit_code == 0, res.output
    assert "data root" in res.output
    assert "schema version" in res.output


def test_doctor_missing_root_exits_1(tmp_data_root, tmp_path):
    s = tmp_data_root.model_copy(
        update={"paths": tmp_data_root.paths.model_copy(update={"data_root": tmp_path / "nope"})}
    )
    res = runner.invoke(app, ["doctor"], obj=s)
    assert res.exit_code == 1


def test_doctor_lists_dataset_rows_and_last_date(tmp_data_root):
    # 2 rows, newest date 2026-02-12.
    def frame(dates):
        return pl.DataFrame({"date": dates, "close": [1.0] * len(dates)})

    pq = io.write_partition(tmp_data_root, "india_vix", "all", frame([date(2026, 1, 5), date(2026, 2, 12)]))
    assert pq.is_file()
    res = runner.invoke(app, ["doctor"], obj=tmp_data_root)
    line = next(ln for ln in res.output.splitlines() if ln.strip().startswith("india_vix"))
    assert "2026-02-12" in line
    assert " 2" in line  # 2 rows


def test_doctor_reports_secret_names_not_values(tmp_data_root):
    res = runner.invoke(app, ["doctor"], obj=tmp_data_root)
    assert "angel_api_key" in res.output
    assert "unset" in res.output or "set" in res.output


def test_doctor_heartbeat_age(tmp_data_root, tmp_data_dir):
    (tmp_data_dir / "state" / "heartbeat.json").write_text("{}", encoding="utf-8")
    res = runner.invoke(app, ["doctor"], obj=tmp_data_root)
    assert "heartbeat" in res.output


@pytest.mark.parametrize("cmd,phase", sorted(NOT_IMPL.items()))
def test_unimplemented_commands_exit_2(tmp_data_root, cmd, phase):
    res = runner.invoke(app, [cmd], obj=tmp_data_root)
    assert res.exit_code == 2
    assert f"not implemented yet (phase {phase})" in res.output


def test_version_still_works():
    assert runner.invoke(app, ["version"]).exit_code == 0
