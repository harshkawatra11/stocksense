from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from typer.testing import CliRunner

from stocksense.cli import app
from stocksense.data import io

runner = CliRunner()

NOT_IMPL = {
    "ingest": 2,
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


def test_ingest_eq_is_a_real_subcommand_and_bare_ingest_stays_stubbed(tmp_data_root):
    assert runner.invoke(app, ["ingest", "eq", "--help"], obj=tmp_data_root).exit_code == 0
    assert runner.invoke(app, ["ingest"], obj=tmp_data_root).exit_code == 2
    res = runner.invoke(app, ["ingest", "eq", "--from", "2020-01-01", "--to", "nonsense"], obj=tmp_data_root)
    assert res.exit_code != 0


def test_build_panel_runs_all_four_builds(tmp_path_factory):
    from tests.fixtures.synth import _settings_for, make_market, write_market

    root = tmp_path_factory.mktemp("cli_panel")
    write_market(make_market(), root)
    s = _settings_for(root)
    s = s.model_copy(
        update={
            "universe": s.universe.model_copy(
                update={
                    "u500_size": 15,
                    "u500_exit_rank": 20,
                    "u200_size": 6,
                    "u200_exit_rank": 8,
                    "min_med_turnover_inr": 1.0,
                    "u200_min_med_turnover_inr": 1.0,
                }
            )
        }
    )
    res = runner.invoke(app, ["build-panel"], obj=s)
    assert res.exit_code == 0, res.output
    assert "panel_daily" in res.output and "gap_src" in res.output
    for name in ("security_master", "panel_daily", "panel_universe", "panel_market", "adj_events"):
        assert io.read(s, name).collect().height > 0


def test_announcement_ingest_commands_exist(tmp_data_root):
    for cmd in ("announcements", "board-meetings", "results"):
        res = runner.invoke(app, ["ingest", cmd, "--help"], obj=tmp_data_root)
        assert res.exit_code == 0, res.output
