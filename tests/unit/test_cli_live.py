from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from stocksense.cli import app
from stocksense.core.state import State
from stocksense.live.preflight import daemon_lock
from stocksense.portfolio.construct import book_stop_key

runner = CliRunner()
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def test_resume_clears_the_book_stop_flag(tmp_data_root):
    st = State(tmp_data_root)
    st.migrate()
    st.kv_set(book_stop_key("main"), "1")
    res = runner.invoke(app, ["resume", "--book", "main"], obj=tmp_data_root)
    assert res.exit_code == 0 and "cleared" in res.output
    assert st.kv_get(book_stop_key("main")) == "0"
    assert runner.invoke(app, ["resume", "--book", "nope"], obj=tmp_data_root).exit_code == 2


def test_job_rejects_an_unknown_name(tmp_data_root):
    res = runner.invoke(app, ["job", "no_such_job"], obj=tmp_data_root)
    assert res.exit_code == 2 and "unknown job" in res.output


def test_run_once_with_the_lock_held_exits_zero_saying_already_running(tmp_data_root):
    lock = daemon_lock(tmp_data_root)
    assert lock.acquire()
    try:
        res = runner.invoke(app, ["run", "--once"], obj=tmp_data_root)
    finally:
        lock.release()
    assert res.exit_code == 0 and "already running" in res.output


def test_task_scripts_follow_the_runbook_contract():
    inst = (SCRIPTS / "install-task.ps1").read_text(encoding="utf-8")
    for needle in (
        "Register-ScheduledTask -TaskName 'StockSense'", "-AtLogOn", "-Daily -At 07:25", "-WakeToRun",
        "-StartWhenAvailable", "-RestartCount 999", "-MultipleInstances IgnoreNew", "Start-ScheduledTask",
        "run-daemon.ps1",
    ):  # fmt: skip
        assert needle in inst, needle
    un = (SCRIPTS / "uninstall-task.ps1").read_text(encoding="utf-8")
    assert (
        "Stop-ScheduledTask" in un and "Unregister-ScheduledTask -TaskName 'StockSense' -Confirm:$false" in un
    )
    run = (SCRIPTS / "run-daemon.ps1").read_text(encoding="utf-8")
    assert "PYTHONUTF8" in run and "-m stocksense run" in run
