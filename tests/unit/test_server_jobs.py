"""Job registry tests (Phase F1). subprocess.Popen is mocked -- what's
under test is the allowlist/parameter validation (never arbitrary
command construction from a renderer-supplied string), the single-job
enforcement (DuckDB's single-writer constraint made explicit, not
discovered as a confusing lock error), stdout capture into the ring
buffer, and durability writes to `ui_jobs` at start/finish."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from stocksense.data.store import Store
from stocksense.server.jobs import (
    COMMANDS,
    JobAlreadyRunningError,
    JobRegistry,
    MissingParamError,
    UnknownCommandError,
    build_command,
)


@pytest.fixture()
def registry(tmp_path):
    return JobRegistry(tmp_path / "test.duckdb")


def _fake_popen(lines: list[str], returncode: int = 0, pid: int = 4242):
    """A Popen-shaped mock whose .stdout is an iterable of lines (like a
    real pipe would be when iterated), and whose .wait() finalizes
    returncode -- close enough to drive the real _pump_output logic
    without a real subprocess."""
    m = MagicMock()
    m.pid = pid
    m.stdout = iter(l + "\n" for l in lines)
    m.returncode = returncode
    m.wait = MagicMock(return_value=returncode)
    return m


# ---- build_command: the closed-set validation ----

def test_build_command_rejects_unknown_parameters() -> None:
    spec = COMMANDS["backfill-corporate-actions"]
    with pytest.raises(MissingParamError):
        build_command(spec, {"start": "2024-01-01", "end": "2024-01-31", "rm": "-rf /"})


def test_build_command_rejects_missing_required_parameters() -> None:
    spec = COMMANDS["backfill-corporate-actions"]
    with pytest.raises(MissingParamError):
        build_command(spec, {"start": "2024-01-01"})  # end missing


def test_build_command_renders_flags_correctly() -> None:
    spec = COMMANDS["backfill-nse-archive"]
    args = build_command(spec, {"start": "2024-01-01", "end": "2024-01-31", "kind": "cm"})
    assert args == ["backfill-nse-archive", "--start", "2024-01-01", "--end", "2024-01-31", "--kind", "cm"]


def test_build_command_handles_positional_parameters() -> None:
    spec = COMMANDS["statement-ingest"]
    args = build_command(spec, {"file": "statements/foo.xlsx"})
    assert args == ["statement-ingest", "statements/foo.xlsx"]


def test_build_command_no_params_needed_for_simple_commands() -> None:
    spec = COMMANDS["foreman-assess"]
    args = build_command(spec, {})
    assert args == ["foreman", "assess"]


# ---- Phase G/B1: train-candidate and reconcile params were previously
# unreachable (empty param_flags) -- regression coverage that they're
# now real, correctly-flagged parameters, not just present in the dict. ----

def test_train_candidate_params_are_reachable() -> None:
    spec = COMMANDS["train-candidate"]
    args = build_command(spec, {"horizon": 10, "top_n": 30, "cost_bps": 15.0})
    assert args == ["train-candidate", "--horizon", "10", "--top-n", "30", "--cost-bps", "15.0"]


def test_train_candidate_works_with_no_params_too() -> None:
    spec = COMMANDS["train-candidate"]
    assert build_command(spec, {}) == ["train-candidate"]


def test_reconcile_params_are_reachable() -> None:
    spec = COMMANDS["reconcile"]
    args = build_command(spec, {"horizon": 5, "lifecycle": "shadow"})
    assert args == ["reconcile", "--horizon", "5", "--lifecycle", "shadow"]


# ---- Phase G/B2: predict, tax-summary, universe-as-of were entirely
# absent from the allowlist -- previously unreachable from the UI. ----

def test_predict_command_registered() -> None:
    spec = COMMANDS["predict"]
    args = build_command(spec, {"horizon": 20, "lifecycle": "live"})
    assert args == ["predict", "--horizon", "20", "--lifecycle", "live"]


def test_tax_summary_requires_fy_bounds() -> None:
    spec = COMMANDS["tax-summary"]
    with pytest.raises(MissingParamError):
        build_command(spec, {"fy_start": "2024-04-01"})  # fy_end missing
    args = build_command(spec, {"fy_start": "2024-04-01", "fy_end": "2025-03-31"})
    assert args == ["tax-summary", "--fy-start", "2024-04-01", "--fy-end", "2025-03-31"]


def test_universe_as_of_requires_date() -> None:
    spec = COMMANDS["universe-as-of"]
    with pytest.raises(MissingParamError):
        build_command(spec, {})
    args = build_command(spec, {"as_of": "2020-06-30"})
    assert args == ["universe-as-of", "--as-of", "2020-06-30"]


# ---- JobRegistry: allowlist enforcement ----

def test_start_rejects_unregistered_command(registry) -> None:
    with pytest.raises(UnknownCommandError):
        registry.start("rm -rf /", {})


@patch("stocksense.server.jobs.subprocess.Popen")
def test_start_rejects_bad_params_before_spawning_anything(mock_popen, registry) -> None:
    with pytest.raises(MissingParamError):
        registry.start("backfill-corporate-actions", {"start": "2024-01-01"})  # missing 'end'
    mock_popen.assert_not_called()


# ---- JobRegistry: spawn, capture, durability ----

@patch("stocksense.server.jobs.subprocess.Popen")
def test_start_spawns_and_records_ui_job(mock_popen, registry, tmp_path) -> None:
    mock_popen.return_value = _fake_popen(["line one", "line two"])
    job_id = registry.start("foreman-assess", {})

    time.sleep(0.2)  # let the pump thread drain the fake stdout
    store = Store(registry._duckdb_path)
    row = store.read_ui_job(job_id)
    store.close()

    assert row is not None
    assert row["command"] == "foreman-assess"
    assert row["status"] == "completed"
    assert row["pid"] == 4242


@patch("stocksense.server.jobs.subprocess.Popen")
def test_tail_returns_captured_output(mock_popen, registry) -> None:
    mock_popen.return_value = _fake_popen(["alpha", "beta", "gamma"])
    job_id = registry.start("foreman-assess", {})
    time.sleep(0.2)
    assert registry.tail(job_id) == ["alpha", "beta", "gamma"]


@patch("stocksense.server.jobs.subprocess.Popen")
def test_failed_subprocess_recorded_as_failed(mock_popen, registry) -> None:
    mock_popen.return_value = _fake_popen(["oops"], returncode=1)
    job_id = registry.start("foreman-assess", {})
    time.sleep(0.2)

    store = Store(registry._duckdb_path)
    row = store.read_ui_job(job_id)
    store.close()
    assert row["status"] == "failed"


# ---- JobRegistry: single-writer enforcement ----

@patch("stocksense.server.jobs.subprocess.Popen")
def test_second_job_rejected_while_one_is_running(mock_popen, registry) -> None:
    """DuckDB allows only one write-holding process at a time -- this is
    that constraint enforced explicitly rather than surfacing as a
    confusing 'file locked' error from a second subprocess."""
    never_ending = MagicMock()
    never_ending.pid = 111
    never_ending.stdout = iter(())  # yields nothing, thread blocks on nothing but proc.wait() below hangs it "running"
    never_ending.returncode = None

    def _wait_forever():
        time.sleep(10)
        return 0
    never_ending.wait = _wait_forever

    mock_popen.return_value = never_ending
    registry.start("foreman-assess", {})
    time.sleep(0.1)  # ensure the first job is registered as running

    with pytest.raises(JobAlreadyRunningError):
        registry.start("foreman-assess", {})


@patch("stocksense.server.jobs.subprocess.Popen")
def test_stop_kills_the_process(mock_popen, registry) -> None:
    proc = MagicMock()
    proc.pid = 555
    proc.stdout = iter(())

    def _wait_forever():
        time.sleep(10)
        return 0
    proc.wait = _wait_forever

    mock_popen.return_value = proc
    job_id = registry.start("foreman-assess", {})
    time.sleep(0.1)

    with patch("stocksense.server.jobs.subprocess.run") as mock_run, \
         patch("stocksense.server.jobs.os.name", "nt"):
        stopped = registry.stop(job_id)
        assert stopped
        mock_run.assert_called_once()
        killed_cmd = mock_run.call_args[0][0]
        assert "taskkill" in killed_cmd
        assert "555" in killed_cmd


def test_stop_returns_false_for_unknown_job(registry) -> None:
    assert registry.stop("does-not-exist") is False
