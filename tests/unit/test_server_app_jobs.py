"""Endpoint tests for Phase F1's job-trigger/watch API and F4's settings
API, via FastAPI's TestClient. subprocess.Popen is mocked -- what's
under test is HTTP-level behavior: status codes for unknown/invalid/
conflicting job requests, the job list merging live+durable status, the
WebSocket stream, and settings round-tripping through the store."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.duckdb"
    monkeypatch.setenv("STOCKSENSE_DUCKDB_PATH", str(db_path))

    import stocksense.server.app as app_mod
    app_mod._registry = None  # force a fresh JobRegistry bound to this test's tmp db
    return TestClient(app_mod.app)


def _fake_popen(lines, returncode=0, pid=999):
    m = MagicMock()
    m.pid = pid
    m.stdout = iter(l + "\n" for l in lines)
    m.returncode = returncode
    m.wait = MagicMock(return_value=returncode)
    return m


def test_job_commands_lists_the_allowlist(client) -> None:
    resp = client.get("/api/job-commands")
    assert resp.status_code == 200
    names = {c["name"] for c in resp.json()["commands"]}
    assert "backfill-corporate-actions" in names
    assert "foreman-assess" in names


def test_trigger_unknown_command_returns_404(client) -> None:
    resp = client.post("/api/jobs/not-a-real-command", json={})
    assert resp.status_code == 404


def test_trigger_missing_required_param_returns_422(client) -> None:
    resp = client.post("/api/jobs/backfill-corporate-actions", json={"start": "2024-01-01"})
    assert resp.status_code == 422


@patch("stocksense.server.jobs.subprocess.Popen")
def test_trigger_valid_command_returns_job_id(mock_popen, client) -> None:
    mock_popen.return_value = _fake_popen(["hello"])
    resp = client.post("/api/jobs/foreman-assess", json={})
    assert resp.status_code == 200
    assert "job_id" in resp.json()


@patch("stocksense.server.jobs.subprocess.Popen")
def test_second_concurrent_job_returns_409(mock_popen, client) -> None:
    never_ending = MagicMock()
    never_ending.pid = 1
    never_ending.stdout = iter(())
    import time as _time
    never_ending.wait = lambda: _time.sleep(5)
    mock_popen.return_value = never_ending

    r1 = client.post("/api/jobs/foreman-assess", json={})
    assert r1.status_code == 200
    r2 = client.post("/api/jobs/foreman-assess", json={})
    assert r2.status_code == 409


def test_stop_unknown_job_returns_404(client) -> None:
    resp = client.post("/api/jobs/does-not-exist/stop")
    assert resp.status_code == 404


@patch("stocksense.server.jobs.subprocess.Popen")
def test_list_jobs_includes_completed_run(mock_popen, client) -> None:
    import time

    mock_popen.return_value = _fake_popen(["output line"])
    trigger = client.post("/api/jobs/foreman-assess", json={})
    job_id = trigger.json()["job_id"]
    time.sleep(0.3)

    resp = client.get("/api/jobs")
    assert resp.status_code == 200
    jobs = resp.json()["jobs"]
    assert any(j["job_id"] == job_id for j in jobs)


@patch("stocksense.server.jobs.subprocess.Popen")
def test_websocket_streams_output_and_signals_done(mock_popen, client) -> None:
    mock_popen.return_value = _fake_popen(["alpha", "beta"])
    trigger = client.post("/api/jobs/foreman-assess", json={})
    job_id = trigger.json()["job_id"]

    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        messages = []
        for _ in range(10):
            msg = ws.receive_json()
            messages.append(msg)
            if msg.get("done"):
                break
        lines = [m["line"] for m in messages if "line" in m]
        assert "alpha" in lines
        assert "beta" in lines
        assert any(m.get("done") for m in messages)


# ---- Settings ----

def test_settings_get_returns_current_effective_values(client, monkeypatch) -> None:
    """AUDIT FIX regression: GET must reflect what get_settings() ACTUALLY
    returns (defaults + real env overrides), not a disconnected DB table
    -- the original implementation "worked" but was reading/writing a
    table nothing else in the codebase ever consulted."""
    monkeypatch.setenv("STOCKSENSE_PLANNER_MODEL", "sonnet")
    resp = client.get("/api/settings")
    assert resp.json()["settings"]["planner_model"] == "sonnet"


def test_settings_get_uses_defaults_when_unset(client) -> None:
    resp = client.get("/api/settings")
    assert resp.json()["settings"]["planner_model"] == "opus"  # Settings' own default


def test_settings_put_writes_env_file_never_the_real_one(client, tmp_path, monkeypatch) -> None:
    """The real .env holds live Upstox credentials -- this test proves
    the write lands in an isolated file, not REPO_ROOT/.env."""
    import stocksense.server.app as app_mod

    fake_repo_root = tmp_path / "fake_repo"
    fake_repo_root.mkdir()
    monkeypatch.setattr(app_mod, "REPO_ROOT", fake_repo_root)

    resp = client.put("/api/settings", json={"planner_model": "sonnet", "planner_effort": "high"})
    assert resp.status_code == 200

    env_content = (fake_repo_root / ".env").read_text(encoding="utf-8")
    assert "STOCKSENSE_PLANNER_MODEL=sonnet" in env_content
    assert "STOCKSENSE_PLANNER_EFFORT=high" in env_content
    # and definitively not the real repo's .env
    assert not (Path(__file__).resolve().parents[2] / ".env").read_text(encoding="utf-8") == env_content


def test_settings_put_merges_preserving_unrelated_env_lines(client, tmp_path, monkeypatch) -> None:
    import stocksense.server.app as app_mod

    fake_repo_root = tmp_path / "fake_repo"
    fake_repo_root.mkdir()
    (fake_repo_root / ".env").write_text("STOCKSENSE_UPSTOX_API_KEY=some-existing-key\n", encoding="utf-8")
    monkeypatch.setattr(app_mod, "REPO_ROOT", fake_repo_root)

    client.put("/api/settings", json={"planner_model": "sonnet"})

    env_content = (fake_repo_root / ".env").read_text(encoding="utf-8")
    assert "STOCKSENSE_UPSTOX_API_KEY=some-existing-key" in env_content  # untouched
    assert "STOCKSENSE_PLANNER_MODEL=sonnet" in env_content


def test_settings_get_masks_secret_fields(client, monkeypatch) -> None:
    monkeypatch.setenv("STOCKSENSE_UPSTOX_API_KEY", "super-secret-real-key-value")
    resp = client.get("/api/settings")
    value = resp.json()["settings"]["upstox_api_key"]
    assert "super-secret-real-key-value" not in value
    assert value.endswith("alue")  # last 4 chars only, for the user to confirm which key is active


# ---- Claude auth endpoints ----

def _mock_auth_proc(logged_in=True, email="x@example.com", plan="pro"):
    m = MagicMock()
    m.returncode = 0
    m.stdout = __import__("json").dumps({"loggedIn": logged_in, "email": email, "subscriptionType": plan})
    return m


@patch("stocksense.agent.access.subprocess.run")
@patch("stocksense.agent.access.shutil.which", return_value="claude")
def test_claude_auth_status_reports_logged_in_and_not_yet_authorized(mock_which, mock_run, client) -> None:
    mock_run.return_value = _mock_auth_proc()
    resp = client.get("/api/claude/auth")
    data = resp.json()
    assert data["logged_in"] is True
    assert data["email"] == "x@example.com"
    assert data["access_granted"] is False  # logged in != authorized


@patch("stocksense.agent.access.subprocess.run")
@patch("stocksense.agent.access.shutil.which", return_value="claude")
def test_authorize_then_status_reflects_it(mock_which, mock_run, client) -> None:
    mock_run.return_value = _mock_auth_proc()
    resp = client.post("/api/claude/authorize")
    assert resp.status_code == 200
    assert resp.json()["access_granted"] is True

    resp = client.get("/api/claude/auth")
    assert resp.json()["access_granted"] is True


@patch("stocksense.agent.access.subprocess.run")
@patch("stocksense.agent.access.shutil.which", return_value="claude")
def test_authorize_fails_when_not_logged_in(mock_which, mock_run, client) -> None:
    mock_run.return_value = _mock_auth_proc(logged_in=False, email=None)
    resp = client.post("/api/claude/authorize")
    assert resp.status_code == 409


@patch("stocksense.agent.access.subprocess.run")
@patch("stocksense.agent.access.shutil.which", return_value="claude")
def test_decline_revokes_previously_granted_access(mock_which, mock_run, client) -> None:
    mock_run.return_value = _mock_auth_proc()
    client.post("/api/claude/authorize")

    resp = client.post("/api/claude/decline")
    assert resp.json()["access_granted"] is False

    resp = client.get("/api/claude/auth")
    assert resp.json()["access_granted"] is False


# ---- Usage gauge (Phase F3) ----

def test_usage_endpoint_empty_when_no_projects_dir(client, tmp_path, monkeypatch) -> None:
    import stocksense.agent.usage_tracker as usage_mod

    monkeypatch.setattr(usage_mod, "CLAUDE_PROJECTS_DIR", tmp_path / "no_such_dir")
    resp = client.get("/api/claude/usage")
    assert resp.status_code == 200
    data = resp.json()
    assert data["measured_not_official"] is True
    assert data["window_5h"]["total_tokens"] == 0


def test_usage_soft_alarm_roundtrip(client, tmp_path, monkeypatch) -> None:
    import stocksense.agent.usage_tracker as usage_mod

    # never scan the real ~/.claude/projects tree in a unit test -- slow
    # (608MB+ in practice) and not hermetic
    monkeypatch.setattr(usage_mod, "CLAUDE_PROJECTS_DIR", tmp_path / "no_such_dir")

    resp = client.put("/api/claude/usage/soft-alarm", params={"tokens_5h": 500000})
    assert resp.status_code == 200
    assert resp.json()["soft_alarm_tokens_5h"] == 500000

    resp = client.get("/api/claude/usage")
    assert resp.json()["soft_alarm_tokens_5h"] == 500000
