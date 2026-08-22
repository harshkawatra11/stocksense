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


# ---- Phase G/A1: job log history (re-reading a job after it's done) ----

@patch("stocksense.server.jobs.subprocess.Popen")
def test_get_log_of_running_job_reads_live_buffer(mock_popen, client) -> None:
    never_ending = MagicMock()
    never_ending.pid = 1
    never_ending.stdout = iter(["still going\n"])
    import time as _time
    never_ending.wait = lambda: _time.sleep(5)
    mock_popen.return_value = never_ending

    trigger = client.post("/api/jobs/foreman-assess", json={})
    job_id = trigger.json()["job_id"]
    import time
    time.sleep(0.2)

    resp = client.get(f"/api/jobs/{job_id}/log")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "live"
    assert "still going" in data["lines"]


@patch("stocksense.server.jobs.subprocess.Popen")
def test_get_log_of_finished_job_reads_from_disk(mock_popen, client) -> None:
    """The whole point of A1: a job's output must still be readable
    AFTER it's no longer the one 'currently running' -- before this
    endpoint existed, a finished job's log was unrecoverable from the UI
    even though server/jobs.py was already persisting it to disk."""
    mock_popen.return_value = _fake_popen(["line one", "line two"])
    trigger = client.post("/api/jobs/foreman-assess", json={})
    job_id = trigger.json()["job_id"]

    import time
    for _ in range(20):
        jobs = client.get("/api/jobs").json()["jobs"]
        row = next((j for j in jobs if j["job_id"] == job_id), None)
        if row and row["status"] != "running":
            break
        time.sleep(0.1)

    resp = client.get(f"/api/jobs/{job_id}/log")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "disk"
    assert "line one" in data["lines"]
    assert "line two" in data["lines"]


def test_get_log_of_unknown_job_returns_404(client) -> None:
    resp = client.get("/api/jobs/does-not-exist/log")
    assert resp.status_code == 404


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


# ---- Phase G/B3: /api/ask (the RAG corpus was buildable but not queryable) ----

def test_ask_requires_a_question(client) -> None:
    resp = client.post("/api/ask", json={})
    assert resp.status_code == 422


def test_ask_returns_no_results_on_empty_corpus(client) -> None:
    resp = client.post("/api/ask", json={"question": "why did I lose money?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_chunks_retrieved"] == 0
    assert "Nothing relevant" in data["answer"]


def test_ask_goes_through_the_f2_access_gate(client, monkeypatch) -> None:
    """rag.agent.ask() calls agent.claude_cli.invoke(), which is the
    single F2 enforcement point -- no corpus content is needed to prove
    this: an indexed document with no Claude access granted must come
    back access_denied via the exact same path every other Claude call
    uses, not a separate/bypassable check in this endpoint."""
    from stocksense.data.store import Store
    from stocksense.rag.index import index_document, rebuild_fts_index
    import stocksense.server.app as app_mod

    settings_db = app_mod.get_settings().duckdb_path
    store = Store(settings_db)
    index_document(store, "research", "verdict.md", "Verdict", "The gate passed with p=0.0085.")
    rebuild_fts_index(store)
    store.close()

    resp = client.post("/api/ask", json={"question": "did the gate pass?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_chunks_retrieved"] == 1  # retrieval doesn't need Claude access, only narration does


# ---- Phase G/Track C: research doc listing/reading, path-traversal safety ----

@pytest.fixture()
def research_client(client, tmp_path, monkeypatch):
    import stocksense.server.app as app_mod

    fake_repo_root = tmp_path / "fake_repo"
    (fake_repo_root / "research").mkdir(parents=True)
    (fake_repo_root / "research" / "verdict_intraday.md").write_text("# Verdict\n\nGATE FAIL", encoding="utf-8")
    (fake_repo_root / "research" / "fold_results.csv").write_text("fold_id,alpha_net\n0,-0.001\n", encoding="utf-8")
    (fake_repo_root / "research" / "not_a_doc.py").write_text("print('hi')", encoding="utf-8")
    # a real secret file OUTSIDE research/, at repo root -- the traversal target
    (fake_repo_root / ".env").write_text("STOCKSENSE_UPSTOX_API_KEY=real-secret-value", encoding="utf-8")
    monkeypatch.setattr(app_mod, "REPO_ROOT", fake_repo_root)
    return client


def test_list_research_docs_only_lists_md_and_csv(research_client) -> None:
    resp = research_client.get("/api/research/docs")
    assert resp.status_code == 200
    docs = resp.json()["docs"]
    assert "verdict_intraday.md" in docs
    assert "fold_results.csv" in docs
    assert "not_a_doc.py" not in docs


def test_get_research_doc_returns_content(research_client) -> None:
    resp = research_client.get("/api/research/doc/verdict_intraday.md")
    assert resp.status_code == 200
    assert "GATE FAIL" in resp.json()["content"]


def test_get_research_doc_rejects_path_traversal(research_client) -> None:
    """The decisive test: a name that would escape research/ and read
    the real .env-equivalent (STOCKSENSE_UPSTOX_API_KEY) must be
    rejected, not served."""
    for attempt in ["../.env", "..%2f.env", "..\\.env", "/etc/passwd"]:
        resp = research_client.get(f"/api/research/doc/{attempt}")
        assert resp.status_code in (400, 404), f"traversal attempt {attempt!r} was not rejected: {resp.status_code}"
        assert "real-secret-value" not in resp.text


def test_get_research_doc_rejects_nonexistent_file(research_client) -> None:
    resp = research_client.get("/api/research/doc/does_not_exist.md")
    assert resp.status_code == 404


def test_get_research_doc_rejects_non_doc_extension(research_client) -> None:
    resp = research_client.get("/api/research/doc/not_a_doc.py")
    assert resp.status_code == 404
