"""Claude CLI access-gate tests (Phase F2). `claude auth status` is
mocked -- what's under test is: the gate never touches credentials
directly, an account change auto-revokes rather than silently carrying
authorization over, and the fail-closed behavior when the check itself
can't be confirmed."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from stocksense.agent.access import (
    ClaudeAccessNotGranted,
    check_claude_auth,
    grant_access,
    is_access_granted,
    require_claude_access,
    revoke_access,
)
from stocksense.data.store import Store


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def _mock_auth_proc(logged_in=True, email="user@example.com", plan="pro", returncode=0):
    m = MagicMock()
    m.returncode = returncode
    m.stdout = json.dumps({"loggedIn": logged_in, "email": email, "subscriptionType": plan})
    return m


# ---- check_claude_auth: never touches credentials directly ----

@patch("stocksense.agent.access.subprocess.run")
@patch("stocksense.agent.access.shutil.which", return_value="claude")
def test_check_claude_auth_calls_only_the_status_subcommand(mock_which, mock_run) -> None:
    mock_run.return_value = _mock_auth_proc()
    check_claude_auth()
    cmd = mock_run.call_args[0][0]
    assert cmd == ["claude", "auth", "status", "--json"]
    # never references the credentials file directly
    assert not any("credential" in str(a).lower() for a in cmd)


@patch("stocksense.agent.access.shutil.which", return_value=None)
def test_check_claude_auth_handles_missing_binary(mock_which) -> None:
    status = check_claude_auth()
    assert not status.logged_in
    assert "not found" in status.raw_error


# ---- grant / revoke ----

@patch("stocksense.agent.access.subprocess.run")
@patch("stocksense.agent.access.shutil.which", return_value="claude")
def test_grant_access_records_the_currently_logged_in_email(mock_which, mock_run, tmp_store) -> None:
    mock_run.return_value = _mock_auth_proc(email="harsh@example.com")
    status = grant_access(tmp_store)
    assert status.email == "harsh@example.com"
    assert is_access_granted(tmp_store)


@patch("stocksense.agent.access.subprocess.run")
@patch("stocksense.agent.access.shutil.which", return_value="claude")
def test_grant_access_fails_when_not_logged_in(mock_which, mock_run, tmp_store) -> None:
    mock_run.return_value = _mock_auth_proc(logged_in=False, email=None)
    with pytest.raises(ClaudeAccessNotGranted):
        grant_access(tmp_store)


def test_revoke_access_clears_the_flag(tmp_store) -> None:
    tmp_store.set_app_setting("claude_access_granted", "true")
    tmp_store.set_app_setting("claude_access_approved_email", "x@example.com")
    revoke_access(tmp_store)
    assert not is_access_granted(tmp_store)


# ---- the core safety property: account change auto-revokes ----

@patch("stocksense.agent.access.subprocess.run")
@patch("stocksense.agent.access.shutil.which", return_value="claude")
def test_account_change_auto_revokes_access(mock_which, mock_run, tmp_store) -> None:
    """The single most important property: approving access for one
    Claude account must NOT silently carry over if the logged-in
    account later changes -- this is what makes 'my Pro account may
    change' safe by construction."""
    mock_run.return_value = _mock_auth_proc(email="old-account@example.com")
    grant_access(tmp_store)
    assert is_access_granted(tmp_store)

    mock_run.return_value = _mock_auth_proc(email="new-account@example.com")
    assert not is_access_granted(tmp_store)  # auto-revoked, not silently still-granted

    # and it stays revoked -- the flag itself was actually cleared, not just this one check
    mock_run.return_value = _mock_auth_proc(email="old-account@example.com")  # even if the original account logs back in
    assert not is_access_granted(tmp_store)


@patch("stocksense.agent.access.subprocess.run")
@patch("stocksense.agent.access.shutil.which", return_value="claude")
def test_logout_revokes_access(mock_which, mock_run, tmp_store) -> None:
    mock_run.return_value = _mock_auth_proc(email="x@example.com")
    grant_access(tmp_store)

    mock_run.return_value = _mock_auth_proc(logged_in=False, email=None)
    assert not is_access_granted(tmp_store)


def test_never_granted_by_default(tmp_store) -> None:
    assert not is_access_granted(tmp_store)


def test_require_claude_access_raises_when_not_granted(tmp_store) -> None:
    with pytest.raises(ClaudeAccessNotGranted):
        require_claude_access(tmp_store)


@patch("stocksense.agent.access.subprocess.run")
@patch("stocksense.agent.access.shutil.which", return_value="claude")
def test_require_claude_access_passes_silently_when_granted(mock_which, mock_run, tmp_store) -> None:
    mock_run.return_value = _mock_auth_proc(email="x@example.com")
    grant_access(tmp_store)
    require_claude_access(tmp_store)  # must not raise


# ---- fail-closed on ambiguity ----

def test_is_access_granted_fails_closed_on_store_error(tmp_store) -> None:
    class _BrokenStore:
        def get_app_setting(self, key):
            raise RuntimeError("db busy")

    assert not is_access_granted(_BrokenStore())


# ---- the single-choke-point property: invoke() itself enforces the gate ----

def test_invoke_refuses_to_call_claude_when_access_not_granted(tmp_store) -> None:
    """The load-bearing property: invoke() must never reach subprocess.run
    for `claude` at all when access isn't granted -- not just log a
    warning and proceed. A caller (planner/codegen/assess/report/rag)
    that forgets to check the gate itself must still be blocked here."""
    from stocksense.agent.claude_cli import AgentRequest, invoke

    with patch("stocksense.agent.claude_cli.subprocess.run") as mock_run:
        result = invoke(AgentRequest(prompt="hi"), store=tmp_store)
        mock_run.assert_not_called()
    assert result.status == "access_denied"


@patch("stocksense.agent.access.shutil.which", return_value="claude")
def test_invoke_proceeds_when_access_granted(mock_which, tmp_store) -> None:
    from stocksense.agent.claude_cli import AgentRequest, invoke

    # `subprocess.run` is ONE shared module attribute regardless of which
    # module's namespace patches it (`import subprocess` binds the same
    # global module object everywhere) -- so a single dispatching mock is
    # used here instead of two separately-patched call sites, which would
    # silently clobber each other (confirmed directly while debugging this
    # test: the second patch overwrote the first's mock for BOTH modules).
    def _dispatch(cmd, **kwargs):
        if cmd[1:4] == ["auth", "status", "--json"]:
            return _mock_auth_proc(email="x@example.com")
        return MagicMock(returncode=0, stdout='{"result": "ok"}', stderr="")

    with patch("stocksense.agent.access.subprocess.run", side_effect=_dispatch) as mock_run:
        grant_access(tmp_store)
        result = invoke(AgentRequest(prompt="hi"), store=tmp_store)
        assert mock_run.call_count == 3  # grant's check + invoke's access re-check + the actual claude call
    assert result.status == "ok"


def test_invoke_opens_its_own_store_and_still_denies_when_caller_passes_none(tmp_path, monkeypatch) -> None:
    """A caller that doesn't thread a store through (store=None) must
    not silently bypass the gate -- invoke() opens its own connection
    just to check."""
    from stocksense.agent.claude_cli import AgentRequest, invoke

    monkeypatch.setenv("STOCKSENSE_DUCKDB_PATH", str(tmp_path / "no_access.duckdb"))
    with patch("stocksense.agent.claude_cli.subprocess.run") as mock_run:
        result = invoke(AgentRequest(prompt="hi"), store=None)
        mock_run.assert_not_called()
    assert result.status == "access_denied"
