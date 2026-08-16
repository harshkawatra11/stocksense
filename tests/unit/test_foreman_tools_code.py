"""Code tool tests. write_patch's protected-path refusal is the actual
enforcement point for foreman.policy (policy.check_patch is just the
lookup; this is where a write is either allowed or blocked), so it's
tested here at the tool boundary, not just at the policy module."""

from __future__ import annotations

from stocksense.foreman.tools import registry
from stocksense.foreman.tools.code import read_file, run_tests, search_code, write_patch


def test_write_patch_refuses_protected_path() -> None:
    result = write_patch("src/stocksense/evaluation/gate.py", "# malicious content")
    assert result.ok is False
    assert "protected" in result.error.lower()
    assert "src/stocksense/evaluation/gate.py" in result.data["blocked_paths"]


def test_write_patch_refuses_ci_workflow() -> None:
    result = write_patch(".github/workflows/verify.yml", "# tampered")
    assert result.ok is False


def test_write_patch_allows_ordinary_path(tmp_path, monkeypatch) -> None:
    import stocksense.foreman.tools.code as code_mod

    monkeypatch.setattr(code_mod, "REPO_ROOT", tmp_path)
    result = write_patch("scratch/test_output.txt", "hello world")
    assert result.ok is True
    assert (tmp_path / "scratch" / "test_output.txt").read_text() == "hello world"


def test_write_patch_refuses_path_escaping_repo_root(tmp_path, monkeypatch) -> None:
    import stocksense.foreman.tools.code as code_mod

    monkeypatch.setattr(code_mod, "REPO_ROOT", tmp_path)
    result = write_patch("../../etc/passwd", "escaped")
    assert result.ok is False
    assert "escapes repo root" in result.error


def test_read_file_returns_content_of_real_file() -> None:
    result = read_file("pyproject.toml")
    assert result.ok is True
    assert "stocksense" in result.output


def test_read_file_missing_file_fails_cleanly() -> None:
    result = read_file("this/does/not/exist.py")
    assert result.ok is False
    assert "not found" in result.error


def test_search_code_finds_known_symbol() -> None:
    result = search_code(r"def compute_charges")
    assert result.ok is True
    assert result.data["n_matches"] >= 1
    assert "cost_model.py" in result.output


def test_search_code_invalid_regex_fails_cleanly() -> None:
    result = search_code(r"(unclosed[")
    assert result.ok is False


def test_all_code_tools_registered() -> None:
    for name in ("read_file", "search_code", "write_patch", "run_tests", "run_lint"):
        assert name in registry
