"""Adversary tests: each check verified against a hand-written file that
embodies the exact anti-pattern it exists to catch, and a clean file
that must NOT trigger a false positive. Written to a real temp file
(not exercised via AST fragments) so the check runs against something
close to what it will actually see in a real changed-files list."""

from __future__ import annotations

import pytest

from stocksense.foreman.adversary import (
    check_assertionless_tests,
    check_hardcoded_test_expectations,
    check_research_result_seed_sensitivity,
    check_swallowed_exceptions,
    has_blocking_finding,
    red_team,
)


@pytest.fixture()
def repo_root(tmp_path, monkeypatch):
    import stocksense.foreman.adversary as adv_mod

    monkeypatch.setattr(adv_mod, "REPO_ROOT", tmp_path)
    return tmp_path


def _write(root, rel_path, content):
    p = root / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return rel_path


def test_catches_test_with_no_assertion(repo_root) -> None:
    path = _write(repo_root, "tests/unit/test_fake.py", """
def test_something():
    x = 1 + 1
    print(x)
""")
    findings = check_assertionless_tests([path])
    assert len(findings) == 1
    assert findings[0].severity == "blocking"
    assert "test_something" in findings[0].detail


def test_does_not_flag_test_with_real_assertion(repo_root) -> None:
    path = _write(repo_root, "tests/unit/test_real.py", """
def test_something():
    assert 1 + 1 == 2
""")
    findings = check_assertionless_tests([path])
    assert findings == []


def test_does_not_flag_test_using_pytest_raises(repo_root) -> None:
    path = _write(repo_root, "tests/unit/test_raises.py", """
import pytest

def test_something():
    with pytest.raises(ValueError):
        raise ValueError("x")
""")
    findings = check_assertionless_tests([path])
    assert findings == []


def test_catches_bare_except_pass(repo_root) -> None:
    path = _write(repo_root, "src/module.py", """
def risky():
    try:
        do_something()
    except Exception:
        pass
""")
    findings = check_swallowed_exceptions([path])
    assert len(findings) == 1
    assert findings[0].severity == "concern"


def test_does_not_flag_except_that_logs_or_reraises(repo_root) -> None:
    path = _write(repo_root, "src/module.py", """
def risky():
    try:
        do_something()
    except Exception as e:
        log.warning("failed", error=str(e))
""")
    findings = check_swallowed_exceptions([path])
    assert findings == []


def test_catches_tautological_assertion(repo_root) -> None:
    path = _write(repo_root, "tests/unit/test_fake2.py", """
def test_fake():
    result = compute()
    assert result == result
""")
    findings = check_hardcoded_test_expectations([path])
    assert len(findings) == 1
    assert findings[0].severity == "blocking"


def test_does_not_flag_real_comparison(repo_root) -> None:
    path = _write(repo_root, "tests/unit/test_real2.py", """
def test_real():
    result = compute()
    assert result == 42
""")
    findings = check_hardcoded_test_expectations([path])
    assert findings == []


def test_seed_sensitivity_flags_sign_flip() -> None:
    findings = check_research_result_seed_sensitivity({42: 0.005, 7: -0.002, 123: 0.003})
    assert len(findings) == 1
    assert findings[0].severity == "blocking"


def test_seed_sensitivity_ok_when_consistent() -> None:
    findings = check_research_result_seed_sensitivity({42: 0.005, 7: 0.004, 123: 0.006})
    assert findings == []


def test_seed_sensitivity_flags_insufficient_seeds() -> None:
    findings = check_research_result_seed_sensitivity({42: 0.005})
    assert len(findings) == 1
    assert findings[0].severity == "concern"


def test_red_team_aggregates_all_checks(repo_root) -> None:
    _write(repo_root, "tests/unit/test_a.py", "def test_a():\n    pass\n")
    findings = red_team(["tests/unit/test_a.py"])
    assert len(findings) >= 1


def test_has_blocking_finding_true_when_any_blocking() -> None:
    from stocksense.foreman.adversary import AdversaryFinding

    findings = [
        AdversaryFinding("x", "f.py", "detail", "note"),
        AdversaryFinding("y", "f.py", "detail", "blocking"),
    ]
    assert has_blocking_finding(findings) is True


def test_has_blocking_finding_false_when_none_blocking() -> None:
    from stocksense.foreman.adversary import AdversaryFinding

    findings = [AdversaryFinding("x", "f.py", "detail", "note"), AdversaryFinding("y", "f.py", "detail", "concern")]
    assert has_blocking_finding(findings) is False
