"""Agent bridge safety guardrails, tested without invoking the real CLI:
secret redaction and the unverified-numbers tripwire. These are the
structural defenses named in docs/11-agent-harness.md — the whole point
is that they hold even if a prompt is worded carelessly."""

from __future__ import annotations

from stocksense.agent.claude_cli import (
    _build_prompt,
    _check_unverified_numbers,
    _extract_fact_numbers,
    redact_secrets,
    AgentRequest,
)


def test_redact_secrets_removes_api_key_pattern() -> None:
    text = "api_key: sk-abcdef1234567890abcdef"
    out = redact_secrets(text)
    assert "sk-abcdef1234567890abcdef" not in out
    assert "[REDACTED]" in out


def test_redact_secrets_removes_long_opaque_tokens() -> None:
    text = "token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcdefghij"
    out = redact_secrets(text)
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in out


def test_redact_secrets_leaves_normal_text_alone() -> None:
    text = "Your total cost drag was 34.5% of gross P&L."
    assert redact_secrets(text) == text


def test_build_prompt_includes_facts_as_fenced_json() -> None:
    req = AgentRequest(prompt="Summarize this.", facts={"cost_drag": 0.345, "gross_pnl": 1000.0})
    full = _build_prompt(req)
    assert "```json" in full
    assert "0.345" in full
    assert "ONLY source of numbers" in full


def test_build_prompt_redacts_secrets_in_facts() -> None:
    req = AgentRequest(prompt="Do not leak this.", facts={"api_key": "sk-abcdef1234567890abcdef"})
    full = _build_prompt(req)
    assert "sk-abcdef1234567890abcdef" not in full


def test_extract_fact_numbers_finds_nested_values() -> None:
    facts = {"a": 82.68, "b": {"c": 25, "d": [10.0, 20.0]}}
    numbers = _extract_fact_numbers(facts)
    assert "25" in numbers
    assert "10.0" in numbers


def test_unverified_numbers_flags_invented_figure() -> None:
    facts = {"cost_drag_pct": 34.5, "gross_pnl": 1000.0}
    output_bad = "Your cost drag was 87.2% which is very high."  # 87.2 not in facts
    assert _check_unverified_numbers(output_bad, facts) is True


def test_unverified_numbers_passes_when_grounded() -> None:
    facts = {"cost_drag_pct": 34.5, "gross_pnl": 1000.0}
    output_good = "Your cost drag was 34.5% of a gross P&L of 1000."
    assert _check_unverified_numbers(output_good, facts) is False


def test_unverified_numbers_ignores_small_numbers() -> None:
    # small numbers (ranks, counts under 10) are too noisy to police reliably
    facts = {"gross_pnl": 1000.0}
    output = "This is your #3 worst trade out of 7."
    assert _check_unverified_numbers(output, facts) is False
