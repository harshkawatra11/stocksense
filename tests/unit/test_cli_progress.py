"""Tests for _emit_progress (Phase F1): the throttled, parseable
progress line long-running backfills print so a UI job console has a
finer signal than 'still running' for a multi-hour job."""

from __future__ import annotations

from stocksense.cli.main import _PROGRESS_EVERY, _emit_progress


def _capture(capsys):
    return capsys.readouterr().out


def test_emits_at_the_configured_cadence(capsys) -> None:
    for i in range(1, _PROGRESS_EVERY + 1):
        _emit_progress(i, total=10_000)
    out = _capture(capsys)
    lines = [l for l in out.splitlines() if l.startswith("PROGRESS:")]
    assert len(lines) == 1
    assert lines[0] == f"PROGRESS: {_PROGRESS_EVERY}/10000 ({round(100 * _PROGRESS_EVERY / 10000)}%)"


def test_does_not_emit_between_cadence_points(capsys) -> None:
    _emit_progress(1, total=10_000)
    _emit_progress(2, total=10_000)
    out = _capture(capsys)
    assert out == ""


def test_always_emits_on_forced_completion(capsys) -> None:
    _emit_progress(7, total=10_000, force=True)
    out = _capture(capsys)
    assert "PROGRESS: 7/10000" in out


def test_emits_on_exact_completion_even_off_cadence(capsys) -> None:
    _emit_progress(37, total=37)  # 37 is not a multiple of _PROGRESS_EVERY
    out = _capture(capsys)
    assert "PROGRESS: 37/37 (100%)" in out


def test_percentage_never_exceeds_100(capsys) -> None:
    _emit_progress(10, total=10, force=True)
    out = _capture(capsys)
    assert "(100%)" in out
    assert "101%" not in out


def test_zero_total_is_a_noop(capsys) -> None:
    _emit_progress(1, total=0, force=True)
    out = _capture(capsys)
    assert out == ""
