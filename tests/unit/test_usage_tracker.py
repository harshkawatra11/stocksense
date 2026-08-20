"""Usage tracker tests (Phase F3). No real ~/.claude data is touched --
every test scans a synthetic tmp_path tree shaped like real session
transcripts (verified against an actual file during planning: one JSON
object per line, `type: "assistant"` entries carry `message.usage` +
`timestamp`). What's under test: incremental byte-offset scanning (never
re-reading what's already been counted, never consuming a partial
trailing line), deterministic dedup, rolling-window math, and that one
malformed file doesn't blind the scan to every other session."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from stocksense.agent.usage_tracker import (
    SOFT_ALARM_SETTING_KEY,
    _read_new_lines,
    get_usage_summary,
    scan_and_update,
)
from stocksense.data.store import Store


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def _assistant_line(ts: datetime, model="claude-sonnet-4-6", input_tokens=10, output_tokens=20,
                     cache_creation=0, cache_read=0) -> str:
    return json.dumps({
        "type": "assistant", "timestamp": ts.isoformat().replace("+00:00", "Z"),
        "message": {
            "model": model,
            "usage": {
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation, "cache_read_input_tokens": cache_read,
            },
        },
    })


def _write_session(path: Path, lines: list[str], trailing_newline: bool = True) -> None:
    """newline="" disables Python's platform newline translation -- real
    JSONL transcripts use bare \\n regardless of OS (Node.js writers
    don't translate), and without this, Windows' text-mode write_text
    would turn every \\n into \\r\\n, corrupting the fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + ("\n" if trailing_newline else "")
    path.write_text(content, encoding="utf-8", newline="")


# ---- _read_new_lines: offset mechanics ----

def test_read_new_lines_from_zero_offset(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    _write_session(path, ["a", "b", "c"])
    lines, offset = _read_new_lines(path, 0)
    assert lines == ["a", "b", "c"]
    assert offset == path.stat().st_size


def test_read_new_lines_only_reads_appended_content(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    _write_session(path, ["a", "b"])
    _, offset_after_first = _read_new_lines(path, 0)

    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write("c\n")
    lines, new_offset = _read_new_lines(path, offset_after_first)
    assert lines == ["c"]
    assert new_offset == path.stat().st_size


def test_read_new_lines_does_not_consume_a_partial_trailing_line(tmp_path) -> None:
    """A session actively being written to may have an incomplete final
    line at the moment of scanning -- it must be left for the NEXT scan,
    not parsed as-is (which would either fail or silently truncate it)."""
    path = tmp_path / "s.jsonl"
    _write_session(path, ["complete_line"], trailing_newline=True)
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write('{"partial": "not yet terminat')  # no trailing newline -- still being written

    lines, offset = _read_new_lines(path, 0)
    assert lines == ["complete_line"]
    assert offset < path.stat().st_size  # the partial line's bytes were NOT consumed

    # a later scan, once the line completes, picks it up correctly
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write('ed"}\n')
    lines2, offset2 = _read_new_lines(path, offset)
    assert lines2 == ['{"partial": "not yet terminated"}']
    assert offset2 == path.stat().st_size


# ---- scan_and_update: parsing, incrementality, dedup ----

def test_scan_extracts_assistant_usage_events(tmp_store, tmp_path) -> None:
    projects = tmp_path / "projects"
    now = datetime.now(timezone.utc)
    _write_session(projects / "p1" / "s1.jsonl", [
        _assistant_line(now, input_tokens=10, output_tokens=20),
        json.dumps({"type": "user", "timestamp": now.isoformat()}),  # not an assistant turn -- must be skipped
        _assistant_line(now, input_tokens=5, output_tokens=15),
    ])
    n = scan_and_update(tmp_store, projects_dir=projects)
    assert n == 2

    events = tmp_store.read_usage_events()
    assert len(events) == 2
    assert set(events["input_tokens"]) == {10, 5}


def test_rescan_is_incremental_no_double_counting(tmp_store, tmp_path) -> None:
    projects = tmp_path / "projects"
    now = datetime.now(timezone.utc)
    path = projects / "p1" / "s1.jsonl"
    _write_session(path, [_assistant_line(now, input_tokens=100)])

    scan_and_update(tmp_store, projects_dir=projects)
    n_second_scan = scan_and_update(tmp_store, projects_dir=projects)  # nothing new appended
    assert n_second_scan == 0
    assert len(tmp_store.read_usage_events()) == 1

    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(_assistant_line(now, input_tokens=50) + "\n")
    n_third_scan = scan_and_update(tmp_store, projects_dir=projects)
    assert n_third_scan == 1
    assert len(tmp_store.read_usage_events()) == 2


def test_scan_skips_a_malformed_file_without_blinding_the_rest(tmp_store, tmp_path) -> None:
    projects = tmp_path / "projects"
    now = datetime.now(timezone.utc)
    _write_session(projects / "p1" / "good.jsonl", [_assistant_line(now, input_tokens=42)])
    bad_path = projects / "p1" / "bad.jsonl"
    bad_path.write_bytes(b"\xff\xfe not valid utf-8 or json at all \x00\x01")

    n = scan_and_update(tmp_store, projects_dir=projects)
    assert n >= 1
    events = tmp_store.read_usage_events()
    assert 42 in list(events["input_tokens"])


def test_scan_missing_projects_dir_returns_zero(tmp_store, tmp_path) -> None:
    n = scan_and_update(tmp_store, projects_dir=tmp_path / "does_not_exist")
    assert n == 0


# ---- get_usage_summary: rolling windows, model mix, soft alarm ----

def test_usage_summary_separates_5h_and_7d_windows(tmp_store, tmp_path) -> None:
    projects = tmp_path / "projects"
    now = datetime.now(timezone.utc)
    recent = now - timedelta(hours=1)
    old_but_within_week = now - timedelta(days=3)
    too_old = now - timedelta(days=10)

    _write_session(projects / "p1" / "s1.jsonl", [
        _assistant_line(recent, input_tokens=100, output_tokens=100),
        _assistant_line(old_but_within_week, input_tokens=200, output_tokens=200),
        _assistant_line(too_old, input_tokens=999, output_tokens=999),
    ])

    scan_and_update(tmp_store, projects_dir=projects)
    summary = get_usage_summary(tmp_store, rescan=False)

    assert summary["window_5h"]["total_tokens"] == 200  # only the 1-hour-ago event
    assert summary["window_7d"]["total_tokens"] == 200 + 400  # 1h-ago + 3d-ago, not the 10d-old one
    assert summary["measured_not_official"] is True


def test_usage_summary_computes_model_mix(tmp_store, tmp_path) -> None:
    projects = tmp_path / "projects"
    now = datetime.now(timezone.utc)
    # distinct token counts per line -- otherwise two calls with identical
    # (ts, model, usage) would hash to the same deterministic event_id and
    # get correctly deduped as "the same event", exactly as real re-reads
    # should be, but that's not what THIS test wants to exercise.
    _write_session(projects / "p1" / "s1.jsonl", [
        _assistant_line(now, model="claude-opus-5", input_tokens=1),
        _assistant_line(now, model="claude-opus-5", input_tokens=2),
        _assistant_line(now, model="claude-sonnet-5", input_tokens=3),
    ])
    scan_and_update(tmp_store, projects_dir=projects)
    summary = get_usage_summary(tmp_store, rescan=False)
    mix = summary["window_7d"]["model_mix"]
    assert mix["claude-opus-5"] == pytest.approx(2 / 3, abs=0.01)


def test_soft_alarm_trips_when_threshold_exceeded(tmp_store, tmp_path) -> None:
    projects = tmp_path / "projects"
    now = datetime.now(timezone.utc)
    _write_session(projects / "p1" / "s1.jsonl", [
        _assistant_line(now, input_tokens=600, output_tokens=0),
    ])
    scan_and_update(tmp_store, projects_dir=projects)
    tmp_store.set_app_setting(SOFT_ALARM_SETTING_KEY, "500")

    summary = get_usage_summary(tmp_store, rescan=False)
    assert summary["soft_alarm_tripped"] is True


def test_soft_alarm_not_tripped_when_unset(tmp_store, tmp_path) -> None:
    projects = tmp_path / "projects"
    now = datetime.now(timezone.utc)
    _write_session(projects / "p1" / "s1.jsonl", [_assistant_line(now, input_tokens=10**9)])
    scan_and_update(tmp_store, projects_dir=projects)

    summary = get_usage_summary(tmp_store, rescan=False)
    assert summary["soft_alarm_tokens_5h"] is None
    assert summary["soft_alarm_tripped"] is False


def test_empty_events_summary_has_zeroed_windows(tmp_store) -> None:
    summary = get_usage_summary(tmp_store, rescan=False)
    assert summary["window_5h"]["total_tokens"] == 0
    assert summary["window_5h"]["message_count"] == 0
    assert summary["window_5h"]["model_mix"] == {}
