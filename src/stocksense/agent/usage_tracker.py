"""
Claude usage tracker (Phase F3). Anthropic publishes no numeric usage or
rate-limit API for a Pro/Max CLI session -- `claude auth status --json`
gives login/plan info, nothing about tokens or session windows. What IS
real and verified: every session transcript under
`~/.claude/projects/**/*.jsonl` carries a `message.usage` object
(input/output/cache tokens) and a timestamp on every assistant turn --
the same local data community tools like `ccusage` parse.

This module aggregates that into REAL, MEASURED rolling-window usage. It
cannot and does not claim to be "% of your official limit" -- Anthropic
doesn't publish that formula for Pro/Max plans, so callers must present
this as measured usage, not a certified quota, and the desktop UI labels
it that way explicitly rather than only in code comments.

Incremental by design: a byte offset per file (claude_usage_offsets)
means a re-scan only reads what's been APPENDED since the last check,
never re-parsing the 600MB+ of history that can accumulate across a
long-lived Claude Code installation. Usage is account-wide, not
per-project -- every session file under the whole ~/.claude/projects
tree is scanned, not just this repo's.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"

SOFT_ALARM_SETTING_KEY = "claude_usage_soft_alarm_tokens_5h"


def _read_new_lines(path: Path, start_offset: int) -> tuple[list[str], int]:
    """Reads only bytes appended since `start_offset`, and deliberately
    leaves a not-yet-newline-terminated trailing line unconsumed (a
    session actively being written to may have a partial line at EOF) --
    the returned offset only advances past complete lines, so the next
    scan picks up exactly where this one left off, never mid-line."""
    with open(path, "rb") as f:
        f.seek(start_offset)
        chunk = f.read()
    if not chunk:
        return [], start_offset

    text = chunk.decode("utf-8", errors="ignore")
    if text.endswith("\n"):
        lines = [l for l in text.split("\n") if l]
        new_offset = start_offset + len(chunk)
    else:
        parts = text.split("\n")
        lines = [l for l in parts[:-1] if l]
        trailing_partial_bytes = len(parts[-1].encode("utf-8"))
        new_offset = start_offset + len(chunk) - trailing_partial_bytes
    return lines, new_offset


def _event_id(file_path: str, line: str) -> str:
    """Deterministic, not random -- if the same line were ever read
    twice (it shouldn't be, given offset tracking, but idempotency is
    cheap insurance), it produces the same id and ON CONFLICT DO NOTHING
    dedupes it rather than double-counting usage."""
    return hashlib.sha256(f"{file_path}|{line}".encode()).hexdigest()[:24]


def _parse_usage_events(file_path: str, lines: list[str]) -> list[dict]:
    events = []
    for line in lines:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") != "assistant":
            continue
        message = d.get("message") or {}
        usage = message.get("usage") or {}
        ts_raw = d.get("timestamp")
        if not usage or not ts_raw:
            continue
        try:
            ts = pd.Timestamp(ts_raw)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
        except (ValueError, TypeError):
            continue

        events.append({
            "event_id": _event_id(file_path, line),
            "ts": ts,
            "model": message.get("model"),
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cache_creation_tokens": int(usage.get("cache_creation_input_tokens") or 0),
            "cache_read_tokens": int(usage.get("cache_read_input_tokens") or 0),
        })
    return events


def scan_and_update(store, projects_dir: Path | None = None) -> int:
    """Scans every session transcript for newly-appended usage events
    since the last scan, writes them, and advances each file's offset.
    Returns the number of new events written. A single unreadable/
    malformed file is logged and skipped, not fatal to the whole scan --
    one corrupt transcript must not blind the tracker to every other
    session's real usage."""
    projects_dir = projects_dir or CLAUDE_PROJECTS_DIR
    if not projects_dir.exists():
        return 0

    n_new = 0
    for path in projects_dir.glob("**/*.jsonl"):
        path_str = str(path)
        try:
            offset = store.get_usage_offset(path_str)
            lines, new_offset = _read_new_lines(path, offset)
            if lines:
                events = _parse_usage_events(path_str, lines)
                if events:
                    store.insert_usage_events(pd.DataFrame(events))
                    n_new += len(events)
            if new_offset != offset:
                store.set_usage_offset(path_str, new_offset)
        except Exception as e:  # noqa: BLE001 -- one bad file must not blind the whole scan
            log.warning("usage_scan_file_failed", file=path_str, error=str(e))
            continue
    return n_new


def _window_summary(events: pd.DataFrame) -> dict:
    if events.empty:
        return {
            "input_tokens": 0, "output_tokens": 0, "cache_creation_tokens": 0, "cache_read_tokens": 0,
            "total_tokens": 0, "message_count": 0, "model_mix": {},
        }
    total_tokens = int(
        events["input_tokens"].sum() + events["output_tokens"].sum()
        + events["cache_creation_tokens"].sum() + events["cache_read_tokens"].sum()
    )
    model_mix = events["model"].value_counts(normalize=True).round(3).to_dict()
    return {
        "input_tokens": int(events["input_tokens"].sum()),
        "output_tokens": int(events["output_tokens"].sum()),
        "cache_creation_tokens": int(events["cache_creation_tokens"].sum()),
        "cache_read_tokens": int(events["cache_read_tokens"].sum()),
        "total_tokens": total_tokens,
        "message_count": int(len(events)),
        "model_mix": model_mix,
    }


def get_usage_summary(store, rescan: bool = True) -> dict:
    """The rolling 5-hour and 7-day windows, plus the self-configured
    soft-alarm threshold and whether it's currently tripped. Rescans
    first by default so callers get fresh numbers without a separate
    step -- rescan is incremental (see scan_and_update), so this stays
    cheap on repeated calls."""
    if rescan:
        scan_and_update(store)

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    events_7d = store.read_usage_events(since=now - timedelta(days=7))
    events_5h = events_7d[events_7d["ts"] >= (now - timedelta(hours=5))] if not events_7d.empty else events_7d

    soft_alarm_raw = store.get_app_setting(SOFT_ALARM_SETTING_KEY)
    soft_alarm_tokens = int(soft_alarm_raw) if soft_alarm_raw else None
    window_5h = _window_summary(events_5h)
    tripped = soft_alarm_tokens is not None and window_5h["total_tokens"] >= soft_alarm_tokens

    return {
        "measured_not_official": True,
        "window_5h": window_5h,
        "window_7d": _window_summary(events_7d),
        "soft_alarm_tokens_5h": soft_alarm_tokens,
        "soft_alarm_tripped": tripped,
    }
