from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import polars as pl

from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.core.errors import LockboxError
from stocksense.core.paths import Paths
from stocksense.validation.trials import git_sha

Purpose = Literal["research", "production", "lockbox"]
_PURPOSES = ("research", "production", "lockbox")

# True only while inside open_lockbox(...); a ContextVar so threads and tasks do not leak into each other.
_LOCKBOX_OPEN: ContextVar[bool] = ContextVar("stocksense_lockbox_open", default=False)


def research_end(settings: Settings) -> date:
    """Last research date: the day before lockbox_start."""
    return settings.research.lockbox_start - timedelta(days=1)


def guard[FrameT: (pl.DataFrame, pl.LazyFrame)](
    frame: FrameT, date_col: str, purpose: str, settings: Settings
) -> FrameT:
    """Apply a purpose to a frame: research drops rows after research_end, production keeps all,
    lockbox keeps all but only inside open_lockbox(...)."""
    if purpose not in _PURPOSES:
        raise ValueError(f"purpose must be one of {_PURPOSES}, got {purpose!r}")
    if purpose == "production":
        return frame
    if purpose == "lockbox":
        if not _LOCKBOX_OPEN.get():
            raise LockboxError("purpose='lockbox' is only allowed inside open_lockbox(...)")
        return frame
    return frame.filter(pl.col(date_col).cast(pl.Date) <= research_end(settings))


def _touches_path(settings: Settings) -> Path:
    return Paths(settings).state / "lockbox_touches.jsonl"


@contextmanager
def open_lockbox(settings: Settings, clock: Clock, candidate_id: str, reason: str) -> Iterator[None]:
    """Allow purpose='lockbox' loads for one candidate, once ever; logs the touch on entry."""
    path = _touches_path(settings)
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("candidate_id") == candidate_id:
                raise LockboxError(f"lockbox already opened for candidate {candidate_id!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": clock.now().isoformat(),
        "candidate_id": candidate_id,
        "reason": reason,
        "git_sha": git_sha(),
    }
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    token = _LOCKBOX_OPEN.set(True)
    try:
        yield
    finally:
        _LOCKBOX_OPEN.reset(token)
