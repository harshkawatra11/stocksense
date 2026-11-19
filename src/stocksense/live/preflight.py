"""Process-level plumbing of the daemon: single-instance lock, GPU lock, heartbeat file."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from stocksense.core.config import Settings
from stocksense.core.paths import Paths


class FileLock:
    """A non-blocking exclusive lock on one byte of a file (msvcrt on Windows, fcntl elsewhere)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fh: Any = None

    def acquire(self) -> bool:
        if self._fh is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")  # noqa: SIM115 - held until release()
        try:
            if sys.platform == "win32":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass  # closing the handle drops the lock anyway
        fh.close()

    def __enter__(self) -> FileLock:
        if not self.acquire():
            raise BlockingIOError(f"lock held: {self.path.name}")
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def daemon_lock(settings: Settings) -> FileLock:
    """`state/daemon.lock`: only one daemon per data root."""
    return FileLock(Paths(settings).state / "daemon.lock")


def gpu_lock(settings: Settings) -> FileLock:
    """`state/gpu.lock`: GPU jobs hold it (`with gpu_lock(s):`) so only one runs at a time."""
    return FileLock(Paths(settings).state / "gpu.lock")


def write_heartbeat(
    path: Path, *, ts: datetime, pid: int, current_job: str | None, last_ok_job: str | None, degraded: bool
) -> None:
    """Write heartbeat.json atomically (temp file in the same folder + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "ts": ts.isoformat(), "pid": pid, "current_job": current_job, "last_ok_job": last_ok_job,
        "degraded": degraded,
    }  # fmt: skip
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(body), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def read_heartbeat(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (OSError, ValueError):
        return None
