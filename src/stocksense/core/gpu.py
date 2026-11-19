"""The GPU lock (Program G2 rule 7): one GPU process at a time, holding `state/gpu.lock`.

Every GPU entry point wraps its work in `with gpu_lock(settings):`; a second holder gets
`RuntimeError("gpu busy")` at once instead of waiting. The lock is a non-blocking msvcrt/fcntl byte lock
(`live.preflight.FileLock`), so it is released when the holder exits or dies.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from stocksense.core.config import Settings
from stocksense.core.paths import Paths


@contextmanager
def gpu_lock(settings: Settings) -> Iterator[None]:
    from stocksense.live.preflight import FileLock

    lock = FileLock(Paths(settings).state / "gpu.lock")
    if not lock.acquire():
        raise RuntimeError("gpu busy")
    try:
        yield
    finally:
        lock.release()
