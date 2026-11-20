"""The GPU lock (Program G2 rule 7): one GPU process at a time, holding `state/gpu.lock`.

Every GPU entry point wraps its work in `with gpu_lock(settings):`; a second holder gets
`RuntimeError("gpu busy")` at once instead of waiting. The lock is a non-blocking msvcrt/fcntl byte lock
(`live.preflight.FileLock`), so it is released when the holder exits or dies.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext

from stocksense.core.config import Settings
from stocksense.core.paths import Paths
from stocksense.models.base import ModelSpec, build_model


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


def spec_uses_gpu(spec: ModelSpec) -> bool:
    """True when the model that `spec` builds runs on CUDA (its `device` attribute); CPU families (gbm,
    linear, dens, ...) have none or a cpu one and never need the GPU lock."""
    device = getattr(build_model(spec), "device", None)
    return getattr(device, "type", str(device or "")) == "cuda"


def gpu_lock_if(settings: Settings, uses_gpu: bool) -> AbstractContextManager[None]:
    """`gpu_lock(settings)` for a GPU model, a no-op context for a CPU one."""
    return gpu_lock(settings) if uses_gpu else nullcontext()
