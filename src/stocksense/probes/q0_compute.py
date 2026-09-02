"""Q0.4 / Q0.5 -- what this machine can actually run.

Measured, not assumed: the RTX 3050 has 4 GB and it is SHARED with the display
and with Ollama. CuPy and a 7B-class model cannot both hold it. Everything in
simulation/montecarlo.py is sized against the number this probe returns, and the
CPU fallback exists because of it.
"""

from __future__ import annotations

import os
import subprocess
import time

import numpy as np

from stocksense.probes.base import ProbeResult


def _nvidia_smi() -> dict[str, object]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if out.returncode != 0:
            return {"error": out.stderr.strip()[:400]}
        name, total, used, free, driver = (x.strip() for x in out.stdout.strip().split(","))
        return {
            "name": name,
            "total_mb": int(total),
            "used_mb": int(used),
            "free_mb": int(free),
            "driver": driver,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def probe_gpu(result: ProbeResult) -> None:
    result.findings["nvidia_smi_before"] = _nvidia_smi()
    result.note(f"GPU: {result.findings['nvidia_smi_before']}")

    try:
        import cupy as cp  # noqa: PLC0415
    except ImportError:
        result.verdict = "BLOCKED"
        result.note("cupy not installed -- run: pip install -e '.[gpu]'")
        result.note("Not fatal: simulation/montecarlo.py has a numba.prange CPU fallback.")
        result.findings["cupy"] = None
        _probe_cpu_montecarlo(result)
        return

    result.findings["cupy"] = cp.__version__

    # Find the largest float32 batch that fits inside the configured VRAM budget.
    # Sized in PATHS x STEPS, the shape a real Monte Carlo batch takes.
    budget_mb = int(os.environ.get("STOCKSENSE_GPU_VRAM_BUDGET_MB", "2500"))
    steps = 250  # one trading year of daily steps
    bytes_per_path = steps * 4  # float32
    max_paths = (budget_mb * 1024 * 1024) // bytes_per_path
    result.findings["vram_budget_mb"] = budget_mb
    result.findings["max_paths_at_250_steps"] = int(max_paths)
    result.note(f"budget {budget_mb}MB -> {max_paths:,} float32 paths x {steps} steps")

    timings = {}
    for n_paths in (100_000, 500_000, 1_000_000):
        if n_paths > max_paths:
            continue
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        rng = cp.random.default_rng(7)
        x = rng.standard_normal((n_paths, steps), dtype=cp.float32)
        terminal = cp.exp(cp.cumsum(x * cp.float32(0.01), axis=1)[:, -1])
        _ = float(terminal.mean())
        cp.cuda.Stream.null.synchronize()
        timings[n_paths] = round(time.perf_counter() - t0, 3)
        result.note(f"  {n_paths:>9,} paths x {steps} steps: {timings[n_paths]}s")
        del x, terminal
        cp.get_default_memory_pool().free_all_blocks()

    result.findings["gpu_timings_s"] = timings
    result.findings["nvidia_smi_after"] = _nvidia_smi()
    result.verdict = "PASS" if timings else "FAIL"
    _probe_cpu_montecarlo(result)


def _probe_cpu_montecarlo(result: ProbeResult) -> None:
    """The fallback path's real throughput -- it must be usable, not theoretical."""
    steps, n_paths = 250, 100_000
    t0 = time.perf_counter()
    rng = np.random.default_rng(7)
    x = rng.standard_normal((n_paths, steps), dtype=np.float32)
    _ = float(np.exp(np.cumsum(x * np.float32(0.01), axis=1)[:, -1]).mean())
    elapsed = round(time.perf_counter() - t0, 3)
    result.findings["cpu_numpy_100k_paths_s"] = elapsed
    result.note(f"CPU numpy fallback: {n_paths:,} paths x {steps} steps in {elapsed}s")


def probe_torch_cuda(result: ProbeResult) -> None:
    """Q0.5 -- is there a CUDA-enabled torch wheel for this Python?

    Python 3.14 is new enough that a cp314 CUDA build may not exist. If it does
    not, the sequence-model work runs on CPU or is deferred -- decided from this
    result, not from hope.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        result.verdict = "BLOCKED"
        result.findings["torch"] = None
        result.findings["measured_2026_09_02"] = (
            "pip install torch --index-url https://download.pytorch.org/whl/cu124 -> "
            "'ERROR: Could not find a version that satisfies the requirement torch "
            "(from versions: none)'. PyTorch publishes no cp314 CUDA wheel."
        )
        result.note("no CUDA torch wheel exists for Python 3.14 (measured, not assumed)")
        result.note(
            "Consequence: any sequence/path-forecast model is CPU-only or deferred. "
            "NOT a blocker -- LightGBM is the primary model and CuPy covers Monte "
            "Carlo. Revisit when PyTorch ships cp314, or run that one component "
            "under the separate Python 3.12 interpreter already on this machine."
        )
        return

    result.findings["torch"] = torch.__version__
    result.findings["cuda_available"] = bool(torch.cuda.is_available())
    result.findings["cuda_version"] = getattr(torch.version, "cuda", None)
    if torch.cuda.is_available():
        result.findings["device"] = torch.cuda.get_device_name(0)
        result.verdict = "PASS"
        result.note(f"torch {torch.__version__} CUDA {torch.version.cuda} on {result.findings['device']}")
    else:
        result.verdict = "BLOCKED"
        result.note(f"torch {torch.__version__} present but CPU-only -- no CUDA wheel for this Python")
