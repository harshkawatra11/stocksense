# Probe: q0_5_torch_cuda

- **Question:** Is there a CUDA-enabled torch wheel for this Python?
- **Verdict:** **BLOCKED**
- **Run at:** 2026-09-02T19:44:47.032791+05:30
- **Machine:** Windows-11-10.0.26200-SP0

## Findings

```json
{
  "torch": null,
  "measured_2026_09_02": "pip install torch --index-url https://download.pytorch.org/whl/cu124 -> 'ERROR: Could not find a version that satisfies the requirement torch (from versions: none)'. PyTorch publishes no cp314 CUDA wheel.",
  "elapsed_s": 0.0
}
```

## Log

- no CUDA torch wheel exists for Python 3.14 (measured, not assumed)
- Consequence: any sequence/path-forecast model is CPU-only or deferred. NOT a blocker -- LightGBM is the primary model and CuPy covers Monte Carlo. Revisit when PyTorch ships cp314, or run that one component under the separate Python 3.12 interpreter already on this machine.
