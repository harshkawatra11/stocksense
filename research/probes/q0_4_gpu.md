# Probe: q0_4_gpu

- **Question:** How many Monte Carlo paths fit in the 4GB VRAM budget?
- **Verdict:** **PASS**
- **Run at:** 2026-09-02T19:44:46.977851+05:30
- **Machine:** Windows-11-10.0.26200-SP0

## Findings

```json
{
  "nvidia_smi_before": {
    "name": "NVIDIA GeForce RTX 3050 4GB Laptop GPU",
    "total_mb": 4096,
    "used_mb": 0,
    "free_mb": 3965,
    "driver": "596.36"
  },
  "cupy": "14.2.0",
  "vram_budget_mb": 2500,
  "max_paths_at_250_steps": 2621440,
  "gpu_timings_s": {
    "100000": 0.233,
    "500000": 0.106,
    "1000000": 0.19
  },
  "nvidia_smi_after": {
    "name": "NVIDIA GeForce RTX 3050 4GB Laptop GPU",
    "total_mb": 4096,
    "used_mb": 77,
    "free_mb": 3889,
    "driver": "596.36"
  },
  "cpu_numpy_100k_paths_s": 0.523,
  "elapsed_s": 3.56
}
```

## Log

- GPU: {'name': 'NVIDIA GeForce RTX 3050 4GB Laptop GPU', 'total_mb': 4096, 'used_mb': 0, 'free_mb': 3965, 'driver': '596.36'}
- budget 2500MB -> 2,621,440 float32 paths x 250 steps
-     100,000 paths x 250 steps: 0.233s
-     500,000 paths x 250 steps: 0.106s
-   1,000,000 paths x 250 steps: 0.19s
- CPU numpy fallback: 100,000 paths x 250 steps in 0.523s
