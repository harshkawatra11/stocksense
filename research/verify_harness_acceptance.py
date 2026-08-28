"""
Phase K2.2's acceptance gate -- run this manually (real database, several
minutes) before trusting any new result out of research/harness.py.

Reproduces ONE cell of research/verdict_bhavcopy_rerun.md exactly:
full_pit / horizon=10 / top_n=10 / cost=25bps -- 25 folds, mean net alpha
+0.01672262970534454 (+1.672%/rebalance). That number is committed history,
produced by the original hand-written bhavcopy_rerun_sweep.py; if this
refactored harness cannot reproduce it, the harness has a bug and nothing
built on top of it (the search loop, any new sweep) should be trusted yet.

Usage: python research/verify_harness_acceptance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stocksense.core.config import get_settings
from stocksense.data.store import Store
from stocksense.research.harness import SweepConfig, run_sweep

EXPECTED_MEAN_ALPHA_NET = 0.01672262970534454
EXPECTED_N_FOLDS = 25
TOLERANCE = 1e-6


def main() -> int:
    settings = get_settings()
    settings.price_source = "bhavcopy"
    settings.use_point_in_time_universe = True

    store = Store(settings.duckdb_path, read_only=True)
    try:
        config = SweepConfig(
            cap_bands=(("full_pit", None),),
            horizon_grid=(10,),
            top_n_grid=(10,),
            cost_grid_bps=(25.0,),
            gate_cost_bps=25.0,
        )
        result = run_sweep(settings, store, config, random_seed=42)
    finally:
        store.close()

    cell = result.fold_results[
        (result.fold_results["cap_band"] == "full_pit")
        & (result.fold_results["horizon_bars"] == 10)
        & (result.fold_results["top_n"] == 10)
        & (result.fold_results["cost_bps"] == 25.0)
    ]
    n_folds = len(cell)
    mean_alpha_net = float(cell["alpha_net"].mean()) if n_folds else float("nan")

    print(f"n_folds       : {n_folds} (expected {EXPECTED_N_FOLDS})")
    print(f"mean_alpha_net: {mean_alpha_net:.14f} (expected {EXPECTED_MEAN_ALPHA_NET:.14f})")

    ok_folds = n_folds == EXPECTED_N_FOLDS
    ok_alpha = abs(mean_alpha_net - EXPECTED_MEAN_ALPHA_NET) < TOLERANCE
    if ok_folds and ok_alpha:
        print("ACCEPTANCE GATE: PASS -- harness reproduces the committed result exactly.")
        return 0
    print("ACCEPTANCE GATE: FAIL -- the harness does NOT reproduce the committed result. "
          "Do not trust any new result from research/harness.py until this is fixed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
