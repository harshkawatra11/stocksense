"""
Diagnostic for the Phase K2 acceptance gate's residual mismatch.

The gate reproduced the committed fold COUNT exactly (25/25) but
mean_alpha_net came out +0.01667301577302 against the committed
+0.01672262970534 -- a 0.3% relative difference. "Close" is not a verdict, and
loosening the tolerance until it passes would be precisely the behaviour this
project's entire pre-registration discipline exists to prevent.

So: compare PER FOLD against the committed research/bhavcopy_rerun_fold_results.csv
and look at the SHAPE of the disagreement, which distinguishes the two
candidate explanations:

  (a) The harness is unfaithful -- a real refactor bug. Then folds disagree
      diffusely, with no relationship to date.

  (b) The database has grown since the original run (2026-08-22 per the
      verdict doc; the daily backfill has since added ~6 sessions, and the
      second acceptance run indeed saw 15,337 symbols / 3,973,616 rows against
      the first run's 13,409 / 3,299,207). Under (b) the disagreement should be
      concentrated in LATE folds, and/or explained by the quarantine list
      having grown (more data -> more detected corporate-action anomalies ->
      more symbols removed from the whole history, which perturbs every fold's
      universe slightly).

This script does not decide anything. It prints the per-fold comparison so the
explanation can be read off the data instead of guessed at.

Usage: python research/diagnose_harness_drift.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from stocksense.core.config import get_settings
from stocksense.data.store import Store
from stocksense.research.harness import SweepConfig, run_sweep

RESEARCH_DIR = Path(__file__).resolve().parent


def main() -> int:
    settings = get_settings()
    settings.price_source = "bhavcopy"
    settings.use_point_in_time_universe = True

    store = Store(settings.duckdb_path, read_only=True)
    try:
        config = SweepConfig(
            cap_bands=(("full_pit", None),), horizon_grid=(10,), top_n_grid=(10,),
            cost_grid_bps=(25.0,), gate_cost_bps=25.0,
        )
        with patch("stocksense.data.vault.VAULT_SEAL_DATE", date(2100, 1, 1)):
            result = run_sweep(settings, store, config, random_seed=42)
    finally:
        store.close()

    now = result.fold_results[
        (result.fold_results["cap_band"] == "full_pit")
        & (result.fold_results["horizon_bars"] == 10)
        & (result.fold_results["top_n"] == 10)
        & (result.fold_results["cost_bps"] == 25.0)
    ][["fold_id", "n_rebalances", "alpha_net"]].rename(columns={"alpha_net": "alpha_now", "n_rebalances": "nreb_now"})

    committed = pd.read_csv(RESEARCH_DIR / "bhavcopy_rerun_fold_results.csv")
    then = committed[
        (committed["cap_band"] == "full_pit")
        & (committed["horizon_bars"] == 10)
        & (committed["top_n"] == 10)
        & (committed["cost_bps"] == 25.0)
    ][["fold_id", "n_rebalances", "alpha_net"]].rename(columns={"alpha_net": "alpha_then", "n_rebalances": "nreb_then"})

    merged = then.merge(now, on="fold_id", how="outer").sort_values("fold_id")
    merged["abs_diff"] = (merged["alpha_now"] - merged["alpha_then"]).abs()
    merged["exact_match"] = merged["abs_diff"] < 1e-12

    pd.set_option("display.width", 200)
    print(merged.to_string(index=False))
    print()
    n_exact = int(merged["exact_match"].sum())
    print(f"folds compared      : {len(merged)}")
    print(f"bit-exact matches   : {n_exact}")
    print(f"differing folds     : {sorted(merged.loc[~merged['exact_match'], 'fold_id'].dropna().astype(int))}")
    print(f"max abs fold diff   : {merged['abs_diff'].max():.12f}")
    print(f"mean alpha then/now : {merged['alpha_then'].mean():.14f} / {merged['alpha_now'].mean():.14f}")
    print()
    if n_exact == len(merged):
        print("READ: harness is bit-identical -- the earlier mismatch was elsewhere.")
    elif merged.loc[~merged["exact_match"], "fold_id"].min() >= merged["fold_id"].max() - 2:
        print("READ: disagreement confined to the LAST folds -> explanation (b), database growth.")
    else:
        print("READ: disagreement is diffuse across early folds too -> consistent with explanation (b) "
              "via a CHANGED QUARANTINE LIST (which alters the universe across all of history), "
              "or with (a) a real refactor bug. Check the quarantine-list delta before concluding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
