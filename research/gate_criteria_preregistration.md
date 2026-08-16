# Gate Criteria Pre-Registration

**Date:** 2026-08-13
**Committed before:** the corrected-methodology sweep re-run (fixed embargo, see `d4ab097`) and the survivorship-bound simulation. Neither has been run yet as of this commit. This ordering is the entire point of the document — see below.

## Why this document exists

The first version of `evaluation/gate.py` (commit `85f11cb`) used `min_pct_folds_positive=0.6` and `n_best_folds_to_drop=2`. Both were chosen after already seeing Run 2/3 results, which happened to show 77–82% of folds positive under a 2-fold-removal test. The gate could not have failed the model it was built to describe.

`docs/10-evaluation.md` §17 names this exact failure mode — tuning against results you can already see — as the deepest risk in the whole evaluation design. It was committed in the same session that wrote that warning. This document is the fix: criteria fixed by principle, in writing, before the numbers that will be judged against them exist.

## The criteria, and why each was chosen without reference to Phase 0's results

| Criterion | Value | Justification (not data-fitted) |
|---|---|---|
| `min_mean_alpha_net` | `> 0.0` | Definitional — a model that doesn't clear costs on average has no claim to being deployed at all. Not a tuned number; there is no other value this could sensibly be. |
| `hit_rate_significance_alpha` | `0.10` | A one-sided exact binomial test against the null "folds are positive by pure chance" (p=0.5). 0.10 (not the conventional 0.05) is chosen because walk-forward folds are few and serially correlated — a stricter alpha would make the test nearly powerless at the fold counts this project can realistically produce (10–25). This is a standard small-sample tradeoff, not a threshold picked to make a specific result pass. |
| `best_fold_drop_fraction` | `0.15` | Scale-invariant by construction — a fraction, not a count. Chosen as "enough to remove a small cluster of lucky/unlucky periods without gutting the sample," independent of how many folds any specific run happens to produce. At 11 folds this rounds to 2 (coincidentally matching the old fixed value); at 20 folds it becomes 3. The rule stays fixed; only its application to a given fold count changes. |
| `min_folds_required` | `10` | Below this, neither the binomial test nor the drop-fraction test has enough samples to mean much. Set from the mechanics of the tests themselves, not from how many folds Phase 0 happened to produce (Phase 0's h=20 run had 11 — just above this floor, not tuned to sit comfortably above it). |
| Incumbent comparison | candidate must strictly beat incumbent's `mean_alpha_net` | Unchanged from the original version; this one was never in question — a replacement model that isn't actually better has no reason to replace anything. |

## What counts as a pass or fail — written before knowing the answer

The corrected-methodology sweep (fixed embargo) and the survivorship-bound simulation have **not been run** as of this document's commit. Whatever they produce, evaluated against the table above, is the real result. Specifically:

- If the corrected sweep's h=20 configuration produces ≥10 folds, mean net alpha > 0, a hit rate whose one-sided binomial p-value ≤ 0.10, and a drop-worst-15%-of-best-folds mean still > 0 — **the gate passes**, and that is evidence, not a foregone conclusion, because these thresholds were fixed before the run.
- If it does not clear all four — **the gate fails**, and per the kill criteria already recorded in `research/phase0_verdict.md`, the correct response is to stop and re-research the signal, not to adjust the thresholds until it passes.

## What this does not fix

Pre-registering the *threshold values* stops the specific overfitting failure that happened once. It does not stop a subtler version: if this exact gate is run repeatedly against re-tunings of the *model* (different features, different hyperparameters) until one clears it, the gate becomes an overfitting instrument again — just one level removed. `docs/10-evaluation.md`'s evaluation-attempt counter (OQ-11, not yet built) is the actual long-term defense against that; this document only closes the specific hole found in this audit.
