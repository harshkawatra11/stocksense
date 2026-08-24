# F&O Positioning Signal: Pre-Registration

**Date:** 2026-08-24
**Committed before:** `research/fno_signal_sweep.py` has been run against real data. `bhavcopy_fo` is being backfilled as this document is written; `features/fno.py`'s functions have only ever been exercised against synthetic fixtures in `tests/unit/test_fno_features.py`, never against a real trained model or a real walk-forward fold. No result of any kind exists yet. This ordering is the entire point of the document, per this project's own established pattern (`gate_criteria_preregistration.md`, `preregistration_intraday.md`, `preregistration_bhavcopy_rerun.md`).

## Why this run exists

The user asked for daily-movement (same/next-day) prediction. That exact hypothesis — direction over 1 trading bar, from OHLCV-derived features — has been tested **five independent times** on this codebase's data and failed every time:

1. v1's original diagnosis
2. Run 1 (2010–2026, 98 symbols)
3. Run 2 (2000–2026, same universe)
4. Run 3 (data-bug-corrected)
5. Run 4 (embargo-corrected, 191 folds) — gross alpha ≈ +0.044%/rebalance against a 7.3bps break-even; **indistinguishable from zero**, not merely uneconomic
6. Phase E4's dedicated intraday track (real Upstox minute bars, MIS costs, same-day label) — **gross expectancy negative in all 15 folds, before a single rupee of cost**

Presented with this record directly, the user chose — explicitly, in writing — **one further real attempt, using genuinely new data rather than a new model on the same price data, with the odds disclosed before running it.** This document is that disclosure.

The new data: `src/stocksense/data/nse_archive.py`'s `fetch_fo_bhavcopy` and `src/stocksense/features/fno.py`'s `build_oi_features`/`build_put_call_ratio`/`classify_oi_quadrant`/`days_to_expiry` have existed in this codebase since an earlier phase, fully built and unit-tested in isolation, but **never backfilled (`bhavcopy_fo` held 0 rows before this phase) and never wired into a trained model.** `fno.py`'s own docstring already states the correct standard: "these features are admitted only if they beat an ablation, same as delivery.py" — that ablation has never actually been run. This closes that gap.

## What has already been seen (full disclosure)

Before writing this document: `features/fno.py`'s functions were exercised only against small synthetic fixtures in `tests/unit/test_fno_features.py` (4-quadrant classification, OI aggregation excluding options from the futures-only sum, a hand-computed PCR case) — confirming the *arithmetic* is correct, never that the *signal* is real. No real `bhavcopy_fo` data has been queried, joined to price features, or trained against as of this commit. This is disclosed in full per this project's standing practice, specifically because synthetic-fixture correctness checks cannot bias a real-data hyperparameter or feature choice in either direction.

## The parameters, and where each one comes from

| Parameter | Value | Source |
|---|---|---|
| `horizon_bars` | 1 | Identical to every one of the five prior h=1 attempts (`labels.forward_return.add_forward_return_labels(horizon_bars=1)`, unchanged) — matches the user's actual ask ("profit in a single trading day") and preserves direct comparability to the prior negative results. |
| Universe | F&O-eligible symbols only, point-in-time | Not a choice — a hard constraint. NSE lists single-stock derivatives for roughly 180–200 names; this hypothesis can never apply outside that set. Point-in-time eligibility is derived from `bhavcopy_fo` itself (a symbol counts as F&O-eligible on date `d` if it has a `FUTSTK` row on or before `d` within a trailing window, mirroring `universe_pit.py`'s own no-future-data discipline), intersected with `universe_pit.universe_as_of`'s existing liquidity floor — reused unchanged, not re-derived. |
| Feature set A (baseline) | `features/engine.py`'s existing price features, on the F&O-restricted universe | The correct control: NOT the old 98-symbol or full-PIT h=1 numbers (different universe, would confound the comparison), but the SAME price features re-run on this smaller, F&O-eligible-only universe. Isolates "does F&O data help" from "is this just a different, more liquid universe." |
| Feature set B (treatment) | Feature set A, plus `build_oi_features`, `build_put_call_ratio`, `classify_oi_quadrant`, `days_to_expiry` | Joined point-in-time-safe: a date `d`'s features are built ONLY from `bhavcopy_fo` rows dated `< d` (yesterday's close OI/PCR predicting today's forward return) — the identical "features known at T predict return T→T+h" pattern every other horizon in this codebase already uses, not new methodology, just a new feature source at h=1. |
| Cost grid | `settings.cost_grid_bps`, unchanged | Reused exactly as every prior h=1 sweep used it — not an MIS/intraday cost model, deliberately, so this test isolates the feature-set change from an execution-assumption change. |
| Fold construction | `evaluation.walkforward.make_folds`, defaults unchanged | Same mechanism as every prior sweep in this project. |
| Ranker | `CrossSectionalRanker`/`RankerConfig(random_state=settings.random_seed)`, unchanged | Deliberately the SAME model class as every prior test. A model-class change (CatBoost, an ensemble) is explicitly NOT bundled into this run — see "What this does not do" below. |
| Gate | `evaluation/gate.py`, **unmodified**; criteria per `gate_criteria_preregistration.md`, unchanged | Same exact gate as every prior run in this project. |

## A real limitation, named rather than hidden

Point-in-time F&O eligibility is derived from `bhavcopy_fo`'s own contract listings, not from an independent, authoritative NSE eligibility-list archive — a symbol that briefly had a thinly-traded future listed and delisted could introduce noise into the universe boundary. This is a coarser point-in-time reconstruction than `universe_pit.py`'s equity-side liquidity filter, which is built from the fuller `bhavcopy_eq` turnover history. Flagged here as a known imprecision, not something to quietly assume away.

## What counts as a pass or fail — written before the full-scale result exists, odds disclosed

- **The prior**: 5 of 5 same-horizon attempts on price-only features failed, one decisively (negative gross alpha before costs). This changes the feature set, not the underlying scarcity of same-day equity signal. This should be read as a real but long-odds attempt — genuinely untested, but base rates for discovering a new tradeable alpha source at this horizon are low, and options-derived signals in particular have mixed, often non-robust support in the broader literature once realistic costs are applied.
- **PASS**: feature set B clears `evaluation/gate.py`'s criteria (≥10 folds, mean net alpha net > 0, one-sided binomial hit-rate p ≤ 0.10, mean after dropping best 15% of folds still > 0) AND beats feature set A's own result on the same universe — both conditions, not either alone, since B passing while A also passes at similar magnitude would mean the F&O features added nothing.
- **FAIL** (either B doesn't clear the gate, or B doesn't measurably beat A): report the result and stop pursuing same-day prediction on this data combination — per this project's own kill-criteria precedent (`phase0_verdict.md`, `preregistration_intraday.md`), not adjust the feature set, horizon, or model until something passes. The 10-day mid/small-cap system (Phase H, already live) remains the one result in this project with real forward evidence behind it, regardless of this outcome.

No parameter in the table above may be changed after this commit based on the sweep's result.
