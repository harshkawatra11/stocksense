# Bhavcopy Point-in-Time Gate Re-Run: Pre-Registration

**Date:** 2026-08-22
**Committed before:** `research/bhavcopy_rerun_sweep.py` has been run against the real 8.17M-row `bhavcopy_eq` spine. Only the wiring (`universe_pit.py`'s `turnover_rank_band` parameter, `cli/main.py`'s `_load_candles` threading it through, and this sweep script itself) has been written and unit-tested as of this commit — no full-universe backtest number of any kind has been seen. This ordering is the entire point of the document, per this project's own established pattern (`gate_criteria_preregistration.md`, `preregistration_intraday.md`).

## Why this run exists

`research/phase0_verdict.md` Run 4's h=10/h=20 GATE PASS was measured on `PHASE0_UNIVERSE`: 98 hand-picked, **currently-listed, currently-liquid** large-cap symbols. Two caveats have stood on that result since Run 3:

1. **Survivorship bias**, previously bounded only by a Monte Carlo synthetic-delisting simulation (`survivorship_bound.py`), not measured on a real point-in-time universe.
2. **The universe has never once included a mid or small cap.** The user's actual product requirement — top 5–10 names skewed mid/small-cap — has never been tested through this pipeline.

`universe_pit.universe_as_of` and `filter_to_point_in_time_universe` were built and unit-tested for exactly this (`data/universe_pit.py`, `docs/STATUS.md`'s "Data spine" entry) but, per that same entry, had "exactly one caller (a display CLI command)" before this — every walk-forward fold to date ran on the full un-filtered symbol set or the hardcoded 98. This run closes both gaps in one experiment: point-in-time-correct **and** cap-band-restricted.

## What has already been seen (full disclosure)

Before writing this document: the new `turnover_rank_band` parameter (`universe_pit.py`) and its wiring through `_load_candles` (`cli/main.py`) were built and unit-tested on **synthetic 5–10 symbol fixtures** (`tests/unit/test_universe_pit.py`, `tests/unit/test_price_source_wiring.py`) to confirm the rank-band slicing logic is correct and point-in-time-safe. No real bhavcopy backtest — full-universe or cap-restricted — has been run or inspected. This is disclosed in full per this project's own standing rule, specifically because unit tests on synthetic fixtures cannot bias a hyperparameter choice on real market data one way or the other.

## The parameters, and where each one comes from

| Parameter | Value | Source |
|---|---|---|
| `price_source` | `"bhavcopy"` | `core/config.py` — the already-built, already-tested alternative to the 98-symbol `candles` path. |
| `use_point_in_time_universe` | `True` | Closes HIGH-4 (survivorship bias) for real rather than by Monte Carlo bound. |
| `horizon_grid` | `(10, 20)` | The two configurations that passed the pre-registered gate in Run 4 on the old universe (`phase0_verdict.md`). Not re-opening the full `(1,3,5,10,20)` sweep here — h=1 already failed independently 5 times and is out of scope; h=3/h=5 were never gate-passing configurations. Testing exactly the two winners on the corrected universe is the direct, minimal test of whether Run 4's result survives the fix. |
| `top_n_grid` | `(10, 20)` | Matches each horizon's own winning `top_n` in Run 4 (h=10→top_n=10, h=20→top_n=20), plus the cross-check of the other value, unchanged from `core/config.py`'s existing default set restricted to the two values that mattered. |
| `cost_grid_bps` | `(10.0, 15.0, 25.0, 35.0)` | Unchanged from `core/config.py`'s existing `cost_grid_bps` default — reused, not re-derived. |
| `min_avg_daily_turnover_inr` | ₹50,00,000 | `core/config.py`'s existing default, unchanged. |
| `min_price_inr` | ₹5.00 | `core/config.py`'s existing default (penny-stock floor), unchanged. |
| `lookback_days` | 60 | `universe_pit.py`'s existing default, unchanged. |
| **Cap bands (`turnover_rank_band`)** | `None` (full PIT universe), `(0.8, 1.0]` "large", `(0.5, 0.8]` "mid", `(0.15, 0.5]` "small" | Percentile slices of that date's turnover-ranked, already-liquidity-filtered universe — a **liquidity-rank proxy for market cap**, not market cap itself (bhavcopy carries no shares-outstanding). Boundaries are round numbers chosen for interpretability before any result exists, not fitted to produce a particular band's outcome. `None` is included as the direct apples-to-apples comparison against Run 4 with only the survivorship fix applied and no cap restriction, isolating the two effects (PIT-universe vs. cap-band) from each other. |
| Fold construction | `make_folds`, `test_window = max(21, horizon * 12)`, embargo per `walkforward.py`'s existing (already-corrected, Run 4) logic | Reused completely unchanged — this run tests a different universe, not a different walk-forward methodology. |
| Ranker | `CrossSectionalRanker`/`RankerConfig(random_state=settings.random_seed)`, unchanged | Same model class and seed as every prior sweep; this run isolates the universe/cap-band effect, not a model change. |
| Gate | `evaluation/gate.py`, **unmodified**; criteria per `gate_criteria_preregistration.md` (`min_mean_alpha_net>0`, one-sided binomial `p<=0.10`, `best_fold_drop_fraction=0.15`, `min_folds_required=10`) | Same exact gate as every prior run in this project. No incumbent comparison is applicable — there is no currently-live model trained on a cap-restricted bhavcopy universe to beat, matching how `verdict_intraday.md` also reports `incumbent_mean_alpha_net: None` for a track with no live incumbent. |

## A real limitation, named rather than hidden

`turnover_rank_band` is a **liquidity-rank proxy for market cap**, not market capitalization. NSE bhavcopy carries no shares-outstanding field, so true market cap cannot be computed from this source alone. A thinly-traded large-cap name could theoretically rank below a heavily-traded mid-cap one on a given date. The band boundaries (0.8/0.5/0.15) are round-number approximations, not calibrated against an actual index constituent list. Real index membership (NIFTY Midcap 150 / Smallcap 250, ingested from NSE's published constituent files) is the correct long-run fix and is **not** built here — this run uses the best available proxy from data already in hand, disclosed as a proxy rather than presented as exact.

## What counts as a pass or fail — written before the full-scale result exists

Evaluated independently per (horizon, top_n, cap_band) combination:

- If a combination produces ≥10 folds, mean net alpha per rebalance > 0, a hit rate whose one-sided binomial p-value ≤ 0.10, and a drop-worst-15%-of-folds mean still > 0 — **that combination's gate passes.**
- If the `None` (full PIT universe, no cap restriction) combination fails where Run 4's old-universe result passed, that is direct, disclosed evidence that Run 4's headline number was inflated by survivorship bias beyond what the Monte Carlo bound estimated — reported as such, not explained away.
- If every mid/small-cap combination fails while large/full-universe combinations pass, that is direct evidence the user's mid/small-cap requirement is not servable by this pipeline as built — reported as such. Per this project's own kill-criteria precedent (`phase0_verdict.md`, `preregistration_intraday.md`), the correct response to any FAIL here is to report the finding and stop, or scope a specifically-targeted follow-up investigation (e.g., real index-membership data) — not to adjust `horizon_grid`/`top_n_grid`/the cap-band boundaries/the gate thresholds until a FAIL becomes a PASS.

No parameter in the table above may be changed after this commit based on the sweep's result.
