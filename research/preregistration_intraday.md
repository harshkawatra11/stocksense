# Intraday Gate Pre-Registration

**Date:** 2026-08-20
**Committed before:** the full-scale sweep (`research/intraday_sweep.py`) over the real 91.2M-bar Upstox spine. Only a small pipeline smoke test has been run as of this commit (see "What has already been seen" below, disclosed in full). This ordering is the entire point of the document — see `research/gate_criteria_preregistration.md` for why this project takes it seriously: the first version of `evaluation/gate.py` used thresholds chosen after seeing results, and that is exactly the failure this ordering prevents.

## What has already been seen (full disclosure)

Before writing this document, a **pipeline smoke test** was run on 8 of 244 symbols, 3 of ~45 folds, purely to confirm the trade-shaped backtest loop (`evaluation/intraday_backtest.py`, committed at `6aa66e4`) runs end-to-end without crashing and emits `gate.py`-compatible output. It was not run to evaluate the edge.

Result: all 3 folds showed negative net alpha (≈ −0.13% per fold, per-trade). This is disclosed here in full rather than omitted, specifically **because** it is negative — nobody adjusts a threshold to make their own smoke test look worse, so this result cannot be the source of any parameter chosen below. The full sweep below uses the complete universe and all folds; a 3-fold/8-symbol slice is not a statistically meaningful preview of that result either way.

## The parameters, and where each one comes from

| Parameter | Value | Source |
|---|---|---|
| `stop_pct` | 1.0% | Close to the real average intraday loss magnitude in the user's own closed positions: 0.97% (`positions` table, 441 intraday round trips, `AVG(ABS(exit_price-entry_price)/entry_price) WHERE net_pnl<0`). Not exact — see limitation below — but grounded, not arbitrary. |
| `target_pct` | 1.5% | Close to the real average intraday win magnitude: 1.24% (same table, `net_pnl>0`). |
| `max_holding_minutes` | 60 | **Not derivable from real data** — see limitation below. A session-management default (long enough to let a real setup develop, short enough that "did this trade even work" is knowable same-session), not a fitted number. |
| `top_n` | 10 | Matches the existing daily-pipeline sweep grid's own `top_n=10` best-alpha configuration (`research/phase0_viability_surface_2010_2026.csv`), for continuity with a value that already had honest support elsewhere in this project — not re-derived from intraday data, since intraday capacity constraints (open-position skip in `simulate_intraday_trades_for_fold`) are a different mechanism than the daily rebalance. |
| `exposure_inr` | ₹75,000 | The user's own stated capital × leverage (₹15,000 × 5×), per the arithmetic already fixed in the Phase E context document. |
| Fold unit | whole trading sessions | Never split mid-session — `session_split` compares each bar's normalized session date, not a raw timestamp, against fold boundaries (fixes a real boundary-exclusion bug found during E4 development). |
| `min_train_sessions` / `test_window_sessions` / `embargo_sessions` | 500 / 42 / 1 | ~2 years train, ~1 quarter test, 1 full session gap — the session-unit reinterpretation of the daily pipeline's own `make_folds` defaults (`evaluation/walkforward.py`), reused completely unchanged. |
| Entry fill | next 5-min bar's open + half-spread (2.5bps), participation-capped at 10% of that bar's volume, circuit-lock excluded | `execution/fill_model.simulate_fill`, Phase E3 — never the signal bar's own close. |
| Exit | first-touch stop/target/time against the true 1-minute path, conservative stop-wins on same-bar ambiguity | `labels/intraday_labels.first_touch_label`, Phase E2. |
| Costs | `execution/cost_model.compute_charges(segment="equity_intraday")` | Verified correct against Zerodha's published charge sheet (STT 2.5bps sell-leg-only, stamp duty buy-leg-only). |
| Ranking target (training only) | cross-sectional relative session return over `max_holding_minutes` bars, path-**independent** | `add_relative_session_return` — the path-dependent first-touch outcome is only ever computed at evaluation time, never fed to the ranker as a training signal (it doesn't exist until an entry has already been decided). |
| Universe | every symbol with intraday bars (244, the full backfilled set) | No further liquidity filter applied at this stage — the backfill's own top-250-by-turnover selection (`data/upstox_intraday.py`) already is the liquidity filter. |
| Gate | `evaluation/gate.py`, **unmodified** | Same exact binomial hit-rate test, 0.15 best-fold drop fraction, `min_folds_required=10`, incumbent comparison. |

## A real limitation, named rather than hidden

`positions.holding_seconds` is currently unusable for grounding `max_holding_minutes`: querying the real data shows **median and mean holding time for all 441 intraday positions rounds to 0.0 minutes**, which is not plausible for real trades. This traces to the Angel One statement export (`statements/parsers/angel.py`) apparently not carrying reliable intraday time-of-day granularity — `open_time`/`close_time` are present but too coarse to compute a meaningful `holding_seconds` for same-day round trips. Win/loss **price magnitudes** (which come from `entry_price`/`exit_price`, unaffected by this) remain reliable and are what grounds `stop_pct`/`target_pct` above. This gap is a real fix for a later phase (tightening the Angel parser's time extraction), not something papered over here by inventing a holding-time number that looks derived but isn't.

## What counts as a pass or fail — written before the full-scale result exists

- If the full sweep (244 symbols, ~45 folds) produces ≥10 folds, mean net alpha per trade > 0, a hit rate whose one-sided binomial p-value ≤ 0.10, and a drop-worst-15%-of-folds mean still > 0 — **the gate passes**, and per the Phase E plan, E5 (risk controls) and E6 (the daily loop) proceed.
- If it does not clear all criteria — **the gate fails**, and per this project's own kill-criteria precedent (`research/phase0_verdict.md`), the correct response is to report the finding and stop, not to adjust `stop_pct`/`target_pct`/`top_n`/the fold parameters until it passes.

No parameter in the table above may be changed after this commit based on the full sweep's result.
