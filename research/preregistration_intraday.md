# Intraday Gate Pre-Registration

**Date:** 2026-08-19
**Committed before:** `research/intraday_sweep.py` has been run. No fold result, no expectancy number, no gate verdict exists yet as of this commit. That ordering is the entire point of this document — see `research/gate_criteria_preregistration.md` for why the project adopted this discipline in the first place (a prior gate's thresholds were chosen after seeing results that happened to clear them; this document exists so that mistake cannot repeat here).

Every parameter below is fixed now, in writing, and **may not be adjusted after seeing a sweep result.** If the sweep fails the gate, the correct response is to report the failure and stop — not to loosen a threshold until it passes.

---

## Universe and data

| Parameter | Value | Source |
|---|---|---|
| Symbol universe | 244 symbols with intraday bar history | `intraday_bars`, resolved via `backfill-intraday --top-n 250` (247 resolved, 3 unmapped to an Upstox instrument) |
| Point-in-time filter | `universe_pit.filter_to_point_in_time_universe`, unchanged | Existing liquidity/price floors, not re-derived for this phase |
| Date range | 2022-01-03 → 2026-08-17 | 1,146 trading days, verified count |
| Bar grain (research) | 5-minute, derived from stored 1-minute | `features/intraday.resample_to_bars_sql`, proven byte-identical to the pandas reference on both fixtures and real data before use |
| Bar grain (label simulation) | 1-minute (native) | `labels/intraday_labels.first_touch_label` walks the true 1-min path, not the 5-min research grain — a stop/target crossed between two 5-min bars must not be missed |

## Fold construction

| Parameter | Value | Reasoning |
|---|---|---|
| Fold unit | Whole trading sessions | A fold boundary inside a session would leak morning into afternoon; extends `walkforward.make_folds`' bar-based logic to session-based |
| `min_train` | ~500 sessions (~2 years) | Matches the existing daily-horizon convention's `min_train_bars` philosophy scaled to sessions |
| `test_window` | ~42 sessions (~2 months) | Small enough to yield ≥10 folds over 1,146 sessions; large enough that one fold isn't dominated by a single week's noise |
| Embargo | ≥1 full session between train and test | The intraday analogue of the daily code's purge+embargo gap — see `walkforward.py`'s own docstring for why feature lookback itself is not a leakage vector but label resolution crossing the boundary is |
| Expected fold count | ~15 | `(1146 − 500) / 42 ≈ 15`, comfortably clears `gate.py`'s `min_folds_required=10` |

## Entry / exit rule

**Entry:** top-N by `CrossSectionalRanker` score at each 5-minute rebalance point, filtered through `fill_model.simulate_fill` (next-bar-open + half-spread fill, participation cap, circuit-lock filter, MIS-leverage lookup — all four checks from Phase E3, unchanged).

**Exit — stop/target/max-hold, grounded in real trading behavior, not invented:**

`positions.holding_seconds` is currently unusable for this purpose — confirmed live: all 441 intraday positions in the `positions` table show `holding_seconds = 0`, because Angel One's `Trades_History` export (the source `statement-ingest` currently parses) carries trade **date** only, no execution **time**. This is a genuine gap in the ingested data, not a bug in FIFO reconstruction, and it also explains why 4 of 13 Kundli behavioral diagnostics (`disposition_effect`, `revenge_trading`, `time_of_day_edge`, `opening_bell_bleed`) currently report `NaN` — tracked as a follow-up (extending the Angel parser to also read `Trading_Insights.xlsx`, which does carry entry/exit timestamps), not fixed as part of this phase.

For this document, the real entry/exit-timestamped data in `statements/Trading_Insights.xlsx` ("Trade" sheet, Angel's own matched round-trips, 138 rows spanning 2026-02-20 → 2026-08-18 — a shorter, more recent window than the full FIFO-reconstructed history) was read directly to ground these parameters:

| Statistic | Value |
|---|---|
| Median holding time | 12.1 min |
| P75 holding time | 25.9 min |
| P90 holding time | 120.8 min |
| Median \|return\| | 0.29% |
| P75 \|return\| | 0.65% |
| P90 \|return\| | 1.33% |
| Win rate | 56.5% |
| Median win | +0.26% |
| Median loss | −0.35% |

**The uncomfortable finding these numbers already show:** median loss magnitude (0.35%) exceeds median win magnitude (0.26%) despite a >50% win rate — the real account's cost-drag dosha (`critical`, 152.8% of gross P&L) and averaging-down dosha (`notable`) are consistent with this. A systematic model that simply reproduced this trader's own instinctive stop/target placement would not be a meaningful test.

**Chosen parameters** (deliberately different from the trader's own realized average, for the reason above — stated here so the choice is auditable, not hidden):

| Parameter | Value | Reasoning |
|---|---|---|
| `stop_pct` | 1.0% | ≈3× the real median loss magnitude — wide enough to absorb ordinary intrabar noise (median realized move is 0.29%), tight enough to be a real risk control, not a number chosen to flatter the sweep |
| `target_pct` | 1.5% | Enforces a 1.5:1 reward:risk ratio at entry — deliberately better than the trader's own realized ratio (0.26:0.35 ≈ 0.74:1), since the point of testing a systematic entry/exit is to see whether disciplined risk:reward — not the trader's own instinct — produces a viable edge |
| `max_holding_minutes` | 60 | Covers the P75 real holding time (25.9 min) with headroom; well short of the P90 (120.8 min) tail, which the session-close backstop (15:30, non-negotiable) already bounds regardless |

These are fixed now. If the sweep result would look better with a different stop/target, that is not a reason to change them post hoc — it is the finding.

## Costs

`execution.cost_model.compute_charges(segment="equity_intraday")`, unchanged — already verified correct in this project: STT 2.5bps sell-leg-only, stamp duty buy-leg-only.

## Benchmark for alpha

Mean return of the tradeable cross-section over the same holding window, matching the daily pipeline's existing relative-return convention (`labels/forward_return.add_relative_forward_return`) — a candidate's alpha is measured against what an average tradeable name did in the same window, not against zero.

## Gate

`evaluation/gate.py`, **completely unchanged**: `min_mean_alpha_net > 0`, one-sided binomial hit-rate test at `alpha=0.10`, best-15%-of-folds-dropped stress test, `min_folds_required=10`. No new criteria, no relaxed criteria.

## What counts as pass or fail — written before the sweep runs

If `research/intraday_sweep.py` produces ≥10 session-based folds, mean net alpha > 0, a hit rate significant at the pre-registered binomial threshold, and a positive mean after dropping the best 15% of folds — **the gate passes**, and per `evaluation/gate.py`'s existing behavior, the candidate is promoted to `shadow`, not `live` (shadow trial is a separate, later earn).

If it does not clear all four — **the gate fails**, and the correct response, per this project's own established precedent, is to report the failure honestly in `research/verdict_intraday.md` and stop. Phase E5 (risk controls) and E6 (daily loop) do not get built on a failed gate.

## Known limitations, named in advance

1. **2022–2026 is one broad market regime** — no 2020-style crash in the sample. Fold count will be healthy; regime diversity will not be.
2. **Upstox bars ≠ Angel One fills.** `fill_model.py` narrows this gap (real spread/participation/circuit checks); only live paper trading against real fills closes it.
3. **Stop/target are chosen from 138 recent round-trips** (`Trading_Insights.xlsx`'s covered window), not the full 444-position history — the best available real data, but a smaller and more recent sample than the full FIFO reconstruction. If Phase E5 later fixes the holding-time gap and the full-history distribution differs materially, that's a legitimate reason to *pre-register a revised set of parameters before a follow-up sweep* — never to adjust these after seeing this sweep's result.
4. **Real Upstox instrument-master gaps:** `DIACABS`, `JBCHEPHARM`, `MTARTECH` never resolved to an Upstox instrument during the E1 backfill and are absent from the universe. Immaterial at 244/247 symbols, named for completeness.
