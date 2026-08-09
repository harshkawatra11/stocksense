# 05 — The Nightly Pipeline

## The sequence

Fifteen steps, strictly ordered. Each has defined inputs, defined outputs, and defined failure behavior.

```
 1  Ingest              ──┐
 2  Validate              │  DATA        hard-fail zone
 3  Reconstruct trades  ──┘
 4  Grade predictions   ──┐
 5  Grade decisions       │  LEARNING
 6  Rebuild features      │
 7  Relabel regimes     ──┘
 8  Train candidates    ──┐
 9  Evaluate (adversarial)│  MODEL       gated zone
10  GATE                ──┘
11  Shortlist           ──┐
12  Investigate           │  OUTPUT      degrade-visibly zone
13  Synthesize            │
14  Deliver             ──┘
15  Record heartbeat        ALWAYS RUNS
```

The three zones behave differently on failure, and the difference is deliberate. Steps 1–3 hard-fail because proceeding on bad data corrupts everything downstream. Steps 8–10 are gated because an unvalidated model must never reach the user. Steps 11–14 degrade visibly because enrichment failures should cost quality, not the whole night.

---

## Step 1 — Ingest

**In:** Upstox API (Analytics surface). **Out:** raw candles, F&O snapshots, order fills for the session.

Fetches the day's incremental data: OHLCV for the active universe, F&O snapshots for derivative instruments, and the user's own orders, fills, positions, and funds. Uses the **V3 historical endpoints** ([02-data-layer.md](02-data-layer.md)) — V2 would silently deliver far shallower history.

Idempotent by date: re-running for the same session upserts rather than appends.

**On failure — HARD FAIL.** Abort the run. A night without data is not a night that can be partially salvaged: features would be computed on gaps, grading would score against absent outcomes, and training would fit noise. Mark the cycle failed, surface it, retry per the backoff policy in [08-operations.md](08-operations.md).

---

## Step 2 — Validate

**In:** newly ingested data. **Out:** validation report, quarantine flags.

Runs the full check set from [02-data-layer.md](02-data-layer.md): trading-day expectation, candle completeness, OHLC sanity, price continuity against corporate actions, universe freshness, F&O linkage, duplicate detection.

**On failure — HARD FAIL for structural problems** (missing sessions, unapplied corporate actions, OHLC violations). **Quarantine for isolated ones** — a handful of suspect instruments are flagged and excluded from this night's downstream processing rather than killing the run.

The governing principle, stated in [02-data-layer.md](02-data-layer.md) and repeated because it is the one most likely to be compromised under time pressure: **skip a night loudly rather than train on corrupt data quietly.**

---

## Step 3 — Reconstruct trades

**In:** raw order fills. **Out:** `trade_ledger` rows.

Raw fills are a flat list; they carry no notion of a trade. This step groups them into whole trades — the ₹830 → ₹827 → ₹833 → ₹850 lifecycle with its re-entries and scale-ins reconstructed as one object rather than six unrelated events.

Grouping is by instrument and session, with a new trade beginning when position returns to flat. For each reconstructed trade it computes: the ordered entry/exit sequence, realized P&L net of costs, holding duration, maximum favorable excursion, and maximum adverse excursion.

MFE and MAE are what make step 5 possible. Without knowing the best and worst prices available while the position was open, "you exited early" is an assertion rather than a measurement.

**On failure — HARD FAIL.** Trade data is account truth; a reconstruction that cannot complete indicates either a data problem or a logic bug, and both need attention before the ledger is written to.

---

## Step 4 — Grade predictions (Track A)

**In:** `predictions` rows whose horizon closed today, plus actual outcomes. **Out:** grades appended to those rows.

Scores each matured prediction on direction (was the sign right), magnitude (how far off was the expected move), timing (did it happen within the horizon), calibration contribution (feeding the aggregate reliability measurement in [06-retraining-rigor.md](06-retraining-rigor.md)), and path quality (did a "correct" prediction survive a drawdown that would have stopped a real position out).

Shadow-mode predictions are graded identically to live ones — that is the entire point of shadow mode.

**Predictions are immutable.** Grading appends outcome columns; it never edits the original forecast. A system that can revise its own history cannot be audited.

**On failure — HARD FAIL.** Grading is the mechanism by which the system learns. A night that skips it is a night that learned nothing while appearing to run.

---

## Step 5 — Grade decisions (Track B)

**In:** `trade_ledger`, `predictions`. **Out:** decision grades, behavioral pattern updates.

This is the "what could have happened" engine, and it is emphatically **not** a P&L report.

For each reconstructed trade it asks: relative to the model's assessment at the moment of each entry and exit, was that decision sound? Both directions are evaluated, and both are reported:

```
Case A — exit was early
  Exit ₹833 · subsequent max ₹850 · model P(continuation) at exit: 71%
  → Holding was statistically justified. Left ₹17 on the table.

Case B — exit was correct
  Exit ₹833 · subsequent low ₹810 · model P(continuation) at exit: 38%
  → Exit avoided a larger loss. Good decision.
```

Case B matters as much as Case A. A grader that only flags missed upside teaches the user to hold losers, which is a worse habit than the one it corrected. The system must be able to say *"your trade was profitable but the decision was poor"* and *"your trade lost money but the decision was sound"* — that separation is the whole value of Track B.

Behavioral patterns accumulate across trades: early-exit tendency, post-loss overtrading, re-entry expectancy by attempt number, sizing drift after winning streaks, time-of-day performance.

**Output goes to the Synthesizer as coaching context and nowhere else.** No path connects this step to step 8. That is the Track A / Track B invariant ([00-overview.md](00-overview.md)).

**On failure — DEGRADE.** Coaching is valuable but not load-bearing. Log the failure, mark the review as missing its behavioral section, continue.

---

## Step 6 — Rebuild features

**In:** validated market data. **Out:** versioned feature rows.

Computes the full taxonomy from [03-feature-engineering.md](03-feature-engineering.md) for the new session. Incremental — but with a trailing window sufficient to satisfy the longest lookback, since a 200-day feature needs 200 days behind it.

**Point-in-time correctness is enforced here**, and a violation is a silent catastrophe: leakage produces beautiful backtests and worthless live predictions. Feature computation must be strictly backward-looking.

**On failure — HARD FAIL.** No features, no training, no inference.

---

## Step 7 — Relabel regimes

**In:** features, volatility measures. **Out:** regime labels at instrument and market level.

Applies the deterministic labeling rules ([04-model-brain.md](04-model-brain.md)) to produce the ground-truth labels the regime classifier trains against.

**On failure — HARD FAIL.** Regime labels are training targets; without them step 8 has nothing to fit the router to.

---

## Step 8 — Train candidates

**In:** feature rows, regime labels, matured outcomes. **Out:** candidate models (unpromoted).

Trains the regime classifier and each per-regime specialist. Every candidate is recorded in the model registry as `candidate` with its full reproducibility manifest — training window, feature schema version, hyperparameters, random seed ([06-retraining-rigor.md](06-retraining-rigor.md)).

**Candidates are never used for inference.** They are trained, then judged. The currently-live model continues serving until and unless the gate says otherwise.

**On failure — DEGRADE.** Training failure is not a catastrophe: the live model is untouched and still valid. Log it, skip to step 11, generate tonight's shortlist with the incumbent, and surface the failure prominently.

---

## Step 9 — Evaluate

**In:** candidate models, held-out data, episode library. **Out:** scorecard, gate verdicts.

The Evaluator ([10-evaluation.md](10-evaluation.md)) runs its suite against every candidate: purged and embargoed walk-forward validation, net of realistic transaction costs, stratified by regime, measured for calibration as well as accuracy, and compared against the baseline gauntlet.

**Nightly runs a subset; the full suite runs on cadence.** The complete evaluation — every episode, every era, Monte Carlo, the full stress battery — is hours of compute and does not belong in every night's window. Nightly runs the fast, decisive checks: walk-forward on recent folds, cost-adjusted comparison against the incumbent, calibration, regime stratification, and the **Baseline 8 ablation** (LightGBM-only), which is cheap and answers the most important question about whether the LLM layers are earning their place.

The full suite runs weekly, on demand from the Control Room, and always before any promotion that would move a model down the production staircase toward real capital.

**A candidate that has only passed the nightly subset is marked as such.** Partial evaluation never counts as full evaluation, and the scorecard states which it was.

**On failure — DEGRADE, and reject.** An unevaluatable candidate is not promotable. Keep the incumbent.

---

## Step 10 — GATE

**In:** evaluator verdicts, incumbent metrics. **Out:** promotion decision, registry update.

The decision point the entire thesis rests on. The Gate does not invent criteria — it **executes the verdict the Evaluator produced** ([10-evaluation.md](10-evaluation.md)). Criteria live in one place; the mechanism lives here.

```
        evaluator verdicts vs live model
                        │
        ┌───────────────┴────────────────┐
        │  ALL hard gates PASS:           │
        │   • beats incumbent net of costs│
        │   • no regime regression        │
        │   • calibration within tolerance│
        │   • passes stress battery       │
        │   • beats Baseline 8 ablation   │
        │   • statistically significant   │
        │   • reproducibility manifest OK │
        └───────────────┬────────────────┘
                        │
             ┌──────────┴──────────┐
            PASS                  FAIL
             │                     │
    promote → shadow          keep incumbent
    archive previous          log reason in full
    registry updated          registry updated
```

Two properties matter more than the criteria themselves.

**Failure is normal and must not be treated as an error.** Most nights should reject. A gate that passes every candidate is not a gate — it is blind retraining wearing a gate's clothes, which is precisely the failure mode this design exists to prevent.

**Rejection reasons are recorded in detail and surfaced in the Control Room**, not buried in a log file. "Rejected: −0.4% net of costs vs incumbent; calibration drifted in high-volatility regime" is diagnostic information. A silent rejection teaches nothing.

The Gate is the only writer to the model registry ([01-architecture.md](01-architecture.md)).

**On failure — DEGRADE toward safety.** Any error inside the gate itself resolves to *keep the incumbent*. The safe default is always the model that has already been proven.

---

## Step 11 — Shortlist

**In:** live model, today's features. **Out:** ranked, filtered, budget-sized candidate list.

Runs inference across the universe with whichever model is now live, then applies the filters from [04-model-brain.md](04-model-brain.md): tradeability, universe validity, confidence floor, sector diversification, and the Layer 2 budget cut.

Every prediction generated here is written to `predictions` with its model version and regime, so that step 4 can grade it when its horizon matures. **This is the write that closes the loop** — tonight's shortlist becomes the material the system learns from in a few days' time.

**On failure — HARD FAIL.** No shortlist, no brief. There is nothing to send.

---

## Step 12 — Investigate

**In:** shortlist, news, F&O, trade history. **Out:** structured verdicts.

Layer 2 ([04-model-brain.md](04-model-brain.md)) investigates each candidate via Ollama Cloud, in rank order, within the adaptive free-tier budget.

**On failure or exhaustion — DEGRADE VISIBLY.** Uninvestigated candidates receive `inconclusive` verdicts and are explicitly flagged as uninvestigated in the brief. Silent truncation is prohibited: a candidate that was never examined must never be presented as one that passed examination.

---

## Step 13 — Synthesize

**In:** investigated shortlist, grades, regime summary, gate outcome, degradation flags. **Out:** brief text and review text.

Layer 3 ([04-model-brain.md](04-model-brain.md)) composes both documents via one to two Claude Code CLI invocations.

**On failure — DEGRADE.** Fall back to a templated brief assembled directly from structured output, clearly marked as a fallback. Layer 3 must verify it received a real response; the Windows invocation hazards documented in [04-model-brain.md](04-model-brain.md) fail silently and would otherwise produce an empty brief that looks like a quiet night.

---

## Step 14 — Deliver

**In:** brief text, review text. **Out:** Telegram messages.

Sends the evening review after processing, and the morning brief on schedule before market open.

**On failure — DEGRADE + RETAIN.** Retry with backoff; if delivery still fails, store both documents and surface them in the Control Room on next launch, marked undelivered. A brief that was generated but never delivered is a recoverable situation, provided the user is told.

---

## Step 15 — Record heartbeat

**In:** everything that happened. **Out:** `job_runs` rows.

Writes one record per pipeline run and one per step: timings, status, rows processed, error detail.

**This step always runs**, including after a hard failure — especially after a hard failure. The heartbeat is how the user answers *"did last night actually work?"*, and how missed-cycle detection works on next launch ([01-architecture.md](01-architecture.md)).

Four statuses, deliberately distinct:

| Status | Meaning |
|---|---|
| `completed` | Ran to the end, whether or not the gate promoted anything |
| `failed` | A hard-fail step aborted the run |
| `aborted` | The user pressed emergency stop |
| `interrupted` | The process died — app closed, crash, power loss |

"The gate rejected a candidate" and "the app closed mid-retrain" must never look alike the next morning.

---

## Idempotency

The pipeline is safely re-runnable. A crash at step 9 followed by a fresh run must not double-count candles, duplicate trades, re-grade already-graded predictions, or leave orphaned registry rows.

Requirements:

- **Ingestion upserts by natural key** — instrument, interval, timestamp. Never blind-appends.
- **Trade reconstruction is deterministic** — the same fills produce the same trades, and reconstruction replaces rather than accumulates.
- **Grading is guarded** — an already-graded prediction is skipped, not re-scored.
- **Feature computation is deterministic and replaceable** — recomputing a date overwrites cleanly.
- **The registry is transactional** — a candidate is fully recorded with its manifest, or not recorded at all. No half-written model versions.
- **Steps checkpoint** — a re-run can resume from the last completed step rather than redoing hours of work.

---

## Manual and partial runs

Every step is independently triggerable from the Control Room ([07-control-room.md](07-control-room.md)), because debugging requires running one thing without running everything.

Dependencies are enforced rather than assumed: triggering step 9 when step 8 never produced a candidate fails with an explanatory message instead of doing something undefined.

Manual runs are recorded in the heartbeat log exactly like scheduled runs, and marked as manual — so that a morning's confusing state can be traced to the debugging session that produced it.
