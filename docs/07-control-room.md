# 07 — The Control Room

## Why it is a control room, not a dashboard

A dashboard displays. A control room operates.

The user is the developer, the QA tester, and the trader ([00-overview.md](00-overview.md)). At 11pm they need to know why a candidate was rejected and re-run step 9 in isolation. At 9am they need to read a brief and decide whether to act. Same person, same application, and the application must be good at both.

The consequence: **everything the system does must be observable and controllable from here.** If a state exists only in a log file, or an action is only possible by editing a config and restarting, the Control Room has failed at its job.

Two hard rules shape the whole UI:

- **No hidden state.** If it affects behavior, it is visible.
- **No unreachable action.** If the system can do it, the user can trigger it.

---

## Screen map

```
┌───────────────────────────────────────────────────────────┐
│  StockSense          ● SANDBOX          Last run: ✓ 02:14  │  ← always visible
├──────────┬────────────────────────────────────────────────┤
│ Today    │                                                  │
│ Pipeline │                                                  │
│ Models   │              (active screen)                     │
│ Trades   │                                                  │
│ Data     │                                                  │
│ Logs     │                                                  │
│ Settings │                                                  │
├──────────┴────────────────────────────────────────────────┤
│  [ ■ EMERGENCY STOP ]                    backend ● healthy  │  ← always visible
└───────────────────────────────────────────────────────────┘
```

Three things are visible on every screen without exception: **the Sandbox/Live indicator**, **the last run's status**, and **the emergency stop**. Each is there because the cost of it being one click away is unacceptable.

---

## 1. Today

The morning screen. Opens here by default.

- **The brief** — the same content Telegram delivered, rendered properly. Each candidate shows direction, calibrated probability, expected move, regime, setup rationale, and its Layer 2 verdict.
- **Uninvestigated markers** — candidates Layer 2 could not reach are visually distinct, never blended in with investigated ones ([04-model-brain.md](04-model-brain.md)).
- **Degradation banner** — if any step degraded last night, it says so here, in plain language, above the content. A brief produced without investigation or without Claude synthesis must announce that fact before the user acts on it.
- **Yesterday's evening review** — prediction grades, decision grades, behavioral notes.
- **Undelivered notice** — if Telegram failed, the brief is here and marked undelivered ([05-nightly-pipeline.md](05-nightly-pipeline.md)).

The design constraint: a user who reads only this screen must never be misled about how much the system actually knows.

---

## 2. Pipeline

The operational screen: what ran, what it did, and how to run it again.

### Service health

Live status per component — Ingestion, Validation, Trade Reconstruction, Prediction Grader, Decision Grader, Feature Engine, Regime Labeling, Trainer, Gate, Shortlister, Investigator, Synthesizer, Notifier.

Each shows state (`idle` / `running` / `error`), last run time, last duration, and last error if any.

### Run timeline

The 15 steps of the last run ([05-nightly-pipeline.md](05-nightly-pipeline.md)) as a vertical timeline with per-step status, duration, and rows processed. Expanding a step shows its detail — for step 10, the full gate reasoning; for step 12, the budget consumed and how many candidates were reached.

### Manual triggers

A button per step, plus "run full nightly cycle now."

Dependencies are enforced, not assumed. Triggering the backtest when no candidate exists fails with an explanation rather than doing something undefined. Manual runs are recorded in the heartbeat log and marked manual, so a confusing state next morning can be traced to the debugging session that caused it.

### Missed-cycle detection

If the expected schedule did not produce a run — laptop asleep, off, captive portal ([01-architecture.md](01-architecture.md)) — this screen says so prominently and offers a catch-up run. The app must never open looking normal while the user acts on a stale brief.

### Scheduler configuration

When the nightly cycle runs, owned by the Electron main process so the schedule is visible here rather than buried in Windows Task Scheduler ([08-operations.md](08-operations.md)).

---

## 3. Models

Where the learning claim is either substantiated or exposed.

### Registry

Every model version ever trained: ID, type (regime classifier or named specialist), training window, feature schema version, lifecycle state (`candidate` / `shadow` / `live` / `archived` / `rolled_back`), and headline metrics.

Selecting a version shows its full record — complete metrics including per-regime and per-window breakdowns, calibration curve, stress replay results, the full reproducibility manifest, and the gate decision with its reason.

### Gate history

Every promotion decision, with reasons. **Rejections are as prominent as promotions** — arguably more useful, since "rejected: −0.4% net of costs; calibration drifted in high-volatility regime" is diagnostic information that a rejection buried in a log file would waste.

**Promotion rate is displayed as a metric.** Per [06-retraining-rigor.md](06-retraining-rigor.md), a gate that rarely rejects is not gating. If this number climbs toward 100%, the gate has quietly stopped working and the user needs to see that without being told to go looking.

### Calibration and accuracy

Predicted probability versus observed frequency, bucketed, per regime. Rolling accuracy and Brier score over time. Drift indicators.

This is where a model that is confidently wrong becomes visible — the failure mode accuracy alone cannot reveal ([06-retraining-rigor.md](06-retraining-rigor.md)).

### Rollback

One click to restore any archived version, with confirmation showing what is being replaced and what is being restored. Atomic, recorded with timestamp and optional reason. The rolled-back model is marked and will not be silently re-promoted later.

### Shadow tracking

Models in shadow, their accumulated graded track record, and their progress toward the trial-period threshold that would make them user-facing.

---

## 4. Trades

The trade journal — Track B's surface.

- **Reconstructed trades**, not raw fills: the full entry/exit sequence as one object, with realized P&L, duration, MFE, and MAE.
- **Decision grade per trade**, with the reasoning shown in both directions — exits that were early relative to the model, and exits that correctly avoided a subsequent loss ([05-nightly-pipeline.md](05-nightly-pipeline.md)).
- **Trade detail view**: the price path with entries and exits marked, MFE/MAE bands, what the model predicted at each decision point, and what happened afterward.
- **Behavioral patterns**: early-exit tendency, post-loss behavior, re-entry expectancy by attempt number, sizing drift after winning streaks, time-of-day performance.

The screen must make the distinction the whole track exists for legible at a glance: **profitable and well-decided are different axes.** A profitable trade with a poor decision grade and a losing trade with a sound one should both be immediately visible as such.

---

## 5. Data

Store health and universe correctness.

- **Coverage** — what history exists, per interval, with gaps visualized. Daily from 2000, intraday from 2022 ([02-data-layer.md](02-data-layer.md)).
- **Universe** — active instruments, recent additions and deactivations, unnamed or suspicious entries. This screen exists because a stale universe silently poisons training, and the prior incarnation of this project carried thousands of delisted tickers before anyone noticed.
- **Validation results** — the latest report, quarantined instruments, and why each was quarantined.
- **Corporate actions** — recent actions and confirmation that adjustments were applied.
- **Backfill control** — trigger, monitor, pause, and resume the historical backfill, with per-instrument progress. Deliberately separate from nightly ingestion so it can never be triggered by accident.
- **Feature schema** — current version, and the control to trigger a full rebuild. This action carries an explicit warning: after a schema change, every existing model is stale by definition ([03-feature-engineering.md](03-feature-engineering.md)).

---

## 6. Logs

Live tail of the current or most recent run, streamed from the backend over SSE ([01-architecture.md](01-architecture.md)).

Filterable by component and level, searchable, with the ability to jump to a specific step's log segment from the Pipeline screen. Exportable, so a problem can be examined outside the app.

The requirement is that debugging never demands leaving the application to open a file — if the answer is in a log, it is reachable here.

---

## 7. Settings

### Credentials

Upstox tokens and the Telegram bot token, held in OS-level secure storage via Electron's `safeStorage` — never plaintext config, never the repository ([01-architecture.md](01-architecture.md)). The UI shows connection status per credential and offers a test action, never the stored value.

### Sandbox / Live

The most consequential control in the application, and treated accordingly.

- **Sandbox is the default** on a fresh install
- The current mode is visible on **every screen**, not just this one
- Switching to Live requires explicit confirmation and is recorded in the heartbeat log
- Live mode is **visually unmistakable** — a persistent, high-contrast indicator that cannot be mistaken for Sandbox at a glance

### Algo Trading gate

Read-mostly. Displays each gate condition from [02-data-layer.md](02-data-layer.md) with its current measured status — sessions accumulated, calibration standing, regime coverage, net-of-cost edge.

**The gate's status is computed, not asserted.** The user cannot tick these boxes manually. When all conditions are met, the per-session opt-in becomes available; it expires rather than persisting, so leaving execution enabled requires a deliberate act each time rather than a forgotten setting.

Until then, this screen's honest job is to show exactly how far the system is from earning the capability.

### Layer 2 budget

Ollama Cloud tier, measured throughput from recent runs, current shortlist sizing, and the manual override. Displays what the adaptive budget actually observed rather than what it assumed ([04-model-brain.md](04-model-brain.md)).

### Prediction horizon

The configurable horizon, with a warning that changing it invalidates comparability across models trained on different horizons.

---

## Emergency stop

Present on every screen, styled as the destructive action it is.

Kills the backend process tree immediately, taking any spawned `ollama` or `claude` subprocess with it. The in-flight run is marked `aborted` in the heartbeat log — distinct from `failed` and from `interrupted`, because "the user stopped it" and "it crashed" are different facts ([05-nightly-pipeline.md](05-nightly-pipeline.md)).

It exists because unattended automation occasionally does something visibly wrong at an inconvenient hour, and the correct response must be one click rather than hunting for a process ID.

---

## Design principles

**Honesty over polish.** A degraded brief must look degraded. An uninvestigated candidate must look uninvestigated. A missed cycle must be impossible to miss. The UI's job is to convey the system's real epistemic state, not to look confident.

**Diagnosis over summary.** Every aggregate number is drillable to the detail that produced it. A rejected candidate that shows only "rejected" wastes the most useful information the night generated.

**Safety through friction, precisely placed.** Live mode, rollback, backfill, and feature-schema rebuild all require confirmation. Nothing else does — friction everywhere is friction nowhere, and a user who reflexively dismisses confirmations has lost the protection they were meant to provide.

**No dead ends.** Every error message names what failed and offers the next action — retry, view logs, roll back, open the relevant screen.
