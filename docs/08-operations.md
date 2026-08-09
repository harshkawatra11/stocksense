# 08 — Operations

## Scheduling

### Who owns the schedule

The **Electron main process** owns it, not Windows Task Scheduler.

Both were considered. Task Scheduler has one real advantage — it can wake a sleeping machine — but it puts the schedule somewhere the application cannot see, edit, or explain. A user asking "when does this run?" would have to leave the app to find out, and a schedule that drifted out of sync with the app's assumptions would be invisible until a morning went wrong.

An in-app scheduler keeps configuration, execution, and visibility in one place ([07-control-room.md](07-control-room.md)). The cost is that the app must be running, which is already required for the backend to exist at all.

The accepted consequence: **if the app is closed or the machine is asleep, the cycle does not run.** No cloud fallback in v1. Mitigation is detection, not prevention — see below.

### Timing

```
15:30  Market close
   │
   ├─ (settle window — let the broker finalize the day's data)
   │
20:00  EVENING CYCLE  ── steps 1-10 ─────────────────┐
   │   ingest · validate · reconstruct · grade ×2     │
   │   features · regimes · train · backtest · GATE   │
   │                                                   │
   │   → evening review delivered                      │
   │                                                   │
   ├─ (Layer 2 investigation window — hours available) │
   │                                                   │
07:00  MORNING CYCLE  ── steps 11-15 ────────────────┘
   │   shortlist · investigate · synthesize · deliver
   │
08:15  Morning brief delivered
   │
09:15  Market open
```

Splitting the pipeline at step 10 is deliberate. The heavy, deterministic work — ingestion through gating — runs in the evening when its inputs are fresh and its failures leave a full night for recovery. The shortlist runs in the morning so it reflects the promoted model and the most recent data.

The investigation window between them exists because Layer 2's adaptive budget benefits from time ([04-model-brain.md](04-model-brain.md)). Free-tier session limits reset on a multi-hour cycle, so spreading investigation across the night can reach more candidates than compressing it would.

Both times are configurable. These are defaults, not constants.

### Missed cycles

On launch, the app compares the last successful `job_runs` record against the expected schedule. A gap means a missed cycle, which is surfaced prominently on the Pipeline screen with a catch-up option.

What must never happen: the app opens looking normal while the Today screen shows a three-day-old brief that appears current.

A catch-up run ingests the missed sessions in order. Grading, training, and gating proceed normally — the pipeline is idempotent and date-driven ([05-nightly-pipeline.md](05-nightly-pipeline.md)), so catching up is a matter of running the sequence per missed date rather than a special code path.

---

## Secrets

| Secret | Storage | Notes |
|---|---|---|
| Upstox API key / secret | `safeStorage` (OS-backed) | Per environment — Sandbox and Live credentials are distinct entries |
| Upstox access token | `safeStorage`, with expiry tracked | Refreshed via the SDK's auth flow; expiry surfaced before it bites |
| Telegram bot token | `safeStorage` | — |
| Telegram chat ID | Config (not secret) | — |
| Ollama Cloud auth | Managed by the `ollama` CLI itself | StockSense does not store or handle it |
| Claude auth | Managed by the Claude Code CLI itself | Subscription session; StockSense does not store or handle it |

Rules:

- **Nothing secret in the repository.** `.env`, credential files, and token caches are gitignored. This has bitten this project before: a prior commit history includes an ignore rule added specifically because a locally-installed agent skill directory could carry credentials.
- **Nothing secret in plaintext on disk.** OS-level secure storage only.
- **The backend receives credentials from Electron at spawn time**, rather than reading them from disk itself. One component owns secret retrieval.
- **Nothing secret in logs.** Log redaction is a requirement, not a convention — tokens appearing in an SSE stream would put them in the renderer and in any exported log file.
- **Token expiry is surfaced before it fails.** An Upstox token that expires overnight produces a hard failure at step 1; a warning the previous evening prevents it.

---

## Logs and files

```
%APPDATA%/StockSense/
  config.json           ← non-secret configuration
  logs/
    app.log             ← Electron main
    backend.log         ← Python backend
    runs/
      2026-08-08-evening.log
      2026-08-08-morning.log
  data/
    stocksense.duckdb
    parquet/…
    models/…
  briefs/
    2026-08-08-morning.md   ← retained even when delivery fails
```

Per-run logs are separate files, named by date and cycle, so investigating a specific night does not mean scrolling a single ever-growing file. Rotation and retention are configurable, with per-run logs retained long enough to investigate a problem noticed a week later.

The data directory is the valuable one. Everything else can be rebuilt from source; twenty-six years of ingested history and the full model registry cannot be rebuilt quickly.

---

## Backup

There is no cloud component, which means there is no backup unless the user makes one.

| Asset | Rebuildable? | Priority |
|---|---|---|
| `stocksense.duckdb` | No — contains trade ledger, predictions, registry, heartbeat | **Critical** |
| Episode library + holdout definitions | No — attempt history and locked eras cannot be reconstructed | **Critical** |
| `parquet/` | Yes, by re-running backfill (slow, rate-limited) | High |
| `models/` | Only if the manifest and data survive | High |
| `briefs/` | No | Medium |
| `config.json` | Trivially | Low |

The episode library ranks alongside the database for a non-obvious reason: rebuilding the *episodes* is mechanical, but the **record of how many times each holdout has been tested** is not. Lose that and every subsequent evaluation silently loses its overfitting guard (OQ-11).

The trade ledger and predictions log are irreplaceable — they are the accumulated record the entire learning claim rests on. A recommended practice is a periodic copy of the data directory to external or synced storage, taken while the app is closed so the DuckDB file is not mid-write.

---

## Runbook

### "Did last night run?"

Pipeline screen. The run timeline shows every step with status and duration. `job_runs` distinguishes `completed`, `failed`, `aborted`, and `interrupted` — check which, because the remedies differ.

### Step 1 failed — Upstox

Most common causes, in order of likelihood:

1. **Expired access token** — re-authenticate in Settings. The most frequent cause by a wide margin.
2. **Rate limited** — check whether a backfill was running concurrently with the nightly cycle. They should not overlap.
3. **Upstox outage or maintenance** — verify independently; retry.
4. **Network unavailable at run time** — check for a missed-cycle marker instead.

The incumbent model is untouched in all cases. Re-run step 1 from the Pipeline screen once resolved.

### Step 2 quarantined instruments

Open the Data screen's validation results. Common causes: a corporate action not yet reflected, a symbol change producing an apparent gap, or a genuinely halted instrument.

If the universe is stale, refresh it. If the corporate action is missing, ingest it and re-run validation. Quarantine is not an error state to dismiss — it is the system declining to train on data it cannot vouch for.

### Cross-source disagreement quarantined a field

Open the Data screen's validation results and check which sources disagreed and by how much ([02-data-layer.md](02-data-layer.md)).

Usual causes, in order: an unapplied corporate action (the most common by far — one source has adjusted, another has not), a symbol collision after a rename, a stale universe entry, or yfinance data quality on a thinly traded name.

Resolve the underlying cause and re-run validation. Do **not** widen the tolerance to make the warning go away — the tolerance is calibrated against logged history (OQ-9), and loosening it to silence a real discrepancy is how corrupt data enters the store.

### The gate rejected again

**This is normal.** Per [06-retraining-rigor.md](06-retraining-rigor.md), rejection is the expected outcome and a gate that rarely rejects has stopped gating.

Read the recorded reason on the Models screen. A rejection citing a specific regime regression or calibration drift is the system working. Investigate only if rejections change character — for instance, if every candidate suddenly fails on a metric that used to pass, which points at data rather than at the model.

### The evaluation scorecard looks too good

Treat this as a symptom, not a result. In backtesting, spectacular numbers are far more often a bug than an edge.

Check in this order:

1. **Fidelity tier** — did the edge appear only at T3 (modeled microstructure)? If it cannot be seen at T1 or T2, it is a property of the slippage model, not the market ([10-evaluation.md](10-evaluation.md)).
2. **Leakage** — run the point-in-time correctness tests. Lookahead is the single most common cause of beautiful results.
3. **Baseline 8** — does it beat LightGBM-only by an implausible margin? A large gap attributable to the LLM layers deserves suspicion before celebration.
4. **Attempt count** — how many candidates have been tested against this holdout? A high count means the result is partly fitted to the evaluator (OQ-11).
5. **Universe** — is the test running on today's instrument list rather than a point-in-time reconstruction? That is survivorship bias, and it flatters everything.
6. **Costs** — confirm the full stack was applied, not just brokerage.

### Predictions look wrong

1. Models screen → calibration view. Is the model confidently wrong, or just wrong? Those need different responses.
2. Check drift indicators. Feature drift points at data or at a changed market; concept drift points at a stale model.
3. Check the regime classifier's confusion matrix — routing errors propagate to every downstream prediction and can look like specialist failure.
4. Data screen → validate the inputs. Bad features produce bad predictions with perfect internal consistency.
5. If the live model is genuinely misbehaving, **roll back** ([07-control-room.md](07-control-room.md)). Rollback is a normal operation, not an admission of failure.

### Brief arrived without investigation

Layer 2 was unreachable or out of budget. Check the Layer 2 panel in Settings for measured throughput and consumption.

If quota exhaustion is chronic rather than occasional, apply the documented options in order ([04-model-brain.md](04-model-brain.md)): reduce shortlist size, reduce per-candidate prompt size, move to a smaller cloud model. Upgrading the tier breaks a stated non-goal and should be a deliberate decision, not a reflex.

### Brief arrived as a template

Layer 3 failed. Before assuming an outage, check the two Windows invocation hazards from [04-model-brain.md](04-model-brain.md) — an unresolved `claude.CMD` shim path and an invalid system-prompt flag both fail silently and have cost this project time before.

Verify the CLI works interactively from a terminal. If it does, the fault is in the invocation, not the service.

### Telegram silent

The brief is retained and shown in the Control Room marked undelivered. Verify the bot token and chat ID in Settings, then retry delivery. The content is not lost.

### Something is visibly wrong right now

Emergency stop. The run is marked `aborted`, the incumbent model is untouched, and no state is left half-written — the registry is transactional and ingestion upserts ([05-nightly-pipeline.md](05-nightly-pipeline.md)).

---

## Development and QA

The user is also the developer and the tester, so the boundary between operating and developing is thin by design — but two separations are not thin, and must not be blurred.

**Sandbox first.** All trading-logic development and QA runs against the Upstox Sandbox surface. Sandbox is the default on a fresh install, and the mode is visible on every screen ([07-control-room.md](07-control-room.md)).

**The backend is independently runnable.** The full pipeline executes from a terminal with no Electron present ([01-architecture.md](01-architecture.md)). This matters when the UI itself is what is broken, and it makes headless testing straightforward.

Testing priorities, in rough order of consequence:

1. **Point-in-time correctness** — leakage tests on the feature engine. This is the highest-value test surface in the entire codebase, because leakage produces beautiful backtests and worthless predictions, and nothing else in the system will catch it.
2. **Idempotency** — running any step twice produces the same state as running it once.
3. **Gate logic** — a candidate that should fail must fail. Test with deliberately worse candidates.
4. **Trade reconstruction** — synthetic fill sequences with scale-ins, scale-outs, re-entries, and same-day reversals.
5. **Cost model** — verified against actual contract notes, not assumed.
6. **Failure paths** — every row of the failure table in [06-retraining-rigor.md](06-retraining-rigor.md) should be exercisable deliberately.

---

## Upgrades

Two upgrade paths carry real risk and both are Control Room operations with explicit warnings ([07-control-room.md](07-control-room.md)):

**Feature schema changes** invalidate every existing model. A model trained against schema v3 and served schema v4 features is receiving inputs whose meaning has silently changed. After a schema bump, a full feature rebuild is required and the registry must reflect that existing models are stale.

**Prediction horizon changes** break comparability. Models trained on different horizons cannot be compared by the gate, and predictions graded against different horizons cannot be pooled for calibration. The horizon is recorded per prediction precisely so that grading always compares like with like.

Neither should happen casually, and neither should happen implicitly.
