# 01 — Architecture

## Top-level view

```
┌──────────────────────────────────────────────────────────────────┐
│                         THE USER'S LAPTOP                         │
│                                                                    │
│  ┌────────────────────────────┐    ┌───────────────────────────┐ │
│  │   ELECTRON (Node/Chromium)  │    │   PYTHON BACKEND          │ │
│  │                              │    │   (FastAPI, localhost)     │ │
│  │  • Control Room UI           │◄──►│                            │ │
│  │  • Process supervision       │HTTP│  • Ingestion               │ │
│  │  • Scheduler (nightly cron)  │    │  • Feature engine          │ │
│  │  • Secret storage            │    │  • Trade reconstruction    │ │
│  │  • Emergency stop            │    │  • Regime labeling         │ │
│  │                              │    │  • Training + gating       │ │
│  │  NO market logic.            │    │  • Investigator client     │ │
│  │  NO model code.              │    │  • Synthesizer client      │ │
│  └────────────────────────────┘    └────────────┬──────────────┘ │
│                                                    │                │
│                                       ┌────────────▼─────────────┐ │
│                                       │  DuckDB + Parquet         │ │
│                                       │  (local disk, single      │ │
│                                       │   source of truth)         │ │
│                                       └──────────────────────────┘ │
└───────────────────────────────────┬──────────────────────────────┘
                                     │  (network, nightly only)
        ┌────────────────┬───────────┼───────────────┬──────────────┐
        ▼                ▼           ▼               ▼              ▼
   ┌─────────┐    ┌───────────┐ ┌──────────┐  ┌───────────┐  ┌──────────┐
   │ UPSTOX  │    │  OLLAMA    │ │  CLAUDE   │  │ TELEGRAM  │  │  NEWS     │
   │ NSE ARCH│    │  CLOUD     │ │  CODE CLI │  │ BOT API   │  │  SOURCES  │
   │ YFINANCE│    │            │ │           │  │           │  │           │
   │ data +  │    │ Layer 2    │ │ Layer 3   │  │ delivery  │  │ RSS/feeds │
   │ account │    │ investigate│ │ synthesize│  │           │  │           │
   └─────────┘    └───────────┘ └──────────┘  └───────────┘  └──────────┘
```

## The two-process split

StockSense is two processes, and the boundary between them is strict.

### Electron — the shell

Electron owns the user interface, process lifecycle, scheduling, and secrets. It is the Control Room ([07-control-room.md](07-control-room.md)).

Electron contains **no market logic and no model code**. It does not compute a feature, does not call Upstox, does not decide anything about a stock. If a reader of the source finds a moving-average calculation in the Electron tree, that is a bug in the architecture, not a shortcut.

Electron's responsibilities:

- **Renderer process** — the Control Room UI. Displays state, issues commands. Context-isolated, no direct Node access, communicates with main via IPC only.
- **Main process** — spawns and supervises the Python backend as a child process, owns the nightly scheduler, holds credentials in OS-level secure storage, and implements the emergency stop by killing the process tree.

### Python backend — the brain

A FastAPI service bound to `127.0.0.1` on a fixed local port, spawned by Electron's main process at app launch and terminated with it. It is not exposed to the network and it is not a server in the deployment sense — it is a local compute worker that happens to speak HTTP.

Everything real happens here: ingestion, validation, feature engineering, trade reconstruction, regime labeling, model training, gating, investigation calls, synthesis calls, Telegram dispatch, and all reads and writes to the data store.

Python was chosen because that is where the entire quantitative ecosystem lives — LightGBM, Polars/pandas, DuckDB's Python client, the official Upstox SDK. Attempting any of this in Node would mean reimplementing or shelling out to Python anyway, with worse ergonomics.

### Why split at all

A single-process Electron app would need the model stack in Node (it is not there) or embedded Python (fragile packaging). A single-process Python app would need a GUI toolkit the user does not know.

The split lets each side do what it is good at, and it produces a useful property: **the backend is independently runnable**. The entire nightly pipeline can be executed from a terminal with no UI present. That matters for debugging, for headless testing, and for the case where the UI itself is what is broken.

### The communication contract

Electron → Python is HTTP over localhost. Python → Electron is server-sent events over the same connection, for log streaming and job progress.

The contract is narrow and stable:

| Direction | Purpose | Shape |
|---|---|---|
| Electron → Python | Trigger a job (full nightly run, or one step) | `POST /jobs/{name}` |
| Electron → Python | Query state (health, registry, predictions, trades) | `GET /...` |
| Electron → Python | Mutate config (schedule, sandbox/live mode) | `POST /config/...` |
| Python → Electron | Job progress, log lines, step status | SSE stream |

Electron never reaches into the data store directly. All reads go through the backend API, so there is exactly one component that understands the schema. This avoids the failure mode where a UI query and a pipeline query disagree about what a table means.

## Component responsibilities

Each component below has one job, a defined input, and a defined output. Nothing reads from Upstox except Ingestion. Nothing writes to the model registry except the Gate.

| Component | Reads | Writes | Job |
|---|---|---|---|
| **Ingestion** | Upstox API, NSE archives, yfinance | raw candles, F&O snapshots, delivery data, order fills | The only component permitted to call external data sources. Fetch, normalize, reconcile across sources, record provenance. |
| **Validation** | raw ingested data | validation report, quarantine flags | Detect missing candles, stale/renamed/delisted symbols, unapplied corporate actions, cross-source disagreement. |
| **Feature Engine** | candles, F&O, index data | versioned feature rows | Turn raw market data into the numeric rows models train on. |
| **Trade Reconstruction** | order fills | trade ledger | Group raw fills into whole trades with scale-ins, scale-outs, re-entries. |
| **Regime Labeling** | candles, volatility measures | regime labels | Assign trending / sideways / high-volatility per stock per day. |
| **Prediction Grader** | predictions log, actual outcomes | graded predictions | Track A. Score yesterday's predictions on direction, magnitude, timing, calibration. |
| **Decision Grader** | trade ledger, predictions log | decision grades | Track B. Score the user's entries and exits. Never touches training data. |
| **Trainer** | feature rows, regime labels, graded outcomes | candidate models | Fit the regime classifier and per-regime specialists. |
| **Evaluator** | candidate models, held-out history, episode library | scorecard, gate verdicts | **Adversarial peer system.** Owns the simulator, episodes, baselines, and hard gates. Tries to falsify the brain ([10-evaluation.md](10-evaluation.md)). |
| **Gate** | evaluator verdicts, live model | model registry, promotion decision | The only writer to the registry. Executes the promote/reject decision the evaluator's verdicts determine. |
| **Shortlister** | live model, today's features | candidate shortlist | Run inference, rank, cut to the budget the Investigator can afford. |
| **Investigator** | shortlist, news, F&O, trade history | structured verdicts | Layer 2. Calls Ollama Cloud. |
| **Synthesizer** | shortlist + verdicts + grades | brief text | Layer 3. Calls Claude Code CLI. |
| **Notifier** | brief text | — | Send to Telegram. |
| **Job Runner** | all of the above | job-run heartbeat log | Sequence the pipeline, enforce idempotency, record every step. |

Layers 1–3 (Trainer/Shortlister, Investigator, Synthesizer) are specified in detail in [04-model-brain.md](04-model-brain.md).

## Data flow

```
UPSTOX ──┐
NSE ARCH ─┼─► Ingestion ──► Reconcile ──► Validation ──► DuckDB store
YFINANCE ─┘
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              ▼                             ▼                             ▼
      Feature Engine              Trade Reconstruction            Regime Labeling
              │                             │                             │
              │                             ▼                             │
              │                    Decision Grader ──► coaching output    │
              │                    (TRACK B — DEAD END                    │
              │                     for training data)                    │
              │                                                            │
              └──────────────┬─────────────────────────────────────────────┘
                             ▼
                     Prediction Grader
                     (TRACK A)
                             ▼
                          Trainer
                             ▼
                     EVALUATOR (peer system)
                     simulator · episodes ·
                     baselines · hard gates
                             ▼
                          GATE ──► model registry
                             ▼
                      Shortlister
                             ▼
                   Investigator (Ollama Cloud)
                             ▼
                   Synthesizer (Claude CLI)
                             ▼
                        Telegram
```

Note the shape of Track B in that diagram. It terminates. The coaching output flows to the Synthesizer as text for the user, and nowhere else. There is no arrow from Trade Reconstruction to Trainer, and there must never be one — the reasoning is in [00-overview.md](00-overview.md).

## Network dependencies and failure behavior

The application requires network access at four points. None of them are optional, and none of them are silent when they fail.

| Dependency | Used for | When | On failure |
|---|---|---|---|
| **Upstox API** | Candles, F&O, account fills | Every night, step 1 | **Hard fail.** No data means no valid night. Abort the run, mark the cycle failed, surface in Control Room, retry per backoff policy. Never proceed on partial data. |
| **NSE archives** | Delivery data, corporate actions, universe | Every night, step 1 | **Degrade.** Delivery features become null for the affected date rather than fabricated; the gap is recorded and backfilled on a later run. |
| **yfinance** | Cross-source reconciliation | Every night, step 1 | **Degrade.** Reconciliation proceeds with fewer voices; affected fields are flagged as single-sourced. |
| **Ollama Cloud** | Layer 2 investigation | Every night, step 12 | **Degrade.** Investigation is enrichment, not correctness. On quota exhaustion or outage, emit `inconclusive` verdicts for uninvestigated candidates, mark them clearly, and continue. The brief still ships, flagged as un-investigated. |
| **Claude Code CLI** | Layer 3 synthesis | Every night, step 13 | **Degrade.** Fall back to a templated, non-LLM brief built directly from model output and verdicts. Less readable, still correct and still delivered. |
| **Telegram Bot API** | Delivery | Every night, step 14 | **Degrade + retain.** Retry with backoff. If it still fails, the brief is stored and shown in the Control Room on next launch, marked undelivered. |
| **News sources** | Investigator context | Every night, step 12 | **Degrade.** Investigation proceeds on market data alone, verdict notes the missing context. |

The rule behind this table: **the pipeline hard-fails only where proceeding would corrupt state or produce a wrong answer.** Missing data is corrupting. Missing enrichment is not. Everything that degrades must degrade *visibly* — a brief built without investigation must say so, because a silently-degraded brief that looks normal is worse than no brief.

### Offline at scheduled time

If the laptop is asleep, powered off, or behind a captive portal when the schedule fires, that night's cycle does not run. There is no cloud fallback in v1; this is an accepted limitation of choosing a desktop architecture.

The requirement is detection, not prevention. On next launch, the app compares the last successful job-run record against the expected schedule, and if a cycle was missed it says so prominently in the Control Room and offers a catch-up run. What must not happen is the app opening as if nothing were wrong while the user acts on a three-day-old brief.

## The evaluator is a peer system

The Evaluator is not a stage inside the brain. It is a **separate subsystem with its own data access path**, and the separation is structural rather than conventional.

```
┌──────────────────┐                    ┌──────────────────────┐
│   QUANT BRAIN     │   proposes ─────►  │      EVALUATOR        │
│                   │                    │                        │
│  features         │  ◄──── falsifies   │  simulator             │
│  regimes          │                    │  episode library       │
│  training         │                    │  baselines             │
│  inference        │                    │  hard gates            │
│                   │                    │                        │
│  CANNOT read      │                    │  OWNS held-out eras    │
│  held-out data    │                    │  and ground truth      │
└──────────────────┘                    └──────────────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │    GATE      │  executes the verdict
                     └─────────────┘
```

Three rules give the separation teeth:

**The evaluator owns ground truth.** Held-out eras, the episode library, and the final-validation set belong to the evaluator. The brain's training code has no read path to them. This is enforced by module boundaries, not by discipline.

**No shared scoring code.** If the brain and the evaluator computed returns through the same function, a bug there would be invisible to both. Independent implementations mean a disagreement surfaces rather than cancels.

**The evaluator can veto.** Its hard gates are not inputs to a weighted score — any one failing blocks promotion regardless of headline performance ([10-evaluation.md](10-evaluation.md)).

### The production staircase

Promotion is a sequence of checkpoints, each able to send a candidate backward:

```
Quant Brain → Strategy → Backtester → Adversarial Validator
                                              ↓
              Production ← Risk Gate ← Live Shadow ← Paper Market
```

The naive path this replaces, and exists to prevent:

```
LLM → BUY → real money
```

The **gap between consecutive stages is the primary diagnostic**. A candidate showing Sharpe 2.1 in backtest, 1.4 on paper, and 0.6 in shadow has not been unlucky — it has met reality, and the shape of the decay says where the backtest was optimistic.

## Interface boundary: one control surface

The system has exactly **one control surface — the Electron Control Room** ([07-control-room.md](07-control-room.md)). Telegram is **delivery only**.

Telegram carries the morning brief and the evening review outward. It accepts nothing inward: no bot commands, no `/retrain`, no `/rollback`, no configuration, no acknowledgements that change state.

The reasoning is that a second control surface creates split-brain risk. Two places to change a setting means two places that can disagree, two code paths to keep synchronized, and a category of bug where the app's displayed state and the system's actual state diverge because a command arrived through the other door. For a single-user application whose primary requirement is total internal visibility, that trade is strictly bad — the user is never far from the machine that runs the system, and the convenience of triggering a retrain from a phone does not pay for an authority split in the architecture.

A useful side effect: Telegram's inbound surface being closed means a compromised or spoofed chat cannot drive the system. Delivery-only is also the safer posture.

## Storage

DuckDB with Parquet for bulk time-series, on local disk. Rationale and layout in [02-data-layer.md](02-data-layer.md).

The architecturally relevant point: DuckDB is embedded. There is no database server to install, start, supervise, or firewall. Electron spawns one child process (Python), and Python opens a file. That is the entire runtime topology, and it is the main reason this choice beats PostgreSQL for a single-user desktop application.

## Process lifecycle

```
App launch
   ├─ Electron main starts
   ├─ Spawns Python backend as child process
   ├─ Waits for /health to return ready (with timeout)
   ├─ Renderer connects, Control Room paints
   └─ Scheduler arms

Nightly trigger (or manual)
   ├─ Scheduler POSTs /jobs/nightly
   ├─ Backend runs pipeline, streams progress via SSE
   └─ Control Room shows live step status and logs

Emergency stop
   ├─ Electron kills the backend process tree
   ├─ Any spawned subprocess (ollama, claude) dies with it
   └─ Partial run marked aborted in the heartbeat log

App quit
   └─ Backend terminated; a running job is marked interrupted, not completed
```

An interrupted job is never recorded as successful. The heartbeat log distinguishes *completed*, *failed*, *aborted*, and *interrupted*, because "the app closed mid-retrain" and "the retrain rejected the candidate" are very different facts and must not look alike the next morning.

## Security posture

Single-user, single-machine, so the threat model is narrow — but two things still matter.

**Credentials** (Upstox tokens, Telegram bot token) are held in OS-level secure storage via Electron's `safeStorage`, never in plaintext config files, never in the repository. The Python backend receives them from Electron at spawn time rather than reading them from disk itself.

**The backend binds to `127.0.0.1` only.** It is never exposed on `0.0.0.0`. There is no authentication on the local API, and that is acceptable precisely because it is not reachable off-machine — but it means the binding must not be widened without adding auth first.

A note on data egress: Layer 2 sends candidate analysis to Ollama Cloud and Layer 3 sends summarized output to Anthropic via the Claude Code CLI. What leaves the machine is market data and derived signals. The user's broker credentials never leave the machine, and the trade ledger leaves only as aggregated coaching context — never as raw account identifiers.
