# StockSense

A single-user, locally-run quantitative trading brain for the NSE, packaged as an Electron desktop application.

It studies the market and its own mistakes every night, and hands the user a researched plan every morning.

**Status: documentation phase.** No implementation code has been written. The documentation set is the contract implementation will be built against.

## Start here

**[docs/00-overview.md](docs/00-overview.md)** — what this is, what it is not, and why.

## Documentation set

| Document | Covers |
|---|---|
| [00-overview.md](docs/00-overview.md) | Thesis, dual-track learning invariant, scope, non-goals |
| [01-architecture.md](docs/01-architecture.md) | Two-process split, component contracts, network dependencies |
| [02-data-layer.md](docs/02-data-layer.md) | Upstox surfaces, historical depth, DuckDB store, entities |
| [03-feature-engineering.md](docs/03-feature-engineering.md) | Full feature taxonomy and the output contract |
| [04-model-brain.md](docs/04-model-brain.md) | The three layers and their strict interfaces |
| [05-nightly-pipeline.md](docs/05-nightly-pipeline.md) | The 15-step sequence and per-step failure behavior |
| [06-retraining-rigor.md](docs/06-retraining-rigor.md) | Validation, gating, calibration, drift, rollback |
| [07-control-room.md](docs/07-control-room.md) | Electron UI specification |
| [08-operations.md](docs/08-operations.md) | Scheduling, secrets, logs, runbooks |
| [09-open-questions.md](docs/09-open-questions.md) | Unresolved items with defaults and resolution criteria |
| [10-evaluation.md](docs/10-evaluation.md) | The adversarial evaluation system — simulator, episodes, baselines, hard gates |

## The shape of it, briefly

```
Upstox + NSE archives + yfinance
              ↓
        ingest · reconcile · validate · reconstruct trades
              ↓
        grade predictions (Track A) · grade decisions (Track B)
              ↓
        features · regimes · train candidates
              ↓
              ┌─────────────────────────┐
              │  ADVERSARIAL EVALUATOR   │  tries to prove it wrong
              │  simulator · episodes ·  │
              │  baselines · hard gates  │
              └─────────────────────────┘
              ↓
                  ┌──────────┐
                  │   GATE   │   promote only if it genuinely wins
                  └──────────┘
              ↓
        LightGBM shortlist  →  Ollama Cloud investigates
                            →  Claude synthesizes
                            →  Telegram
```

**Two learning tracks, kept strictly separate.** The model learns from the market's outcomes; the user gets coached on their own decisions. Neither is allowed to contaminate the other.

**Nothing reaches real money without earning it:**

```
Backtest → Paper → Live shadow → Small capital → Scaled
```

## What it does not do

No automated order execution — Upstox Algo Trading access exists but is gated off behind a measurable condition. No latency-competitive execution (intraday *horizons* are supported; racing to the exchange is not). No cloud hosting. No second paid LLM account.
