# Documentation Status

The 11 documents in `docs/` (`00`–`10`) were written **before** any code existed, as a complete architectural specification. Code has since been built, tested, and — critically — **evidence has emerged that contradicts some of the documents' own load-bearing assumptions.**

This file exists because a future reader (including a future session of this assistant) who trusts `docs/` at face value would build the wrong system. Read this first. Then `research/phase0_verdict.md` for what the evidence actually says. Only then `docs/`, filtered through both.

## The two contradictions that matter most

**1. Daily cadence → monthly cadence.** The docs (`04`, `05`, `06`) are written around a nightly retrain / daily-horizon prediction cycle. Phase 0 measured this directly: a 1-day holding horizon cannot clear realistic transaction costs, confirmed independently four times (v1's own historical output, and three re-runs in this codebase on different data). The horizon that survives costs is **~20 trading bars (roughly monthly)**. Every daily-cadence detail in `04`–`08` should be read as *the wrong cadence*, not an implementation detail to fill in later.

**2. The three-layer LLM brain → unbuilt, unproven, and possibly unjustified.** `04-model-brain.md` centers on LightGBM → Ollama investigation → Claude synthesis, with an explicit ablation (Baseline 8: LightGBM alone) specified as the test for whether the LLM layers earn their cost. **That ablation has never been run**, because the LLM layers don't exist yet. What's built is Baseline 8 itself — a LightGBM-only cross-sectional ranker — with nothing to compare it against. The docs describe the LLM layers as the differentiator; the only thing proven to work so far is the layer the docs treat as a baseline.

## Per-document status

| Doc | Status | What's actually true |
|---|---|---|
| `00-overview.md` | **PARTIAL** | The dual-track (market-mistakes vs trader-mistakes) invariant is sound design and still the target. The core mechanism it describes — the reconcile/learning loop — **has zero implementation** (see below). Horizon-agnostic framing already correctly updated and matches evidence. |
| `01-architecture.md` | **ASPIRATIONAL** | Describes an Electron + Python two-process system with the evaluator as a peer service. Built: a single-process Python CLI. No Electron, no IPC, no separate evaluator process. |
| `02-data-layer.md` | **PARTIAL, ONE CLAIM ACTIVELY WRONG** | Multi-source reconciliation (Upstox/NSE/yfinance) with per-field provenance: not built — only yfinance is wired, with no provenance tracking. Point-in-time universe: explicitly **not** met — this is the current #1 blocker (see `research/phase0_verdict.md`, survivorship section). The adjustment-validation discipline the doc calls for is real and built (`data/validate.py`), just narrower than specified. |
| `03-feature-engineering.md` | **PARTIAL** | Price/candlestick/volume/market-context categories: built. F&O and news/events categories: explicitly out of scope, correctly marked as such in the doc itself. Horizon-agnostic label design: built and matches evidence. |
| `04-model-brain.md` | **MOSTLY ASPIRATIONAL** | Layer 1 (LightGBM) exists as a single flat cross-sectional ranker — **not** the regime-gated specialist architecture the doc describes (no regime classifier, no per-regime models). Layers 2 (Ollama) and 3 (Claude) do not exist. The Baseline 8 ablation this doc says must be run before trusting the LLM layers has literally never happened. |
| `05-nightly-pipeline.md` | **ASPIRATIONAL** | The 15-step scheduled nightly sequence, Telegram delivery, automatic shortlisting: none of this runs. What exists: manually-invoked CLI commands (`train-candidate`, `predict`, `registry`) that a human runs on demand. |
| `06-retraining-rigor.md` | **PARTIAL, WITH A FOUND METHODOLOGY BUG** | Purged/embargoed walk-forward: built, and an actual over-conservative embargo bug was found and fixed during audit (see `research/phase0_verdict.md`). Cost-aware backtesting, best-trade-removal, parameter-perturbation stress tests: built and run. Calibration tracking, drift detection, the full stress battery, rollback CLI: **not** built. Shadow lifecycle state exists in the registry schema but nothing populates or reads it as an actual trial. |
| `07-control-room.md` | **0% BUILT** | Entirely aspirational. No UI of any kind exists. |
| `08-operations.md` | **ASPIRATIONAL** | Describes operating a scheduled service. Nothing runs as a service; there is no scheduler, no secrets management beyond `.env`, no log rotation. |
| `09-open-questions.md` | **PARTIALLY RESOLVED BY EVIDENCE** | OQ-1 (Ollama free tier) and historical-depth questions: resolved as documented. **OQ-4 (which horizon carries edge) has since been answered empirically: ~20 bars, monthly** — this should be treated as closed, not open, though the docs file itself has not yet been edited to reflect it. |
| `10-evaluation.md` | **PARTIAL, ONE PIECE MISSING IS THE PRODUCT ITSELF** | Walk-forward validation, best-trade-removal, parameter perturbation: built and run for real, not simulated. Monte Carlo: attempted, found methodologically flawed (terminal return is order-invariant under the reshuffle method used — needs redoing, drawdown-focused). Episode library, baseline gauntlet (10 baselines — only 1 exists), Quant IQ scorecard, regime-stratified evaluation, reasoning evaluation: **not built**. **The immutable prediction ledger — arguably the single most important artifact this document specifies — does not exist.** The `predictions` table is created in the schema and never written to. |

## What "not built" means for anyone picking this up

If you are extending this codebase: **do not assume any document above describes working code.** Check `src/stocksense/` directly, and cross-reference `research/phase0_verdict.md` for what the evidence actually supports before trusting a documented design decision — several were made before evidence existed to test them, and at least two (daily cadence, LLM-centrality) are now known to be wrong or unproven respectively.

The documents are not being deleted or rewritten wholesale, because the reasoning in them — the dual-track invariant, the adversarial-evaluator philosophy, the gate-not-average principle, the point-in-time discipline — is sound and still the target. They describe *intent*, validated in places and contradicted in others. Treat them accordingly.
