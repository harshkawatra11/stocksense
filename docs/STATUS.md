# Documentation Status

The 11 documents in `docs/` (`00`–`10`) were written **before** any code existed, as a complete architectural specification. Code has since been built, tested, and — critically — **evidence has emerged that contradicts some of the documents' own load-bearing assumptions.**

This file exists because a future reader (including a future session of this assistant) who trusts `docs/` at face value would build the wrong system. Read this first. Then `research/phase0_verdict.md` for what the evidence actually says. Only then `docs/`, filtered through both.

## StockSense v2 build (2026-08-16, plan: statement forensics + agent harness)

A second build phase started after Phase 0/1's monthly-horizon research concluded (GO, gated, haircut-quantified). Per `C:\Users\harsh\.claude\plans\indexed-discovering-summit.md`. **Built and tested (103 tests passing):**

- **Statement forensics ("Kundli"):** `stocksense/statements/` — Zerodha + Upstox tradebook parsers (fuzzy column matching, golden-file tested), FIFO position reconstruction (short sells, partial fills, multi-day carries), a 13-metric behavioral diagnostics catalogue ("doshas": cost drag, disposition effect, revenge trading, overtrading, sizing chaos, martingale escalation, concentration, drawdown, tail dependence, time-of-day edge, expectancy), a 7-scenario counterfactual engine ("what if"), and a report assembler. CLI: `stocksense statement-ingest <file>`, `stocksense kundli`.
- **Exact Indian cost model:** `execution/cost_model.compute_charges` — verified against Zerodha's published charge sheet to the paisa (₹82.68/8.3bps intraday round-trip on ₹100k vs ₹222.48/22.2bps delivery — intraday genuinely cheaper, confirming the retraction in `research/phase0_verdict.md`).
- **Agent bridge:** `stocksense/agent/claude_cli.py` — Windows-safe Claude CLI invocation (`shutil.which`, `--append-system-prompt`), enforcing **Python computes every number, Claude writes every narrative**: facts are serialized as the only permitted source of figures, and a numeric tripwire flags any unattributable number in the response. Every invocation is logged to `agent_runs`. Secret redaction is tested and already caught one real gap (JSON's closing quote broke the original regex).
- **Harness:** `stocksense/harness/` — a minimal job graph (topological order, cycle detection) and runner (resume-after-failure, idempotency-key-based skip-on-rerun, `job_runs` written even on hard failure).
- **Schema:** extended (not replaced) with `statements`, `trades`, `trade_charges`, `positions`, `diagnostics`, `counterfactuals`, `rag_documents`, `rag_chunks`, `agent_runs`, plus a defensive migration adding grading columns to `predictions` (still not populated by a live loop — that's Phase 5 of the new plan, unbuilt).

**A real bug was found and fixed during this build**, the same discipline that caught ADANIENT: `Store.write_trades`/`write_positions`/etc. used `INSERT ... SELECT *`, which binds by column **position**, not name. Once `statement_id` was appended to a DataFrame after parsing (out of DDL order), trade times silently landed in the trade_date column instead of erroring. Fixed by binding all `write_*` methods to explicit column lists; a regression test (`test_cli_statements.py`) guards it end-to-end.

**Not yet built** (remaining plan phases at that point, unstarted): the multi-source NSE/yfinance data spine and point-in-time universe (Phase 1B), the 14-skill suite (Phase 2 — only the agent bridge exists, no `skills/*.md` files yet), the RAG agent (Phase 3), the portfolio optimizer (Phase 4), the autonomous harness loops incl. the reconcile loop that finally writes `predictions` (Phase 5 — CRITICAL-1 from the original audit is therefore still open), the intraday research track (Phase 6), and any UI beyond the CLI (Phase 7).

## The Foreman: self-building harness (2026-08-16, same day, second build phase)

The plan pivoted from "build the remaining phases by hand" to "build a harness that builds them" — see `C:\Users\harsh\.claude\plans\indexed-discovering-summit.md`. All prior phases (data spine, skills, RAG, optimizer, loops, intraday, the Electron desktop app) become the Foreman's **backlog** rather than a manual sequence.

**Built and tested (188 tests passing, up from 103):**

- **CI** — `.github/workflows/verify.yml`, the independent referee. Runs the full suite plus the leakage and determinism suites as separately-named steps on every push/PR, and checks `.env` never entered history. The Foreman may only self-merge on a **green remote** run (via `gh run list`), never its own local test pass.
- **`foreman/policy.py`** — the single most important file in this build. A literal allowlist of protected paths (`evaluation/gate.py`, `walkforward.py`, `cost_model.py`, the leakage/determinism/gate tests, any preregistration doc, and the policy file and CI workflow themselves). Touching one is a routing decision — the goal branches to a human-reviewed PR and halts, never a silent write.
- **`foreman/tools/`** — a closed, typed tool registry (`base.py`) so a hallucinated tool name fails fast in the executor. `code.py` (read/search/write/test/lint) and `git_tools.py` (branch/commit/push/PR/CI-check) are the first two groups; `write_patch` is the sole write path and checks `policy` before touching disk.
- **`foreman/planner.py`** — Claude decomposes a goal into a plan, validated against the tool registry and converted into a `harness.Graph` (reusing the Phase 0 graph/runner rather than a second execution engine).
- **`foreman/executor.py`** — runs the plan, routes on the verifier's verdict: protected path → PR always; unprotected + fully green including remote CI → auto-merge; anything else → `blocked` with a written reason, never retried past `max_attempts`.
- **`foreman/verifier.py`** — the sequential gate chain (protected-paths → local tests → leakage → determinism → remote CI → pre-registered gate for research goals), ordered so expensive remote checks never run on a result that already failed something cheap.
- **`foreman/adversary.py`** — a red-team pass that tries to falsify a result before it's accepted: assertionless tests, swallowed exceptions, tautological assertions, and (for research goals) seed-sensitivity of the headline number. A blocking finding downgrades an otherwise-merged result.
- **`foreman/assess.py`** — reads real signals (test count, which backlog markers exist, recent goal outcomes, protected-violation count) and asks Claude to propose ranked goals with reasons — the self-assessment that makes this "build itself" rather than execute a fixed list.
- **`foreman/budget.py`** — daily invocation cap + a cooperative kill switch. Cadence is on-demand per the user's choice; scheduling is future work, not built now.
- CLI: `stocksense foreman run "<goal>"`, `foreman assess`, `foreman status`, `foreman ledger`.

**Bugs found and fixed by this build's own tests, all real, none cosmetic:**
1. `policy._normalize` used `str.lstrip("./")`, which strips arbitrary leading `.`/`/` *characters*, not a literal prefix — silently ate the dot off `.github/workflows/...` and made the CI-workflow protection pattern never match. The one test written specifically to guard the CI workflow's own protection caught it before anything shipped.
2. `foreman/tools/__init__.py` only re-exported `base`, so importing `stocksense.foreman.tools` without also importing `.code`/`.git_tools` left the registry empty — a planner-only test file failed six ways until the package `__init__` was made to import every tool module explicitly for its registration side effects.
3. `adversary.check_assertionless_tests` checked for the literal substring `"pytest.raises"` inside `ast.dump()` output, which renders attribute access as nested `Attribute(...)` nodes, not dotted text — so it never matched and every legitimate `pytest.raises`-based test was misflagged as assertionless. Fixed with `ast.unparse` to get real source text. Notable because this bug was *in the module whose entire purpose is catching bugs like this*.

**Not yet built:** the loop scheduler (on-demand only for now), a real invocation of `foreman run` against a live goal (only exercised in this session via mocked git/network calls — no real branch, commit, push, or PR has been created yet), and the backlog items themselves (Electron app, reconcile loop, data spine, skills, RAG, optimizer, intraday track) — those are queued as the Foreman's first real work, not done.

### Fixed same day: the two-model split wasn't actually wired

Caught by direct question, not by testing: `AgentRequest` had a `model` field nobody ever set, no `effort` field existed at all, and the planner was asking one undifferentiated agent call to both decompose a goal into steps AND write full file contents inline — not the "Opus plans, Sonnet executes" split the plan document names in its own byline.

Fixed:
- `agent/claude_cli.py` — added `effort` to `AgentRequest`, wired to `claude --effort <level>` (confirmed the flag exists via `claude --help`); both `model` and `effort` now logged into `agent_runs.input_json` for auditability.
- `foreman/planner.py` — decomposition now explicitly runs at `PLANNER_MODEL="opus"`, `PLANNER_EFFORT="low"`. The prompt no longer asks for literal file content in `write_patch` steps — it asks for a `spec` (what the file must do), enforced by instruction, not just convention.
- `foreman/codegen.py` (new) — `generate_file_content` runs at `CODEGEN_MODEL="sonnet"`, `CODEGEN_EFFORT="medium"`, resolving a `spec` into real file contents. Wired into `planner.plan_to_graph`: a `write_patch` step with `spec` (not literal `content`) is resolved through Sonnet at graph-execution time, using upstream step outputs as context — so Opus's plan never contains generated code, only the intent to generate it.
- `foreman/assess.py` — self-assessment's goal proposal also routed through `PLANNER_MODEL`/`PLANNER_EFFORT`, since ranking/prioritization is a judgment call at the same tier as decomposition, not a code-generation task.

7 new tests (`test_agent_model_effort.py`, `test_foreman_codegen.py`, plus 2 added to `test_foreman_planner.py`) assert the flags actually reach the subprocess command line, and that a `write_patch` step with literal `content` skips codegen entirely (only `spec`-only steps trigger the extra call) — 198 tests passing, up from 188.

## CRITICAL-1 closed: the reconcile loop (2026-08-16, same day)

**`predictions` is finally written to.** `stocksense/harness/loops.py` adds `record_predictions` (scores today's live/shadow model, freezes one row per symbol with a `feature_snapshot_hash` so a later reviewer can confirm nothing was recomputed with hindsight) and `grade_matured_predictions` (once a prediction's horizon has actually elapsed in the candle data, grades it against realized relative forward return computed with the **exact same function training uses** — `labels.forward_return` — so grading and training can never silently disagree about what "correct" means).

`build_reconcile_graph` wraps both as a two-node `harness.Graph` (grade before record), idempotency-keyed by calendar date — running `stocksense reconcile` twice in one day is a no-op the second time, the property `docs/05-nightly-pipeline.md` requires of every step. This is the harness's own graph/runner infrastructure (built idle in the earlier Foreman phase) doing real work for the first time.

CLI: `stocksense record-predictions`, `stocksense grade`, `stocksense reconcile` (the graph-wired version of both, recommended). 9 new tests on real synthetic data with a genuinely trained LightGBM model (not mocked) — 207 tests passing, up from 198.

**What this does not yet do:** grading does not feed back into the gate — a graded prediction's outcome doesn't currently influence whether a model stays live or gets rolled back. That's the natural next extension, not built here. Confidence/calibration tracking (Brier score, reliability curves per `docs/06`) also remains unbuilt; `predictions.confidence` is written as `NULL` for now since the LightGBM regressor doesn't yet produce a calibrated interval.

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
