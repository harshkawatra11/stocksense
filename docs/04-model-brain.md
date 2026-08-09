# 04 — The Model Brain

## Three layers, three different jobs

```
   feature rows (03)
          │
          ▼
┌───────────────────────────────────────────────┐
│  LAYER 1 — QUANT ENGINE          LightGBM      │
│  local · CPU · seconds-to-minutes · explainable│
│                                                 │
│  Answers: "which of ~2000 instruments are      │
│            statistically worth looking at?"     │
│                                                 │
│  OUT: ranked shortlist + probability + expected │
│       move + regime.  NOT a trade decision.     │
└──────────────────────┬────────────────────────┘
                        │  ~5–30 candidates
                        ▼
┌───────────────────────────────────────────────┐
│  LAYER 2 — INVESTIGATOR       Ollama Cloud     │
│  remote inference · slow OK · budget-bounded   │
│                                                 │
│  Answers: "does anything outside the numbers   │
│            confirm or contradict this?"         │
│                                                 │
│  OUT: structured verdict per candidate          │
│       corroborated / contradicted / inconclusive│
└──────────────────────┬────────────────────────┘
                        │  investigated shortlist
                        ▼
┌───────────────────────────────────────────────┐
│  LAYER 3 — SYNTHESIZER      Claude Code CLI    │
│  subscription auth · 1–2 calls per night        │
│                                                 │
│  Answers: "what should the human actually be    │
│            told, and how?"                      │
│                                                 │
│  OUT: morning brief + evening review, in prose  │
└───────────────────────────────────────────────┘
```

The layers are strictly ordered and strictly separated. Each has a defined input, a defined output, and a job the others do not do. **No layer may do another layer's job** — Layer 1 must not attempt reasoning, Layer 3 must not attempt statistics.

The reason for three layers rather than one: they are good at genuinely different things. Statistics over thousands of instruments is a tree-model job and a terrible LLM job. Reading a resignation announcement and understanding its implication is an LLM job and an impossible tree-model job. Deciding what a human needs to hear at 9am is a different LLM job again, and the most expensive one.

### Layers 2 and 3 must prove they earn their place

That argument is plausible, and plausible is not sufficient. **Baseline 8 in the evaluation gauntlet is Layer 1 alone, with no LLM layers at all** ([10-evaluation.md](10-evaluation.md)), and it runs on every evaluation.

If the full stack cannot beat LightGBM-only on net-of-cost performance, then Layers 2 and 3 are expensive decoration and the honest conclusion is to ship the simpler system. This is the single most uncomfortable number the evaluator produces, which is exactly why it is reported prominently rather than buried.

---

## Layer 1 — Quant Engine

### Model family: LightGBM

Gradient-boosted decision trees, and specifically **not** deep learning, transformers, or reinforcement learning in v1.

The reasoning, stated once so it does not have to be relitigated:

**Data shape favors trees.** The training set is tabular — a few hundred engineered numeric columns, on the order of millions of instrument-days at daily resolution. Gradient-boosted trees are the reference-standard performer on tabular data at this scale. Neural architectures earn their advantage on high-dimensional unstructured input (images, language, raw sequences) or on datasets orders of magnitude larger.

**Debuggability under solo operation.** The user is the only engineer. At 6am, facing a prediction that looks wrong, the required capability is answering *why* — which feature drove it, is that feature computed correctly, does the split make sense. Feature importance and SHAP values give that directly. A transformer gives an attention map and a shrug.

**Retraining budget.** The full model set must train, validate, and gate every night on a laptop, inside a window that leaves room for everything else. LightGBM does this in minutes on CPU. Nightly neural retraining on a 4GB GPU would dominate the entire night and deliver less.

**And the claim is testable.** Baseline 9 in the evaluation gauntlet ([10-evaluation.md](10-evaluation.md)) is a LightGBM+XGBoost ensemble, so "trees are enough" is a hypothesis the evaluator checks rather than an assumption the architecture protects. If a different model family demonstrably beats it through the same gates, the reasoning above stops applying.

**RL is rejected for a specific reason, not a vague one.** Reinforcement learning is the obvious-sounding fit for "learns from trade outcomes," and it is a trap here. RL needs enormous interaction data, is notoriously unstable to train, and — decisively — is extremely difficult to validate safely. This system's entire credibility rests on a gate that can prove a candidate is better than the incumbent ([06-retraining-rigor.md](06-retraining-rigor.md)). A model class that resists that proof is disqualified regardless of ceiling.

This is a v1 decision, not a permanent one. Revisiting it requires evidence that the gate is being cleared consistently and the tree models have plateaued — not novelty.

### Regime-gated, not an ensemble vote

The instinctive way to use several models is to average them. That is wrong here, and understanding why is the core insight of Layer 1.

Trending markets and range-bound markets reward **opposite** behavior. In a trend, a breakout on strong volume is a continuation signal. In a range, that same breakout is most often a false break that reverts. A single model trained across both learns the average of two contradictory relationships — which is approximately no relationship. Averaging an ensemble of such models does not fix it; it entrenches it.

So the structure is a router, not a committee:

```
              feature row
                   │
                   ▼
        ┌────────────────────┐
        │ REGIME CLASSIFIER   │   LightGBM, multiclass
        │ trending / sideways │
        │ / high-volatility   │
        └─────────┬──────────┘
                  │ routes to exactly one
      ┌───────────┼───────────┐
      ▼           ▼           ▼
 ┌─────────┐ ┌─────────┐ ┌─────────┐
 │TRENDING │ │SIDEWAYS │ │HIGH-VOL │   LightGBM, one per regime,
 │specialist│ │specialist│ │specialist│  each trained ONLY on its
 └─────────┘ └─────────┘ └─────────┘   own regime's history
```

Each specialist sees only its own regime's data, so it is free to learn that regime's true relationships without them being diluted by the others.

### Regime definitions

Regimes are assigned by explicit, deterministic rules over volatility and trend-strength features ([03-feature-engineering.md](03-feature-engineering.md)). The classifier is then trained to *predict* those labels from features — which is what makes forward-looking routing possible, since tomorrow's label is not yet observable.

| Regime | Character |
|---|---|
| **Trending** | Sustained directional movement; MAs aligned and sloping; higher-highs/higher-lows structure holding; volatility normal or moderately elevated |
| **Sideways** | Directionless; price oscillating within a bounded range; MAs flat and interleaved; low ADX-equivalent trend strength |
| **High-volatility** | Realized volatility in a high percentile of the instrument's own history; range expansion; large gaps; often event- or crisis-driven, and behaviorally distinct regardless of direction |

Regime is assigned at **both** instrument and market level. A stock can be trending inside a chaotic market, and the difference matters — [03-feature-engineering.md](03-feature-engineering.md) exposes both to the specialists.

Boundaries are inherently fuzzy. The classifier outputs class probabilities, and a low-confidence regime assignment is itself a signal: an instrument the router cannot confidently place is an instrument whose prediction deserves less trust. Regime confidence is carried forward into the shortlist and into grading.

### Horizon

Every model is trained and evaluated at **one declared horizon and one declared resolution** ([03-feature-engineering.md](03-feature-engineering.md)). Horizon is a parameter of the model, not a property of the system: daily-resolution models train across the full 2000→ history, intraday models from 2022→, and both are first-class.

Multiple horizons mean multiple models with separate registry entries, never one model with a blurred target.

### Outputs

Per instrument, per bar:

| Output | Meaning |
|---|---|
| Regime | Assigned class plus confidence |
| Direction | Predicted sign over the configured horizon |
| Probability | **Calibrated** probability of continuation |
| Expected magnitude | Predicted return over the horizon |
| Expected adverse excursion | Predicted worst drawdown before the horizon closes |
| Model version | Which registry entry produced this — required for grading |
| Top contributing features | Feature attributions, for the Control Room and for Layer 2 context |

The probability must be **calibrated**, not merely ranked. A model that says 80% must be right about 80% of the time. Ranking-only accuracy is insufficient for a system that presents confidence numbers to a human who will size positions by them. Calibration is measured and enforced in [06-retraining-rigor.md](06-retraining-rigor.md).

### Shortlisting

Layer 1's deliverable is a **shortlist**, not a decision. Ranking runs across the universe, then filters apply:

1. **Tradeability** — minimum liquidity and turnover ([03-feature-engineering.md](03-feature-engineering.md)). A brilliant signal on an untradeable instrument is noise.
2. **Universe validity** — active, non-delisted, correctly named ([02-data-layer.md](02-data-layer.md)).
3. **Confidence floor** — below a threshold, nothing is worth a human's morning attention.
4. **Diversification** — cap per sector, so a single sector move does not consume the entire brief.
5. **Budget cut** — trim to what Layer 2 can actually investigate tonight. This is the adaptive part, specified below.

---

## Layer 2 — The Investigator

### Ollama Cloud, via the local CLI

Layer 2 runs on [Ollama Cloud](https://docs.ollama.com/cloud): the local `ollama` CLI invokes a model suffixed `-cloud`, which downloads only a small manifest and executes inference on Ollama's GPUs. Ollama v0.32.6 is already installed on the target machine.

This resolves a constraint that shaped earlier drafts of this design. The machine's RTX 3050 has 4GB of VRAM, which cannot comfortably host a model capable of the reasoning this layer needs. Cloud execution removes the ceiling.

It also introduces two facts the rest of this documentation must stay honest about:

- **Inference data leaves the machine.** Candidate market data and derived signals are sent to Ollama's infrastructure. This is why [00-overview.md](00-overview.md) is careful that "offline desktop app" means *no self-hosted server*, not *airgapped*.
- **The free tier is bounded.** Roughly one cloud model, with session limits that reset on a multi-hour cycle and weekly caps on top. Per the project's non-goals, **v1 stays on the free tier** — no second billing account. That constraint is not worked around; it is designed for, below.

### The job

For each shortlisted candidate, Layer 2 asks the question Layer 1 structurally cannot: *does anything outside the numbers confirm or contradict this?*

| Investigation axis | What it examines |
|---|---|
| News and announcements | Recent company news, corporate actions, filings — is this move information-driven or mechanical? |
| Scheduled events | Earnings or board meetings imminent? A high-confidence signal two days before earnings carries risk the model cannot see. |
| Sector and peer coherence | Are peers moving similarly? A lone mover in a flat sector is a different case from sector-wide strength. |
| F&O contradiction | Does derivatives positioning agree with the cash signal? Short covering dressed as strength is the canonical trap. |
| Historical analogy | In the trade ledger, has this user been burned by this setup before? ([02-data-layer.md](02-data-layer.md)) |
| Macro overlay | Index regime, volatility environment, anything that makes today unusual |

### Output: structured, not prose

Layer 2 emits a fixed structure per candidate, because Layer 3 must consume it reliably:

```
{
  instrument, verdict, confidence_adjustment,
  supporting_factors[], contradicting_factors[],
  risk_flags[], one_line_rationale
}
```

`verdict` ∈ `corroborated` | `contradicted` | `inconclusive`.

`confidence_adjustment` is a bounded modifier, never a replacement. **Layer 2 may adjust the quant model's confidence; it may not overwrite it.** The bound exists because an LLM's felt certainty is not a calibrated probability, and allowing it to overwrite a measured, gated, calibrated number with an unmeasured one would discard the most rigorously validated part of the system in favor of the least.

Free-text prose from Layer 2 is confined to `one_line_rationale`. Everything else is enumerated. Prose is Layer 3's job.

### Adaptive budget — the free-tier design

The shortlist is sized to fit the quota, rather than the quota being assumed to fit the shortlist.

```
Before investigating:
  ├─ Read recorded throughput from recent runs
  │  (candidates/hour, tokens/candidate, observed limits)
  ├─ Read remaining window before market open
  ├─ Compute affordable_count
  └─ Trim shortlist to affordable_count, highest-ranked first

During investigation:
  ├─ Track consumption per candidate
  ├─ On approaching limits → stop cleanly, do not thrash retries
  └─ Record actuals for the next night's estimate

On exhaustion or outage:
  ├─ Remaining candidates → verdict = inconclusive
  ├─ Mark them explicitly as UNINVESTIGATED
  └─ Continue the pipeline
```

Three rules govern this:

**Never silently truncate.** If only six of eleven candidates were investigated, the brief says so. An uninvestigated candidate presented identically to an investigated one is a lie of omission, and it is exactly the failure mode that erodes trust in the system.

**Investigate in rank order.** Budget goes to the strongest signals first.

**Learn the budget empirically.** Night one measures actual throughput on the free tier; subsequent nights size themselves from recorded history rather than a guess. The initial estimate is deliberately conservative. Measured throughput is tracked in [09-open-questions.md](09-open-questions.md) until enough runs exist to make it a settled number.

If the free tier proves too tight to be useful even at a small shortlist, the documented options — in preference order — are: reduce shortlist further, reduce per-candidate prompt size, move to a smaller cloud model, or (breaking a stated non-goal, and therefore requiring an explicit decision rather than a drift) upgrade the tier.

### Failure behavior

Layer 2 is **enrichment, not correctness**. Its absence degrades the brief; it does not invalidate it. Outage, quota exhaustion, or malformed output all result in `inconclusive` verdicts, a visible flag, and a completed pipeline ([01-architecture.md](01-architecture.md)).

Output that fails schema validation is retried once, then treated as `inconclusive`. Malformed LLM output is never parsed leniently into a verdict — a guessed verdict is worse than an honest absence.

---

## Layer 3 — The Synthesizer

### Claude Code CLI, subscription-authenticated

Layer 3 invokes the `claude` CLI non-interactively (`claude -p`), authenticated by the user's existing Claude subscription. **Not the metered Anthropic API** — no API key, no second billing account, consistent with the project's non-goals.

Two Windows-specific invocation hazards are documented here because they have already cost this project time in a previous incarnation, and both fail *silently*:

- The CLI is installed as an npm shim (`claude.CMD`). A subprocess spawn that does not resolve the executable path properly will fail to find it. Resolve via `shutil.which` rather than assuming a bare `claude` is executable.
- System-prompt injection uses `--append-system-prompt`. There is no `--system` flag; passing one produces a no-op rather than an error.

An implementation that gets either wrong produces a pipeline that appears to run and quietly never calls Claude at all. Layer 3 must therefore verify it received a real response, and treat absence as a failure rather than as empty output.

### The call budget, and why it is the reason Layer 2 exists

**One to two invocations per night. Total.**

Subscription authentication is designed around interactive human sessions, not high-volume unattended automation. A nightly job that called Claude once per candidate across twenty instruments would be both abusive of that model and operationally fragile — the first rate-limit response at 2am would take down the brief.

This single constraint is the structural reason the system has three layers instead of two. Per-candidate deep investigation is genuinely valuable, and it genuinely cannot be done at Claude-subscription volume — so it is done by Layer 2, on infrastructure priced for volume, and Claude receives only the already-investigated result.

### Input contract

Layer 3 receives **compact structured data only**:

- The investigated shortlist — predictions, calibrated probabilities, expected moves, regimes, verdicts, factors, risk flags
- Yesterday's prediction grades (Track A)
- Yesterday's trade decision grades and behavioral patterns (Track B)
- Market regime summary and gate outcome
- Explicit flags: which candidates were uninvestigated, which pipeline steps degraded

It never receives raw candles, full news text, or unfiltered logs. Those are Layer 1 and Layer 2 inputs. Layer 3's job is judgment and expression over digested facts — feeding it raw data would waste the budget and invite it to redo, badly, work already done properly.

### Outputs

**Morning brief** — the top opportunities with confidence and setup rationale, the market regime read, explicit risk flags, and a plain statement of what the system does *not* know. Delivered before market open.

**Evening review** — how yesterday's predictions actually scored, how the user's trades scored on decision quality (both directions: exits that were early, and exits that correctly avoided a loss), behavioral patterns worth naming, and what the retraining gate decided and why.

Both are written for a human who will act on them, which means: state confidence honestly, flag uncertainty rather than smoothing it, and never manufacture conviction the underlying numbers do not support.

### Failure behavior

If Claude is unreachable or the invocation fails, Layer 3 falls back to a **templated brief** built directly from Layer 1 and Layer 2 output. Less readable, fully correct, still delivered — and clearly marked as a fallback ([01-architecture.md](01-architecture.md)).

---

## What the brain does not do in v1

- **It does not place orders.** Algo Trading is gated ([02-data-layer.md](02-data-layer.md)).
- **It does not size positions.** Sizing depends on capital, conviction, and risk appetite the system does not model in v1.
- **It does not predict future candlestick sequences.** Predicting a full OHLC path is a substantially harder problem than predicting direction and magnitude, and attempting it before the simpler problem is demonstrably solved would be building on sand.
- **It does not learn from the user's trades.** Track B coaches the human; it never reaches `model.fit()`. This is the invariant from [00-overview.md](00-overview.md), and it is enforced as a data-access rule in [02-data-layer.md](02-data-layer.md).
