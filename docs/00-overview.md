# 00 — Overview

> **Status: PARTIAL — see [STATUS.md](STATUS.md).** The dual-track invariant here is sound and still the target. The reconcile/learning loop this document centers on has zero implementation. Horizon framing is correct and matches evidence (monthly, not daily).

## What StockSense is

StockSense is a single-user, locally-run quantitative trading brain for the NSE, packaged as an Electron desktop application.

It observes the market and the user's own trades, predicts what is likely to happen next, records what actually happened, grades both its own predictions and the user's decisions against reality, and retrains itself — every night, through a validation gate.

The one-sentence definition:

> A desktop application that studies the market and its own mistakes every night, and hands the user a researched plan every morning.

## What StockSense is not

It is not a latency-competitive trading system. It does not race other participants to the exchange, and it never will — that is a different discipline requiring co-location and infrastructure this project explicitly refuses to build. The nightly batch cadence is what makes the rest possible: slow cloud inference, multi-minute training, hours of unattended investigation.

**This is a statement about latency, not about horizon.** The system is horizon-agnostic by design ([03-feature-engineering.md](03-feature-engineering.md)) — features, labels, models, simulator, and evaluator all take horizon as a parameter. Daily-horizon models train across the full history back to 2000; intraday-horizon models train from 2022 forward, where intraday data begins. Both are first-class. What StockSense will not do is compete on execution speed.

It is also not an alert app. An alert app tells you a stock crossed a threshold. StockSense tells you what it predicted, what actually happened, whether its reasoning held, what you did about it, and whether your decision was sound — and then changes itself based on the answer.

## The core thesis

**Every night, the system learns from its own mistakes.**

This is the reason the project exists, and it is the single claim every other document here must remain consistent with. But "learning from mistakes" is a slogan until it is made mechanical, and made mechanical it splits into two tracks that must never be conflated.

### Track A — Market mistakes

The model predicted something. Reality disagreed. That gap becomes a labeled training example.

```
Yesterday: model predicted STOCK X → 78% continuation
Today:     actual outcome → reversed at ₹843

                    ↓
        ERROR = predicted vs actual
   (direction? magnitude? timing? confidence?)
                    ↓
   Becomes a labeled row in tonight's training set
                    ↓
        Candidate model retrains on it
                    ↓
   GATE: does the candidate genuinely generalize
   better on data it has never seen, net of costs?
                    ↓
            PASS → promote      FAIL → reject, log why
```

The gate is not optional decoration. Without it, "learning from mistakes" and "overfitting to yesterday's noise" are the same operation. A system that retrains blindly every night does not get smarter — it gets more confident about whatever happened most recently, which is the opposite of the goal. See [06-retraining-rigor.md](06-retraining-rigor.md) for the full set of requirements that make the gate meaningful.

### Track B — Trader mistakes

The user exited at ₹833; the stock went to ₹850. Or the user exited at ₹833 and the stock collapsed to ₹810. Both are learning material — about the *user*, not about the market.

This track produces coaching, delivered in the evening review: patterns like *exits winners early*, *overtrades after a loss*, *third re-entry into the same stock has poor expectancy*, *position sizing inflates after a winning streak*.

### The invariant

**Track B must never influence model training weights.**

This is an architectural invariant, not a preference. The reasoning: the user's behavioral errors are not market truth. If the trade ledger were allowed to flow into the market model's training data as signal, the model would begin learning to reproduce the user's habits — including the bad ones — and would then recommend them back with the false authority of a quantitative system. The user's mistakes would be laundered into predictions and returned as advice.

So the two tracks share the same nightly job and the same data store, and they share nothing else. Track A reads market outcomes and writes to the model registry. Track B reads the trade ledger and writes to the coaching output. No path connects the trade ledger to `model.fit()`.

A useful consequence of keeping them separate is that the system can say things a single-track system cannot:

- "The model was right, but you exited too early."
- "The model was wrong, and your manual exit reduced the loss."
- "The model was right about direction and wrong about magnitude."
- "Your trade was profitable, but statistically it was a poor decision."

That last one matters most. **Profitability and decision quality are not the same thing**, and a system that cannot tell them apart cannot coach.

## The evaluation standard

A learning claim that cannot be falsified is marketing. So the system is built alongside an **adversarial evaluator** whose job is to try to prove the brain wrong ([10-evaluation.md](10-evaluation.md)).

The question it exists to answer is not *"does StockSense know finance?"* — any language model can produce competent-sounding market commentary. It is:

> Placed into a historically accurate market, under the information constraints that actually existed at that moment and with realistic execution costs applied, does StockSense beat strong quantitative baselines — and does it stay good when the regime changes?

The evaluator is a **peer system, not a subsection of the brain**. It owns the historical simulator, the episode library, the held-out eras, and the hard gates. The brain cannot read them. And its gates are vetoes rather than weights: a candidate with spectacular P&L and failing risk behavior does not deploy.

The consequence for how anything reaches real money:

```
Quant Brain → Strategy → Backtester → Adversarial Validator
                                              ↓
              Production ← Risk Gate ← Live Shadow ← Paper Market
```

rather than `LLM → BUY → real money`.

## Single-user consequences

The same person is the developer, the QA tester, and the trader. This is not a footnote — it changes the architecture in two opposite directions at once.

**What it removes.** No multi-user authentication. No role separation. No permission model. No tenancy. No account system anywhere in the application.

**What it demands.** Total internal visibility. The person debugging a suspicious prediction at 11pm and the person acting on a signal at 9am are the same person using the same application, and the app must serve both. Every service, model, and job must be observable and controllable from the UI: health status, live logs, manual triggers, model registry with rollback, an emergency stop.

This is why the Electron app is described throughout this documentation as a **Control Room** and not a dashboard. A dashboard displays. A control room operates. See [07-control-room.md](07-control-room.md).

## The shape of a day

```
15:30  Market closes
         ↓
  (evening, scheduled)
         ↓
  Ingest → Validate → Reconstruct trades → Grade predictions
  → Grade user decisions → Rebuild features → Relabel regimes
  → Train candidates → Walk-forward test → GATE → Shortlist
  → Investigate (Ollama Cloud) → Synthesize (Claude) → Telegram
         ↓
09:00  Morning brief arrives on the user's phone
         ↓
  User trades — or does not. The user decides.
         ↓
15:30  Market closes. The loop closes with it.
```

The full step-by-step specification, including per-step failure behavior, is in [05-nightly-pipeline.md](05-nightly-pipeline.md).

## Explicit non-goals for v1

These are decisions, not omissions. Each is listed so a future reader does not mistake absence for oversight.

| Non-goal | Reason |
|---|---|
| Latency-competitive execution | Racing to the exchange is a different discipline with different infrastructure. Intraday *horizons* are supported; intraday *speed competition* is not. |
| Automated order execution | Upstox Algo Trading access exists and is **gated off by default**. v1 produces recommendations; the user decides. See [02-data-layer.md](02-data-layer.md) for the concrete condition that would enable it. |
| Deep learning / transformers / reinforcement learning | Data volume favors gradient-boosted trees; RL is hard to validate safely and hard to debug alone at 6am. See [04-model-brain.md](04-model-brain.md). |
| Cloud hosting / always-on server | Everything runs on the user's machine. No VPS, no hosting bill, no remote ops burden. |
| A second paid LLM billing account | Ollama Cloud free tier + Claude Code CLI (existing subscription) only. This constraint actively shapes the design of Layer 2. |
| Multi-user support, auth, roles | Single user. See above. |
| Web dashboard | The Electron Control Room is the interface. Telegram is the delivery channel. Nothing is hosted. |

## An honest note on "offline"

StockSense is a desktop application with no hosted server component. It is **not** an airgapped application.

Three things reach the network every night: the Upstox API (market and account data), Ollama Cloud (Layer 2 inference — meaning candidate analysis leaves the machine), and the Telegram Bot API (delivery). Claude Code CLI also requires connectivity.

"Offline desktop app" in this project means *no self-hosted server, no cloud infrastructure to maintain, no hosting cost, all state and all training on local disk*. It does not mean the application functions without internet. [01-architecture.md](01-architecture.md) specifies exactly what fails and how when connectivity is absent.

## Document map

| Document | Covers |
|---|---|
| [01-architecture.md](01-architecture.md) | Process split, component contracts, network dependencies |
| [02-data-layer.md](02-data-layer.md) | Upstox surfaces, historical depth, DuckDB store, entities |
| [03-feature-engineering.md](03-feature-engineering.md) | Full feature taxonomy and the output contract |
| [04-model-brain.md](04-model-brain.md) | The three layers and their strict interfaces |
| [05-nightly-pipeline.md](05-nightly-pipeline.md) | The 15-step sequence and per-step failure behavior |
| [06-retraining-rigor.md](06-retraining-rigor.md) | Validation, gating, calibration, drift, rollback |
| [07-control-room.md](07-control-room.md) | Electron UI specification |
| [08-operations.md](08-operations.md) | Scheduling, secrets, logs, runbooks |
| [09-open-questions.md](09-open-questions.md) | Unresolved items with defaults and resolution criteria |
| [10-evaluation.md](10-evaluation.md) | The adversarial evaluation system — simulator, episodes, baselines, hard gates |
