# 10 — Quant Brain Evaluation

> **Status: PARTIAL, ONE MISSING PIECE IS THE PRODUCT ITSELF — see [STATUS.md](STATUS.md).** Walk-forward, best-trade-removal, parameter perturbation: built and run for real. Monte Carlo: attempted, found methodologically flawed (needs redoing, drawdown-focused). Episode library, 10-baseline gauntlet (1 exists), Quant IQ scorecard, regime stratification, reasoning evaluation: not built. **The immutable prediction ledger — arguably this document's most important artifact — does not exist; the `predictions` table is never written to.**

## The standard

The wrong question is *"does StockSense know finance?"* Any language model can produce competent-sounding market commentary, and competent-sounding commentary is worth nothing.

The right question:

> **Placed into a historically accurate market, under the information constraints that actually existed at that moment and with realistic execution costs applied, does StockSense make better decisions than strong quantitative baselines — and does it stay good when the market changes?**

That is a much harder standard, and it is the one this document enforces.

## Separation of powers

The evaluator is a **peer system to the brain, not a subsection of it.**

```
┌──────────────┐                      ┌──────────────────┐
│  QUANT BRAIN │  proposes ────────►  │    EVALUATOR      │
│              │                      │                    │
│  generates   │  ◄──────── falsifies │  tries to prove    │
│  hypotheses, │                      │  the brain wrong   │
│  models,     │                      │                    │
│  strategies  │                      │  owns: simulator,  │
│              │                      │  episodes, gates   │
└──────────────┘                      └──────────────────┘
```

The brain's job is to be clever. The evaluator's job is to be **hostile**. It does not try to confirm that a model works; it tries to break it, and promotion happens only when it fails to.

Three structural rules make that adversarial relationship real rather than rhetorical:

1. **The evaluator owns the ground truth.** The simulator, the episode library, and the held-out eras belong to the evaluator. The brain cannot read them, only be tested against them.
2. **No shared code paths.** If the brain and the evaluator computed returns with the same function, a bug in that function would be invisible to both. Independent implementations of the scoring math mean disagreement surfaces rather than cancels.
3. **The evaluator can veto.** Its gates are not advisory inputs to a weighted score. Any hard gate failing blocks deployment regardless of how good the P&L looks.

### The production path

```
Quant Brain → Strategy → Backtester → Adversarial Validator
                                              ↓
              Production ← Risk Gate ← Live Shadow ← Paper Market
```

Explicitly replacing the naive path this project exists to avoid:

```
LLM → BUY → real money
```

Every arrow is a checkpoint that can send the candidate backward.

---

## 1. Capability decomposition

**No single score.** A blended number lets a strength in one dimension hide a fatal weakness in another — which is precisely how systems with beautiful backtests and catastrophic risk behavior get deployed.

Seven capabilities, graded separately:

| Capability | The question it answers |
|---|---|
| **Market understanding** | Does it correctly interpret what actually happened? |
| **Prediction** | Can it forecast future returns and movement? |
| **Trade selection** | Given many options, does it choose the best ones? |
| **Risk** | Does it avoid dangerous positions and size sensibly? |
| **Execution** | Does its edge survive costs, slippage, and latency? |
| **Adaptation** | Does it improve when regimes change, or degrade? |
| **Portfolio intelligence** | Does it understand existing positions and avoid contradicting itself? |

Portfolio intelligence deserves a note because it is the one most systems omit. A brain that recommends buying a stock it already holds a full position in, or recommends two names that are 0.9-correlated as though they were diversification, does not understand the portfolio — it understands only individual instruments. That failure is invisible to per-trade metrics and shows up only in portfolio-level evaluation.

---

## 2. The historical market simulator

The most important component in this document. It is a time machine with one prime directive:

> **The agent must never see the future.**

### Replay discipline

The simulator advances chronologically and exposes only the information set that genuinely existed at that timestamp:

```
10:01:00
  Agent sees:  candles ≤ 10:01 · F&O ≤ 10:01 · news published ≤ 10:01
               · index and breadth state ≤ 10:01 · its own open positions

  Agent decides:  BUY XYZ · ₹830 · qty 100 · confidence 82%

10:02 → 10:03 → 10:04 → …    simulator advances

  Outcome revealed only after the horizon closes.
```

This is where most impressive-looking trading systems quietly fall apart. The known failure modes, all of which the simulator must structurally prevent rather than merely discourage:

| Leak | How it happens |
|---|---|
| **Lookahead in features** | A window that includes data after the decision point ([03-feature-engineering.md](03-feature-engineering.md)) |
| **Survivorship bias** | Testing on today's instrument list, which excludes everything that delisted or collapsed |
| **Restatement leakage** | Using corporate-action-adjusted or corrected data that was not public at the time |
| **News timestamp drift** | Using publication time when the market learned later, or vice versa |
| **Universe leakage** | Filtering by liquidity measured over the whole period, including the future |
| **Optimistic execution** | Assuming fills at prices that did not have volume behind them |

Point-in-time universe reconstruction matters as much as point-in-time prices: an evaluation of 2008 must run against the instruments that were listed and liquid *in 2008*, including the ones that no longer exist.

### Fidelity tiers

The simulator declares its tier for every run, and each tier states plainly what it can and cannot claim. This is the honest response to a real data constraint rather than a pretence of precision the data cannot support.

| Tier | Data | Can claim | Cannot claim |
|---|---|---|---|
| **T1 — Daily bar** | Daily OHLCV, 2000→ | Regime behavior, multi-day edge, cost-adjusted returns | Anything about intraday path or entry timing |
| **T2 — Minute bar** | 1-min candles, 2022→ | Intraday timing, opening-range behavior, realistic-ish entry prices | True spread, queue position, partial fills |
| **T3 — Modeled microstructure** | T2 + slippage/latency/fill models | Sensitivity of the edge to execution assumptions | That the modeled fills are what would actually have happened |

**There is no true order-book replay tier.** NSE Level 2/3 and tick-by-tick order data are separate commercial products, not available through the Upstox historical API ([02-data-layer.md](02-data-layer.md)). Spread, queue position, and partial fills are therefore **modeled, not replayed** — and every report says so. The T3 interface is designed so a real order-book feed could be plugged in later without rewriting the evaluator, but nothing in this project currently claims that fidelity.

The rule that follows: **a strategy whose edge appears only at T3 and cannot be seen at T1 or T2 is treated as unproven**, because at that point the edge is a property of the slippage model rather than of the market.

---

## 3. The episode library

Thousands of frozen historical scenarios — the exam papers. Each episode is an immutable triple: an information snapshot, a question, and an outcome that is revealed only after the answer is locked.

| Episode class | Example |
|---|---|
| **Open-ended** | "Here is NSE on this morning. What do you do?" |
| **Continuation** | "A stock has risen 2.3%, volume has exploded, OI is rising. What happens next?" |
| **Position management** | "You bought at ₹830. It is now ₹827. What should you do?" |
| **Decision review** | "You sold at ₹833. It later reached ₹850. Was the decision bad?" |
| **Regime shift** | "The market has entered high volatility. Which positions should be reduced?" |
| **Portfolio conflict** | "You hold a full position in this name and the model now signals it again. What now?" |
| **News shock** | "This announcement just landed. Is the pre-announcement signal still valid?" |
| **Trap** | Setups that historically looked strong and failed — short covering dressed as strength |

The decision-review class is where the evaluator tests the distinction the whole system is built on: **profitability and decision quality are different axes** ([00-overview.md](00-overview.md)). An answer of "yes, that was a bad decision" to a sell at ₹833 that was followed by a collapse to ₹810 is *wrong*, even though the price did later touch ₹850 on some other path.

Trap episodes are deliberately over-represented relative to their natural frequency. A library sampled purely at random is dominated by unremarkable days, and unremarkable days do not discriminate between a good brain and a mediocre one.

---

## 4. Execution economics

A directionally correct call that loses money after costs is **not a partially correct call**. It scores zero or negative. Never partial credit.

```
Signal: BUY.  Outcome: +0.20%.
Costs:  brokerage + STT + exchange + SEBI + stamp + GST + slippage + spread
Net:    negative
Score:  ≤ 0
```

The full Indian cost stack is specified in [06-retraining-rigor.md](06-retraining-rigor.md). What the evaluator adds is the execution layer around it:

| Modeled | Treatment |
|---|---|
| **Spread** | Estimated from intraday range and liquidity at T2/T3 |
| **Slippage** | A function of order size relative to available liquidity **and** current volatility regime — never a constant |
| **Latency** | Injected delay between decision and fill, swept as a stress parameter |
| **Partial fills** | Modeled against traded volume; large orders in thin names do not fill completely |
| **Liquidity ceiling** | Position size capped by a fraction of the instrument's actual traded volume |
| **Position limits** | Exposure caps enforced inside the simulation, not assumed away |

The reason slippage cannot be a constant: it scales with size and widens sharply in turbulent markets, so a fixed assumption systematically flatters exactly the trades most likely to disappoint — large positions, thin instruments, volatile sessions. A constant-slippage backtest is a backtest of a market that does not exist.

---

## 5. Two scoreboards, kept apart

Prediction quality and trading quality are different questions and must never be collapsed into one number.

### Prediction quality

If the brain says 70% ten thousand times, roughly seven thousand should occur.

| Metric | Measures |
|---|---|
| **Log loss** | Probabilistic accuracy, punishing confident errors hardest |
| **Brier score** | Aggregate probabilistic accuracy |
| **Calibration error** | Reliability across confidence buckets |
| **Precision / recall** | Performance at actionable thresholds |
| **ROC-AUC** | Ranking quality, independent of threshold |
| **Directional accuracy** | Raw hit rate |
| **Information coefficient** | Correlation between predicted and realized returns — the classic quant measure |

### Trading quality

| Metric | Measures |
|---|---|
| **Net P&L** | After the full cost stack |
| **Sharpe / Sortino** | Risk-adjusted return; Sortino because upside volatility is not a problem |
| **Profit factor** | Gross profit ÷ gross loss |
| **Maximum drawdown** | Worst peak-to-trough — the number that decides whether a strategy is survivable in practice |
| **Expectancy** | Average outcome per trade |
| **Turnover** | Cost and capacity implications |
| **Win rate, avg win / avg loss** | Distribution shape, not just central tendency |

**Both scoreboards are reported. Neither substitutes for the other.** Excellent calibration that does not convert into net profit means the edge is real but too small to trade. Strong P&L on poor calibration means the returns came from a few outliers and the confidence numbers cannot be trusted for sizing — which is arguably more dangerous, because it invites oversizing.

---

## 6. The baseline gauntlet

A brain evaluated alone will always look impressive. It needs opponents.

| # | Baseline | What beating it proves |
|---|---|---|
| 1 | Buy and hold | Any edge at all over doing nothing |
| 2 | Nifty benchmark | Edge over the market itself |
| 3 | Random entries | Edge over chance, at matched turnover |
| 4 | Simple momentum | Edge over the most obvious factor |
| 5 | Mean reversion | Edge over the second most obvious factor |
| 6 | VWAP strategy | Edge over a standard execution heuristic |
| 7 | **The user's own manual strategy** | That the system is worth using at all |
| 8 | **LightGBM only, no LLM layers** | **That the LLM layers earn their cost** |
| 9 | LightGBM + XGBoost ensemble | Edge over a stronger pure-ML stack |
| 10 | Full StockSense | — |

Two of these carry disproportionate weight.

**Baseline 7** is the honest one. If the full system cannot beat the user trading manually, it has no reason to exist regardless of its Sharpe ratio.

**Baseline 8 is the decisive ablation.** It isolates exactly what the Ollama investigation layer and the Claude synthesis layer contribute. If StockSense-with-LLMs cannot beat LightGBM-alone on net-of-cost performance, then Layers 2 and 3 are expensive decoration and the honest conclusion is to ship the simpler system. It is cheap to compute, so it runs **nightly** as part of the fast evaluation subset ([05-nightly-pipeline.md](05-nightly-pipeline.md)) rather than waiting for the full suite. Its result is reported prominently rather than buried — it is the single number most likely to be uncomfortable, which is precisely why it must be visible.

All baselines run through the identical simulator, cost model, and universe. A baseline given easier conditions is not a baseline.

---

## 7. The Quant IQ scorecard

```
════════════════════════════════════════════
             STOCKSENSE QUANT IQ
════════════════════════════════════════════

  Market Understanding          91/100
  Prediction Calibration        87/100
  Directional Prediction        84/100
  Trade Selection               89/100
  Risk Management               93/100
  Execution Quality             78/100
  Regime Adaptation             81/100
  Portfolio Intelligence        90/100

────────────────────────────────────────────
  Composite                     87/100
════════════════════════════════════════════

  HARD GATES

  Prediction quality            PASS
  Risk management               PASS
  Execution realism             PASS
  Out-of-sample validity        PASS
  Regime robustness             PASS
  Statistical significance      PASS
  Baseline 8 ablation           PASS

              ↓
      DEPLOYMENT ELIGIBLE
════════════════════════════════════════════
```

**The composite is descriptive, not decisive.** It exists to make trends visible across versions, and it has no authority.

Authority sits with the hard gates, and gates are **vetoes, not weights**. A model with a 95 composite and a failing risk gate is not deployed. Averaging would let spectacular P&L purchase forgiveness for dangerous risk behavior, which is the exact trade that destroys trading systems.

**Statistical significance** is a gate because an edge measured over too few trades is not an edge. A strategy with a beautiful Sharpe over 40 trades has demonstrated nothing that could not be luck, and the gate requires enough independent observations to distinguish the two.

---

## 8. Regime-stratified evaluation

An aggregate Sharpe over 2000–2026 is nearly useless. It averages a system that may be excellent in trends and worthless in chop, and tells you neither.

**By era** — using the daily history available from January 2000 ([02-data-layer.md](02-data-layer.md)):

| Era | Character |
|---|---|
| 2000–2003 | Dot-com aftermath |
| 2004–2007 | Sustained bull |
| 2008 | Global financial crisis |
| 2009–2013 | Recovery, then stagnation |
| 2014–2019 | Demonetisation, GST, structural change |
| 2020 | COVID crash and violent recovery |
| 2021 | Liquidity-driven bull |
| 2022 | Rate shock |
| 2023–2026 | Recent regimes |

**By market condition:**

```
  Bull market          88/100
  Bear market          76/100
  Sideways             61/100      ← the brain is weakest here
  High volatility      93/100
  Low volatility       68/100
  News shock           54/100      ← and worst here
```

The purpose is not the average. **The purpose is knowing when the brain is stupid.** A system that scores 54 on news shocks is not a bad system — it is a system that should reduce confidence or stand aside when news is driving the tape, and that is actionable intelligence the aggregate score would have hidden entirely.

A regime whose score is both low and high-variance is a regime where the model should not be trading.

---

## 9. Walk-forward evaluation

Never train on 2005–2025 and test on data the model has indirectly seen.

```
TRAIN 2000 ────────── 2012   ║embargo║   TEST 2013
TRAIN 2000 ─────────── 2013  ║embargo║   TEST 2014
TRAIN 2000 ──────────── 2014 ║embargo║   TEST 2015
                                              →  and onward
```

Expanding window, strict forward ordering, with the purge-and-embargo discipline specified in [06-retraining-rigor.md](06-retraining-rigor.md) — training samples whose labels extend into the test window are removed, and a gap of at least the label horizon plus the longest feature lookback separates the two.

Each fold reports independently. **Aggregate performance across folds is reported alongside per-fold variance**, because a model that wins in 2013 and 2019 while losing in 2016 and 2022 has not demonstrated an edge — it has demonstrated regime dependence that the average conceals.

---

## 10. Adversarial stress tests

Here the evaluator actively tries to destroy the strategy.

| Test | Passing behavior |
|---|---|
| **Slippage ×2** | Edge survives |
| **Slippage ×5** | Degrades gracefully; does not invert |
| **Latency +1s / +5s** | Edge survives at the target horizon |
| **Entry degraded 0.1% / 0.2%** | Edge survives |
| **Remove best 10 trades** | Still profitable — the edge is not a lottery ticket |
| **Remove worst 10 trades** | Performance was not merely the absence of disasters |
| **Parameter perturbation ±20%** | No collapse |
| **Universe perturbation** | Dropping a random 10% of instruments does not break it |
| **Start-date shift** | Beginning the test a month earlier or later does not change conclusions |

The two removal tests are a matched pair and answer opposite questions. If removing the best ten trades destroys profitability, the "edge" was a handful of outliers and will not repeat. If removing the worst ten transforms mediocre results into excellent ones, the strategy's real problem is tail risk it does not control.

**Parameter sensitivity is the sharpest fragility test.** If changing a lookback from 14 to 13 collapses performance, the parameter was fitted to noise. The correct response is to reject the strategy — not to keep 14 because it works better.

---

## 11. Monte Carlo

One equity curve is one sample from a distribution. It is hypnotic and nearly meaningless on its own.

Given the strategy's trades, reshuffle their sequence thousands of times and simulate the resulting paths:

```
  Expected CAGR              28%
  95% range          −4%  →  41%

  P(drawdown > 20%)          18%
  P(drawdown > 40%)           3%
  P(losing year)             11%
```

This reframes the question from *"what did it return?"* to *"what range of outcomes was I actually exposed to, and could I have survived the bad ones?"* A strategy with an attractive mean and an 18% chance of a 20% drawdown is a strategy the user needs to know about before capital is committed, not after.

Sequence risk is the point: the same trades in a different order can mean a comfortable year or a margin call.

---

## 12. Reasoning evaluation

Layers 2 and 3 ([04-model-brain.md](04-model-brain.md)) produce judgment and language, which the metrics above cannot score. They get their own rubric.

Given a scenario — *"you hold ₹100,000 of XYZ, it is up 1.2%, momentum has weakened but the sector remains strong; what do you do?"* — the reasoning is scored on:

| Criterion | Failure looks like |
|---|---|
| Evidence identification | Missing the momentum deterioration entirely |
| Irrelevance rejection | Anchoring on a fact that does not bear on the decision |
| Arithmetic correctness | Getting position value or percentage moves wrong |
| Self-consistency | Recommending hold while arguing the case for exit |
| **Hallucination** | Citing news, earnings, or numbers that do not exist |
| Risk-constraint obedience | Recommending size beyond stated limits |
| Uncertainty expression | False confidence where the evidence is genuinely thin |
| Quant consistency | Contradicting the model's own probability without justification |

Hallucination is graded as a **hard failure**, not a deduction. A layer that invents a fact once will invent one again, and a brief containing a fabricated earnings date is worse than no brief.

Scoring can be assisted by an independent evaluator model, kept separate from the layer being judged.

The governing rule, stated plainly:

> **Beautifully written analysis that loses money is still bad quant work.**

Reasoning scores never substitute for trading performance. They diagnose *why* the system succeeded or failed; they do not decide whether it did.

---

## 13. The immutable prediction ledger

The question behind every honest backtest: **"would I actually have known this at the time?"**

Every prediction freezes its complete information snapshot at the moment it was made:

```
PREDICTION 982731
  Timestamp      09:47:03.241
  Instrument     XYZ
  Horizon        (as configured for this model)

  Information snapshot
    OHLCV        ≤ timestamp
    F&O          ≤ timestamp
    News         ≤ timestamp
    Breadth · Nifty state · sector state
    Feature schema version · model version

  Prediction     BUY · p=0.81 · expected +0.34%
  Snapshot hash  <content hash>

  ─── later, appended, never edited ───
  Actual         +0.41%
  Grade          correct · net-of-cost positive
```

Three properties are absolute:

- **Immutable.** Outcomes are appended; the original prediction and its snapshot are never edited ([02-data-layer.md](02-data-layer.md)).
- **Hashed.** The snapshot hash makes silent retroactive modification detectable.
- **Complete.** Enough context to fully reproduce the decision.

This ledger is one of the project's most valuable long-term assets. Models can be retrained and code rewritten, but an accumulating, tamper-evident record of what the system believed and what actually happened cannot be reconstructed after the fact. It is the raw material for every calibration measurement, every drift detection, and every claim the system will ever make about improving.

---

## 14. The promotion staircase

```
  Backtest
     ↓        ← historical, full evaluation suite
  Paper market
     ↓        ← live data, simulated fills, no capital
  Live shadow
     ↓        ← live data, real-time, still no capital
  Small capital
     ↓        ← real money, hard caps
  Scaled deployment
```

Each stage runs long enough to produce statistically meaningful results, and **the gap between consecutive stages is the primary diagnostic.**

A strategy that shows Sharpe 2.1 in backtest, 1.4 on paper, and 0.6 in shadow has not encountered bad luck — it has encountered reality, and the decay pattern tells you exactly where the backtest was lying. Large backtest-to-paper decay points at leakage or optimistic fills. Large paper-to-shadow decay points at timing, data latency, or universe differences.

**Decay is expected**; the gate is on its magnitude. A candidate whose performance collapses at any transition returns to research rather than continuing down the staircase.

Progression to real capital additionally requires the Algo Trading gate in [02-data-layer.md](02-data-layer.md), which is a separate and stricter condition.

---

## 15. Learning verification

The system claims to learn. The evaluator's job is to make that claim falsifiable rather than decorative.

Every model carries full lineage:

| Recorded | Why |
|---|---|
| Training data range | Reproducibility |
| Feature set and schema version | Semantics change ([03-feature-engineering.md](03-feature-engineering.md)) |
| Hyperparameters, seed, architecture | Determinism |
| Validation results | Per regime, per fold |
| Deployment dates | What was live when |
| Live performance | What actually happened afterward |

The rule that gives the claim teeth:

> **If v17 does not outperform v16 under identical evaluation, v17 does not deploy.**

And the loop the evaluator grades, end to end:

```
prediction → outcome → error → diagnosis → hypothesis
    → new model → validation → deploy?
```

Making money is not sufficient evidence of learning — a model can profit from a favorable regime while getting worse. What counts is **demonstrated improvement on held-out data under identical conditions.**

---

## 16. The evaluation report

```
══════════════════════════════════════════════
            STOCKSENSE MODEL 42
          QUANT BRAIN EVALUATION
══════════════════════════════════════════════

  Period                        2000–2026
  Out-of-sample folds                   17
  Instruments                       2,300+
  Predictions                    8,742,192
  Simulator fidelity          T1 + T2 (T3 modeled)
──────────────────────────────────────────────
  Prediction calibration            91.2%
  Directional accuracy              68.4%
  Trade selection                   73.8%
  Risk score                        94.1%
  Execution robustness              82.7%
──────────────────────────────────────────────
  Net return · Sharpe · Sortino
  Max drawdown · Profit factor
──────────────────────────────────────────────
  Bull 88 · Bear 79 · Sideways 64
  High-vol 91 · Low-vol 71 · News shock 54
──────────────────────────────────────────────
  vs Baseline 7 (manual)             PASS
  vs Baseline 8 (LightGBM only)      PASS
──────────────────────────────────────────────
  Slippage ×2 / ×5                   PASS
  Latency +5s                        PASS
  Parameter perturbation             PASS
  Best/worst trade removal           PASS
  Monte Carlo                        PASS
──────────────────────────────────────────────
  Paper trading                      PASS
  Live shadow                        PASS
══════════════════════════════════════════════
              DEPLOYMENT: PASS
══════════════════════════════════════════════
```

Every report states its fidelity tier and every degradation. A report that cannot show its limitations is not evidence.

---

## 17. What the evaluator cannot do

An evaluation system that overstates its own authority is as dangerous as no evaluation at all.

**No true order-book replay.** Spread, queue position, and partial fills are modeled. Where an edge depends on microstructure, the evaluator cannot confirm it — only fail to refute it (§2).

**Regimes are small samples.** There is one 2008 and one COVID crash. Crisis-era results are single observations, not distributions, and should never be quoted with the confidence that a thousand ordinary days would justify.

**Survivorship is mitigated, not eliminated.** Point-in-time universe reconstruction depends on the completeness of delisting records.

**News history is incomplete.** Archives thin out going backward, so news-driven episodes in early eras under-represent what a participant would actually have known.

**Backtests cannot model the user.** They assume signals are followed. The trader who hesitates or oversizes is a variable the simulator does not contain — which is why Track B exists separately ([00-overview.md](00-overview.md)).

### The deepest risk: overfitting to the evaluator

Every time a model is tuned against the evaluator and re-tested, some of the evaluator's held-out data leaks into the design process — not through code, but through the researcher's decisions. Run this loop enough times and the evaluation becomes a training set with extra steps.

Mitigations, and they are disciplines rather than features:

- **A locked final-validation era**, never examined during development, used once per major version.
- **Evaluation-attempt counting.** Every candidate tested against the same holdout is recorded. Many attempts on one holdout means results should be discounted, and the system should say so.
- **Rotating episode subsets**, so the same scenarios are not optimized against repeatedly.
- **The staircase as external check** (§14). Paper and shadow stages use data that did not exist when the model was built, which no amount of historical overfitting can contaminate.

The honest framing: **the evaluator cannot prove a strategy works. It can only fail to prove that it does not.** Everything downstream of that — the staircase, the risk gate, the position caps — exists because that distinction is real.
