# 06 — Retraining Rigor

## Why this document exists

Every other component in StockSense fails loudly. Ingestion throws, validation quarantines, the CLI errors out — you notice.

Retraining is different. **A badly-validated retraining pipeline fails silently and looks like success.** Metrics improve, the model promotes, the brief ships, and the system reports that it is learning. Months later the live results do not resemble the backtests, and by then the cause is buried under a hundred promotions that each looked fine.

This is the failure mode the whole project is built to avoid. "Every night the system learns from its own mistakes" is either a rigorous claim or a marketing slogan, and the difference is entirely in this document.

The governing principle:

> **Retraining is a promotion decision, not an automatic overwrite.**

Every night produces a *candidate*. Every candidate must earn its place. Most should fail.

---

## 1. Purged, embargoed walk-forward validation

### The problem

Standard k-fold cross-validation shuffles rows into folds. On time-series data this trains on the future and tests on the past, which is not validation — it is leakage with a scoreboard.

Two subtler forms survive naive time-splitting:

**Label overlap.** A prediction made on day D with a 5-day horizon has a label determined by data through D+5. If the test set begins at D+1, the training label already contains test-period information.

**Feature overlap.** A 20-day feature computed on the first test day draws on 20 days of training-period data. Some bleed is unavoidable; unbounded bleed is not.

### The requirement

```
│──────── train ────────│═ embargo ═│──── test ────│
                          ↑
              gap ≥ label horizon + max feature lookback
```

- **Strictly forward.** Train on data up to T, test only after T. Never shuffled.
- **Purged.** Training samples whose labels extend into the test window are removed entirely.
- **Embargoed.** A gap of at least the label horizon plus the longest feature lookback separates train from test.
- **Rolling.** Multiple sequential windows across history, not one split. A candidate that wins in 2021 and loses in 2023 has not won.
- **Reported per window.** Aggregate metrics hide instability. A candidate whose per-window performance swings wildly is fragile regardless of its average.

---

## 2. Transaction-cost-aware backtesting

A strategy with a pre-cost edge and no post-cost edge is not a strategy. Indian equity costs are not trivial and must be modeled explicitly, not approximated by a single round-number percentage:

| Component | Notes |
|---|---|
| Brokerage | Per actual plan, including flat-fee structures |
| STT | Different rates for delivery and intraday, buy and sell side |
| Exchange transaction charges | Per exchange |
| SEBI turnover fees, stamp duty, GST | On brokerage and charges |
| **Slippage** | The one most often ignored and most often decisive |

Slippage deserves its own note: it is not a constant. It scales with position size relative to the instrument's liquidity and widens sharply in high-volatility regimes. A cost model with fixed slippage systematically flatters exactly the trades most likely to disappoint — large positions in thin names during turbulent markets.

**Gate rule:** a candidate must beat the incumbent **net of all costs**. A pre-cost win is not a pass and is not a partial credit.

---

## 3. Regime-stratified evaluation

Aggregate improvement can conceal regime-specific regression. A candidate that gains substantially in sideways markets while degrading in trending ones may show a better blended number while being strictly worse where it matters most.

Since the architecture routes by regime ([04-model-brain.md](04-model-brain.md)), blended metrics are doubly misleading: each specialist only ever operates inside its own regime, so its blended score describes a situation it will never face.

**Requirements:**

- Every metric reported per regime, at minimum trending / sideways / high-volatility
- Each specialist evaluated on its own regime
- The regime classifier evaluated separately as a multiclass problem, including its confusion matrix — routing errors propagate to every downstream prediction
- **No regime regression beyond tolerance, regardless of aggregate gain.** This is a veto, not a weighted factor.

---

## 4. Calibration, not just accuracy

### Why this ranks with correctness

StockSense presents probabilities to a human who will size decisions by them. "82% continuation" is a claim about frequency: over many such calls, about 82 in 100 should work out.

A model that is directionally accurate but says 80% when it means 55% is broken in a way accuracy metrics cannot detect. It will systematically induce oversizing. **A miscalibrated model is more dangerous than a less accurate but honest one**, because it corrupts the user's risk decisions rather than merely their entries.

### Requirements

| Measure | Purpose |
|---|---|
| **Brier score** | Aggregate probabilistic accuracy; tracked per regime |
| **Reliability curve** | Predicted probability vs observed frequency, bucketed — the diagnostic that shows *where* calibration breaks |
| **Expected calibration error** | Single-number summary for gating |
| **High-confidence bucket check** | Specific attention to 70%+ predictions, since those drive the largest positions |

Calibration is measured **out-of-sample only**, on the walk-forward test windows.

**Gate rule:** calibration must be within tolerance and must not have degraded versus the incumbent. A candidate with better accuracy and worse calibration **fails**.

Where a model is otherwise strong but poorly calibrated, post-hoc calibration (isotonic or Platt scaling, fit on validation data only) may be applied — and when it is, the calibration mapping becomes part of the model artifact and part of the reproducibility manifest.

---

## 5. Drift detection

Waiting for performance degradation to become visible means acting after weeks of bad predictions. Drift detection catches the cause instead of the symptom.

**Feature drift** — the distribution of live features has moved away from the training distribution. Measured by population stability index and Kolmogorov–Smirnov tests per feature, run nightly against the live model's training distribution.

**Concept drift** — the relationship between features and outcomes has changed. Detected as sustained degradation in rolling out-of-sample accuracy or calibration, distinct from ordinary variance.

**Regime-distribution shift** — the market has moved into a regime the live model saw little of. This is the most common form in practice and the most tractable: it explains why a model that worked all year suddenly does not, and it argues for a regime-specific retrain rather than a wholesale one.

Drift does not promote anything. It **triggers a retrain-and-revalidate cycle** and raises a Control Room alert. Detection and promotion remain separate concerns — drift says "look again," the gate says "this is better."

---

## 6. Stress replays

A model validated only on calm markets has been validated on the easy half of the problem.

Before promotion, candidates are replayed against a curated library of difficult historical periods. The daily history reaching back to **January 2000** ([02-data-layer.md](02-data-layer.md)) makes this possible without any additional data sourcing:

| Scenario | Why it is in the library |
|---|---|
| 2008 global financial crisis | Sustained high-volatility collapse |
| COVID crash and recovery | Violent drawdown followed by violent reversal |
| Demonetisation | Domestic policy shock |
| GST implementation | Structural transition |
| Election-cycle volatility | Recurring event-driven turbulence |
| Individual high-volatility days | Gap risk, circuit behavior |
| Low-liquidity periods | Slippage stress |
| Corporate-action days | Adjustment-handling correctness |

Stress replays are evaluated as **loss-limitation, not profit-generation**. The question is not "did it make money in March 2020" — it is "did it fail gracefully, or did it produce confident nonsense?" A model that goes quiet and low-confidence during a crash is behaving correctly. A model that maintains 85% confidence through a collapse is dangerous and must not promote.

Note the honest limitation: stress replays run at **daily resolution** for pre-2022 periods, because intraday history does not exist before January 2022 ([02-data-layer.md](02-data-layer.md)). Since v1 trains on daily features, this does not currently constrain the work.

---

## 7. Shadow mode

The gate and the user are two separate audiences, and a model earns them separately.

```
candidate ──GATE PASS──► shadow ──TRIAL PERIOD──► live (user-facing)
    │                       │                        │
    │                  predicts &                 predicts &
  rejected            gets graded,               reaches the
                      never shown                morning brief
```

**Passing the gate earns promotion to shadow, not to the user.** A shadow model runs the full inference path nightly and its predictions are written to `predictions` flagged as shadow, graded identically to live ones ([05-nightly-pipeline.md](05-nightly-pipeline.md)) — and never surfaced.

The distinction matters because backtest performance and live performance diverge for reasons backtests structurally cannot capture: real data arrival timing, real universe composition, real corporate actions, real edge cases. Shadow mode is the only way to observe a model behaving under genuinely live conditions without any consequence to the user.

Promotion from shadow to user-facing requires a defined trial period of graded live predictions, sustained accuracy and calibration, and no regime regression. This is also the mechanism the Algo Trading gate depends on ([02-data-layer.md](02-data-layer.md)).

---

## 8. Reproducibility

Every model in the registry must be rebuildable from its recorded inputs. A model that cannot be reproduced cannot be debugged, and a system whose failures cannot be investigated cannot be improved.

The manifest stored with each model:

| Field | Why |
|---|---|
| Feature schema version | Feature definitions change; a model means nothing without the semantics it learned ([03-feature-engineering.md](03-feature-engineering.md)) |
| Training window | Exact date range |
| Universe snapshot | Which instruments were active and eligible at training time |
| Hyperparameters | Complete set, not defaults-plus-diffs |
| Random seed | Fixed, recorded |
| Library versions | LightGBM and dependencies |
| Label definition | Horizon and label type |
| Cost model version | Backtest results are meaningless without it |
| Full metrics | Aggregate, per-regime, calibration, per-window, stress results |
| Gate decision and reason | Including rejections |

**Determinism is a hard requirement.** Retraining with an identical manifest must produce an identical model. Nondeterminism means every metric is noise of unknown magnitude, and the gate's comparisons become meaningless.

---

## 9. Failure-mode and chaos expectations

The pipeline runs unattended at night. Every dependency will eventually fail, and the required behavior in every case is: **fail visibly, preserve the incumbent, never leave the system in an undefined state.**

| Failure | Required behavior |
|---|---|
| Upstox timeout / rate limit | Hard fail step 1; incumbent untouched; retry per backoff |
| Corrupt or partial data | Validation quarantines or aborts; nothing downstream runs on it |
| Training OOM or crash | Candidate discarded; incumbent serves; failure surfaced |
| Backtest crash | Candidate unvalidatable → rejected by definition |
| Registry write interrupted | Transactional: fully written or not at all. No half-models. |
| Ollama Cloud quota or outage | `inconclusive` verdicts, visibly flagged, pipeline continues |
| Claude CLI unreachable | Templated fallback brief, marked as fallback |
| Telegram failure | Retry, then retain and surface in Control Room |
| Disk full | Hard fail before any write that could corrupt the store |
| Process killed mid-run | Heartbeat records `interrupted`; next launch detects and offers catch-up |

The invariant across every row: **the live model is never left in a worse state than before the run started.** The worst outcome of any failed night is that yesterday's model keeps running — which is, by construction, a model that already passed the gate.

---

## 10. Rollback

Automated gates are necessary and insufficient. A model can pass every check and still behave in a way the user, watching real market conditions, distrusts. Judgment must be able to override statistics.

**One-click rollback from the Control Room** ([07-control-room.md](07-control-room.md)) to any archived registry version, independent of the gate.

Requirements:

- Every superseded model is retained, not deleted
- Rollback is atomic — the live pointer moves, or nothing happens
- Rollback is recorded with a timestamp and an optional reason
- A rolled-back model is marked `rolled_back` and **will not be automatically re-promoted** by a later night, because a model the user rejected must not silently return
- Predictions made by a rolled-back model retain their grades; history is never rewritten

---

## The gate, assembled

Every criterion must pass. There is no weighted score and no partial credit — weighting invites a marginal candidate to compensate for a real regression with a cosmetic gain.

```
PROMOTE the candidate only if ALL hold:

  ☐  Beats incumbent on primary metric, net of all costs
  ☐  No regime shows regression beyond tolerance
  ☐  Calibration within tolerance and not worse than incumbent
  ☐  Walk-forward performance stable across windows
  ☐  Stress replays show graceful degradation, not confident failure
  ☐  Reproducibility manifest complete and valid
  ☐  Regime classifier confusion matrix within tolerance

ANY failure  →  REJECT · keep incumbent · log the specific reason
```

And the cultural point that makes all of it work:

> **Rejection is the expected outcome. A gate that rarely rejects is not gating.**

If most nights promote, the gate is too loose and the system has quietly reverted to blind retraining — the exact failure this document exists to prevent. Promotion rate is itself a metric worth watching in the Control Room.
