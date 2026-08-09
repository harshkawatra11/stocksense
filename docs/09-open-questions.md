# 09 — Open Questions

Every item here is genuinely unresolved. Each carries a **recommended default** so implementation is never blocked, and a **resolution criterion** so it does not stay open by inertia.

Anything resolved is moved to the "Closed" section at the bottom with the evidence that closed it, rather than deleted — a decision whose reasoning is lost tends to get relitigated.

---

## OQ-1 — Ollama Cloud free-tier throughput

**Question.** How many candidates can Layer 2 actually investigate per night on the free tier?

**What is known.** [Ollama Cloud](https://docs.ollama.com/cloud) free tier provides roughly one cloud model, with session limits resetting on a multi-hour cycle and weekly caps on top. Concurrency is limited to a single model. Usage is measured as GPU time rather than tokens, so consumption scales with model size and request duration — which means the answer depends on prompt design, not just count.

**What is not known.** The real number of candidates reachable per night at the intended prompt size, and how it varies across the session-reset cycle.

**Direction settled.** The user has confirmed the **free tier is acceptable** and is not to be treated as a blocker. This is no longer a question about whether to pay; it is a measurement task.

**Recommended default.** Begin at **5 candidates per night**, deliberately conservative. Record actual consumption and duration per candidate. Let the adaptive budget ([04-model-brain.md](04-model-brain.md)) size subsequent nights from measured history rather than the guess.

**Resolution criterion.** Ten completed nights of recorded throughput. At that point the observed ceiling becomes a documented number and the conservative default is replaced.

**If throughput proves tight** — the escalation order is: shrink the prompt, then use a smaller cloud model, then reduce the shortlist. Upgrading to a paid tier breaks a stated non-goal ([00-overview.md](00-overview.md)) and would need to be a deliberate, recorded decision rather than a quiet drift.

---

## OQ-2 — Pre-2022 intraday history

**Question.** How should the absence of intraday data before January 2022 be handled now that the system is horizon-agnostic?

**What is known.** Upstox V3 provides daily and above from January 2000, minutes and hours from January 2022 ([02-data-layer.md](02-data-layer.md)). Since horizon is now a parameter rather than a fixed property, both daily and intraday models are first-class — but they have very different historical depth available to them.

**Recommended default.** **Asymmetric confidence, not asymmetric capability.** Build both; claim differently. A daily-horizon model can be stress-replayed against 2008 and COVID and may carry crisis-robustness claims. An intraday-horizon model cannot, and its evaluation reports must say so via the fidelity tier ([10-evaluation.md](10-evaluation.md)).

The practical consequence: **daily-horizon models are the ones that graduate to real capital first**, because they are the only ones the evaluator can test against genuine crisis regimes.

**Resolution criterion.** Revisit if intraday-horizon models demonstrate sustained edge and the missing crisis coverage becomes the binding constraint on promoting them. At that point the question is whether a paid historical vendor is worth it — a cost decision that should not be made preemptively.

---

## OQ-3 — Algo Trading gate thresholds

**Question.** Are the specific numeric thresholds in the execution gate correct?

**What is known.** The gate's *structure* is settled ([02-data-layer.md](02-data-layer.md)): shadow track record, sustained accuracy, calibration, net-of-cost edge, regime coverage, per-session opt-in, hard caps. All must pass simultaneously.

**What is not known.** Whether 60 sessions is the right window, and what the accuracy and calibration tolerances should be. These were set by judgment, not by evidence, because no evidence exists yet.

**Recommended default.** Keep the documented thresholds. They are deliberately strict; the failure mode of a too-strict gate is a delayed capability, and of a too-loose one is automated money loss.

**Resolution criterion.** Six months of graded live predictions. Thresholds are then set from the observed distribution of the system's real performance rather than from a guess.

**A standing constraint.** Thresholds may be revised **upward** on evidence at any time. Revising them **downward** requires recording why here — the specific risk this document guards against is a future user quietly lowering the bar to unlock a feature they want today.

---

## OQ-4 — Which horizons carry edge

**Question.** Which of the supported horizons actually carry a tradeable edge?

**What is settled.** Horizon is a **parameter**, not an architectural choice ([03-feature-engineering.md](03-feature-engineering.md)). The system supports many; each model declares one; each is evaluated separately. This is no longer a decision to make once — it is a search over a space.

**What is not known.** Where the edge actually lives. Too short and costs dominate the signal; too long and it stops matching how the user trades.

**Recommended default.** Begin the search at **1–5 bars at daily resolution**, matching the nightly cadence — a prediction made tonight resolves within the working week. Expand the search once the evaluation harness exists to compare horizons fairly.

**Resolution criterion.** Run the horizon sweep as a first-class evaluation ([10-evaluation.md](10-evaluation.md)): identical features, identical cost model, net-of-cost edge and calibration compared across horizons. This is a permanent capability, not a one-time answer — the horizon that carries edge may itself change with regime.

---

## OQ-5 — Regime boundary definitions

**Question.** Where exactly do trending, sideways, and high-volatility divide?

**What is known.** Three regimes, assigned by deterministic rules over trend-strength and volatility features, at both instrument and market level. The classifier learns to predict those labels. Low-confidence assignments are themselves treated as signal ([04-model-brain.md](04-model-brain.md)).

**What is not known.** The threshold values, and whether three regimes is the right number. Too few and each specialist learns a blend; too many and each trains on too little data.

**Recommended default.** **Three regimes**, with thresholds set from percentiles of each instrument's own history rather than absolute constants — absolute volatility thresholds do not transfer between a large-cap and a small-cap.

**Resolution criterion.** Evaluate label stability and per-regime specialist performance during initial backtesting. A regime whose specialist cannot beat a blended model is a regime that is not carrying its weight.

---

## OQ-6 — Universe scope

**Question.** How many instruments should the system actually track?

**What is known.** Liquidity filters gate tradeability at shortlist time ([04-model-brain.md](04-model-brain.md)). Universe correctness is a training-data integrity issue, not a convenience ([02-data-layer.md](02-data-layer.md)).

**What is not known.** Whether to train across the full active NSE equity universe or a liquidity-filtered subset. A wider universe gives more training data; it also gives more illiquid noise that will never be traded.

**Recommended default.** **Train on a liquidity-filtered universe** — instruments meeting a minimum median turnover — while retaining full history in the store. Training on instruments the system would never recommend spends model capacity on irrelevant patterns.

**Resolution criterion.** Compare model performance trained on filtered versus full universes during initial backtesting.

---

## OQ-7 — Cost model accuracy

**Question.** Do the modeled transaction costs match reality?

**What is known.** The components are enumerated ([06-retraining-rigor.md](06-retraining-rigor.md)): brokerage, STT, exchange charges, SEBI fees, stamp duty, GST, and slippage. The gate requires a net-of-cost win.

**What is not known.** Whether the modeled numbers match the user's actual contract notes, and — more importantly — whether the slippage model is realistic. Slippage is the component most often underestimated and the one that most reliably decides whether a marginal edge survives.

**Recommended default.** Model explicit charges from the user's actual Upstox plan. Model slippage as a function of position size relative to instrument liquidity and current volatility regime, **not** as a constant.

**Resolution criterion.** Reconcile modeled costs against real contract notes across at least twenty of the user's actual trades. Until that reconciliation happens, treat every backtest's net-of-cost figure as provisional.

---

## OQ-8 — Shadow trial length

**Question.** How long must a model run in shadow before becoming user-facing?

**What is known.** Shadow mode is architecturally distinct from gate promotion ([06-retraining-rigor.md](06-retraining-rigor.md)). Shadow predictions are graded identically to live ones and never surfaced.

**What is not known.** The trial length. Too short and it proves nothing; too long and good models are withheld while the user acts on worse ones.

**Recommended default.** **20 trading sessions**, roughly one month — long enough to accumulate meaningful graded predictions, short enough that improvements reach the user within a reasonable window.

**Resolution criterion.** After several models have completed shadow trials, check whether shadow performance predicted live performance. If it did not, the trial is too short or measuring the wrong thing.

---

## OQ-9 — Cross-source reconciliation tolerance

**Question.** How far can Upstox, NSE archives, and yfinance disagree before a field is quarantined?

**What is known.** Precedence is settled ([02-data-layer.md](02-data-layer.md)): NSE archives are authoritative for delivery and corporate actions, Upstox for OHLCV and F&O, yfinance never overrides either. Provenance is recorded per field. Disagreement beyond tolerance quarantines rather than silently picking a winner.

**What is not known.** The tolerance. Too tight and every night drowns in false quarantines from rounding and timing differences; too loose and genuine corruption — an unapplied split, a symbol collision — passes through as data.

**Recommended default.** Start **tight and noisy**: a small relative tolerance on prices, exact match on volumes and delivery. Log every disagreement without acting on it for the first weeks, then set the threshold from the observed distribution of real disagreements.

**Resolution criterion.** One month of logged reconciliation results. The tolerance is placed above the noise floor and below the smallest disagreement that turned out to be a real error.

**Note.** The disagreement log is worth keeping permanently. A source that develops a systematic bias will show up there long before it shows up in model performance.

---

## OQ-10 — Episode library composition

**Question.** How many episodes, and in what mix?

**What is known.** Episodes are frozen information snapshots with revealed outcomes, spanning eight classes from open-ended to trap setups ([10-evaluation.md](10-evaluation.md)). Trap episodes are deliberately over-represented relative to natural frequency, because a randomly sampled library is dominated by unremarkable days that do not discriminate between a good brain and a mediocre one.

**What is not known.** Total size, per-class proportions, and how much over-representation of hard cases is useful before the library stops resembling the market at all.

**Recommended default.** Build **breadth before volume** — every class populated across every era, a few hundred episodes total, before scaling any single class into the thousands. A large library that is 90% ordinary days is a slow way to learn nothing.

**Resolution criterion.** Measure discriminative power: an episode class where every candidate model scores identically is not testing anything and should be re-weighted or replaced.

---

## OQ-11 — Guarding against evaluator overfitting

**Question.** How is the holdout protected from being consumed by repeated tuning?

**What is known.** This is the deepest risk in the whole evaluation design ([10-evaluation.md](10-evaluation.md)). Every time a model is tuned and re-tested against the same holdout, some of that holdout leaks into the design process — not through code, but through the researcher's decisions. Enough iterations and the evaluation becomes a training set with extra steps.

**Mitigations already specified.** A locked final-validation era used once per major version; evaluation-attempt counting with results discounted as the count rises; rotating episode subsets; and the paper/shadow stages as an external check on data that did not exist when the model was built.

**What is not known.** How many attempts against one holdout is too many, and how steeply results should be discounted as the count climbs.

**Recommended default.** Surface the attempt count prominently in the Control Room ([07-control-room.md](07-control-room.md)) and treat **any holdout with more than a few dozen attempts as compromised** — retire it and cut a fresh one from unused history.

**Resolution criterion.** Compare holdout performance against subsequent paper/shadow performance across several models. A widening gap that correlates with attempt count is direct evidence of holdout decay, and it sets the real threshold.

**Standing note.** This is the one open question that never fully closes. It is a discipline to be maintained, not a problem to be solved once.

---

## Closed

### ✅ Historical data depth — closed 2026-08-08

**Was:** Does the system need NSE bhavcopy backfill for pre-2015 history?

**Closed by:** The [Upstox V3 Historical Candle API](https://upstox.com/developer/api-documentation/announcements/enhanced-historical-candle-data-apis-v3/) provides daily, weekly, and monthly data **from January 2000** and intraday **from January 2022**.

**Decision:** No bhavcopy backfill. Twenty-six years of daily history covers the dot-com aftermath, the 2008 crisis, demonetisation, GST, COVID, and the post-COVID rate cycle — sufficient for regime research at daily resolution. The residual intraday gap is tracked as OQ-2.

**Implementation note:** ingestion must target **V3**. V2 caps daily history at roughly one year, and an implementation that silently used it would train on a fraction of the intended history while appearing to work correctly.

### ✅ Layer 2 model hosting — closed 2026-08-08

**Was:** Can a 4GB-VRAM GPU host a model capable of the investigation layer's reasoning?

**Closed by:** [Ollama Cloud](https://docs.ollama.com/cloud). The local `ollama` CLI runs `-cloud`-suffixed models on Ollama's GPUs, downloading only a small manifest. Ollama v0.32.6 is already installed on the target machine, with no local models pulled and no running instance.

**Decision:** Layer 2 runs on Ollama Cloud, free tier, with an adaptive shortlist budget. The VRAM ceiling no longer constrains model choice.

**Consequences accepted and documented:** inference data leaves the machine, so "offline desktop app" means *no self-hosted server*, not *airgapped* ([00-overview.md](00-overview.md)). Free-tier throughput remains open as OQ-1.

### ✅ Storage technology — closed 2026-08-08

**Was:** PostgreSQL or DuckDB/Parquet?

**Decision:** **DuckDB with Parquet.**

**Reasoning:** PostgreSQL is a server — it must be installed, started, supervised, and handled as a startup failure state, all in exchange for concurrency a single-user desktop app does not need. DuckDB is a library: the runtime stays at two processes, and Python simply opens a file. Its columnar engine also matches the actual query shape, since retraining scans decades of history reading few columns at a time.

**Accepted trade-off:** DuckDB permits a single writer. Survivable because every write already routes through the one Python backend and the pipeline is sequential by design. Concurrent writers would force a revisit.
