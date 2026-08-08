# 09 — Open Questions

Every item here is genuinely unresolved. Each carries a **recommended default** so implementation is never blocked, and a **resolution criterion** so it does not stay open by inertia.

Anything resolved is moved to the "Closed" section at the bottom with the evidence that closed it, rather than deleted — a decision whose reasoning is lost tends to get relitigated.

---

## OQ-1 — Ollama Cloud free-tier throughput

**Question.** How many candidates can Layer 2 actually investigate per night on the free tier?

**What is known.** [Ollama Cloud](https://docs.ollama.com/cloud) free tier provides roughly one cloud model, with session limits resetting on a multi-hour cycle and weekly caps on top. Concurrency is limited to a single model. Usage is measured as GPU time rather than tokens, so consumption scales with model size and request duration — which means the answer depends on prompt design, not just count.

**What is not known.** The real number of candidates reachable per night at the intended prompt size, and how it varies across the session-reset cycle.

**Recommended default.** Begin at **5 candidates per night**, deliberately conservative. Record actual consumption and duration per candidate. Let the adaptive budget ([04-model-brain.md](04-model-brain.md)) size subsequent nights from measured history rather than the guess.

**Resolution criterion.** Ten completed nights of recorded throughput. At that point the observed ceiling becomes a documented number and the conservative default is replaced.

**If it resolves badly** — if even five candidates cannot complete reliably — the documented escalation order is: shrink the prompt, then use a smaller cloud model, then reduce to three candidates. Upgrading to a paid tier breaks a stated non-goal ([00-overview.md](00-overview.md)) and must be a deliberate decision recorded here, not a quiet drift.

---

## OQ-2 — Pre-2022 intraday history

**Question.** Does the absence of intraday data before January 2022 constrain the system?

**What is known.** Upstox V3 provides daily and above from January 2000, minutes and hours from January 2022 ([02-data-layer.md](02-data-layer.md)). Stress replays covering 2008 and the COVID crash therefore run at daily resolution only ([06-retraining-rigor.md](06-retraining-rigor.md)).

**Recommended default.** **Accept the limitation.** v1 trains on daily-resolution features and predicts over a daily-scale horizon, so the gap does not currently bind. Document it rather than working around it.

**Resolution criterion.** Revisit only if the prediction horizon moves to intraday scale. At that point the question becomes whether a paid historical vendor is worth it — a decision with cost implications that should not be made preemptively.

---

## OQ-3 — Algo Trading gate thresholds

**Question.** Are the specific numeric thresholds in the execution gate correct?

**What is known.** The gate's *structure* is settled ([02-data-layer.md](02-data-layer.md)): shadow track record, sustained accuracy, calibration, net-of-cost edge, regime coverage, per-session opt-in, hard caps. All must pass simultaneously.

**What is not known.** Whether 60 sessions is the right window, and what the accuracy and calibration tolerances should be. These were set by judgment, not by evidence, because no evidence exists yet.

**Recommended default.** Keep the documented thresholds. They are deliberately strict; the failure mode of a too-strict gate is a delayed capability, and of a too-loose one is automated money loss.

**Resolution criterion.** Six months of graded live predictions. Thresholds are then set from the observed distribution of the system's real performance rather than from a guess.

**A standing constraint.** Thresholds may be revised **upward** on evidence at any time. Revising them **downward** requires recording why here — the specific risk this document guards against is a future user quietly lowering the bar to unlock a feature they want today.

---

## OQ-4 — Prediction horizon

**Question.** What forward horizon should the models actually predict?

**What is known.** The horizon is configurable and recorded per prediction, so grading always compares like with like ([03-feature-engineering.md](03-feature-engineering.md)). Changing it breaks comparability across models ([08-operations.md](08-operations.md)).

**What is not known.** Which horizon carries genuine edge. Too short and costs dominate the signal; too long and it stops being actionable for the user's style.

**Recommended default.** Start with a **1-to-5-day horizon**, which matches the nightly cadence — a prediction made tonight should be resolvable within the working week.

**Resolution criterion.** Compare net-of-cost edge across candidate horizons during initial backtesting, before the first live model is promoted. Pick on evidence, then leave it alone.

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
