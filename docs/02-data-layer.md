# 02 — Data Layer

## Three sources, one store

StockSense ingests from three sources. They are complementary, not redundant, and each is authoritative for something the others cannot provide.

| Source | Authoritative for | Reach |
|---|---|---|
| **Upstox API** | The operational spine — live data, F&O, and the **user's own orders and positions** (no other source has these) | Daily from Jan 2000, intraday from Jan 2022 |
| **NSE archives** | **Delivery quantity and delivery percentage**, bulk/block deals, official corporate actions, point-in-time universe composition | Bhavcopy archives predate Upstox's window |
| **yfinance** | Independent cross-check and gap-fill | Long daily history, convenient access |

The reason for three rather than one: **delivery percentage does not exist in the Upstox API**, and it is one of the more informative features available on Indian equities — it separates positional accumulation from intraday churn, which price and volume alone cannot distinguish. NSE archives are the only practical source. yfinance earns its place as an independent voice for reconciliation: a single source cannot tell you when it is wrong.

### Source precedence and reconciliation

When sources disagree — and they will — precedence decides, and disagreement beyond tolerance quarantines rather than silently picking a winner.

```
For a given instrument-date field:

  1. NSE archives      ← authoritative for delivery, corporate actions,
                          official settlement prices
  2. Upstox            ← authoritative for OHLCV, F&O, and anything
                          the live system will also read
  3. yfinance          ← fallback only; never overrides 1 or 2

  Cross-check: compare all available sources
     ├─ agreement within tolerance  →  accept, record provenance
     └─ disagreement beyond tolerance  →  QUARANTINE, surface in Control Room
```

Rules that make this trustworthy rather than merely layered:

- **Provenance is recorded per field**, not per row. A row whose close came from Upstox and whose delivery percentage came from NSE archives says exactly that. Silent blending of sources is how untraceable data corruption begins.
- **yfinance never overrides a primary source.** It is a validator and a gap-filler. It carries known survivorship bias and documented quality issues, and treating it as authoritative would import both.
- **Disagreement is signal.** A close that differs by more than a rounding tolerance across sources usually means an unapplied corporate action, a symbol collision, or a stale universe entry — all of which are real problems worth catching, not noise worth averaging away.
- **Reconciliation results are logged**, so a persistent per-source bias becomes visible rather than being quietly absorbed each night.

### The point-in-time obligation

NSE archives are also what make honest historical evaluation possible. The evaluator must reconstruct the universe *as it existed* on a past date, including instruments that have since delisted ([10-evaluation.md](10-evaluation.md)). A universe assembled from today's listings would silently exclude every company that failed — the definition of survivorship bias, and one of the most effective ways to make a backtest look brilliant and mean nothing.

---

## Upstox integration

Broker and live market data comes from Upstox via the [official Python SDK](https://github.com/upstox/upstox-python). Ingestion is the only component permitted to call it ([01-architecture.md](01-architecture.md)); everything downstream reads the local store.

### The three API surfaces

The user holds access to three distinct Upstox surfaces. They are not interchangeable and the docs treat them as separate capabilities with separate roles.

**Analytics** — read-only market and account data. This is the workhorse: historical candles, F&O data, quotes, and the user's own orders, positions, holdings, and funds. Every nightly ingestion runs against this surface. It cannot place an order, which makes it the correct default for a system whose v1 explicitly does not trade.

**Sandbox** — a paper environment mirroring the live API. **All development and QA of trading logic happens here first.** The value is that Sandbox exercises the real API shapes — same request/response contracts, same error semantics — without any real-money consequence. Promotion from Sandbox to live is a deliberate, logged, user-initiated action in the Control Room, never an automatic or configuration-drift event. Sandbox is the default mode on a fresh install.

**Algo Trading** — programmatic order placement. **Present but disabled by default in v1.** StockSense v1 produces recommendations; the user places orders manually. This surface exists in the account and will exist in the codebase as a gated path, but it cannot fire.

### The Algo Trading gate

"Later" is not a specification. If automated execution is ever enabled, it is enabled because a measurable condition was met, and the condition is recorded here so that future-user cannot quietly lower the bar for present-user's convenience.

The gate requires **all** of the following, simultaneously:

| Condition | Threshold |
|---|---|
| Shadow-mode track record | ≥ 60 consecutive trading sessions of graded predictions from the live model |
| Directional accuracy | Sustained above the documented baseline over that window, measured on out-of-sample days only |
| Calibration | Brier score and reliability curve within tolerance — an 80% call must be right about 80% of the time ([06-retraining-rigor.md](06-retraining-rigor.md)) |
| Net-of-cost edge | Positive after realistic brokerage, slippage, and STT — a pre-cost edge does not count |
| Regime coverage | The window must include at least one high-volatility regime period. A model proven only in calm markets has not been proven. |
| Explicit opt-in | Per-session, in the Control Room. Not a persisted setting. It expires. |
| Hard caps | Position size ceiling, daily loss limit, and maximum open positions, all enforced in code before any order is constructed |

Any single failure keeps the gate shut. The gate is evaluated by the system and displayed in the Control Room, so its status is a fact rather than a feeling.

### Historical depth — resolved

This was an open question in the original planning; it is now settled by the [Upstox V3 Historical Candle API](https://upstox.com/developer/api-documentation/announcements/enhanced-historical-candle-data-apis-v3/).

| Interval | Available from |
|---|---|
| Days, weeks, months | **January 2000** |
| Minutes, hours | **January 2022** |

Twenty-six years of daily history covers the dot-com aftermath, the 2008 global financial crisis, demonetisation, GST implementation, the COVID crash and recovery, and the post-COVID rate cycle. **NSE archives are not needed to backfill daily OHLCV** — Upstox covers it. Archives are ingested for what Upstox does not provide: delivery data, official corporate actions, point-in-time universe composition, and reach before 2000.

The honest remaining gap is **pre-2022 intraday**. Minute and hour candles do not exist before January 2022 through this API, which means intraday-resolution work cannot reach the 2008 or COVID crashes — those eras are available at daily resolution only.

Because the system is **horizon-agnostic** ([03-feature-engineering.md](03-feature-engineering.md)), this shapes what can be *claimed* rather than what can be *built*. Daily-horizon models train across the full 2000→ history; intraday-horizon models train from 2022→ and are evaluated at a correspondingly lower fidelity tier ([10-evaluation.md](10-evaluation.md)). Every evaluation report states which resolution it ran at, so a conclusion is never quoted with more historical depth than it actually had.

Note also that V2 and V3 differ substantially in depth (V2 caps daily at roughly one year). **Ingestion must target V3.** An implementation that silently uses V2 would train the models on a fraction of the intended history while appearing to work.

### Rate limits and ingestion etiquette

Upstox enforces per-endpoint rate limits. The initial historical backfill — thousands of instruments across twenty-six years — is by far the heaviest thing this system ever does to the API, and it is a one-time operation.

Requirements:

- The backfill is **resumable and checkpointed**. It records per-instrument progress so an interruption resumes rather than restarts.
- It is **rate-limit aware**, with backoff on 429 responses rather than retry-hammering.
- It is **separate from the nightly job**. Nightly ingestion fetches one incremental day and must never accidentally trigger a full refetch.
- Nightly ingestion is **idempotent**: running it twice for the same date produces the same store state, not duplicated rows ([05-nightly-pipeline.md](05-nightly-pipeline.md)).

### The Upstox skill plugin is a build-time aid

The Claude Code Upstox plugin:

```
/plugin marketplace add upstox/upstox-plugin-marketplace
/plugin install upstox-skill@upstox-plugins-official
```

This is **development tooling, not a runtime component.** It teaches an AI coding assistant how to call the Upstox API correctly — auth flow, endpoint shapes, parameter semantics, rate limits. It ships in nobody's Electron bundle and it executes nothing at 2am.

The distinction it embodies is worth stating directly, because conflating the two would be an architectural error:

> **Broker skill knowledge ≠ market knowledge.**

The skill plugin knows how to *talk to the broker*. The trained models know what the *market has done*. The former is plumbing that could be swapped for a different broker without touching a single model; the latter is the entire point of the system. Nothing the plugin knows ever influences a prediction.

## Storage: DuckDB + Parquet

### Why this and not PostgreSQL

The decision hinges on the single-user desktop constraint.

PostgreSQL is a server. Choosing it means the user installs and runs a database server on their laptop, Electron must detect whether it is running, the app must handle "Postgres isn't up" as a first-class startup state, and packaging the application for a clean machine means shipping or scripting a server install. All of that is ops burden in exchange for concurrency guarantees a single-user app does not need.

DuckDB is a library. Python opens a file. The runtime topology stays at exactly two processes ([01-architecture.md](01-architecture.md)), and there is nothing to install, start, supervise, or firewall.

On top of the ops argument, DuckDB is columnar and vectorized, which matches the actual query shape here. Retraining scans twenty-six years across thousands of instruments and reads a handful of columns at a time. That is precisely the workload column stores are built for and row stores are not.

The trade-off, stated honestly: DuckDB permits one writer. That is a real limitation and it is survivable here only because the architecture already routes every write through the single Python backend, and the nightly pipeline is sequential by design. If a future version needed concurrent writers, this decision would need revisiting.

### Layout

```
data/
  stocksense.duckdb          ← catalog, metadata, all non-bulk tables
  parquet/
    candles/
      interval=day/
        year=2000/…
        year=2026/…
      interval=1min/
        year=2022/…
    fno/
      year=2022/…
  models/
    <model_id>/
      model.txt              ← serialized LightGBM booster
      manifest.json          ← reproducibility record
```

Bulk time-series lives in partitioned Parquet, queried by DuckDB in place. Partitioning by interval and year means a training run that wants daily candles for 2008 reads one partition instead of scanning everything. Everything else — the transactional, frequently-updated, relationally-shaped data — lives in DuckDB tables.

## Entities

These are the entities that must exist. Column-level schema is an implementation decision; the entities and their invariants are not.

### `instruments` — the tradeable universe

Every NSE instrument with its Upstox instrument key, symbol, name, series (EQ/BE/etc.), sector, index memberships, listing and delisting dates, and an active flag.

**This table is a correctness dependency, not a convenience lookup.** A stale universe silently poisons training: delisted tickers contribute phantom history, renamed symbols fragment one company's series into two partial ones, and unnamed rows surface in briefs as opaque codes. A prior incarnation of this project carried 2,119 delisted or suspended tickers in its active universe before they were found and deactivated. The universe must be refreshed on a defined cadence and validated every night.

### `candles` — OHLCV

Open, high, low, close, volume, per instrument, per interval, per timestamp. Daily from 2000; intraday from 2022. Stored in Parquet.

**Adjusted and unadjusted closes are stored separately.** Training features must use corporate-action-adjusted prices, because an unadjusted 1:5 split looks exactly like an 80% crash to a momentum feature. Displaying a price to the user, by contrast, should show what actually printed. Conflating these two is a classic and quiet source of garbage training data.

### `delivery` — settlement-side activity

Per instrument, per day, from NSE archives: deliverable quantity, delivery percentage, and total traded quantity. Bulk and block deal records where present.

This entity exists because delivery percentage answers a question volume cannot: **how much of today's trading was people actually taking ownership, versus intraday churn that netted out by the close?** A 5% up-move on high volume and 20% delivery is a different event from the same move on the same volume at 70% delivery — the first is traders, the second is accumulation.

Unavailable through the Upstox API, which is a principal reason NSE archives are ingested at all.

### `fno_snapshots` — derivatives

Per underlying, per expiry, per day: futures price and volume, open interest and OI change, option chain data, put/call ratio, implied volatility, and futures basis. Feeds the F&O feature category in [03-feature-engineering.md](03-feature-engineering.md).

### `corporate_actions`

Splits, bonuses, dividends, symbol changes, mergers, delistings — with ex-dates and ratios. This is what makes the adjusted close column correct, and what makes symbol-change handling possible rather than guesswork.

### `trade_ledger` — reconstructed trades

Output of Trade Reconstruction. Each row is a *whole trade*, not a fill: instrument, session, ordered sequence of entries and exits with quantities and prices, realized P&L, holding duration, maximum favorable excursion, maximum adverse excursion, and the decision grade assigned by Track B.

This is the only table that describes the user rather than the market. It is read by the Decision Grader and by the Investigator (for "has this setup burned me before?" context). **It is never read by the Trainer.** That prohibition is the Track A / Track B invariant from [00-overview.md](00-overview.md) expressed as a data-access rule.

### `predictions` — every prediction ever made

Timestamp, instrument, the model version that produced it, the regime it was classified into, predicted direction, predicted probability, predicted magnitude, target horizon — and, once reality arrives, the actual outcome and the grades derived from it.

Predictions are **immutable once written**. Grading adds outcome columns; it never edits the original prediction. A system that can revise its own past predictions cannot be audited, and calibration measurement ([06-retraining-rigor.md](06-retraining-rigor.md)) depends entirely on this table being an honest historical record.

Shadow-mode predictions live here too, flagged, so that a model can accumulate a track record without its output ever reaching the user.

### `model_registry`

One row per trained model version: model ID, type (regime classifier or a named specialist), training window, feature schema version, hyperparameters, random seed, all validation metrics including regime-stratified breakdowns, the promote-or-reject decision with its reason, and lifecycle state (`candidate`, `shadow`, `live`, `archived`, `rolled_back`).

The Gate is the only writer. Nothing else promotes a model, and no promotion happens without a row explaining why.

### `job_runs` — the heartbeat

One row per pipeline execution, plus one per step: start and end timestamps, status (`completed` / `failed` / `aborted` / `interrupted`), rows processed, and error detail where relevant.

This table answers the question the user asks most often in practice — *did last night actually run, and did it run properly?* — and it is what missed-cycle detection reads on launch. The four statuses are deliberately distinct: a retrain that rejected its candidate and an app that was closed mid-retrain must never be indistinguishable the next morning.

### `news_events`

Ingested announcements, earnings, corporate events, and macro items, with instrument or sector linkage, timestamp, and source. Consumed by the Investigator ([04-model-brain.md](04-model-brain.md)) rather than fed as raw numeric features into LightGBM.

## Validation

Ingestion is followed by validation, and validation has the authority to fail the night. Every check below runs before any downstream component touches the new data.

| Check | Failure means |
|---|---|
| Expected trading day | Data pulled for a holiday or weekend — likely a scheduling bug |
| Candle completeness | Missing sessions for instruments that should have traded |
| OHLC sanity | `high ≥ max(open, close)`, `low ≤ min(open, close)`, non-negative volume |
| Price continuity | Overnight gaps beyond a threshold with no corresponding corporate action — usually an unapplied adjustment, not a real move |
| Universe freshness | Instruments in the feed absent from `instruments`, or active instruments absent from the feed |
| Corporate action application | Every action with an ex-date on or before today reflected in adjusted prices |
| F&O linkage | Every derivative snapshot resolves to a known underlying |
| Duplicate detection | Re-running ingestion for a date produced no duplicate rows |
| **Cross-source agreement** | Sources disagree beyond tolerance on a shared field — usually an unapplied corporate action, a symbol collision, or a stale universe entry |
| **Provenance completeness** | Every field carries a recorded source; no field of unknown origin enters the store |
| **Delivery sanity** | Deliverable quantity never exceeds traded quantity |

Anything that fails is quarantined and surfaced in the Control Room rather than silently dropped. The governing principle: **it is always better to skip a night loudly than to train on corrupted data quietly.** A missed night costs one brief. A night trained on bad data costs however long it takes to notice.
