# Orchestration Audit — 2026-07-13

Read-only audit of the full intended IST trading day: scheduler job registry, backend
lifespan tasks, gating, misfire behavior, process topology, failure surfacing, and
single-machine resource contention. Line numbers are as of this read (parallel agents
are actively editing `scheduler/market_runner.py` and `intelligence/live_confirmation.py`;
expect small drift).

Severity scale: **HIGH** = can silently lose money/protection or silently stop the system;
**MEDIUM** = degraded correctness or visibility; **LOW** = cosmetic/doc drift.

---

## 1. The intended timeline — verified

All jobs exist in `build_scheduler()` (`scheduler/market_runner.py`), timezone
`Asia/Kolkata`, all with `replace_existing=True, coalesce=True, max_instances=1,
misfire_grace_time=300` (market_runner.py:495-497 region, `common` dict).

| Time (IST) | Job id | Cron | Gated market-days? | Evidence |
|---|---|---|---|---|
| 08:00 daily | `ticker_sync` | `hour=8, minute=0` | **No — runs Sat/Sun too** (harmless) | market_runner.py:503-504 |
| 08:45 Mon–Fri | `data_freshness` | `mon-fri 8:45` | Yes | market_runner.py:510-511 |
| 09:15–15:45 Mon–Fri | `signal_pipeline` | `mon-fri hour=9-15 minute=15,45` | Yes | market_runner.py:554-555 |
| 09:25–15:55 Mon–Fri | `position_review` | `mon-fri hour=9-15 minute=25,55` | Yes | market_runner.py:562-563 |
| 09:05–16:05 Mon–Fri | `refresh_weights` | `mon-fri hour=9-16 minute=5` | Yes | market_runner.py:598-599 |
| continuous | intraday stop monitor | asyncio loop, 7s poll — **backend process, not a scheduler job** | not day-gated (runs 24/7; harmless off-hours since quotes stop) | backend/main.py:78, intelligence/intraday_stops.py:38,98-110 |
| every 10 min 09:00–18:59 | `expire_confirmations` | `mon-fri hour=9-18 */10` | Yes + self-gates on `LIVE_CONFIRMATION_ENABLED` | market_runner.py:622-623, task at 381-401 region |
| every 5 min 09:00–18:59 | `reconcile_sandbox_orders` | `mon-fri hour=9-18 */5` | Yes + self-gates | market_runner.py:629-630 |
| 18:30 Mon–Fri | `incremental_ohlcv` | `mon-fri 18:30` | Yes | market_runner.py:531-532 |
| 18:35 Mon–Fri | `upstox_bhavcopy_reconciliation` | `mon-fri 18:35` | Yes | market_runner.py:540-541 |
| 18:45 Mon–Fri | `incremental_fo` | `mon-fri 18:45` | Yes | market_runner.py:547-548 |
| 18:50 Mon–Fri | `eod_review` | `mon-fri 18:50` | Yes | market_runner.py:577-578 |
| 20:00 Mon–Fri | `accuracy_tracker` (+auto-retrain) | `mon-fri 20:00` | Yes | market_runner.py:605-606 |
| Sat 09:00 | `weekend_review` | `sat 9:00` | Sat only | market_runner.py:612-613 |

Retired (commented out, correctly): `groww_intraday` (fake flat candles,
market_runner.py:524-525) and nightly `calibration` (moved into the Saturday flow,
market_runner.py:591-592).

**Ordering is coherent.** Nothing consumes today's close before 18:30:
- `eod_review` was moved from 15:45 to 18:50 precisely for this (comment at the
  eod_review registration, market_runner.py:565-576 region), and
  `eod_review.fetch_actual_closes` now uses a STRICT same-day match — if today's
  bhavcopy row is missing it omits the ticker rather than fabricating a 0.0% return
  (intelligence/eod_review.py:40-66).
- `signal_pipeline` and `queue_fresh_signals` intentionally run on prior-day EOD data
  plus the live quote layer; no same-day-close dependency.

Findings on the timeline itself:

- **MEDIUM — 5-minute gap between incremental_ohlcv (18:30) and reconciliation (18:35),
  20 min before eod_review (18:50).** `task_incremental_ohlcv` walks up to 4 fallback
  sources (Groww → new bhavcopy → old bhavcopy → Angel One, market_runner.py:115-160)
  and can plausibly exceed 5–20 min on a slow night. Downstream jobs degrade *safely*
  (reconciliation returns "no ohlcv_daily rows for today yet — skipping",
  market_runner.py:230-231; eod_review omits unresolved tickers) but a slow bhavcopy
  night silently produces an empty EOD review with no retry that evening.
  *Fix: chain eod_review/reconciliation off incremental_ohlcv completion (or re-check
  at 19:30) instead of fixed clock offsets.*
- **LOW — `ticker_sync` is not day-gated** (runs weekends). Harmless but inconsistent
  with the stated "market days only" intent. *Fix: add `day_of_week="mon-fri"` or leave
  and document.*
- **LOW — no NSE-holiday calendar anywhere.** Every "mon-fri" job fires on exchange
  holidays; pipeline runs on stale data and auto_trade may act on it. *Fix: add a
  holiday-calendar guard helper checked at the top of market-hour tasks.*

## 2. Process topology

Three (four) processes:

- **BACKEND (uvicorn `backend.main:app`)** — lifespan starts (a) the Upstox quote-cache
  feed for indices + held tickers (backend/main.py:72-73 →
  backend/services/quote_cache.py:87-114) and (b) the fast intraday stop/target monitor
  (backend/main.py:78 → intelligence/intraday_stops.py:98). Also serves all APIs incl.
  `/api/confirmations` approve/reject (order placement lives ONLY here).
- **SCHEDULER (`python -m scheduler.market_runner`)** — everything in the table above,
  including signal generation, auto paper-trade/exit, confirmation queueing
  (market_runner.py:316-317, 340-341), data jobs, EOD/weekend reviews, retrain.
- **FRONTEND** (vite dev server) and **DB** (Docker).

`start.ps1` starts all four: Docker db → Ollama check (check only, does NOT start
Ollama) → backend in a new PowerShell window → scheduler in a new window (skippable via
`-NoScheduler`) → frontend (start.ps1:22-83).

If only the BACKEND runs: no signals, no trades, no confirmation queueing, no data
updates, no EOD review — but live quotes and intraday stop enforcement for
*already-held* positions continue. If only the SCHEDULER runs: full pipeline and paper
trading work (DB-driven), but **no live quote cache and therefore no fast intraday stop
enforcement** — stops degrade to the 30-min `position_review` cadence — and no UI/API.

- **MEDIUM — Windows-login autostart is a stale claim.** README.md:447 says
  "`start_stocksense.ps1` … is registered to run at Windows login", but that script now
  lives in `archive\start_stocksense.ps1` (its own line 3 says "see deploy notes") and
  no registration mechanism (schtasks/Startup shortcut/Register-ScheduledTask) exists
  anywhere in the repo. If a login task still points at the archived script it launches
  an outdated stack; if not, nothing autostarts. *Fix: re-register a login task pointing
  at `start.ps1` (or delete the README claim).*

## 3. Sleep/wake Monday 09:40

All jobs share `misfire_grace_time=300` (5 min) with `coalesce=True`
(market_runner.py `common` dict). APScheduler on wake skips any run missed by >5 min.
So waking at 09:40 Monday:

- **Skipped:** 08:00 `ticker_sync`, 08:45 `data_freshness`, 09:15 `signal_pipeline`,
  09:25 `position_review`, 09:05 `refresh_weights`.
- **Next to fire:** `signal_pipeline` 09:45, `position_review` 09:55.

Self-healing assessment:

- Skipping `signal_pipeline` 09:15 is benign (next slot 30 min later).
- **MEDIUM — skipping `data_freshness` removes the self-heal exactly when it's needed.**
  `task_data_freshness` (market_runner.py:163-178) is the only automatic deep-backfill:
  if Friday-evening `incremental_ohlcv` was missed too (machine off), Monday's entire
  trading day runs on stale data with no backfill until Monday 18:30. The Brain tab
  shows freshness, but nothing acts on it. *Fix: give `data_freshness` a large
  `misfire_grace_time` (e.g. 4h) so a late wake still runs it once.*
- **LOW — after multi-day downtime, `incremental_ohlcv` default `days_back=5` may not
  cover the gap;** `data_freshness` compensates (backfills `age_days+3`) but only if it
  actually fires (see above). *Fix: same as above.*

## 4. Failure surfacing

**Scheduler death:** every job wraps in `instrumented()` writing `job_runs`
(market_runner.py:64-108). Readers: Brain tab via `/api/brain/status` (latest run per
job, backend/routers/brain.py:52-62) and `/api/system/health`'s `_scheduler_heartbeat`
(backend/routers/system_health.py:64-100: >2h stale during market hours = degraded,
>26h = down; feeds the frontend SystemHealthBar).

- **HIGH — surfacing is pull-only; nothing alerts.** If the scheduler dies at 11:00,
  the user finds out only if they happen to look at the UI within the day. No Telegram
  ping, no toast, no sound — the exact "invisible for weeks (mid-June incident)"
  failure mode the heartbeat comment cites (system_health.py:73-74) can recur for
  anyone not watching the tab. *Fix: once `backend/services/telegram_bot.py` lands,
  add a backend-side watchdog that Telegram-pings when `_scheduler_heartbeat` flips to
  degraded/down.*

**Upstox WS feed death:**

- `start_quote_cache_feed` catches any exception from `start_feed` with a log line and
  **returns permanently** — no retry until app restart (quote_cache.py:111-114; same
  for the ImportError path at 101-109).
- Consumers degrade silently: `intraday_stops._check_once` skips any position with no
  quote at **debug** log level (intraday_stops.py:53-55) — i.e. with the feed dead,
  fast stop protection is OFF and the only evidence is an invisible debug line.
- **HIGH — a dead feed silently disables intraday stop enforcement with no user-visible
  signal beyond the frontend PriceAgeDot going red.** `/api/system/health` has no
  quote-feed-age component (system_health.py:103-130 rolls up components, freshness,
  Angel One breaker, model age — not `quote_cache.age_seconds`). *Fix: add a
  `quote_feed` component to system_health using `quote_cache.age_seconds` on the
  indices, and warn at INFO/WARNING in intraday_stops when N consecutive cycles have
  no quotes during market hours.*
- **HIGH — the live watchlist is built once at startup** (backend/main.py:29-50,
  called at 72): indices + positions held *at boot*. A position bought at 10:30 is
  never subscribed, so the fast stop monitor can never see its LTP and silently skips
  it all day (intraday_stops.py:53-55); protection falls back to the 30-min slow loop.
  The code comment admits the narrowness (main.py:68-72). *Fix: re-resolve held
  tickers periodically (or on BUY) and re-subscribe the feed.*

## 5. Single-machine contention (16GB RAM / 4GB GPU)

- **MEDIUM — signal_pipeline (:15/:45) vs position_review (:25/:55): 10-min offset is
  an assumption, not a guarantee.** `max_instances=1` is per-job only. A full
  ~2,475-ticker `run_pipeline_multi` with qwen2.5:3b macro calls (memory: needs 300s
  Ollama timeout + warm-up on this GPU) plus Claude CLI synthesis can easily exceed
  10 minutes, at which point `position_review` starts mid-pipeline and both hit Ollama
  concurrently on a GPU that fits one model. The offset comment even cites Kronos
  (retired) as its rationale (market_runner.py:557-560). Worse: `refresh_weights`
  (:05) + `expire_confirmations` (*/10) + `reconcile_sandbox_orders` (*/5) also
  interleave, though those are cheap DB/HTTP jobs. *Fix: an asyncio semaphore/lock
  shared by signal_pipeline and position_review (skip-or-queue), or gate
  position_review on signal_pipeline not running.*
- **LOW — evening chain 18:30→18:35→18:45→18:50→20:00 is well-spaced.** eod_review's
  Claude CLI call (18:50) is CPU/API, the 20:00 auto-retrain
  (intelligence/accuracy_tracker.py:294-296 → `check_and_trigger_retrain`) is
  CPU-bound LightGBM — 70 min apart, and the market pipeline is idle. Saturday 09:00
  weekend_review runs alone. No realistic collision here.
- **LOW — backend's 7-second intraday stop loop opens a fresh asyncpg connection every
  poll** (intraday_stops.py:67) — ~12k connections/day. Works, but wasteful and a
  contention amplifier under load. *Fix: reuse a connection/pool.*

## 6. Doc/label drift (LOW, batched)

- `backend/routers/brain.py:33-34` — JOB_CADENCE still says `eod_review: 15:45` and
  lists retired `calibration: 16:15`; actual is 18:50 / retired. The Brain tab shows
  these labels. *Fix: update the dict (and add the two confirmation jobs).*
- `intelligence/eod_review.py:2` — module docstring still says "Runs at 15:45 IST".
- `start.ps1:59` — hint says `ollama pull qwen2.5:7b`; the configured/feasible model on
  this GPU is `qwen2.5:3b`.
- `intelligence/intraday_stops.py:20-22` — "Paper-only… order execution doesn't exist
  yet" is now stale: sandbox execution exists via confirmations; fast-loop exits feed
  `queue_exit_confirmations` only via the slow loop's reviews, so a fast intraday stop
  breach exits PAPER but does **not** queue a sandbox SELL confirmation. Worth an
  explicit decision (arguably MEDIUM once sandbox positions are real).

---

## Ranked summary

1. **HIGH** Dead quote feed silently disables fast stop enforcement; no retry, no
   health component, debug-only logs (quote_cache.py:111-114, intraday_stops.py:53-55).
2. **HIGH** Live watchlist frozen at backend boot — intraday buys never get live
   quotes or fast stops (backend/main.py:29-50,72).
3. **HIGH** Scheduler-death detection is pull-only; no alerting despite prior
   weeks-long invisible outage (system_health.py:64-100).
4. **MEDIUM** 5-min misfire grace skips `data_freshness` on a late Monday wake —
   the one self-heal that matters most (market_runner.py common dict, 163-178).
5. **MEDIUM** signal_pipeline/position_review 10-min offset can collide on
   Ollama/GPU when the pipeline runs long (market_runner.py:554-563).
6. **MEDIUM** Fixed-clock evening chain: slow bhavcopy → silently empty EOD review,
   no same-evening retry (market_runner.py:531-578).
7. **MEDIUM** Windows-login autostart claim is stale/unverifiable (README.md:447,
   archive/start_stocksense.ps1:3).
8. **MEDIUM/LOW** Fast intraday stop breaches don't queue sandbox SELL
   confirmations (intraday_stops.py vs live_confirmation.queue_exit_confirmations).
9. **LOW** No NSE-holiday gating; ticker_sync runs weekends; JOB_CADENCE/docstring/
   start.ps1 label drift; per-poll DB connections in the stop loop.
