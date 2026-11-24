# StockSense Program M — Market Lab inside StockSense, the money-flow dashboard, order-flow research, and a strategy library

Written 2026-11-23 by the orchestrator (Opus) after four read-only audits:
- Market Lab architecture;
- StockSense state;
- verbatim StockSense interfaces;
- verbatim Market Lab backend, frontend and data schemas.

**Executors are Sonnet-class subagents, one card at a time.** Every card names the exact files, the existing functions to reuse (file:line), a code skeleton for each new function, test cases with hand-computed numbers, the commit message and the acceptance command. If a card and reality disagree, follow Part 0 rule 14 (probe, record, adapt the smallest piece).

In card M0.3 this file is copied verbatim into the repo as `docs/PLAN-M.md`. `docs/PLAN.md` (Parts C, D, F) and `docs/PLAN-G2.md` Part 0 stay binding. This program realises PLAN.md **T12.3** (read-only API on `127.0.0.1:8765`) and **Phase 14** (UI from an existing repo: Market Lab is that repo).

How to read: everyone reads Parts 0–5. Each executor then reads only its own card in Part 6, plus the **Provides** lines of the cards it depends on.

---

## Context

**Why this program exists.** The user wants:
1. **The Market Lab dashboard rebuilt inside StockSense** (`github.com/heysanjeevv/market-lab`, MIT, local clone `C:\Users\harsh\Downloads\market-lab`, commit `ba3f771`).
   - Smart Money tape, Sector Map, charts with smart-money layers, Swing Lab, per-stock research note, per-stock "hover/click and test strategies".
   - It must start with one command and fetch everything itself, with no retraining.
2. **A real answer to "where is the market's money going".** Liquidity and flow:
   - OI buildup, delivery, FII/DII and participant positioning;
   - option positioning and gamma exposure;
   - the 09:00 pre-open auction ("the market opens at 9:00, not 9:15");
   - depth and pending-order imbalance, volume delta and CVD, big trades, absorption.
   Plus order-flow tools: VWAP, volume profile, market profile, footprint, delta, heatmap, auction market theory, time scales.
3. **A much larger strategy library**, every strategy backtested properly:
   - Market Lab's 23 swing + 39 intraday strategies;
   - the user's tier list (F/C/B);
   - the equity-applicable strategies of Kakushadze & Serur, *151 Trading Strategies* (2018; SSRN 3247865; arXiv 1912.04492);
   - strategies built on the flow data.
4. **Learning notes** for every concept, so the user (and future agents) understand what each tool shows and what NSE data allows.
5. **(Added 2026-11-23, user's words: "StockSense will be something through which I can see the market, and it will daily be giving me profitable stocks based on backtesting of thousands of strategies.")**
   - A **Daily Picks** page and API (card M5.9): every evening, the stocks that today's signals of **gate-passing** strategies select, ranked by the strength of each strategy's out-of-sample evidence.
   - Paper-sized from the broker's capital and recorded in the append-only journal.
   - The user also asked that every section of *151 Trading Strategies* be read and implemented wherever possible. The orchestrator read the local PDF (`C:\Users\harsh\Downloads\ssrn-3247865.pdf`, 361 pages) on 2026-11-23, and M5.4 now covers all 174 strategy sections plus Appendix A.

**The honest limit, which the UI and RESULTS must state.**
- Running thousands of strategy variants raises the bar each one must clear: the deflated Sharpe uses the total trial count.
- Many or all may fail the gate. On such a day, Daily Picks shows "no strategy has passed the gate". Exploratory candidates are shown separately and labelled as such.
- Nothing here guarantees profit.

**What the audit of Market Lab found (verified line by line).**
- Its "Smart Money" ranking is **price-only**: `disp = (ltp/open − 1)·100 / mean20((high−low)/close·100)` (`sm/feed.py:162-203`). No OI, no delivery, no volume, no order flow. Sector chips are the mean of % change vs the previous close. **This program keeps that view exactly and adds the real flow measures beside it.**
- Tape 5-minute bars are built from 1-minute *closes* (`sm/feed.py:128-138`), so tape OB/FVG are approximations. Only the pop-up chart uses true OHLCV.
- **Backtest defects that are NOT ported:**
  - a forward-looking 35% split filter (`lab/swing.py:274-278`);
  - each strategy's exit chosen on the full sample, OOS included (`lab/swing_live.build_summary`);
  - survivorship (today's Nifty 500 used for the past);
  - sizing at the open from that day's close NAV (`lab/swing_pf.py:41`);
  - signals on unfinished daily bars when started intraday;
  - a rewritable paper journal (`lab/live.paper_tick`);
  - a fixed cost with brokerage `* 0` (`lab/swing.py:15-19`);
  - chart warm-up 220 vs research 260.
- Engineering defects: no tests, unpinned dependencies, NSE calls without cookie priming (they fail silently), duplicated files (`sx/{market,smc,structure}.py` are byte-identical to `sm/`; the chart modal JS is duplicated in `smart.html` and `sector.html`), pickle storage.
- No broker order code (paper-only compatible). The `"rt"` "Angel One ticks" listener is a dead hook and is not ported.

**User decisions (2026-11-23).**
1. Finish the light G2 work first (merge H1/H4/H5, finish IPCA). The remaining GPU families wait until Program M is done (16 GB RAM).
2. Frontend: **rebuild in React/Next.js**, same look and behaviour, then extend.
3. Data sources: **keep Market Lab's sources** (NSE website JSON, Yahoo Finance, Google News RSS, Wikipedia) for the dashboard. StockSense's own NSE data and brokers add history, OI, delivery and depth. Research and gated backtests use only StockSense's NSE data (CLAUDE.md rule 3 and A.6).
4. **Both backtest views.** An instant per-stock exploratory backtest (net of costs, labelled *exploratory*), and a pre-registered walk-forward over the PIT universe judged by gate v2. The UI shows both side by side.
5. Standing rules from memory:
   - no scheduled task, no auto-start; manual start only;
   - `taskkill /F /T` before restarts;
   - "Kronos" = shiyu-coder/Kronos;
   - plans as detailed execution cards.

**Intended outcome.**
- `run-lab.bat` (or `stocksense serve`) opens `http://127.0.0.1:8765`: one local server serving the Next.js dashboard and fetching everything while it runs.
- A Smart Money tape identical in method to Market Lab and verified side by side.
- A Money Flow page and an Order Flow workbench.
- A strategy library of about 450 pre-registered trials, each with an exploratory per-stock backtest and a gated walk-forward verdict:
  - 62 Market Lab strategies, with the 39 intraday ones × 3 trend filters;
  - about 20 tier strategies;
  - about 75 *151* stock, sector and macro strategies;
  - 116 + 1 *151* option strategies;
  - about 7 flow strategies.
- A Daily Picks page fed only by gate-passing strategies.
- `docs/learn/` notes for every concept.
- No profit promise: the gate decides, and fails are recorded.

---

# PART 0 — Rules for Program M (binding; in addition to CLAUDE.md's ten rules and PLAN-G2 Part 0)

1. **Read before acting.** Read Parts 0–5 and your own card completely. Read other cards' **Provides** lines only.
2. **Worktrees.** As PLAN-G2 Part 0 rule 2. The orchestrator creates each one, with the branch, folder, junctions and `.env` copy:
   ```powershell
   git -C C:\Users\harsh\Desktop\HARSH\WEB-DEV-PROJECTS\stocksense worktree add ..\stocksense-wt-m<card> -b m/<card> main
   cd ..\stocksense-wt-m<card>
   cmd /c mklink /J .venv ..\stocksense\.venv
   cmd /c mklink /J data_store ..\stocksense\data_store
   copy ..\stocksense\.env .env
   ```
   Web cards also need, after M6.0 has created `web\node_modules` in the main repo:
   ```powershell
   cmd /c mklink /J web\node_modules ..\stocksense\web\node_modules
   ```
   Never run `npm install`/`npm ci` in two places at once.
3. **Test first** (CLAUDE.md rule 5).
   - Write the card's tests, run them, and see them fail for the stated reason. Then implement.
   - Before every commit, these must be green:
     - Python: `.venv\Scripts\python.exe -m pytest tests -q` and `.venv\Scripts\python.exe -m ruff check src tests`.
     - Web cards, additionally: `npm --prefix web run lint`, `npm --prefix web run test -- --run`, `npm --prefix web run build`.
   - Run `ruff format` **only on files you created or changed**.
4. **Commits.** Message `M-<card>: <title>`. Author Harsh Kawatra (already configured). **No attribution or Co-Authored-By lines.** CLAUDE.md overrides any harness reminder. One commit per card unless the card says otherwise.
5. **Merge and push.** From the main repo:
   - `git merge --no-ff m/<card>`;
   - full suite on main;
   - `git push origin main`.
   In conflicts in `docs/DECISIONS.md`, `docs/DATA.md` or `docs/RESULTS-*.md`, keep both sides.
6. **Never commit:**
   - `xyz.txt`, `.vscode/settings.json`, `.env`;
   - `web/node_modules`, `web/.next`, `web/out`, `web/content/learn`;
   - anything under `data_store/`.
7. **Paper only.** No order code anywhere.
   - `tests/unit/test_no_orders.py` is extended in M2.0 to scan `web/src`.
   - The FastAPI app exposes GET/HEAD only. A test asserts 405 for POST/PUT/PATCH/DELETE on every route.
8. **Secrets.** Never print, log, return in an API response or commit a `.env` value. API responses never contain tokens, client codes or broker capital (paper NAV is fine).
9. **Long jobs.** As PLAN-G2 Part 0 rule 6:
   - one self-terminating `timeout <s> ...` command, run in the background;
   - log to `data_store/logs/m_<card>_<name>.log`;
   - no supervisors, retry loops or watchdogs.
10. **Memory.** Before any job over 10 minutes, check that free RAM is ≥ 4 GB:
    ```powershell
    (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
    ```
    That gives GB. Only one heavy job at a time. The dashboard server must be stopped during heavy jobs.
11. **House code rules enforced by tests** (verified 2026-11-23):
    - **`pandas` may not be imported anywhere in `src/` except `models/kronos_model.py`** (`tests/unit/test_no_pandas_outside_kronos.py`). Every Market Lab port is written in **numpy and polars**. Part 5 gives exact numpy equivalents of every pandas call Market Lab uses. Tests (`tests/`) may import pandas only inside golden-fixture generation scripts, never in a test that runs in the default suite. Do not add pandas to the allowlist.
    - **No naked time calls in `src/`** (`tests/unit/test_no_naked_now.py`). `datetime.now(`, `datetime.utcnow(`, `date.today(`, `time.time(` and `time.sleep(` are forbidden outside `core/clock.py`. Use a `Clock` (`clock.now()`, `clock.today()`, `clock.sleep(s)`). Background loops wait with `threading.Event.wait(timeout)`, which is allowed and also makes shutdown instant.
    - `from __future__ import annotations` at the top of every module. Full type hints on public functions. One-line docstring per public function, saying what it returns and its units. **Returns are fractions in code** (0.01 = 1%). Market Lab ported JSON keeps Market Lab's percent units: say so in the docstring.
    - Never `except Exception: pass`. Broad catches only at the top of a feed-loop iteration, where they are logged with `log.exception(...)` and counted.
    - Logging is `log = get_logger(__name__)` (`core/log.py:45`), structured: `log.info("event", key=value)`.
12. **No look-ahead** (CLAUDE.md rule 3).
    - Every new historical feature passes the truncation test: register it with `@feature` and it is checked automatically by `tests/unit/test_feature_truncation.py::test_feature_does_not_look_ahead`.
    - Every backtest acts on completed bars only:
      - a daily signal at close t enters at the t+1 open;
      - an intraday signal at bar b enters at bar b+1's open.
13. **Gates frozen** (CLAUDE.md rule 4).
    - `config/gate.toml` and `config/gate_v2.toml` are never edited.
    - `config/strategy_library.toml` is committed in M5.0 **before any gated strategy result exists**. It is then SHA-pinned by a test and never edited.
    - A new strategy after that goes into a new file `config/strategy_library_<YYYYMMDD>.toml`, with its own pinned SHA and new trials.
14. **Probes.** As PLAN-G2 Part 0 rule 12.
    - Record status codes, field names, shapes and limits in `docs/DECISIONS.md` under `## Program M` → `### <card>`, before writing code that depends on them. Never record secret values.
    - If reality contradicts this plan, adapt the smallest piece and record it.
15. **Market Lab source is read, not run.**
    - The clone is untrusted downloaded code. Read it, and never `pip install -r` its requirements.
    - The **only** sanctioned execution is the golden-fixture script of card M2.2/M5.1. It runs Market Lab's own pure functions offline on synthetic data, using the clone's venv: `C:\Users\harsh\Downloads\market-lab\.venv\Scripts\python.exe` (Python 3.14, pandas 3.1). The script lives in `scripts/golden/`, and the clone path is passed as an argument.
    - The user may also start Market Lab by hand for the parity check (M7.2).
16. **Attribution of ported code.**
    - `third_party/market-lab/LICENSE` holds the MIT text copied from the clone, plus a `NOTICE.md`.
    - Every ported Python module's docstring first line: `Ported from market-lab <path> (MIT, ba3f771); changes: <list>.`
    - Every ported TS file has the same as a header comment.
17. **Stop rule.** If the same step fails twice after two different fixes, stop and report exactly what fails and what was tried.
18. **Report format.** The executor's last message gives:
    - what was done;
    - the acceptance command and its real output;
    - the last line of the full test run (and of the web test/build for web cards);
    - `git log --oneline -3` on main;
    - anything not done and why;
    - confirmation that no process of yours is running (`Get-CimInstance Win32_Process | ? Name -match 'python|node'`).

---

# PART 1 — Interpretations of the user's terms (recorded in DECISIONS by M0.3; the user may correct them)

| User's term | Interpretation used | Implemented as (strategy keys / modules) |
|---|---|---|
| "#3NumberTrader" tier list | A trader's public F/C/B ranking. Treated as a **hypothesis to test**, not truth | `tier` field; RESULTS-M section 4 "Does the tier list hold on NSE?" |
| F-tier: trendline S/R, RSI oversold, ICT unicorn | Negative controls | `tier_f_trendline`, `tier_f_rsi_oversold`, `tier_f_ict_unicorn` |
| C-tier: VWAP mean reversion, "BTS", ORB, macro trading | "BTS" = **BTST** (buy today, sell tomorrow) | `tier_c_vwap_mr`, `tier_c_btst`, `tier_c_orb_15`, `tier_c_orb_30`, `tier_c_macro_trend`, `tier_c_macro_fii` |
| "8-tier" | **B-tier** (typo) | — |
| B: adverse order flow | Order flow against price: absorption, and CVD–price divergence | `tier_b_absorption`, `tier_b_cvd_divergence` |
| B: "STR" | **Short-term reversal** (Jegadeesh 1990; Lehmann 1990). SMT divergence is also included in case that was meant | `tier_b_str_1d`, `tier_b_str_5d`, `tier_b_smt` |
| B: auction market theory ("law of the market") | Dalton market profile / AMT: the 80% rule, initial-balance extension, open-drive, poor-extreme revisit | `tier_b_amt_80`, `tier_b_ib_ext`, `tier_b_open_drive`, `tier_b_poor_extreme` |
| B: mechanical trading | The principle: every strategy is coded rules. Plus the Turtle system | `tier_b_turtle` + the whole library |
| "Abs option imbalances" | **Absorption** + **option imbalances** (ΔOI call vs put by strike; PCR) | `orderflow/absorption.py`; `flow/options.py:oi_imbalance` |
| "Market opens at 9:00" | NSE pre-open call auction:<br>• 09:00–09:08 order entry (random close in the last minute)<br>• 09:08–09:12 matching<br>• 09:12–09:15 buffer<br>The indicative price and the buy/sell imbalance are the first flow signal. The block-deal window is 08:45–09:00 | `flow/preopen.py`, pre-open board |
| "Heat map" | Order-book liquidity heatmap (resting depth by price over time) **and** Market Lab's sector heat view | `orderflow/heatmap.py`; Sector Map |
| "Time scales" | Multi-timeframe alignment 5m/15m/60m/D/W | `orderflow/timescales.py` |

---

# PART 2 — Architecture

```
stocksense/
  run-lab.bat                         # manual start: build web if stale, open browser, serve
  config/strategy_library.toml        # pre-registration (M5.0), SHA-pinned
  config/flow_clients.toml            # optional user list of "smart money" deal clients (starts empty)
  docs/PLAN-M.md  docs/API-M.md  docs/RESULTS-M.md  docs/learn/*.md
  third_party/market-lab/LICENSE  third_party/market-lab/NOTICE.md
  scripts/golden/make_inputs.py       # deterministic synthetic bars -> tests/fixtures/marketlab/inputs_*.json
  scripts/golden/run_marketlab.py     # runs Market Lab's own functions on those inputs (clone venv) -> golden_*.json
  scripts/m_parity.py                 # side-by-side parity with a running Market Lab
  src/stocksense/
    api/
      app.py              create_app(settings, *, start_feeds=True, clock=None, store=None) -> FastAPI
      deps.py             get_store, get_settings (FastAPI dependencies reading app.state)
      readonly.py         GET-only middleware
      static.py           mount web/out, fallback page
      routes_status.py    /health, /api/status
      routes_sm.py        /api/sm/live, /api/sm/chart, /api/sm/quote
      routes_sector.py    /api/horizon, /api/sector, /api/stock, /api/search, /api/chart, /api/tick
      routes_research.py  /api/research
      routes_lab.py       /api/lab/trades, /api/lab/swing, /api/lab/summary, /api/lab/journal, /api/lab/matrix, /api/lab/strategies
      routes_flow.py      /api/flow/*
      routes_orderflow.py /api/of/*
    marketlab/            # ports of Market Lab sm/ sx/ lab/research (numpy/polars, Clock, NseClient)
      npx.py              pandas-equivalent numpy helpers (Part 5.1) — shared by marketlab/ and strategies/
      clock.py            market_status, minute_now, session_minutes (Part 5.2)
      store.py            LabStore (RLock), TTLCache
      sources.py          NSE live JSON via NseClient; Yahoo spark/chart; Google News RSS; Wikipedia
      structure.py        port of sm/structure.py
      smc.py              port of sm/smc.py
      tape.py             port of sm/feed.py (Smart Money)
      sectors.py          port of sx/universe.py + sx/live.py + app.py sector functions
      news.py             port of sx/news.py
      research.py         port of lab/research.py
      industries.py       copy of lab/industries.py tables
      feeds.py            FeedManager: background threads + catch-up thread
    flow/                 # NEW: where the money goes
      oi.py delivery.py deals.py participants.py options.py preopen.py sector_flow.py composite.py
    orderflow/            # NEW: order-flow tools on bars and quotes
      vwap.py profile.py delta.py footprint.py bigtrades.py absorption.py heatmap.py timescales.py
    strategies/           # NEW: library + engines
      base.py             StrategySpec, REGISTRY, register()
      indicators.py       port of lab/swing.features (daily) and lab/features.build (intraday)
      exits.py            port of lab/swing.follow (7 exits) and lab/engine.follow (8 exits)
      swing.py            23 Market Lab swing strategies
      intraday.py         39 Market Lab intraday strategies
      tiers.py k151.py flowstrats.py
      costs.py            trade cost fractions from stocksense.costs (Part 3.6)
      explore.py          per-stock exploratory engine
      portfolio.py        k-slot daily-return simulator (gated path)
      scan.py             radar SIGNAL/OPEN/recent/FORMING
      profile.py          "what kind of stocks worked" buckets
      journal.py          append-only lab journal (state table)
      data.py             load daily/intraday frames for one symbol from our store
    validation/strategy_eval.py       # gated walk-forward runner
    cli_lab.py            # `stocksense serve`, `stocksense lab ...` commands (registered from cli.py)
  web/                                # Next.js app (static export -> web/out, served by FastAPI)
```

**Process model.**
- `stocksense serve [--port 8765] [--no-feeds] [--no-catchup]` runs `uvicorn.run(app, host="127.0.0.1", port=port, workers=1)`.
- `create_app` starts `FeedManager` threads when `start_feeds=True`. A FastAPI `lifespan` stops them (sets a `threading.Event`, joins with a 5 s timeout).
- The catch-up thread runs once per start and does the following in order:
  1. `data.ingest_all.catch_up(ctx)` if data is behind;
  2. `panel.build_all(settings)` if the derived data is behind and the time is outside 09:00–15:45;
  3. the radar scan;
  4. the flow EOD aggregates.
- Nothing is scheduled. Closing the window stops everything.
- The ML daemon (`stocksense run`) is separate and untouched.

**Ports.** StockSense uses 8765. Market Lab stays on 8810, so both can run side by side.

**Frontend.**
- Next.js App Router, TypeScript strict, React 19.
- `output: "export"`, `trailingSlash: true`, `images: { unoptimized: true }`, built into `web/out/` and served by FastAPI.
- Charts: `lightweight-charts` **4.2.0** (the same as Market Lab, so its v4 API calls port 1:1).
- CSS: Market Lab's CSS copied verbatim per section, scoped with `postcss-prefix-selector`.
- Canvas/SVG renderers are ported as imperative "islands" inside React components (Part 5.6).

---
# PART 3 — Verified StockSense interfaces (main @ 14cca43; use these exactly)

## 3.1 Settings, paths, clock, calendar, state, log
- `core/config.py:310` `load_settings(config_dir=Path("config"), env_file=Path(".env")) -> Settings`.
  - Fields: `paths.data_root`, `time`, `research` (`dev_start`, `wf_test_start`, `lockbox_start`), `runtime`, `universe`, `costs`, `gate`, `gate_v2`, `strategies`, `risk`, `schedule`, `secrets`.
  - The secrets used here: `angel_api_key`, `angel_client_id`, `angel_pin`, `angel_totp_secret`, `upstox_analytics_token`.
  - `Settings.require(*names)` raises `ConfigError` for unset secrets.
- `core/paths.py:8` `Paths(settings)`.
  - Properties `root, parquet, cache, panel, features, labels, models, preds, state, reports, logs, sqlite`.
  - `dataset_dir(name)`.
  - **Program M adds** a `lab` property returning `root / "lab"`. Add it in M2.1, with a test.
- `core/clock.py`: `IST`, `Clock` protocol (`now() -> tz-aware IST datetime`, `today()`, `sleep(s)`), `SystemClock`, `FixedClock(dt)`, `SteppingClock(start, step)` with `.advance(td)`.
  - Test fixture `fixed_clock` = 2026-11-12 10:00 IST (`tests/conftest.py:36`).
- `core/calendar.py:29` `TradingCalendar`:
  - `from_store(settings, clock)`, `from_dates(past_sessions, holidays=(), special_sessions=(), tz=IST)`;
  - `is_session(d)`, `next_session(d)`, `prev_session(d)`, `sessions(a, b)` (inclusive), `shift(d, n)`, `session_index(d)`, `session_times(d) -> (open_dt, close_dt)`, `last_completed_session(now)`.
- `core/state.py`:
  - `MIGRATIONS: list[tuple[int, str]]` currently has versions 1–4. **Add a table by appending `(max_version + 1, "CREATE TABLE ...;")`.** PLAN D.6 reserves 5 for `drift`. If 5 is still unused at execution time, use 6 and record why in DECISIONS.
  - `State(settings, clock=None)`: `connect()` (autocommit), `migrate()`, `kv_get(key, default=None)`, `kv_set(key, value)`, `job_start`, `job_finish`, `job_status`.
- `core/log.py:45` `get_logger(name)` gives a structured logger (`log.info("event", k=v)`) that redacts secrets.

## 3.2 Data access
- `data/catalog.py`:
  - `DatasetSpec(name, layer, key, date_col, partition, schema, freshness=None, path="", file=None, nested=False, keep_extra=False, nullable_key=())`.
  - Declare one with `_spec(name, layer, "k1, k2", date_col, partition, {col: dtype}, **kw)`.
  - New raw datasets go in `_RAW_NEW`; new derived datasets go in `_DERIVED` with `path=`.
  - dtype aliases: `_D` Date, `_F` Float64, `_S` Utf8, `_I64`, `_I32`, `_B` Boolean, `_TS` = `pl.Datetime("us","Asia/Kolkata")`.
  - `CATALOG`, `get_spec(name)`.
- `data/io.py`:
  - `read(settings, dataset, start=None, end=None, columns=None, sub=None) -> pl.LazyFrame`;
  - `upsert(settings, dataset, df, sub=None, clock=None) -> list[Path]` (merge on key; new rows win);
  - `write_partition(settings, dataset, key, df, sub=None, clock=None)`.
- `data/nse_http.py:36` `NseClient(clock, transport=None, seed=0)`:
  - `get_api(path, params, referer_path) -> Any` (JSON). It warms cookies on 401/403 up to 2 times. 404 raises `DataNotAvailable`. Transport errors, 429 and 5xx are retried 5 times with back-off from 2 to 60 s, which sleeps through the clock.
  - `get_archive(url) -> bytes | None`, `close()`. Minimum spacing 0.6 s + 0.2 s jitter.
- `data/ingest_common.py`: `IngestContext(settings, clock, calendar, nse, state)`; `IngestResult(dataset, date, rows, status, detail="")`.
- `data/ingest_all.py:165` `catch_up(ctx, steps=None) -> list[IngestResult]`. It runs every ingester from its last stored date to the last completed session; a failed step is reported, not raised.
- `data/status.py:50` `status_table(settings, calendar, clock, datasets=None) -> pl.DataFrame`. Columns `dataset, first_date, last_date, rows, sessions_missing, fresh, issues`. **Behind ⇔ `fresh == False`.**
- `data/ingest_misc.py`: `parse_preopen(payload, ts)` (:292), `capture_preopen(ctx)` (:328), `capture_spreads` (:381), `capture_quotes` (:415).
- `data/ingest_intraday.py:34` `load_bars(settings, symbols, start, end, interval="5m"|"1m", basis="adjusted"|"raw_ratio") -> pl.DataFrame`.
- `panel/__init__.py:6` `build_all(settings)`: security master, daily panel, universe, market.
- `panel/security_master.py:288` `symbol_sid_map(settings) -> (symbol, valid_from, valid_to, sid)`.
- CLI helper `cli.py:206` `_ingest_context(ctx) -> (IngestContext, NseClient, SystemClock)`. Program M builds its own context the same way inside `marketlab/feeds.py` (Part 6, M2.7).

## 3.3 Stored data (schemas verified by reading partitions)

| Dataset (io name) | Coverage | Columns that matter here |
|---|---|---|
| `panel_daily` (`panel/daily/part-YYYY`) | 2010→ | `sid, date, symbol, series, open, high, low, close, prev_close, volume(i64), turnover_inr, n_trades, deliv_qty, deliv_pct, gap_factor, adj, adj_open, adj_high, adj_low, adj_close, adj_volume, ret, ret_co, ret_oc, div_ret, tret, flag_*, industry` |
| `panel_universe` | 2011→ | `date, sid, turnover_rank, med_turn_60, u500, u200, in_fno`. Only eligible rows are stored. `in_fno` is a rank ≤150 proxy before 2024-07-08 |
| `panel_market` (`panel/market.parquet`) | 2010→ | `date, ew500_ret, ew500_co, ew500_oc, ew200_ret, nifty50_ret, nifty500_ret, n_u500, breadth_200, xs_dispersion` |
| `fo_bhavcopy` (month) | 2024-07-08→ | `symbol, instrument_type (IDF/IDO/STF/STO), expiry_date, strike_price, option_type, date, open, high, low, close, prev_close, settle_price, oi(i64), change_in_oi(i64), underlying_price`. **OI is in shares, not contracts** (NIFTY OI divides by 65). **On an option's own expiry date `settle_price` = underlying** (final settlement): exclude expiry-day option rows from IV/greeks. No lot-size column |
| `bhavcopy_delivery` | 2019-08→ | `symbol, series, date, deliv_qty, deliv_pct` |
| `participant_oi` (year) | 2019-12→ | `date, client_type (FII/CLIENT/DII/PRO), fut_idx_long, fut_idx_short, fut_stk_long, fut_stk_short, opt_idx_call_long, opt_idx_put_long, opt_idx_call_short, opt_idx_put_short, opt_stk_*, total_long, total_short` |
| `fii_dii` (all) | 2026-09-21→ (17 days) | `date, category (FII/DII), buy_value, sell_value, net_value` |
| `deals` (year) | 2026-11-11→ | `date, kind, symbol, client, side, qty, price` |
| `preopen` (month) | 2026-09-23→2026-10-21 | `symbol, ts, prev_close, indicative_price, final_price, buy_qty, sell_qty, source` |
| `intraday_bars` (5m, month) | 2024-09-13→ (~470 symbols) | `symbol, ts(IST), interval="FIVE_MINUTE", open, high, low, close, volume(i64), source, adj_asof` |
| `bars_1m` (month) | 2026-03→2026-11 (gaps) | `symbol, ts, open, high, low, close, volume, source="upstox", adj_asof` |
| `index_daily` (year) | 2012→ (169 indices incl. every NIFTY sector index, INDIA VIX) | `index_name, date, open, high, low, close, pe, pb, div_yield, volume, turnover_cr` |
| `india_vix` | 2010→ | `date, close` |
| `macro_daily` | to 2026-11-19 | `series ∈ {BRENT, DXY, GOLD, NDX, SPX, US10Y, USDINR}, date, close, available_ts` |
| `results_index` | — | `seq_number, symbol, period_from, period_to, filing_ts, exch_ts, audited, consolidated, relating_to, xbrl_url`. **No EPS values** (T8.6 fundamentals are not built) |
| `sectors` | — | `symbol, industry (22 values), company, isin, source, as_of` |
| `nse_snapshots/oi_spurts` (`YYYY-MM-DD.parquet`) | 2026-09-18→ (16 days) | `symbol, latest_oi, prev_oi, change_in_oi, oi_change_pct, oi_change_pct_recomputed, volume, underlying_value, snapshot_ts, source_timestamp` |
| `labels/v1` (`load_labels(settings, purpose, start, end, columns)`) | 2011→ | `date, sid, y_oc, y_c2o_5, y_c2o_10, y_c2o_20, y_c2c_1, y_o2o_5, end_*`, `z_<y>`, `r_<y>` |
| `spreads`, `quotes_1m` | **empty** | catalog only |

## 3.4 Backtest engine, costs, metrics
- `backtest/fills.py`:
  - `Order(sid, side, product, when: "open"|"close", qty, trade_date=None, priority=0.0, reason="")`;
  - `Fill`, `DayBar`;
  - `try_fill(order, bar, ctx)`.
  - **The engine fills only at the open or close.** It cannot fill at a stop price inside the bar or at intraday bar times. Program M therefore uses its own simulator `strategies/portfolio.py` (Part 6, M5.7), which produces the gate's daily series directly. The engine is not used for the library.
- `costs/india.py`:
  - `trade_cost(side, product, value_inr, sched: Costs, first_sell_of_scrip_today=True) -> CostBreakdown` (`.total` in ₹);
  - `round_trip_cost_frac(product, value_inr, sched) -> float`.
  - Products are `"delivery"` and `"intraday"`.
- `costs/slippage.py`:
  - `SlippageTable.load_latest(settings)`, `.scaled(factor)`;
  - `slip_frac(turnover_rank, vol_21, order_value, med_turn_20, when: "open"|"close"|"now", table, as_of=None) -> float` (adverse move, as a fraction, per side).
- `config/costs.toml` (verbatim values):
  - **delivery:** STT 0.1% each side, stamp 0.015% buy, exchange 0.00297%, SEBI 0.0001%, GST 18%, DP ₹20 per scrip on sell, brokerage 0.1% capped at ₹20, minimum ₹5;
  - **intraday:** STT 0.025% sell, stamp 0.003% buy, otherwise the same with DP 0;
  - **slippage:** impact_k 0.5, open penalty 3 bps, close penalty 2 bps, half-spread by rank `[[50,1.5],[100,2.5],[200,4],[350,7],[500,12]]` bps (beyond the table 20 bps);
  - `grid.capital_inr = [25000, 100000, 500000, 2500000]`.
- `backtest/metrics.py`:
  - `perf_stats(daily_ret, benchmark, periods=252, dates=None) -> dict` (keys incl. `sharpe, excess_tstat, frac_positive_years, max_drawdown`);
  - `ic_stats(pred_frame, label_col) -> dict` (`mean_ic`, `ic_tstat`; needs columns `date, score, <label_col>`; dates with fewer than 50 pairs are skipped);
  - `newey_west_tstat(x, lags=5)`.

## 3.5 Validation
- `validation/trials.py`:
  - `register_trial(state, family, target, spec) -> trial_id`. It is idempotent on the same spec. `trial_id = f"{family}-{target}-{sha1(canonical)[:10]}"`, status "registered".
  - `finish_trial(state, trial_id, metrics, returns_path)`: once only. `metrics` must contain `"sharpe_daily"`.
  - `n_trials(state, family=None)`.
- `validation/gate.py`:
  - `CandidateResult(candidate_id, family, target, daily: pl.DataFrame[date, net, net_2x, benchmark?, gross], ic: {"mean_ic","ic_tstat"} | None, preds=None, label_col=None, trial_ids=())`;
  - `save_candidate_result(candidate, settings) -> Path` (writes `models/<id>/candidate_result.json` + `daily.parquet`);
  - `validate_candidate(candidate_id, settings, state, open_lockbox=False, *, gate_version=None) -> GateReport`;
  - `evaluate_research_gate(candidate, settings, state, version=None)`.
  - **Gate v2 with `exposure_matched` needs a `gross` column** (daily gross exposure fraction) unless the family starts with `g1-s2`/`g2-s2strict`. **PBO fails when a family has fewer than 10 finished trials.**
  - Nine lines:
    1. mean IC ≥ 0.02
    2. IC t ≥ 3
    3. net Sharpe ≥ 1.0
    4. 2× slip Sharpe ≥ 0.5
    5. excess t ≥ 2
    6. positive years ≥ 70%
    7. DSR ≥ 0.95
    8. PBO ≤ 0.30
    9. max DD ≤ 25%
- `validation/splits.py`:
  - `Fold(name, train_end, test_start, test_end)`;
  - `walk_forward_folds(settings, calendar)` (calendar-year folds from `wf_test_start` to the research end);
  - `train_mask(label_end, fold, embargo_sessions=5, *, calendar, dates=None, dev_start=None)`;
  - `cscv_blocks`.
- `validation/g2_eval.py:300` `evaluate_model(trial_id, settings, state, calendar, *, base_id=None, ...) -> dict`.
  - It reads scores from the `preds` dataset with `sub=trial_id` (`date, sid, score(Float32), made_at, mode`).
  - It writes `reports/g2/models/<trial>/eval.json`.
  - **xs-factor strategies are evaluated by writing their scores into `preds/<trial_id>/` and calling `evaluate_model`** (M5.7).
- `research model-eval --trial <id> --base <id>` (`cli_research.py:482`).

## 3.6 Brokers
- `brokers/types.py:33` `Quote(symbol, ts, ltp, open, high, low, prev_close, volume, bid, ask, bid_qty, ask_qty, upper_circuit, lower_circuit, source)`. After the M0.1 merge (H1) it also has `oi, total_buy_qty, total_sell_qty, avg_price, depth_bid, depth_ask, week52_high, week52_low`.
- `brokers/angel.py:88` `AngelClient(settings, clock, transport=None)`:
  - `login()`;
  - `quote_full(tokens) -> list[Quote]` (exchange hard-coded `"NSE"`; 50 tokens per call; 1 call per second);
  - `candles(token, interval, start_dt, end_dt)` (exchange `"NSE"`).
- `brokers/instruments.py:71` `InstrumentMap(settings, clock)`: `refresh()`, `angel_token(symbol)`, `upstox_key(symbol)`. **Equity only.** The raw Angel scrip master cached at `cache/instruments/angel-<date>.json` contains NFO rows. M3.1 adds an NFO futures parser.
- `brokers/quotes.py:144` `QuoteHub(sources, clock).get(symbols, held=None) -> dict[str, Quote]`.

## 3.7 Features, labels, tests
- `features/base.py:50` `feature(name, *, group, kind="xs"|"mkt", window=1, version=1, label="", outputs=None, registry=None)` decorator. The function signature is `fn(ctx: FeatureContext) -> pl.LazyFrame` with columns `date, sid, <name>` (xs) or `date, <name>` (mkt).
  - `FeatureContext(settings, as_of)`: `.panel()`, `.market()`, `.universe()`, `.sessions()`, `.raw(dataset, columns=None, sub=None)`, `.cutoff_ts`.
  - Rolling helpers: `roll_mean`, `roll_std`, `roll_sum`, `roll_min`, `roll_max`.
- `features/registry.py:15` `FEATURE_MODULES` (list of module names). A new feature module must be appended there **only in a feature-version card**. In Program M, new historical feature modules are registered but **left out of `FEATURE_MODULES`** (as E1–E7 are), so production features stay v1. They are checked by importing them explicitly in `tests/unit/test_feature_truncation.py` (it already imports `kronos_feats` explicitly; add the import the same way).
- `features/derivs.py`:
  - `R_FREE = 0.065`, `MIN_DAYS = 3`;
  - `bs_price(s, k, t, r, sigma, call)`;
  - `implied_vol(price, s, k, t, call, r=R_FREE) -> float | None`;
  - `max_pain(strikes, call_oi, put_oi)`;
  - `oi_price_quad(price_chg: pl.Expr, oi_chg: pl.Expr) -> pl.Expr` (+1 long buildup, −1 short buildup, +0.5 short covering, −0.5 long unwinding);
  - `_chain_stats(grp)`.
- Truncation helpers (`tests/unit/test_feature_truncation.py`):
  - `truncation_env` (session fixture) gives `(settings, sessions, cache)`;
  - `_temp_spec(reg, name, fn, kind)`, `_check(spec, env) -> list[str]`, `violations(...)`;
  - `CUT_INDICES = (300, 380, 450, 520, 590)`.
- Synthetic market: `tests/fixtures/synth.py` provides `make_market(n=40, n_sessions=600, seed=7, planted_ic=0.0)`, `write_market(m, root)`, `_settings_for(root)`, `make_tiny_market()`.
- Fixtures: `tmp_data_root` (Settings with a temp data root), `fixed_clock`.
- `pyproject.toml` pytest `addopts = "-q -m 'not realdata and not gpu'"`, markers `integration`, `realdata`, `gpu`; ruff line-length 110, rules `E,F,I,UP,B,SIM`.

## 3.8 CLI
- `cli.py:27` `app = typer.Typer(...)`. `_settings(ctx)` returns Settings (tests inject `obj=Settings`). Sub-apps: `ingest_app`; `cli_research.register(app)` at `cli.py:828`.
- **Program M adds** `src/stocksense/cli_lab.py` with `register(app)`, which adds the `serve` command and a `lab` group (`lab scan`, `lab explore`, `lab golden-check`), called from `cli.py` right after the `cli_research` line:
  ```python
  __import__("stocksense.cli_lab").cli_lab.register(app)
  ```

---

# PART 4 — What NSE data can and cannot show (honest feasibility; M1.1 expands this into `docs/learn/00-data-reality.md`)

| Concept | Needs | We have | Method here | History for backtests |
|---|---|---|---|---|
| True footprint / delta / CVD | Every trade with its aggressor side | Not sold to retail. Angel quotes give LTP, LTQ, volume, best-5 depth, total buy/sell qty | **Estimates:**<br>• BVC on bars: `buy = V·Φ(ΔC/σ_ΔC)` (Easley, López de Prado & O'Hara 2012)<br>• CLV `V·(2C−H−L)/(H−L)`<br>• tick rule on streamed LTP when H3 exists | 5m bars 2024-09→, 1m 2026-03→. Tick: forward only |
| Big trades | Prints with size | Bulk/block deals (EOD, forward only); volume-spike bars | Bar volume > 5× the same-minute 20-session median; reaction at +5/+15/+30 min | 5m 2024-09→ |
| Heatmap | Order book over time | Angel best 5 (H1); 20-level depth to be probed | Depth snapshots every 5 s for ≤ 25 symbols while the server runs | Forward only |
| Absorption | Volume at a level with no progress | Bars | Volume ≥ 3× median with range ≤ 0.5× median at a day extreme, VWAP band or value-area edge | 5m |
| VWAP + bands | Intraday OHLCV | Yes | Typical price × volume; volume-weighted σ bands; anchored VWAP | 5m/1m |
| Volume profile | Volume at price | Bars | Each bar's volume spread uniformly over [L, H] in bins; POC; 70% value area | 5m/1m |
| Market profile / AMT | Price per 30 minutes | Bars | TPO letters from 09:15; IB = 09:15–10:15; Dalton day types; open types | 5m |
| GEX | Option OI by strike + IV | `fo_bhavcopy` EOD 2024-07→ (OI in shares); NSE live option chain | BS gamma from IV (`implied_vol` on settle, non-expiry rows); `GEX_k = Γ_k·OI_k·S²·0.01`, calls +, puts −; zero-gamma level; walls; PCR; max pain | EOD 2024-07→ |
| OI buildup | Futures OI + price | `fo_bhavcopy` STF/IDF 2024-07→; NSE OI spurts live; Angel NFO quote OI (after the M3.1 parser) | `oi_price_quad`; rollover; basis | 2024-07→ |
| Delivery | Delivery qty | `bhavcopy_delivery` 2019-08→ | `deliv_z20`, delivery value | 2019-08→ |
| FII/DII, participants | NSE files | `participant_oi` 2019-12→; `fii_dii` 17 days | Net index-futures long ratio; changes; cash net | 2019-12→ / forward |
| Pre-open 09:00 | NSE pre-open JSON | `preopen` (gap since 2026-10-21); `capture_preopen` | IEP gap, imbalance | forward (short) |
| Time scales | Multi-timeframe bars | Yes | Alignment table | yes |

**Conclusion** (written into the note and shown in the UI): Indian retail data supports **estimated** order flow from bars and snapshots. True footprint needs aggressor-tagged ticks, which retail does not get. Every estimated tool is labelled, e.g. "delta (BVC estimate from 1-minute bars)".

---

# PART 5 — Shared code every porter uses (write it exactly as given)

## 5.1 `src/stocksense/marketlab/npx.py`: numpy equivalents of Market Lab's pandas calls

Market Lab uses pandas in `lab/features.py`, `lab/swing.py`, `sm/*`. pandas is banned in `src/` (Part 0 rule 11), so every port calls these helpers. Their outputs must equal pandas to 1e-12: card M2.2 adds a golden test against pandas run in the clone's venv.

```python
"""Numpy equivalents of the pandas operations used by market-lab (pandas is not allowed in src/)."""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def shift(x: np.ndarray, k: int = 1, fill: float = np.nan) -> np.ndarray:
    """x shifted down by k (k>0) with `fill` in the first k slots; float array."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    if k <= 0:
        out[:] = x
        return out
    out[:k] = fill
    out[k:] = x[:-k]
    return out


def ewm_alpha(x: np.ndarray, alpha: float) -> np.ndarray:
    """pandas Series(x).ewm(alpha=alpha, adjust=False).mean() (NaNs skipped as pandas ignore_na=False does)."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    prev = np.nan
    for i, v in enumerate(x):
        if np.isnan(v):
            out[i] = prev
            continue
        prev = v if np.isnan(prev) else alpha * v + (1 - alpha) * prev
        out[i] = prev
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    """pandas ewm(span=n, adjust=False).mean()."""
    return ewm_alpha(x, 2.0 / (n + 1.0))


def rma(x: np.ndarray, n: int) -> np.ndarray:
    """pandas ewm(alpha=1/n, adjust=False).mean() (Wilder)."""
    return ewm_alpha(x, 1.0 / n)


def _win(x: np.ndarray, n: int) -> np.ndarray:
    return sliding_window_view(np.asarray(x, dtype=float), n)


def roll(x: np.ndarray, n: int, fn: str) -> np.ndarray:
    """pandas Series(x).rolling(n).<fn>() with min_periods=n: fn in mean|std|max|min|sum; first n-1 are NaN; any NaN in window -> NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    w = _win(x, n)
    if fn == "mean":
        r = w.mean(axis=1)
    elif fn == "std":
        r = w.std(axis=1, ddof=1)
    elif fn == "max":
        r = w.max(axis=1)
    elif fn == "min":
        r = w.min(axis=1)
    elif fn == "sum":
        r = w.sum(axis=1)
    else:
        raise ValueError(fn)
    out[n - 1 :] = r
    return out


def pct_change(x: np.ndarray, n: int) -> np.ndarray:
    """x/x[t-n] - 1 (pandas 3 pct_change(n) with fill_method=None)."""
    x = np.asarray(x, dtype=float)
    return x / shift(x, n) - 1.0


def ffill(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs (pandas ffill); leading NaNs stay NaN."""
    out = np.asarray(x, dtype=float).copy()
    last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def factorize(values: np.ndarray) -> np.ndarray:
    """pandas.factorize codes in order of first appearance (values already sorted by time)."""
    _, first_idx, inv = np.unique(values, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first_idx))
    return order[inv]


def group_transform(x: np.ndarray, day: np.ndarray, how: str) -> np.ndarray:
    """groupby(day).transform(how) for how in cumsum|cummax|cummin|first|max|min (groups contiguous)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    starts = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1, len(x)]
    for a, b in zip(starts[:-1], starts[1:]):
        seg = x[a:b]
        if how == "cumsum":
            out[a:b] = np.nancumsum(seg)
        elif how == "cummax":
            out[a:b] = np.fmax.accumulate(seg)
        elif how == "cummin":
            out[a:b] = np.fmin.accumulate(seg)
        elif how == "first":
            out[a:b] = seg[0]
        elif how == "max":
            out[a:b] = np.nanmax(seg) if np.isfinite(seg).any() else np.nan
        elif how == "min":
            out[a:b] = np.nanmin(seg) if np.isfinite(seg).any() else np.nan
        else:
            raise ValueError(how)
    return out
```
**Notes the executor must respect:**
- pandas `groupby.transform("cumsum")` treats NaN as skipped but keeps NaN at that position. `np.nancumsum` writes the running sum instead. Market Lab only calls cumsum on volume and tp×w, which never have NaN, so this difference is harmless. Document it in the docstring.
- `transform("first")` returns the first **non-NaN** value. Market Lab only uses it on open, which never has NaN.
- The rvol pivot (`lab/features.py` build): in M5.1 it is implemented as:
  ```python
  def rvol_same_bar(v, day, k, lookback=10, minp=5):
      cv = group_transform(v, day, "cumsum")
      nd, nk = day.max() + 1, k.max() + 1
      M = np.full((nd, nk), np.nan)
      M[day, k] = cv
      avg = np.full_like(M, np.nan)
      for d in range(nd):
          lo = max(0, d - lookback)
          W = M[lo:d]                       # previous days only (the shift(1))
          cnt = np.sum(~np.isnan(W), axis=0)
          with np.errstate(invalid="ignore"):
              m = np.nanmean(W, axis=0) if len(W) else np.full(nk, np.nan)
          avg[d] = np.where(cnt >= minp, m, np.nan)
      with np.errstate(invalid="ignore", divide="ignore"):
          return cv / avg[day, k]
  ```
  This equals `piv.rolling(10, min_periods=5).mean().shift(1)` followed by the lookup.

## 5.2 `src/stocksense/marketlab/clock.py`: market time, matching Market Lab's minute arithmetic
```python
"""Ported from market-lab sm/market.py (MIT, ba3f771); changes: our TradingCalendar and Clock instead of holiday-master and datetime.now."""
from __future__ import annotations
from datetime import datetime
from typing import Literal
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, Clock

Status = Literal["closed", "pre-open", "open"]

def market_status(clock: Clock, cal: TradingCalendar, t: datetime | None = None) -> Status:
    """Market Lab's status: pre-open 09:00-09:14, open 09:15-15:29, else closed (sessions from our calendar)."""
    t = (t or clock.now()).astimezone(IST)
    if not cal.is_session(t.date()):
        return "closed"
    m = t.hour * 60 + t.minute
    if m < 555:
        return "pre-open" if m >= 540 else "closed"
    return "open" if m < 930 else "closed"

def minute_now(clock: Clock) -> int:
    """Minutes since 09:15 IST (0 at 09:15, 375 at 15:30); negative before the open."""
    t = clock.now().astimezone(IST)
    return t.hour * 60 + t.minute - 555

def session_minute(ts_epoch_s: int) -> int:
    """Minutes since 09:15 IST of an epoch-second timestamp (Market Lab's JS: floor((epoch+19800)%86400/60)-555)."""
    return int(((ts_epoch_s + 19800) % 86400) // 60) - 555
```
**Calendar caveat:** `TradingCalendar.is_session(today)` must be true on a normal weekday before EOD data exists. Check how `from_store` projects future sessions (`core/calendar.py:60`): it projects weekdays minus holidays. If today is not reported as a session, record it and fall back to `today.weekday() < 5 and today not in holidays`.

## 5.3 `src/stocksense/marketlab/store.py`
```python
from __future__ import annotations
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

@dataclass
class FeedStat:
    interval_s: float
    last_ok: datetime | None = None
    last_err: str | None = None       # "<ExceptionName>: <first 120 chars>" (never secrets)
    errors: int = 0

@dataclass
class LabStore:
    """Thread-safe in-memory state shared by the feed threads and the API (one per app)."""
    lock: threading.RLock = field(default_factory=threading.RLock)
    tape: dict[str, Any] = field(default_factory=dict)        # sm/feed STATE equivalent
    sx: dict[str, Any] = field(default_factory=dict)          # sx/live S equivalent
    univ: dict[str, Any] = field(default_factory=dict)        # sx/universe U equivalent
    flow: dict[str, Any] = field(default_factory=dict)
    feeds: dict[str, FeedStat] = field(default_factory=dict)
    catchup: dict[str, Any] = field(default_factory=lambda: {"state": "idle", "steps": []})
    slow_until: dict[str, float] = field(default_factory=dict)   # source -> epoch seconds (429 back-off)

    def ok(self, feed: str, now: datetime) -> None: ...
    def fail(self, feed: str, now: datetime, err: BaseException) -> None: ...
    def snapshot_status(self) -> dict[str, Any]: ...

class TTLCache:
    """Thread-safe get-or-compute cache with per-key expiry (replaces market-lab app.py `cached`)."""
    def __init__(self, clock) -> None: ...
    def get(self, key: Any, ttl_s: float, fn: Callable[[], Any]) -> Any: ...
```
Every Market Lab global (`STATE`, `S`, `U`, `_C`, `GOT`) becomes a field of the one `LabStore` held in `app.state.store`. Read and write it under `store.lock`. Copy the data out (`dict(...)`) before returning it in an API response.

## 5.4 Feed loop pattern (`marketlab/feeds.py`), the only allowed shape
```python
def _loop(name: str, store: LabStore, clock: Clock, stop: threading.Event,
          step: Callable[[], float]) -> None:
    """Run step() until stop is set; step returns the seconds to wait before the next call."""
    while not stop.is_set():
        try:
            wait = step()
            store.ok(name, clock.now())
        except Exception as e:  # feed boundary: log, count, continue (PLAN-M Part 0 rule 11)
            log.exception("feed_error", feed=name)
            store.fail(name, clock.now(), e)
            wait = 30.0
        stop.wait(wait)
```
Threads are `threading.Thread(target=_loop, args=(...), name=f"feed-{name}", daemon=True)`.

## 5.5 HTTP for non-NSE sources (`marketlab/sources.py`)
- One `httpx.Client(timeout=httpx.Timeout(20, connect=10), follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})` per app. The transport can be injected for tests.
- On a 429: set `store.slow_until[source] = now + 120` and raise `SourceThrottled`. Callers skip that source until the time passes.
- NSE website JSON goes through `NseClient.get_api(path, params, referer_path)`. Its warm-up handles cookies. Use referer paths `"market-data/live-equity-market"` (market-watch), `"option-chain"` (option chain), `"market-data/pre-open-market-cm-and-emerge-market"` (pre-open) and `"get-quotes/equity?symbol=<SYM>"` (quote).
- NseClient spacing (0.6 s) applies, so the 4-second index loop makes at most a few NSE calls per iteration. Market Lab's exact intervals are kept (M2.3). If NSE calls cannot keep the interval, the loop simply runs late: never parallelise NSE calls.

## 5.6 Frontend "imperative island" pattern (how canvas/SVG code is ported)
Market Lab renders with template strings, canvas and SVG. To keep the look identical without re-designing:
```tsx
"use client";
import { useEffect, useRef } from "react";
import { drawStage } from "@/lib/ml/smart/stage";     // pure TS port of smart.html draw()/beads()/etc.

export function Stage({ data, hover, onHover, onOpen }: StageProps) {
  const bg = useRef<HTMLCanvasElement>(null), fx = useRef<HTMLCanvasElement>(null);
  useEffect(() => { if (bg.current && data) drawStage(bg.current, data, hover); }, [data, hover]);
  return (<div id="stage"><canvas id="bg" ref={bg} /><canvas id="fx" ref={fx} />{/* … */}</div>);
}
```
- Pure drawing and formatting functions live in `web/src/lib/ml/<page>/*.ts` and are ported line by line from the HTML `<script>` with types added.
- Where Market Lab builds HTML strings with `innerHTML`, use React JSX for simple lists, and an island `div ref` with `innerHTML` **only** for large template blocks (research note, tile details). In that case keep Market Lab's `esc()` on every interpolated value. The test in M6.2 asserts that `esc` is applied, by rendering a symbol containing `<script>`.
- Global cross-part calls (`window.openChart`, `window.labShow`, `window.go`) become a React context, `LabActions { openChart(sym), labShow(sym, strat, tf), go(view, arg) }`.

## 5.7 CSS scoping (replaces Market Lab's Shadow DOM)
- Install `postcss-prefix-selector@<exact>`. Files: `web/src/styles/ml-smart.css`, `ml-sector.css`, `ml-lab.css`, each copied verbatim from the `<style>` block of the matching HTML. `web/postcss.config.mjs`:
  ```js
  import prefixer from "postcss-prefix-selector";
  export default { plugins: [prefixer({ prefix: "", transform(prefix, selector, prefixed, file) {
      const m = /ml-(smart|sector|lab)\.css$/.exec(file || ""); if (!m) return selector;
      const scope = `.ml-${m[1]}`;
      if (selector === ":root" || selector === "html,body" || selector === "html" || selector === "body") return scope;
      if (selector.startsWith("@")) return selector;
      return `${scope} ${selector}`;
  } })] };
  ```
- Each section root: `<section className="ml-smart">…</section>` (and so on). This mirrors index.html's rewrite of `:root` to `:host` and `html,body{` to `:host{display:block;`.
- Fonts: in `web/src/app/layout.tsx` `<head>`, the same Google Fonts link as Market Lab's `index.html`:
  `https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap`.
- The global page background `#edf0f3`, the sticky disclaimer `#disc` and the `.divider` come from index.html's own `<style>`, in `globals.css`.

---
# PART 6 — Task cards

Card format: **Depends** · **Files** · **Spec** (with skeletons) · **Tests** (with numbers) · **Acceptance** · **Commit** · **Provides**.

## PHASE M0 — Finish the light G2 work and set up (sequential; CPU)

### M0.0 Orchestrator housekeeping (orchestrator only)
1. Check for stray processes and the GPU:
   ```powershell
   Get-CimInstance Win32_Process | ? { $_.Name -match 'python|node' } | select ProcessId,CommandLine
   nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
   ```
   Kill any project leftovers with `taskkill /F /T /PID <pid>`.
2. In the main repo:
   - `git restore --staged .vscode/settings.json`;
   - append `xyz.txt` and `.vscode/` to `.git/info/exclude` (local only; do not edit `.gitignore` for these).
3. Remove the merged, clean worktree:
   - `git worktree remove ..\stocksense-wt-g2d1`;
   - then `git branch -d g2/d1`;
   - then confirm `Test-Path data_store\parquet\bhavcopy_eq` is still True.
4. Copy this plan file to `docs/PLAN-M.md` only in M0.3 (not here).

### M0.1 Finish G2-H (H1, H4, H5) and merge
**Where:** worktree `..\stocksense-wt-g2h`, branch `g2/h` (H1 `692b119` and H4 `8331869` are already committed).
**Spec:** finish PLAN-G2 card H5 (`docs/PLAN-G2.md:461`) with the uncommitted files already in the worktree:
- `src/stocksense/data/mcp_verify.py` (`VerifyResult`, `verify`);
- `tests/unit/test_h5_mcp_verify.py`;
- modified `brokers/nse_mcp.py`, `explain/report.py`, `live/jobs.py`, `tests/integration/test_full_day.py`.

**Steps:**
1. `git -C ..\stocksense-wt-g2h status`, then read every modified file and the H5 card.
2. Run `.venv\Scripts\python.exe -m pytest tests/unit/test_h5_mcp_verify.py -q`. Fix until these pass:
   - all-match fake → `VerifyResult(mismatch=0)` and job detail `mismatch=0`;
   - one close off by 0.5 → `mismatch=1`;
   - transport raising → status `skipped`, detail `unreachable`;
   - the job never raises.
3. Full suite and ruff in the worktree. Commit `G2-H5: nightly NSE MCP verification stage` (no attribution).
4. Merge: `git merge main` into `g2/h` if main moved (keep both sides of docs conflicts). Then from the main repo `git merge --no-ff g2/h`, full suite, ruff, push.
5. Tick H1, H4 and H5 in `docs/PLAN-G2.md` Part 9; commit `docs: tick G2 H1 H4 H5`.
6. `git worktree remove ..\stocksense-wt-g2h`.

**Provides:** `Quote` depth/OI/totals fields; `ingest_misc.depth_imbalance`; the new `spreads` columns; `brokers/nse_mcp.py` (`make_call`, `call_many`, `check_fresh`, `fetch_live_index`, `fetch_call_auction`, `call_auction_symbols`); the `mcp_verify` stage.

### M0.2 Finish G2-D7 (IPCA) and merge
**Where:** worktree `..\stocksense-wt-g2d7`, branch `g2/d7`.

**Steps:**
1. Free RAM must be ≥ 6 GB and nothing else heavy may be running.
2. Locate the spec: `C:/Users/harsh/AppData/Local/Temp/d7_ipca_spec.json`.
   - If it is missing, read the spec of trial `g2-ipca-y_c2o_10-045dcfd195` from the registry and write it to `data_store/state/d7_ipca_spec.json`:
     ```
     .venv\Scripts\python.exe -c "import json,sqlite3;c=sqlite3.connect('data_store/state/stocksense.sqlite');print(c.execute(\"select spec_json from trials where trial_id='g2-ipca-y_c2o_10-045dcfd195'\").fetchone()[0])"
     ```
   - The `wf-run` spec JSON is the `spec` field inside. **The re-registered trial id must equal `g2-ipca-y_c2o_10-045dcfd195`.** If it does not, stop and report.
3. Launch from the worktree in the Bash tool, in the background:
   ```bash
   cd /c/Users/harsh/Desktop/HARSH/WEB-DEV-PROJECTS/stocksense-wt-g2d7
   POLARS_MAX_THREADS=4 PYTHONPATH="$(pwd -W)/src" timeout 23400 ../stocksense/.venv/Scripts/python.exe -m stocksense research wf-run \
     --spec-json <spec> --trial-family g2-ipca --folds 2016..2025 --refit-every 1 --max-hours 6 --seeds 1,2,3 \
     > data_store/logs/g2_D7_ipca_wf4.log 2>&1
   ```
   The run resumes per fold (2016–2020 are already done).
4. If it dies with `MemoryError`:
   - in `src/stocksense/models/ipca.py`, cast the `(T, L, L)` weight tensor to `np.float32`;
   - add a test: on the synthetic market, float32 vs float64 predictions agree within 1e-4;
   - commit `G2-D7: ipca float32 weights`;
   - relaunch once.
5. Run `research model-eval --trial g2-ipca-y_c2o_10-045dcfd195 --base s1-gbm-y_c2o_10-e0f7276025` (same PYTHONPATH form).
6. Docs:
   - `docs/DECISIONS.md` `### D7`: numpy ALS (not the pip package); no label winsorizing; ALS weights N_t × date weight; cae as a standalone torch model; λ̂ = date-weighted mean factor return; ipca not converged in 100 iterations; cae best epoch 1–2; the three memory crashes and the fix.
   - `docs/RESULTS-G2.md` section 2, two rows with 3-line verdicts:
     - cae: mean IC 0.0203 (t 6.31), corr base 0.28, orth IC 0.0029 (t 0.4), blend −0.0155 (t −3.3), S1 net 0.07, 2× −0.31, NOT admitted;
     - ipca: from its `eval.json`.
7. Commit `G2-D7: ipca and cae results`. Merge `--no-ff`, full suite, push. Remove the worktree and branch.

**Provides:** the D7 verdict. G2 then **pauses**: D1 MASTER, D2–D6, D8–D10 and D12 resume after Program M (Part 7).

### M0.3 Program M scaffolding and decisions
**Files:** `docs/PLAN-M.md`, `docs/DECISIONS.md`, `docs/PLAN.md` (A.6 table only), `third_party/market-lab/LICENSE`, `third_party/market-lab/NOTICE.md`, `.gitignore`, `tests/unit/test_third_party_notice.py`.

**Spec:**
1. `docs/PLAN-M.md` = this plan file, verbatim.
2. `third_party/market-lab/LICENSE` = a byte copy of `C:\Users\harsh\Downloads\market-lab\LICENSE`.
3. `NOTICE.md`:
   ```markdown
   # market-lab (MIT)
   Upstream: https://github.com/heysanjeevv/market-lab, commit ba3f771 (local clone C:\Users\harsh\Downloads\market-lab).
   Ported into StockSense (rewritten in numpy/polars and TypeScript; behaviour preserved except the fixes listed in docs/PLAN-M.md Context):
   sm/feed.py, sm/smc.py, sm/structure.py, sm/market.py, sx/universe.py, sx/live.py, sx/news.py, lab/swing.py, lab/swing_live.py,
   lab/swing_pf.py, lab/swing_an.py, lab/features.py, lab/strategies.py, lab/engine.py, lab/live.py, lab/research.py, lab/industries.py,
   web/index.html, web/smart.html, web/sector.html, web/lab.html.
   ```
4. `.gitignore` additions: `web/node_modules/`, `web/.next/`, `web/out/`, `web/content/learn/`, `web/next-env.d.ts`.
5. `docs/PLAN.md` A.6: add these rows to the source table.
   - **NSE website live JSON** (`www.nseindia.com/api`): `allIndices`; `NextApi/apiClient/marketWatchApi?functionName=getIndicesData` (F&O list, NIFTY 500, sector indices); `NextApi/apiClient/GetQuoteApi?functionName=getSymbolData` (classification, delivery); `getIntegratedFilingData`; `results-comparision`; option chain; OI spurts; pre-open.
     - Job: the live dashboard (Program M).
     - Not used for: research history.
   - **Yahoo Finance** (`query1/query2.finance.yahoo.com`, spark + chart).
     - Job: the dashboard's live 1-minute tape and intraday chart candles, as Market Lab does; the macro readings in the research note (`BZ=F, USDINR=X, HG=F, ^GSPC`).
     - Not used for: research, features, labels or gated backtests.
   - **Google News RSS** (`news.google.com/rss/search`).
     - Job: dashboard headlines and chart pins.
     - Not used for: trading decisions.
   - **Wikipedia REST** (`en.wikipedia.org`).
     - Job: the research-note company summary.
     - Not used for: anything else.
6. `docs/DECISIONS.md`, new section `## Program M`, `### M0.3`:
   - the Part 1 interpretations table;
   - the user's decisions of 2026-11-23;
   - port 8765;
   - Next.js static export;
   - lightweight-charts 4.2.0;
   - no pandas in `src/` (numpy ports);
   - no pickle;
   - the list of Market Lab defects not ported;
   - "Market Lab's ranking is price-only; the flow layer is new";
   - the engine cannot fill at stop prices, so the library uses `strategies/portfolio.py`.
7. Test `test_third_party_notice.py`:
   ```python
   def test_market_lab_notice():
       root = Path(__file__).resolve().parents[2] / "third_party" / "market-lab"
       assert "MIT License" in (root / "LICENSE").read_text(encoding="utf-8")
       assert "ba3f771" in (root / "NOTICE.md").read_text(encoding="utf-8")
   ```

**Acceptance:** `.venv\Scripts\python.exe -m pytest tests/unit/test_third_party_notice.py -q` gives 1 passed; then the full suite.
**Commit:** `M-0.3: Program M plan, sources, licence notice`.

---

## PHASE M1 — Learning notes (docs only; one executor; web research allowed; runs in parallel with M2)

### M1.1 `docs/learn/`
**Files:** `docs/learn/README.md` plus the notes below, and `tests/unit/test_learn_docs.py`.

**Every note has exactly these `##` headings, in this order:**
1. `What it is`
2. `How it is computed`
3. `What it says about flow and positioning`
4. `What NSE data allows and how StockSense computes it`
5. `Evidence`
6. `How traders misuse it`
7. `Where it appears in the dashboard`
8. `Sources`

**Rules:**
- Formulas use plain text math.
- Section 4 names the module and function from Part 2 and the history available (Part 4).
- Section 5 cites papers with links.
- No paid-course pages as sources. Prefer papers, exchange docs and textbooks.
- Use web search and fetch. If a URL fails, use another source.

**Notes (file: required specifics):**
- `00-data-reality.md`: the Part 4 table expanded, plus why aggressor-tagged ticks are not available to retail in India.
- `01-order-flow.md`: market vs limit orders, aggressor, imbalance; Cont, Kukanov & Stoikov (2014) order-flow imbalance.
- `02-delta-and-cvd.md`: tick rule, Lee–Ready (1991), BVC (Easley, López de Prado & O'Hara 2012), CLV; CVD; divergence.
- `03-footprint.md`: bid×ask volume ladder; stacked imbalances; our estimate from 1-minute BVC.
- `04-big-trades-and-absorption.md`: block and bulk deals (NSE rules and windows), volume spikes, absorption, price reaction.
- `05-heatmap-and-liquidity.md`: resting depth, spoofing caveats, best-5 vs deeper depth.
- `06-vwap.md`: VWAP, bands, anchored VWAP, VWAP mean reversion, institutional VWAP execution.
- `07-volume-profile.md`: POC, VAH/VAL, 70% value area, HVN/LVN.
- `08-market-profile-and-amt.md`: TPO, IB, Dalton day types and open types, the 80% rule, poor highs/lows, single prints, balance vs imbalance ("the law of the market").
- `09-time-scales.md`: fractal timeframes, alignment.
- `10-options-positioning-and-gex.md`: OI by strike, PCR, max pain, IV skew, dealer gamma, GEX sign conventions, zero-gamma, walls; Barbon & Buraschi (2021), Ni, Pearson, Poteshman & White (2021).
- `11-oi-buildup.md`: the four quadrants, rollover, basis.
- `12-delivery-fii-dii-participants.md`.
- `13-preopen-0900.md`: NSE pre-open call auction timings (cite the NSE circular), IEP computation, imbalance, the F&O pre-open if applicable, block-deal windows (08:45–09:00 and 14:05–14:20), why "the market opens at 9:00".
- `14-tier-list-assessment.md`: each tier strategy (Part 1), with evidence for and against and our pre-registered hypothesis.
- `15-151-strategies-map.md`: **a table of every strategy in the paper by section number**: `| § | name | one-line formula (eq. numbers) | asset class | status + reason | StockSense key |`. Status is one of implemented, adapted, informational, await_data, deferred or infeasible, exactly as M5.4 assigns it.
  - The executor reads the user's local copy `C:\Users\harsh\Downloads\ssrn-3247865.pdf` with PyMuPDF, using the extraction command in M5.4. Read **every** page, including Appendix A and the glossary; do not use the abstract or contents alone.
  - Notes `08`, `10`, `11`, `14` and `17` also cite the relevant paper sections.
  - Also add `docs/learn/18-151-glossary-notes.md`: the paper's glossary terms used anywhere in StockSense (e.g. alpha, basis, carry, contango, delta, dollar-neutral, eRank, IBS, Sharpe ratio, VRP), each with a one-line definition written in our own words plus the paper's page number. Do not copy the paper's glossary text. This file is a table, so it is not in `NOTES` (no 8-heading check). A separate test asserts that it exists and has ≥ 40 rows.
- `16-smart-money-concepts.md`: BOS/CHoCH/OB/FVG/sweeps exactly as `marketlab/smc.py` computes them; the absence of academic evidence.
- `17-macro-trading-india.md`: crude, USDINR, US10Y, DXY, FII flows, RBI dates; how the index-timing rules use them.

**Tests:**
```python
NOTES = ["00-data-reality", ..., "17-macro-trading-india"]   # all 18 names above
HEADINGS = ["What it is", "How it is computed", "What it says about flow and positioning",
            "What NSE data allows and how StockSense computes it", "Evidence", "How traders misuse it",
            "Where it appears in the dashboard", "Sources"]
@pytest.mark.parametrize("n", NOTES)
def test_note_shape(n):
    t = (LEARN / f"{n}.md").read_text(encoding="utf-8")
    assert [h for h in re.findall(r"^## (.+)$", t, re.M)] == HEADINGS
    assert "http" in t.split("## Sources", 1)[1]
def test_151_table_complete():
    # K151_SECTIONS: 176 pinned section ids (174 strategy sections of the paper's contents + "A.d0", "A.d1");
    # STATUS = {"implemented","adapted","informational","await_data","deferred","infeasible"}. See M5.4 "Map completeness".
    t = (LEARN / "15-151-strategies-map.md").read_text(encoding="utf-8")
    rows = [l for l in t.splitlines() if l.startswith("| ") and not l.startswith("| §") and not l.startswith("|---")]
    secs = [r.split("|")[1].strip() for r in rows]
    assert sorted(secs) == sorted(K151_SECTIONS) and len(secs) == len(set(secs))
    for r in rows:
        assert r.split("|")[5].strip().split()[0] in STATUS
```
**Commit:** `M-1.1: learning notes`.

---

## PHASE M2 — Backend: the Market Lab port (sequential; one executor at a time)

### M2.0 API skeleton, GET-only, `serve` command, no-orders scan extended
**Depends:** M0.3.
**Files:**
- `src/stocksense/api/{app.py,deps.py,readonly.py,static.py,routes_status.py}`
- `src/stocksense/cli_lab.py`
- `src/stocksense/cli.py` (one line)
- `tests/unit/test_api_skeleton.py`
- `tests/unit/test_no_orders.py` (extend)

**Spec, `api/app.py`:**
```python
"""Read-only HTTP API and dashboard server (PLAN.md T12.3, PLAN-M M2)."""
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from stocksense import __version__
from stocksense.core.clock import Clock, SystemClock
from stocksense.core.config import Settings
from stocksense.marketlab.store import LabStore

WEB_OUT = Path(__file__).resolve().parents[3] / "web" / "out"

def create_app(settings: Settings, *, start_feeds: bool = True, clock: Clock | None = None,
               store: LabStore | None = None, web_out: Path = WEB_OUT) -> FastAPI:
    """FastAPI app: GET-only JSON under /api, /health, and the static dashboard at /."""
    clock = clock or SystemClock()
    store = store or LabStore()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mgr = None
        if start_feeds:
            from stocksense.marketlab.feeds import FeedManager   # lazy: imports httpx clients, brokers
            mgr = FeedManager(settings, clock, store)
            mgr.start()
        app.state.feeds = mgr
        yield
        if mgr is not None:
            mgr.stop()

    app = FastAPI(title="StockSense", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings, app.state.clock, app.state.store = settings, clock, store
    from stocksense.api.readonly import ReadOnlyMiddleware
    app.add_middleware(ReadOnlyMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                       allow_methods=["GET"], allow_headers=["*"])
    from stocksense.api import routes_status
    app.include_router(routes_status.router)
    # later cards: app.include_router(routes_sm.router) etc. — each card adds its own include line here
    from stocksense.api.static import mount_web
    mount_web(app, web_out)
    return app
```
**`api/readonly.py`:**
```python
class ReadOnlyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            return JSONResponse({"error": "read-only API"}, status_code=405)
        resp = await call_next(request)
        if request.url.path.startswith("/api"):
            resp.headers["Cache-Control"] = "no-store"
        return resp
```
(`OPTIONS` is allowed for CORS preflight only.)

**`api/static.py`:**
- `mount_web(app, web_out)`: if `(web_out / "index.html").exists()`, run `app.mount("/", StaticFiles(directory=web_out, html=True), name="web")`.
- Otherwise add `GET /` returning `HTMLResponse("<h1>StockSense</h1><p>Dashboard not built: run run-lab.bat</p>")`.
- Mount **after** all routers.

**`api/routes_status.py`:** `router = APIRouter()`.
- `GET /health` → `{"ok": True, "version": __version__}`.
- `GET /api/status` → `store.snapshot_status()`:
  ```
  {"ready": bool, "feeds": {name: {"interval_s", "last_ok", "errors", "last_err"}},
   "catchup": {...}, "time": iso, "market": status}
  ```
  In M2.0, `market` is `"closed"` until M2.1 lands.

**`api/deps.py`:** `def store(request: Request) -> LabStore: return request.app.state.store`; likewise `settings`, `clock`.

**`cli_lab.py`:**
```python
def register(app: typer.Typer) -> None:
    app.command("serve")(serve)
    app.add_typer(lab_app, name="lab")

def serve(ctx: typer.Context, port: int = 8765, no_feeds: bool = False, no_catchup: bool = False) -> None:
    """Start the local dashboard server on 127.0.0.1 (manual start only; Ctrl+C stops it)."""
    import uvicorn
    from stocksense.api.app import create_app
    s = _settings(ctx)
    app = create_app(s, start_feeds=not no_feeds)
    app.state.catchup_enabled = not no_catchup
    uvicorn.run(app, host="127.0.0.1", port=port, workers=1, log_level="info")
```
`_settings` is imported from `stocksense.cli` lazily inside the function, to avoid a circular import. `cli.py` gets one line after line 828:
```python
__import__("stocksense.cli_lab").cli_lab.register(app)
```

**`test_no_orders.py` extension:** the existing file scans `src/**/*.py`. Add:
```python
def test_no_order_strings_in_web():
    web = REPO / "web" / "src"
    if not web.exists():
        pytest.skip("web not created yet")
    bad = [f"{p}:{i}" for p in web.rglob("*.ts*") for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
           if PATTERN.search(line)]
    assert not bad, bad
```
Reuse the module's existing compiled regex name (read the file; it may be called something other than `PATTERN`).

**Web fixture dump (used by every later API card and by M6):**
- Add a pytest option in `tests/conftest.py`:
  ```python
  def pytest_addoption(parser):
      parser.addoption("--write-web-fixtures", action="store_true", default=False,
                       help="API tests write their JSON responses to web/src/test/fixtures/")
  @pytest.fixture
  def web_fixture(request):
      """Call web_fixture(name, payload) in an API test; writes only when --write-web-fixtures is given."""
      def _w(name: str, payload) -> None:
          if request.config.getoption("--write-web-fixtures"):
              out = Path(__file__).resolve().parents[1] / "web" / "src" / "test" / "fixtures"
              out.mkdir(parents=True, exist_ok=True)
              (out / f"{name}.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
      return _w
  ```
  If `tests/conftest.py` already defines `pytest_addoption`, add the option inside it.
- Every API route test in M2.3–M4.8 calls `web_fixture("<route_name>", r.json())`.

**Tests (`test_api_skeleton.py`):**
```python
@pytest.fixture
def client(tmp_data_root, fixed_clock, tmp_path):
    app = create_app(tmp_data_root, start_feeds=False, clock=fixed_clock, web_out=tmp_path / "noweb")
    with TestClient(app) as c:
        yield c
def test_health(client):
    r = client.get("/health"); assert r.status_code == 200 and r.json()["ok"] is True
@pytest.mark.parametrize("m", ["post", "put", "patch", "delete"])
@pytest.mark.parametrize("path", ["/health", "/api/status", "/"])
def test_write_methods_405(client, m, path):
    assert getattr(client, m)(path).status_code == 405
def test_fallback_page(client):
    assert "run-lab.bat" in client.get("/").text
def test_static_served(tmp_data_root, fixed_clock, tmp_path):
    out = tmp_path / "out"; out.mkdir(); (out / "index.html").write_text("<p>hi</p>", encoding="utf-8")
    with TestClient(create_app(tmp_data_root, start_feeds=False, clock=fixed_clock, web_out=out)) as c:
        assert c.get("/").text == "<p>hi</p>"
def test_api_no_store(client):
    assert client.get("/api/status").headers["cache-control"] == "no-store"
def test_serve_help():
    r = CliRunner().invoke(app, ["serve", "--help"]); assert r.exit_code == 0 and "8765" in r.output
```
**Acceptance:** `.venv\Scripts\python.exe -m pytest tests/unit/test_api_skeleton.py tests/unit/test_no_orders.py -q` all pass; full suite green.
**Commit:** `M-2.0: read-only API skeleton and serve command`.
**Provides:** `create_app`, `LabStore` stub (fields + `snapshot_status`), the router include convention, `stocksense serve`, `cli_lab.register`.

### M2.1 Sources, market clock, store, `Paths.lab` + live probe
**Depends:** M2.0.
**Files:**
- `src/stocksense/marketlab/{__init__.py,npx.py,clock.py,store.py,sources.py}`
- `src/stocksense/core/paths.py` (`lab` property)
- `tests/unit/test_ml_npx.py`, `test_ml_clock.py`, `test_ml_store.py`, `test_ml_sources.py`
- `tests/fixtures/marketlab/probe_*.json`

**Spec:**
1. `npx.py`, `clock.py` and `store.py` exactly as Part 5.1–5.3 (fill in the `...` bodies).
   - `LabStore.ok/fail` update `FeedStat`.
   - `snapshot_status` returns plain dicts with ISO times.
2. `sources.py`. Each function returns parsed Python data or raises `SourceError(source, detail)` / `SourceThrottled(source)`:
   ```python
   class Sources:
       def __init__(self, settings: Settings, clock: Clock, store: LabStore, nse: NseClient | None = None,
                    http: httpx.Client | None = None) -> None: ...
       # NSE (all via self.nse.get_api(path, params, referer_path))
       def nse_all_indices(self) -> dict[str, dict]:                 # port sx/live.all_indices (keys verbatim)
       def nse_watch(self, index_name: str) -> dict[str, dict]:      # port sx/live.watch; "SECURITIES IN F&O" for the F&O list
       def nse_symbol_data(self, sym: str) -> dict:                  # port sx/universe.quote output keys
       def nse_integrated_filing(self, sym: str) -> list[dict]:      # lab/research.quarters raw payload
       def nse_results_comparison(self, sym: str) -> dict:           # lab/research.balance raw payload
       def nse_option_chain(self, symbol: str, is_index: bool) -> dict:   # probe path (see below)
       def nse_oi_spurts(self) -> list[dict]:                         # probe path
       def nse_preopen(self, key: str) -> dict:                       # reuse the path used by ingest_misc.capture_preopen
       # non-NSE (self.http)
       def yahoo_spark(self, symbols: list[str]) -> dict[str, dict]:  # port sm/feed spark: 20 per call, 0.08 s gap via clock.sleep
       def yahoo_chart(self, ysym: str, interval: str, rng: str) -> dict:   # raw chart.result[0]
       def google_news(self, q: str, days: int, limit: int) -> list[dict]:  # port sx/news.search parsing (title strip, dedupe 60 chars)
       def wiki_summary(self, name: str) -> dict | None:              # port lab/research.wiki + stricter match (M2.5)
       def nifty500_list(self) -> dict[str, dict]:                    # ind_nifty500list.csv via self.nse.get_archive
   ```
   - The path for `nse_watch` is `"/api/NextApi/apiClient/marketWatchApi"` with params `{"functionName": "getIndicesData", "symbol": index_name}` and referer `"market-data/live-equity-market"`. If it returns 404/401 after warm-ups, try `"/api/equity-stockIndices"` with `{"index": index_name}` (legacy), adapt the parser (`data[]` rows with the same field names), and record which works.
   - `nse_symbol_data`: `"/api/NextApi/apiClient/GetQuoteApi"` with `{"functionName": "getSymbolData", "marketType": "N", "series": "EQ", "symbol": sym}`. Map the fields as `sx/universe.quote` (61-70) does. Read that function in the clone for the exact JSON paths.
   - Option chain: try `"/api/option-chain-indices"` `{"symbol": "NIFTY"}` and `"/api/option-chain-equities"` `{"symbol": "RELIANCE"}`, referer `"option-chain"`. If both fail, try the newer `"/api/NextApi/apiClient/GetQuoteApi"` `{"functionName": "getOptionChainData", ...}` (inspect what the NSE option-chain page requests, via WebFetch of the page HTML/JS or the Chrome MCP network tab; record).
   - OI spurts: `"/api/live-analysis-oi-spurts-underlyings"`, referer `"market-data/oi-spurts"`.
   - Yahoo symbol map: stocks `SYM.NS`; indices from `sx/universe.YAHOO` (copy the dict) plus `{"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}`. Read `sm/feed.py` and `lab/bars.py:17-20` for the full map.
   - A 429 from Yahoo sets `store.slow_until["yahoo"] = clock.now().timestamp() + 120` and raises `SourceThrottled`. Every Yahoo call first checks `slow_until`.
3. **Probe (live, once, during market hours if possible; otherwise outside them, noting it):**
   - Script `scripts/m_probe_sources.py` calls each `Sources` method once and prints only: status, top-level keys, row counts, and the first row's keys.
   - Save trimmed responses (≤ 30 rows; no personal data) to `tests/fixtures/marketlab/probe_<name>.json`.
   - Write the findings to DECISIONS `### M2.1`.

**Tests:**
- `test_ml_npx.py`, hand values:
  - `ema([1,2,3], 2)`: alpha = 2/3, gives `[1, 1.6667, 2.5556]` (1e-4).
  - `rma([1,2,3], 2)`: alpha = 0.5, gives `[1, 1.5, 2.25]`.
  - `roll([1,2,3,4], 2, "std")` gives `[nan, .7071, .7071, .7071]`.
  - `roll([1,nan,3], 2, "max")` gives `[nan, nan, nan]`.
  - `shift([1,2,3], 1)` gives `[nan, 1, 2]`.
  - `ffill([nan,1,nan,2])` gives `[nan, 1, 1, 2]`.
  - `factorize(["b","b","a","c"])` gives `[0, 0, 1, 2]`.
  - `group_transform([1,2,3,4], [0,0,1,1], "cumsum")` gives `[1, 3, 3, 7]`.
  - `rvol_same_bar`: 7 days × 2 bars, volumes all 10 except day 6 bar 0 = 20. Day 6 bar 0 gives 20/10 = 2.0; days 0–4 give NaN (fewer than 5 previous days).
- `test_ml_clock.py`: `market_status` for a FixedClock on a session (a `TradingCalendar.from_dates` with that date):
  - 08:59 → closed, 09:00 → pre-open, 09:14 → pre-open, 09:15 → open, 15:29 → open, 15:30 → closed;
  - a non-session date → closed;
  - `session_minute(1763266500)` (2025-11-16 04:15 UTC = 09:45 IST) → 30.
- `test_ml_store.py`: `ok/fail` counts; `snapshot_status` JSON-serialisable; `TTLCache` calls fn once within the TTL (SteppingClock) and again after it.
- `test_ml_sources.py` (`httpx.MockTransport`, plus a fake NseClient built with `transport=`):
  - each parser on its probe fixture gives the expected keys (from the probe);
  - a Yahoo 429 sets `slow_until` and raises `SourceThrottled`;
  - spark with 45 symbols makes 3 calls;
  - `nse_watch` falls back to the legacy path when the first returns 404.

**Acceptance:** the four test files pass; full suite green; the probe section exists in DECISIONS.
**Commit:** `M-2.1: market-lab sources, clock, store, numpy helpers`.
**Provides:** `marketlab.npx.*`, `market_status`, `minute_now`, `session_minute`, `LabStore`, `TTLCache`, `Sources`, `Paths.lab`.

### M2.2 Structure and SMC port with golden tests (numpy)
**Depends:** M2.1.
**Files:**
- `src/stocksense/marketlab/{structure.py,smc.py}`
- `scripts/golden/make_inputs.py`, `scripts/golden/run_marketlab.py`
- `tests/fixtures/marketlab/inputs_*.json`, `golden_*.json`
- `tests/unit/test_ml_structure_golden.py`

**Port from:** `sm/structure.py` and `sm/smc.py` (read them fully in the clone). Keep every public function name and every output key:
- `add_indicators`, `swings`, `label_swings`, `structure_events`, `zones`, `session_levels`, `regime`, `analyse`;
- `fvgs`, `order_blocks`, `read`, and the day-events logic.

**Bar frame type.** Market Lab passes pandas DataFrames with a DatetimeIndex. Our port takes a `Bars` dataclass:
```python
@dataclass(frozen=True)
class Bars:
    ts: np.ndarray            # int64 epoch seconds (UTC), ascending
    open: np.ndarray; high: np.ndarray; low: np.ndarray; close: np.ndarray; volume: np.ndarray
    @staticmethod
    def from_polars(df: pl.DataFrame) -> "Bars": ...   # expects columns ts (Datetime IST) or time (int), open, high, low, close, volume
    def ist_dates(self) -> np.ndarray: ...             # datetime64[D] of IST session date per bar
```
- Outputs that Market Lab returns as DataFrames (swings, events) become lists of dicts with the same keys, plus `time` as epoch seconds.
- `read()` returns exactly the dict Market Lab's `smc.read` returns. The frontend reads `read.chart.{vwap,events,obs,fvgs}`, `read.liquidity.{pdh,pdl,or_hi,or_lo,swept}`, `read.structure.{last,breaks_today}`, `read.order_blocks`, `read.fvgs`, `read.buyers_pct`, `read.vwap`, `read.price` and `read.events`.

**Golden method (the one sanctioned execution of Market Lab code):**
1. `scripts/golden/make_inputs.py` (runs with our venv, numpy only) writes deterministic synthetic bars:
   - `inputs_5m_stock.json`: 40 sessions × 75 bars from 2025-01-01, seed 11. GBM close with σ = 0.0015 per bar, high/low = max/min(o, c) × (1 ± |N(0, 0.0008)|), volume lognormal.
   - `inputs_5m_index.json`: the same with volume 0.
   - `inputs_1d_stock.json`: 600 sessions, seed 12, σ = 0.018.
   - The timestamps are real IST session times: 09:15 + 5-minute steps, weekdays.
2. `scripts/golden/run_marketlab.py <clone_path> <fixtures_dir>`:
   - inserts `<clone_path>` at `sys.path[0]`;
   - builds pandas DataFrames from the inputs (`index = pd.to_datetime(ts, unit="s", utc=True)`);
   - calls Market Lab's `sm.smc.read(df, n)` (n=2 for 5m, 3 for 1d), `sm.structure.analyse(df, 3)` and `sm.structure.session_levels(df_15m)`;
   - converts the results to JSON (`default=str`, timestamps to epoch seconds) and writes `golden_smc_<name>.json`.
   Run it with:
   ```
   C:\Users\harsh\Downloads\market-lab\.venv\Scripts\python.exe scripts\golden\run_marketlab.py C:\Users\harsh\Downloads\market-lab tests\fixtures\marketlab
   ```
   **If the clone venv is missing or broken, stop and report.** Do not install anything into it.
3. Commit the inputs and goldens. The default test suite never runs Market Lab: it only compares our port with the committed goldens.
4. `stocksense lab golden-check` (in `cli_lab.py`) re-runs step 2 and diffs. It is for re-verification only.

**Tests (`test_ml_structure_golden.py`):**
- For each of the 3 inputs: our `read(bars, n)` equals `golden_smc_*.json`. Floats compare within 1e-9 (relative 1e-9); ints, strings and lists of keys compare exactly; NaN equals NaN.
- `analyse` likewise for 1d.
- Confirmation property: for every swing, `confirmed_at_index == index + n`. No event's time precedes its swing's `confirmed_at`.

**Acceptance:** the golden tests pass; full suite green.
**Commit:** `M-2.2: smart-money structure port with golden tests`.
**Provides:** `marketlab.structure.*`, `marketlab.smc.read/fvgs/order_blocks`, `Bars`, the golden harness (`scripts/golden/*`).

### M2.3 Smart Money tape (`marketlab/tape.py`) and `/api/sm/*`
**Depends:** M2.2.
**Files:** `src/stocksense/marketlab/tape.py`, `src/stocksense/api/routes_sm.py`, `src/stocksense/marketlab/feeds.py` (FeedManager with the tape loop), `tests/unit/test_ml_tape.py`, `tests/unit/test_api_sm.py`.

**Port from:** `sm/feed.py`. Read it fully; the key lines are:
- `fno_list` 62-68
- `sector_of` + `GROUP` + override sets 44-54
- `load_daily` 117-119
- `bars5` 128-138
- `path` 141
- `cycle` 162-203
- `snapshot` 248-252
- `chart` 266
- `quote`
- `run` 286-322

**Changes, each recorded in DECISIONS `### M2.3`:**
1. `load_daily` reads our `panel_daily` (raw `high, low, close` for the last 20 completed sessions per symbol) instead of 200 Yahoo downloads:
   ```python
   range_pct = mean over last 20 rows of (high - low) / close * 100      # same formula as market-lab feed.py:117-119
   ```
   Also read PDH/PDL from the last completed session. Symbols missing from the panel use the Yahoo daily fallback (Market Lab's original path).
2. Thread-safe store; Clock; `stop.wait` instead of sleep.
3. The NSE F&O list, open and prev come via `Sources.nse_watch("SECURITIES IN F&O")`; ltp and path via `Sources.yahoo_spark`, exactly as Market Lab (Yahoo first, NSE fallback).
4. Intervals identical to Market Lab:
   - indices every 4 s when open;
   - stocks every 12 s when open, 60 s if throttled, 300 s when closed;
   - structure recompute at most every 60 s;
   - F&O list every 60 s live, 1800 s closed;
   - 60 s idle when closed.

**Functions:**
- `TapeEngine(settings, clock, cal, sources, store)` with methods `rebuild_day()` (new day: industry map, F&O list, daily ranges), `cycle(full: bool, structure: bool) -> None`, `snapshot(paths: bool) -> dict`, `chart(sym, tf) -> dict`, `quote(sym) -> dict`.
- `FeedManager` (in `feeds.py`) gets `start()`, `stop()`, and a `tape` thread using the Part 5.4 loop whose `step()` returns the next wait from the intervals above.

**Key computations** (numbers must match `feed.py:162-203`):
```
ltp = spark ltp if fresh else nse ltp; prev = nse prev or spark prev; opn = nse open or first 1m close
fo = (ltp/opn - 1)*100; pct = (ltp/prev - 1)*100; disp = round(fo / (range_pct or 2.0), 2)
bull = sorted(from_open > 0, key=-disp); bear = sorted(from_open <= 0, key=disp); rank = position (1-based); side = "bull"/"bear"
sector chips: per GROUP, mean(pct), up count, down count, n
trail = path value 15 minutes ago
liq = ["PDH" if dayHigh > pdh, "PDL" if dayLow < pdl]
```
**Routes (`routes_sm.py`):**
- `GET /api/sm/live?paths=1|0` → `engine.snapshot(paths=="1")`;
- `GET /api/sm/chart?sym=&tf=1m|5m|15m|60m|1d` → `engine.chart(sym.upper(), tf)`;
- `GET /api/sm/quote?sym=` → `engine.quote(sym)`.

Response key trees are exactly Market Lab's:
```
/api/sm/live → {time, market, errors,
  indices: {NIFTY|BANKNIFTY|SENSEX: {ltp, pct, from_open, path, bars, read, expiry, delay}},
  stocks: [{sym, name, sector, ltp, pct, from_open, disp, path?, value_cr, ffmc, bars?, pdh?, pdl?, liq?, read?, trail, rank, side}],
  sectors: [{name, pct, up, down, n}], bull: [sym], bear: [sym]}
/api/sm/chart → {sym, tf, ltp, prev, candles[{time, open, high, low, close, volume}], read}
```
M3.7 later adds an optional `flow` object per stock row. The engine is created in `create_app` when feeds start; with `start_feeds=False`, tests inject a pre-filled store.

**Golden test:** extend `scripts/golden/run_marketlab.py` to run Market Lab's `feed.cycle` logic on recorded probe inputs. Monkeypatch its `M.get_json` to return the saved spark/watch fixtures and its daily loader to return fixture daily bars. Save `golden_tape_snapshot.json`. Our `TapeEngine` fed the same fixtures through a fake `Sources` must match on `pct, from_open, disp, rank, side` per symbol (1e-9) and on the sector chips.

**Tests (`test_ml_tape.py`):**
- `disp`: ltp 102, open 100, range 4 → fo 2.0 → disp 0.5; with range missing → 1.0.
- Ordering of 4 stocks with from_open (+1, +3, −2, −0.5) and range 2: bull = [+3, +1], bear = [−2, −0.5].
- Chip mean: pct (1, −1, 3) gives mean 1.0, up 2, down 1, n 3.
- Liq: dayHigh 105 > pdh 104 → ["PDH"].
- Feed thread: start, then `stop()` returns within 2 s (SteppingClock + fake sources).

`test_api_sm.py`: TestClient on a store seeded from `golden_tape_snapshot.json` inputs. `/api/sm/live` returns the key tree above with `stocks` sorted bull then bear.

**Commit:** `M-2.3: Smart Money tape port`.
**Provides:** `TapeEngine`, `FeedManager.start/stop`, `/api/sm/*`.

### M2.4 Sector Map (`marketlab/sectors.py`, `news.py`) and `/api/{horizon,sector,stock,search,chart,tick}`
**Depends:** M2.3.
**Port from:**
- `sx/universe.py`: SECTORS (22-45) verbatim, GROUPS, YAHOO, `members` 138-157, `stats`, `sector_series` 138-177, `build_classification` 80-97, `nifty500` 56-58;
- `sx/live.py`: 25-157, i.e. `all_indices`, `watch`, `focus`, `price`, `index_value`, `record`, `run`;
- `sx/news.py`: regexes and QUERY map verbatim, `search`, `sector`, `stock`, `market`;
- `app.py`: `ret`, `sector_move`, `nifty_move`, `ranks_now`, `horizon`, `stock_row`, `sector_view`, `stock_view`, `search`, `chart`, `tick`, `rank_history`, `themes`, `since`, `pins`, `rel_chart`, `series_for`, `all_series`, `sec_members` (lines 33-369).

**Changes:**
1. 1-year history (`U["hist"]`) comes from our `panel_daily`: close = **adjusted** close (`adj_close`) for return math, raw `high` for the 52-week high, and `volume`. It covers the last 260 completed sessions per symbol. `IDX:<name>` series come from `index_daily` (`index_name`, `close`). Yahoo history download (`build_history`) is **not** ported: it is replaced, and recorded.
2. Only completed sessions are used (fixes Market Lab's partial-bar issue). Today's live values come only from `price()`/`index_value()`.
3. Classification (`build_classification`) is kept: weekly refresh via `Sources.nse_symbol_data`, 0.25 s apart through `clock.sleep`. Store it in `data_store/lab/classification.json` (UTF-8 JSON).
4. 5-minute snapshots go to `data_store/lab/days/YYYY-MM-DD.json` (same structure `{"snaps":[{m, nifty, sec:{name:pct}}]}`).
5. Feed threads `sector` (the `sx/live.run` intervals: indices 15 s live / 300 s closed; NIFTY 500 watch 15 s / 300 s; members hourly; spark for YAHOO + focus; record every 5 minutes), `slow` (classification weekly) and `news` (600 s warm: market query + 20 sector queries, `wait=False` thereafter).

**Routes** with response trees exactly as Market Lab's:
```
/api/horizon → {range, nifty{v,last}, sectors[{name,index,group,v,size,n,rank,rank_open,trail,live,path}], groups, climb[{name,from,to}],
               themes[{name,parent,n,beat,pct}], news[{time,title,source,link,tag,sector,since?}], chart{dates,a,pins}|None, snaps[{m,nifty}]}
/api/sector?name=&range= → {name,index,group, ix{last,pct,w1,m1,m3,pe,live}, nifty{pct,w1,m1,m3,pe}, rank, rank_open, n, middle,
               groups[{name,n,mid,above50,stocks[stock_row]}], others[], rank_hist[], inside{above50,above50_wk,rose_week,near_high,busy,n},
               movers[], laggards[], news[], chart}
/api/stock?sym= → {sym,row,cls{sector,industry,basic,pe,mcap,delivery,high52,low52,name},sector,peers,against,rank,lines,news,volume{today,normal,ratio},delivery,dma50,chart}
/api/search?q= → [{type:"sector",name} | {type:"stock",sym,name,pct}]    (≤ 15)
/api/chart?sym=&tf= → {sym,tf,ltp,prev,time,candles,read}
/api/tick?syms=a,b → {stocks{sym:{ltp,pct}}, sectors{name:{v,last}}, nifty{v,last}, time, market}    (syms[:150])
```
`stock_row` = `{sym,name,group,basic,ltp,pct,mcap,live,w1,m1,m3,above50,near_high,vol_ratio,high52,dma50}`.

**Golden:** extend the golden script to run Market Lab's `horizon("today")`, `sector_view("Energy","today")` and `stock_view("RELIANCE")`. Stub Market Lab's module state `U["hist"]`, `U["cls"]`, `L.S` and `N.search` with the same fixture inputs our tests feed (synthetic histories for 60 symbols across 4 sectors, fixed snapshots, a fixed news list). Our functions must match key by key (floats 1e-9).

**Tests:**
- `sector_move`: today uses `pct`; w1 uses NSE `w1` when present, else `ret(series, 5)`.
- `members()`:
  - Railways uses only its fixed list;
  - `not_basic` excludes Private and Public Sector Bank from Fin Services;
  - a stock with `industry="Banks"` is in Bank.
- RUNNING threshold: s.v − nv ≥ 1.
- `themes()`: requires ≥ 5 stocks and a median above Nifty.
- `tag()`: "Q2 results beat" → RESULTS; "RBI policy" → POLICY; "Company launches" → NEWS.
- `tick` caps at 150 symbols.

**Commit:** `M-2.4: Sector Map port`.
**Provides:** `SectorEngine` (horizon, sector_view, stock_view, search, chart, tick), `/api/{horizon,sector,stock,search,chart,tick}`.

### M2.5 Research note (`marketlab/research.py`, `industries.py`) and `/api/research`
**Depends:** M2.4, and M5.1 for technicals. Until M5.1 merges, `tech = {}`; a later card fills it in. Both orders work.
**Port from:** `lab/research.py` (`build` 144-309 verbatim logic; `quarters` 33-54; `balance` 57-65; `wiki` 68-88; `macro` 91-104; `peers` 107-113; `sector_strength` 124-130; `clip`, `lin`) and `lab/industries.py` (copy `FACTOR_NAMES`, `IND`, `MACRO_DEFAULT`, `MARGIN_NORM` and `lookup` verbatim; it is a data file).

**Changes:**
1. `ret(sym, days)` and `sector_strength` read `panel_daily` adjusted closes.
2. Technicals come from `strategies.indicators.daily_features` (M5.1) on our panel.
3. `macro()` keeps Yahoo (`BZ=F, USDINR=X, HG=F, ^GSPC`, 6-month daily, `chg3m = c[-1]/c[-64]-1`) per the user's decision. Cache it 6 h in the TTLCache.
4. Cache the note 12 h at `data_store/lab/research/{SYM}.json`. `fresh=1` bypasses the cache.
5. **Wikipedia fix:** accept a page only if the extract contains a word of the company name longer than 3 letters (after removing limited/ltd/india/industries), otherwise `wiki=None`. Record the fix.
6. The index symbols NIFTY, BANKNIFTY, SENSEX, FINNIFTY and MIDCPNIFTY return `{"error": "indices have no company report"}` (status 200, as Market Lab).

**Output key tree** (exactly Market Lab's):
```
{sym, name, built, macro, sector, industry, basic, index, what, summary[], wiki, mcap_cr, pe, sector_pe, delivery, high52, low52, ltp,
 results[{q,kind,income,pbt,profit,eps,date,label}], ttm{income,profit,margin,g_inc,g_pat,q_yoy,norm,pairs,profit_q,n_q},
 balance{de,icr,as_of,interest,dep}, financial, peers[{sym,name,mcap_cr,pe}], sector_strength{sector,sector_3m,nifty_3m,rel,n},
 economy[{factor,name,effect,why,reading{now,chg3m}|None,push}], tech{uptrend,rs63,from_high,atr_pct,ret1y,ret3m}|{},
 scores{...}, weights, total, grade, swot{Strengths,Weaknesses,Opportunities,Threats}, sources[3]}
```
Defaults are verbatim: "Nothing stands out." / "No obvious weak spot in the numbers." / "No clear tailwind right now." / "No clear headwind right now."

**Tests (hand numbers):**
- `lin(5, 0, 10)` = 5.0; `lin(-1, 0, 10)` = 0; `lin(None, 0, 1)` = None.
- Grade: total 75 → Strong, 74 → Good, 62 → Good, 48 → Average, 35 → Weak, 34 → Poor.
- Valuation: pe 10, spe 20 (r = 0.5) → 9; r = 1.0 → 6; r = 1.5 with g_pat 30 → 3 + 1.5 = 4.5.
- Financial health for non-financials: de 0.2 → 10 and cons 10 → mean 10.
- A SWOT strength fires when g_pat > 20.
- The Wikipedia matcher rejects a fixture extract about an unrelated company.
- Golden: Market Lab's `build("TESTCO")` with stubbed network functions returning fixtures equals ours (floats 1e-9).

**Commit:** `M-2.5: research note port`.
**Provides:** `ResearchEngine.report(sym, fresh)`, `/api/research`.

### M2.6 Lab endpoints on the exploratory engine
**Depends:** M5.6 (explore/scan/profile) and M2.4.
**Files:** `src/stocksense/api/routes_lab.py`, `tests/unit/test_api_lab.py`.

**Routes:**
- `GET /api/lab/trades?sym=&tf=1d|5m|60m&strat=&exit=`
  - For a stock with tf `1d`: `strategies.explore.trades_for(sym, strat_key, exit_key, settings, clock)`. It returns Market Lab's `trades_for` shape (`lab/swing_live.py:65-112`) **plus** `exploratory: true`, `costs_model: "stocksense.costs (delivery|intraday) + slippage model"`, `gated: <verdict|null>`.
  - Strategy and exit parameters accept Market Lab display names (e.g. "VCP tightening break", "rr3_40") **and** our registry keys.
  - For an index (`NIFTY`, `BANKNIFTY`, ...) with tf 1d: return `{"trades": [], "note": "Swing rules are for stocks; pick 5m or 60m for the index rules."}`, as Market Lab does.
  - For intraday tf: the intraday explore path, shaped like Market Lab's `lab/live.chart_trades` output. Read `lab/live.py` for the key list and copy it exactly.
- `GET /api/lab/swing` → `strategies.scan.summary(settings)`. Market Lab's `swing_live.summary` key tree, with the gated verdict per strategy added.
- `GET /api/lab/summary` → the intraday library summary built from the latest gated results: `{list: [names], strategies: {...}}`. Read `app.py:/api/lab/summary` and `lab/report.py` for the keys.
- `GET /api/lab/journal` → `strategies.journal.read(settings, days=30)`.
- `GET /api/lab/matrix?sym=&tf=1d` →
  ```
  {sym, tf, n_tests, expected_max_sharpe_null,
   rows: [{strat, name, tier, source, exit, n, win, avgR, totR, pct, is_avgR, oos_avgR, gated}]}
  ```
- `GET /api/lab/strategies` → `[{key, name, source, tier, timeframe, kind, direction, exits, description, reference, gated}]`.

**Tests:** a fixture symbol on the synthetic panel:
- trades keys ⊇ `{t_in, p_in, stop, target, status}`, and `exploratory` is true;
- matrix row count = Σ allowed exits over the registered 1d strategies;
- an unknown strategy name → 400 `{"error": "unknown strategy"}`.

**Commit:** `M-2.6: lab endpoints on the exploratory engine`.

### M2.7 One-command start, catch-up, `run-lab.bat`
**Depends:** M2.4, M5.6.
**Files:** `src/stocksense/marketlab/feeds.py` (catch-up thread), `run-lab.bat`, `tests/unit/test_ml_catchup.py`.

**Spec, `CatchUp(settings, clock, store, steps: list[tuple[str, Callable[[], str]]])`, run once in a thread:**
1. `"ingest"`: build `IngestContext(settings, clock, TradingCalendar.from_store(settings, clock), NseClient(clock), State(settings, clock))`. If any of `bhavcopy_eq`, `bhavcopy_delivery` or `fo_bhavcopy` has `fresh == False` in `status_table(...)`, call `catch_up(ctx)` and return `f"{n_ok} ok, {n_failed} failed"`; else `"fresh"`.
2. `"panel"`: if `panel_daily` is not fresh **and** the time is outside 09:00–15:45 **and** free RAM ≥ 3 GB, call `panel.build_all(settings)`.
   - In market hours, return `"deferred: market hours"` and re-check every 15 minutes until 15:45.
   - On low RAM, return `"deferred: low memory"`.
   - Free RAM comes from `psutil.virtual_memory().available` if psutil is installed (check first). Else use PowerShell via `subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"], capture_output=True, text=True, timeout=20)` (KB).
3. `"scan"`: `strategies.scan.run_scan(settings, asof=cal.last_completed_session(clock.now()))` if `data_store/lab/scan/<asof>.parquet` is missing.
4. `"flow_eod"`: `flow.composite.build_eod(settings, asof)` (M3.7) if missing.

Each step's state (`running|done|skipped|deferred|failed`, plus detail) goes into `store.catchup["steps"]`. A step's exception marks it `failed` and the next step still runs. `create_app` starts the catch-up only if `app.state.catchup_enabled` (default True; `--no-catchup` sets it False).

**`run-lab.bat`** (verbatim):
```bat
@echo off
cd /d %~dp0
if not exist web\node_modules ( call npm --prefix web ci || goto :err )
for /f %%i in ('powershell -NoProfile -Command "if (-not (Test-Path web\out\index.html) -or ((Get-ChildItem web\src -Recurse -File | Measure-Object LastWriteTime -Maximum).Maximum -gt (Get-Item web\out\index.html).LastWriteTime)) {'build'} else {'ok'}"') do set NEED=%%i
if "%NEED%"=="build" ( call npm --prefix web run build || goto :err )
start "" http://127.0.0.1:8765
.venv\Scripts\python.exe -m stocksense serve
goto :eof
:err
echo Build failed. See the messages above.
pause
```
**Tests:**
- Step order with fake steps: ingest → panel → scan → flow_eod.
- A raising step → `failed`, and the next step still runs.
- Panel at FixedClock 11:00 on a session → `deferred: market hours`.
- Panel with a fake free-RAM of 2 GB → `deferred: low memory`.
- `run-lab.bat` exists and does not contain `schtasks` (case-insensitive).

**Commit:** `M-2.7: one-command start with catch-up`.

### M2.8 Parity harness `scripts/m_parity.py`
**Spec:** with Market Lab on 8810 and ours on 8765 (both started by hand), take 10 samples 30 s apart in market hours. Each sample:
- GET both `/api/sm/live`;
- top-20 bull/bear Jaccard;
- Spearman of `disp` over the common symbols (implemented with numpy ranks);
- chip MAD of `pct` by sector name;
- `/api/horizon?range=today` Kendall tau of the sector order (`scipy.stats.kendalltau`).

Print a table and a pass/fail against the thresholds: Jaccard ≥ 0.85, Spearman ≥ 0.98, MAD ≤ 0.05 pp, tau ≥ 0.9. No test is required, but run `ruff` on it.
**Commit:** `M-2.8: parity script`.

### M2.9 Resource check (orchestrator + user)
The user starts `run-lab.bat`. The orchestrator records the following in DECISIONS `### M2.9`:
- peak RSS of the python process every 5 minutes for 30 minutes (`Get-Process -Id <pid> | select WS`);
- NSE/Yahoo calls per minute (add a counter to `Sources`, exposed in `/api/status`);
- error counts.

If RSS > 800 MB, take `tracemalloc` snapshots at 5 and 30 minutes and fix the top allocator (commit `M-2.9: memory fix <what>`).

---
## PHASE M3 — Money Flow: where the money is going (after M2.3)

Common rules for M3:
- EOD functions are pure polars functions of stored data, unit-tested on small hand frames.
- Live functions take `Sources`/broker clients injected.
- New datasets are declared in `data/catalog.py` `_DERIVED`, each with `path="flow"` and its own name:
  ```python
  _spec("flow_oi_daily", "derived", "date, symbol", "date", "year", {...}, path="flow/oi_daily")
  ```
  Writes go through `io.upsert`, so every EOD job is idempotent by `(date, key)` (CLAUDE.md rule 7).
- Every EOD builder has the signature `build_<name>(settings, start: date, end: date) -> int` (rows written) and is called by `flow.composite.build_eod` for any missing dates.

### M3.1 OI buildup (`flow/oi.py`) + Angel NFO futures tokens
**Files:**
- `src/stocksense/flow/oi.py`
- `src/stocksense/brokers/instruments.py` (NFO futures parser)
- `src/stocksense/brokers/angel.py` (`quote_full(..., exchange="NSE")`)
- `data/catalog.py`
- `tests/unit/test_flow_oi.py`, `tests/unit/test_instruments_nfo.py`

**EOD spec:**
```python
def futures_daily(fo: pl.DataFrame, calendar: TradingCalendar) -> pl.DataFrame:
    """Per (date, symbol) near-month futures close, total futures OI (shares), OI change %, price change %, basis, rollover %, quad label.
    fo = fo_bhavcopy rows with instrument_type in {STF, IDF}. Near month = nearest expiry_date >= date, rolled to the next expiry
    on the session before expiry (expiry - 1 session). Rollover% = OI(next expiry) / OI(all expiries) * 100, only in the last 5
    sessions before the near expiry, else null. price_chg_pct uses near-month settle vs the previous session's near-month settle of the
    SAME contract (no roll jump). oi_chg_pct = total OI vs previous session total OI. basis_pct = near settle / underlying - 1 (x100)."""
```
Output columns:
- `date, symbol, fut_close, fut_oi, oi_chg_pct, price_chg_pct, basis_pct, rollover_pct`;
- `quad` (Float, from `derivs.oi_price_quad(pl.col("price_chg_pct"), pl.col("oi_chg_pct"))`);
- `quad_label` ∈ {"long buildup", "short buildup", "short covering", "long unwinding", null}.

The dataset is `flow_oi_daily`, key `date, symbol`, partitioned by year.

**Live spec:**
- `oi_spurts_live(sources) -> pl.DataFrame` from `Sources.nse_oi_spurts()`, with the same columns as the stored `nse_snapshots/oi_spurts` (`symbol, latest_oi, prev_oi, change_in_oi, oi_change_pct, volume, underlying_value`). Every 180 s in market hours.
- `InstrumentMap.angel_nfo_futures(asof: date) -> dict[str, str]`: underlying symbol → near-month futures token. Parse the cached Angel scrip master rows with `exch_seg == "NFO"` and `instrumenttype in {"FUTSTK", "FUTIDX"}`, and pick the nearest expiry ≥ asof.
  - **Probe first** the actual field names (`token, symbol, name, expiry, strike, lotsize, instrumenttype, exch_seg`) from `cache/instruments/angel-<date>.json`, and record them.
  - Expiry format is likely `"28NOV2026"`; parse it with `datetime.strptime(s, "%d%b%Y")`.
- `AngelClient.quote_full(tokens, exchange: str = "NSE")`: the body becomes `{"mode": "FULL", "exchangeTokens": {exchange: chunk}}`. Existing callers are unchanged (default `"NSE"`). The `Quote.oi` field is filled by the H1 parser.
- The live futures-OI loop (60 s, market hours, the top 50 F&O names by today's `value_cr` from the tape):
  - `quote_full(tokens, exchange="NFO")`;
  - live quad from the futures LTP vs the previous EOD near settle, and OI vs the previous EOD `fut_oi`;
  - store in `store.flow["oi_live"]`.
  - Angel limits: 50 tokens per call, 1 call per second. The loop needs Angel secrets; without them it is marked `skipped: no angel secrets` in feeds.

**Tests:**
- Hand chain for 2 dates × 2 expiries. Day 1: near OI 100, next 20; day 2: near 110, next 40, near settle 101 vs 100:
  - oi_chg_pct = (150 − 120)/120 × 100 = 25.0;
  - price_chg_pct = 1.0;
  - quad = +1 → "long buildup".
- Price down with OI up → "short buildup"; price up with OI down → "short covering"; price down with OI down → "long unwinding".
- Rollover inside the last 5 sessions: next 40 / total 150 = 26.67%.
- Roll at expiry − 1: on the session before expiry, near = the next contract.
- Upsert run twice gives the same rows.
- NFO parser on a 5-row fixture scrip-master JSON picks the nearest expiry.

**Commit:** `M-3.1: OI buildup and NFO futures tokens`.

### M3.2 Delivery and deals (`flow/delivery.py`, `flow/deals.py`)
**Spec:**
```python
def delivery_daily(panel: pl.DataFrame) -> pl.DataFrame:
    """Per (date, sid): deliv_pct, deliv_z20 = (deliv_pct - mean of previous 20 sessions)/std of previous 20 (ddof=1; null if <15 obs),
    deliv_value_cr = deliv_qty*close/1e7, deliv_spike = deliv_z20 > 2 and close > open."""
```
- The statistics use `shift(1)` before rolling, so day t is not in its own baseline.
- Dataset `flow_delivery_daily` (key `date, sid`).

```python
def deals_summary(deals: pl.DataFrame, clients: set[str]) -> pl.DataFrame:
    """Per (date, symbol): buy_value_cr, sell_value_cr, net_value_cr, n_deals, top client by |value|, flagged = client in clients."""
```
`config/flow_clients.toml`:
```toml
# Optional: names of deal clients to highlight (exact NSE spelling). Leave empty to highlight none.
clients = []
```
**Tests:**
- deliv_z20 on 21 rows where deliv_pct = 40 for 20 rows (with alternating 38/42 so std ≈ 2.05) and then 50: z = (50 − 40)/2.0520 = 4.873 (4 dp).
- Spike requires close > open.
- Deals: buy 1000 @ ₹100 and sell 400 @ ₹100 give net 0.006 cr.

**Commit:** `M-3.2: delivery and deals flow`.

### M3.3 Participants (`flow/participants.py`)
**Spec:**
```python
def participants_daily(poi: pl.DataFrame, fii_dii: pl.DataFrame) -> pl.DataFrame:
    """Per date: for each client_type in (FII, DII, PRO, CLIENT): idx_fut_long_ratio = fut_idx_long/(fut_idx_long+fut_idx_short),
    its 1- and 5-session change; idx_call_net = opt_idx_call_long - opt_idx_call_short; idx_put_net likewise;
    plus fii_cash_net, dii_cash_net (₹ cr) from fii_dii (null where missing)."""
```
- Wide output columns: `fii_ratio, fii_ratio_chg1, fii_ratio_chg5, dii_ratio, ..., fii_cash_net, dii_cash_net`.
- **Probe:** NSE `"/api/fiidiiTradeReact"` returns only the latest day. Check whether any NSE archive gives FII/DII cash history (e.g. `https://nsearchives.nseindia.com/content/nsccl/fao_participant_vol_<DDMMYYYY>.csv` is F&O volume, not cash). If none, record "forward-only" in DECISIONS and do nothing more.

**Tests:** FII long 300, short 100 → 0.75; chg5 on a 6-row series 0.5 → 0.75 = 0.25.
**Commit:** `M-3.3: participants and FII/DII flow`.

### M3.4 Options positioning and gamma exposure (`flow/options.py`)
**Spec:**
```python
def bs_gamma(s: float, k: float, t: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma (per 1 rupee of spot): phi(d1)/(s*sigma*sqrt(t)); 0 if t<=0 or sigma<=0."""

def chain_frame(fo: pl.DataFrame, d: date, underlying: str, n_expiries: int = 2) -> pl.DataFrame:
    """Option rows (IDO/STO) for one underlying on date d for the nearest n expiries with expiry_date > d (expiry-day rows excluded,
    because settle_price equals the underlying on expiry day), with dte = (expiry - d).days (calendar days) and t = dte/365."""

def chain_metrics(ch: pl.DataFrame, spot: float, r: float = R_FREE) -> dict:
    """iv (via derivs.implied_vol on settle_price; rows with dte < MIN_DAYS skipped), gamma, gex_k = gamma*oi*spot^2*0.01*(+1 call, -1 put)
    in rupees per 1% move; totals: gex_total_cr = sum/1e7; zero_gamma (spot level where total GEX changes sign on grid spot*(1+x),
    x in -10%..+10% step 0.25%, IV held fixed; None if no sign change); call_wall = strike with max call OI >= spot; put_wall = max put
    OI <= spot; pcr_oi = put OI / call OI; pcr_doi = put change_in_oi / call change_in_oi (None if call doi == 0); max_pain
    (derivs.max_pain over all strikes of the nearest expiry); oi_imbalance per strike = put doi - call doi; atm_iv; skew = put_iv(0.95S) - call_iv(1.05S)."""
```
**Conventions (write them in the docstring and the learning note):**
- OI is in shares, so GEX needs no lot size.
- Dealers are assumed long calls and short puts (the common SpotGamma-style convention): calls positive, puts negative.
- Positive total GEX means dealers buy dips and sell rallies (a dampening, range regime). Negative means they amplify moves (a trending, volatile regime).
- It is an approximation. The true dealer inventory is unknown.

**EOD:** `build_options_daily(settings, start, end)` for NIFTY, BANKNIFTY, FINNIFTY and MIDCPNIFTY (IDO) and every STO underlying. It writes:
- `flow_options_daily` (key `date, underlying`): `spot, atm_iv, skew, pcr_oi, pcr_doi, max_pain, max_pain_dist_pct, call_wall, put_wall, gex_total_cr, zero_gamma, zero_gamma_dist_pct, near_expiry`;
- `flow_options_strikes` (key `date, underlying, expiry, strike`; nearest 2 expiries): `call_oi, put_oi, call_doi, put_doi, call_iv, put_iv, gex_cr`.

Spot is `underlying_price` from the same rows.

**Live:** every 180 s in market hours for NIFTY and BANKNIFTY, and every 600 s for the top 20 F&O stocks by value:
- `Sources.nse_option_chain(...)`; parse it into the same `ch` frame (use the probe's field names: strike, CE/PE `openInterest`, `changeinOpenInterest`, `lastPrice`, `impliedVolatility`, `underlyingValue`, expiry dates);
- `chain_metrics` with `iv` = NSE's IV/100 when present (record which was used), else implied from `lastPrice`.

Note: NSE live `openInterest` is in **contracts**, so multiply by the lot size. **Probe the unit**: compare one strike's live OI × lot with the EOD bhavcopy OI. Get the lot size from `https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv` (probe), cached daily in `data_store/lab/fo_lots.json`.

**Tests:**
- `bs_gamma(100, 100, 0.25, 0.065, 0.2)` vs finite differences of `bs_price` delta: (Δ(S+0.01) − Δ(S−0.01))/0.02 within 1e-4. Compute delta numerically from `bs_price` as well.
- GEX sign: 1 call strike with OI 1000 and 1 put strike with OI 1000 at the same strike, ATM: the call gamma equals the put gamma, so the total is 0.
- Zero-gamma on a toy chain (call OI concentrated at 110, put OI at 90, spot 100): a sign change is found between 90 and 110.
- Expiry-day rows are excluded: a chain at d = expiry returns only later expiries.
- Walls and PCR by hand: put OI 300 / call OI 200 gives 1.5.
- Truncation: metrics at d use only rows with `date == d`. The builder is called with `fo` filtered to `date <= d`, and the result equals the unfiltered call.

**Commit:** `M-3.4: options positioning and gamma exposure`.

### M3.5 Pre-open 09:00 (`flow/preopen.py`)
**Spec:**
- Live loop from 09:00:30 to 09:14:30 on sessions, every 30 s:
  - `Sources.nse_preopen("FO")` and `nse_preopen("NIFTY")`, parsed via the existing `ingest_misc.parse_preopen(payload, ts)`;
  - per symbol: `iep_gap_pct = (indicative_price/prev_close − 1)×100`, `imb = (buy_qty − sell_qty)/(buy_qty + sell_qty)`, `total_qty`;
  - store in `store.flow["preopen"]`;
  - **Timing caveat (found 2026-11-23):** broker and news articles say NSE changed the pre-open phases from 2026-09-07 (orders of both kinds to about 09:05, limit orders only to a random close between 09:08 and 09:10, matching to 09:12, buffer to 09:15). We could not read NSE's own page. So do **not** hard-code 09:08. Poll 09:00:30 to 09:14:30 and call `ingest_misc.capture_preopen(ctx)` once at the first iteration whose payload has a non-null `final_price` for NIFTY constituents (or at 09:12:30 if none appeared), then record the observed times in DECISIONS `### M3.5`.
- After 09:15, `/api/flow/preopen` returns the stored board plus `follow_through` per symbol: `sign(iep_gap) == sign(ltp/open − 1)` at 09:30 from the tape.
- NIFTY indicative open: from the NIFTY key payload.
- Breadth: the counts of positive and negative IEP gaps.

**Tests:**
- Parse the existing preopen fixture (`tests/fixtures/files/*preopen*` if present; else one saved by the probe).
- imb: buy 300 / sell 100 → 0.5.
- The loop runs only inside the window (SteppingClock from 08:59 to 09:16; calls happen only between 09:00:30 and 09:14:30).
- `capture_preopen` is called exactly once (mock).

**Commit:** `M-3.5: pre-open flow`.

### M3.6 Sector money flow (`flow/sector_flow.py`)
**Spec, live (from the tape rows and the sector members of M2.4):**
```
flow_dir = Σ(value_cr_i × sign(from_open_i)) / Σ value_cr_i          (−1..+1)
turnover_share = Σ value_cr(sector) / Σ value_cr(all F&O ∪ NIFTY 500 rows)
share_ratio = turnover_share / mean share of the same sector at the same minute over the last 20 sessions (from lab/days snapshots that
              store per-sector value; extend the M2.4 snapshot with "val":{sector: value_cr}; null until 5 sessions exist)
rotation = "in" if share_ratio > 1.2 and flow_dir > 0.2; "out" if share_ratio > 1.2 and flow_dir < -0.2; else "neutral"
breadth = up/down counts; oi_quads = counts of live OI quad labels among F&O members; deliv_spikes = count of yesterday's spikes
```
- EOD: the same from `panel_daily` (`turnover_inr`, `ret_oc`) for history. Dataset `flow_sector_daily` (key `date, sector`).

**Tests:** 3 stocks with value (10, 20, 30) cr and from_open (+1, −1, +2): flow_dir = (10 − 20 + 30)/60 = 0.3333. Rotation thresholds at the boundaries.
**Commit:** `M-3.6: sector money flow`.

### M3.7 Flow composite, EOD build and `/api/flow/*`
**Spec:**
```python
def stock_card(sym: str, store: LabStore, eod: EodFlow) -> dict:
    """Descriptive flow card for one F&O stock (not a forecast). Keys: disp, rvol, oi_quad, oi_quad_label, oi_chg_pct, deliv_z20_prev,
    depth_imb5, buy_sell_ratio, preopen_imb, cvd_slope_30m, options{pcr, gex_sign, call_wall, put_wall}|None, flow_score, components_used."""
```
- `rvol`: cumulative volume today (NSE market-watch `volume`) / the mean cumulative volume to the same minute over the last 20 sessions. Use the 5m `intraday_bars` history, cached once per day per symbol in `store.flow["vol_curve"]`; null if fewer than 10 sessions.
- `cvd_slope_30m`: the OLS slope of BVC CVD (M4.3) over the last 30 one-minute bars of today, divided by the average 1m volume. The 1m bars come from `Sources.yahoo_chart(f"{sym}.NS", "1m", "1d")` for the ≤ 40 symbols shown (top/bottom 20), refreshed every 60 s.
- `flow_score` = the mean of the available components, each z-scored cross-sectionally across today's F&O rows and clipped to ±3:
  - `disp`;
  - `rvol − 1`;
  - signed quad (+1 long buildup, +0.5 short covering, −0.5 long unwinding, −1 short buildup);
  - `depth_imb5`;
  - `cvd_slope_30m`.
  `components_used` lists the names.
- `build_eod(settings, asof)` runs `futures_daily`, `delivery_daily`, `participants_daily`, `build_options_daily` and the sector EOD for missing dates up to `asof`.
- `EodFlow` caches the latest EOD rows per symbol in memory (reloaded when `asof` changes).

**Routes:** see Part 3 of the API table. Shapes:
```
/api/flow/market → {asof, fii{cash_net, cash_net_5d, idx_fut_ratio, ratio_chg5}, dii{...}, pro{...}, client{...},
                    indices{NIFTY{spot, pcr_oi, gex_total_cr, regime:"positive"|"negative", zero_gamma, call_wall, put_wall, max_pain, atm_iv},
                    BANKNIFTY{...}}, vix{close, chg1}, breadth{adv, dec}, preopen{adv, dec, nifty_iep}|None, live: bool}
/api/flow/stocks?sort=flow_score|disp|rvol|oi_chg_pct → [{sym, name, sector, ltp, pct, ...stock_card}]
/api/flow/oi?scope=stocks|index → [{symbol, quad_label, oi_chg_pct, price_chg_pct, basis_pct, rollover_pct, live_oi_chg_pct?}]
/api/flow/options?und=NIFTY&expiry=YYYY-MM-DD → {und, spot, expiry, expiries[], strikes[{strike, call_oi, put_oi, call_doi, put_doi, call_iv, put_iv, gex_cr}],
                    totals{gex_total_cr, zero_gamma, call_wall, put_wall, max_pain, pcr_oi, pcr_doi}, history[{date, gex_total_cr, pcr_oi, spot}]}
/api/flow/sectors?range=today|w1|m1 → [{name, flow_dir, turnover_share, share_ratio, rotation, up, down, oi_quads{}, deliv_spikes}]
/api/flow/preopen → {phase:"before"|"live"|"after", nifty{iep, gap_pct}|None, rows[{symbol, iep, iep_gap_pct, imb, total_qty, follow_through?}]}
/api/flow/deals?days=5 → [{date, symbol, buy_value_cr, sell_value_cr, net_value_cr, n_deals, top_client, flagged}]
/api/flow/participants?days=60 → [{date, fii_ratio, dii_ratio, pro_ratio, client_ratio, fii_cash_net, dii_cash_net}]
```
- `/api/sm/live` stock rows gain `"flow": stock_card(...)` (null when it cannot be computed).
- Every flow response carries `"descriptive": true`.

**Tests:**
- A missing component is skipped (mean of the 3 present).
- Clip at 3.
- Signs.
- `/api/flow/*` shapes from a seeded store and synthetic EOD datasets.
- `build_eod` is idempotent.

**Commit:** `M-3.7: flow composite and flow endpoints`.

---

## PHASE M4 — Order-flow tools (pure functions first, then endpoints)

All functions take `Bars` (M2.2) for one or more sessions and return numpy/dicts. Every function has a hand-computed test. Every API response includes `method` and `data_source`.

### M4.1 `orderflow/vwap.py`
```python
def session_vwap(b: Bars) -> tuple[np.ndarray, np.ndarray]:
    """(vwap, sigma) per bar, reset each IST session: tp=(h+l+c)/3, w=volume (1.0 where volume==0, as market-lab),
    vwap=cumsum(tp*w)/cumsum(w), sigma=sqrt(cumsum(w*(tp-vwap_t)^2)/cumsum(w)) using the running vwap_t."""
def bands(vwap, sigma, ks=(1, 2, 3)) -> dict[str, np.ndarray]:   # "u1","l1",...
def anchored_vwap(b: Bars, anchor_ts: int) -> np.ndarray:         # NaN before the anchor
```
**Tests:** 3 bars with tp (10, 11, 12) and volume (1, 1, 2): vwap = [10, 10.5, 11.25]; σ at bar 3 = sqrt((1·1.5625 + 1·0.0625 + 2·0.5625)/4) = sqrt(0.6875) = 0.8292. Anchored at the first bar of the session equals the session VWAP.

### M4.2 `orderflow/profile.py`
```python
def bin_size(price: float) -> float:          # max(0.05, round(0.0005*price, 2))
def volume_profile(b: Bars, bin: float | None = None) -> dict:
    """Uniform spread of each bar's volume over [low, high] into price bins; returns {bins(lower edges), vol, poc, vah, val, total}.
    Value area: start at POC, repeatedly add the larger of the next bin above/below (ties -> above) until >= 70% of total."""
def tpo_profile(b: Bars) -> dict:
    """30-minute periods from 09:15 (A=09:15-09:44, B=09:45-10:14, ..., M=15:15-15:29); letters per bin; poc_tpo, vah, val,
    ib_high/ib_low (A+B = 09:15-10:14), single_prints (bins touched by exactly one period, between two multi-period bins),
    poor_high (>=2 periods at the high bin), poor_low."""
def day_type(tpo: dict, atr_d: float) -> str:
    """Dalton: 'trend' if range >= 2*IB range on one side and close in the outer 20%; 'normal' if IB covers >= 85% of range;
    'normal_variation' if extension beyond IB <= 1x IB range; 'neutral' if extension on both sides; 'double_distribution' if two TPO-count
    peaks separated by >= 2 single-print bins; 'non_trend' if IB range < 0.35*atr_d and no extension; else 'normal_variation'.
    Rules are checked in the order listed: double_distribution, trend, neutral, non_trend, normal, normal_variation."""
def open_type(prev_vah: float, prev_val: float, first30: Bars) -> str:
    """'open_drive' (opens and never trades back through the open in the first 30 min, moving > 0.3% away);
    'open_test_drive' (tests one side within 15 min then drives the other way past the open by > 0.3%);
    'open_rejection_reverse' (moves one way > 0.3% then closes the 30 min beyond the open the other way);
    'open_auction_in_range' (opens inside prev value area), else 'open_auction_out_of_range'."""
```
**Tests:**
- 1 bar L = 100, H = 101, V = 100, bin 0.25: 4 bins of 25 each; POC = the lowest bin (ties → lowest index); VA covers ≥ 70.
- A bell distribution (bins 10, 20, 40, 20, 10) gives POC the middle bin and VA = the middle 3 (80%).
- TPO letters for 13 periods; IB from A+B.
- Each day type and open type from a hand-built series (one test per type).

### M4.3 `orderflow/delta.py`
```python
def delta_bvc(b: Bars, window: int = 50) -> np.ndarray:
    """Bulk volume classification: dC = close - prev close; sigma = std(dC over the previous `window` bars, ddof=1; past-only);
    buy = V * Phi(dC/sigma); delta = 2*buy - V; 0 where sigma is NaN or 0."""
def delta_clv(b: Bars) -> np.ndarray:            # V*(2C-H-L)/(H-L); 0 when H==L
def delta_tick(prices: np.ndarray, sizes: np.ndarray, bids: np.ndarray, asks: np.ndarray) -> np.ndarray:
    """Lee-Ready: +size if price > mid, -size if < mid, else tick rule vs previous different price."""
def cvd(delta: np.ndarray, day: np.ndarray) -> np.ndarray:   # session cumulative
def divergence(b: Bars, cvd_: np.ndarray, lookback: int = 20) -> np.ndarray:
    """+1 bullish (close at a new `lookback` low while cvd above its value at the previous low), -1 bearish (mirror), else 0."""
```
**Tests:**
- dC = 0 gives delta 0.
- A rising noisy series (dC = 0.1 + N(0, 0.05), seed 3, 200 bars) gives mean delta > 0 over bars 51–200. A constant-step series has σ = 0, so its delta is 0 by definition: assert that too.
- The past-only property: delta at t is unchanged when bars after t are removed.
- CLV: H 11, L 9, C 11, V 100 gives 100.
- Lee-Ready: price 10.02 with mid 10.00 gives +size.
- Divergence on a constructed series gives +1 at the expected index.

### M4.4 `orderflow/footprint.py`
```python
def footprint(b1m: Bars, tf_minutes: int, bin: float) -> list[dict]:
    """For each display bar of tf_minutes: rows [{price, buy, sell, delta}] by spreading each 1m sub-bar's BVC buy/sell volume uniformly
    over the bins it spans; bar delta, poc, stacked_imbalances (>=3 consecutive bins where buy >= 3*sell of the bin below, or sell >=
    3*buy of the bin above). method='BVC estimate from 1-minute bars'."""
```
**Tests:** conservation (Σ buy + Σ sell = Σ volume within 1e-9); a constructed ladder detects one stacked buy imbalance.

### M4.5 `orderflow/bigtrades.py` and `orderflow/absorption.py`
```python
def big_bars(b: Bars, hist: Bars, mult: float = 5.0, min_sessions: int = 10) -> list[dict]:
    """Bars whose volume > mult x the median volume of the same minute-of-session over the previous 20 sessions in `hist`
    (null if fewer than min_sessions); each with ret_5m/15m/30m after (close-to-close), mfe/mae over 30 min, side=sign(close-open)."""
def absorption(b: Bars, vwap: np.ndarray, sigma: np.ndarray, prev_vah: float | None, prev_val: float | None) -> list[dict]:
    """Bars with volume >= 3x median(volume, previous 20 bars) and range <= 0.5x median(range, previous 20) located within 0.2% of the
    day's low/high so far, a VWAP band (±1σ, ±2σ) or prev VAH/VAL; side = 'bid' (at lows: selling absorbed, bullish) or 'offer'."""
```
**Tests:** hand series for each; the time-of-day median uses past sessions only.

### M4.6 `orderflow/heatmap.py` + depth capture (forward-only)
- **Probe:** with H1 merged, `AngelClient.quote_full` on 3 tokens returns `depth_bid`/`depth_ask` with 5 levels. Check the SmartAPI docs (web) for a 20-level depth mode: REST or WebSocket "depth 20". Record the result. **No WebSocket dependency in this card.**
- Capture loop while the server runs in market hours:
  - watch list ≤ 25 symbols: the tape's top 10 bull + top 10 bear + NIFTY and BANKNIFTY near futures (NFO tokens from M3.1);
  - every 5 s: one `quote_full` per exchange (≤ 50 tokens per call; the 1 call/s limit is respected);
  - rows `ts, symbol, side ("bid"|"ask"), level (0-4), price, qty` go to an in-memory ring buffer of 2 hours per symbol;
  - every 60 s, flush to the dataset `depth_snaps` (`_RAW_NEW`, key `symbol, ts, side, level`, partition `day`).
- `heatmap(snaps: pl.DataFrame, bin: float, step_s: int = 30) -> {times, prices, qty[[...]]}`: the maximum resting quantity per (time bucket, price bin).
- **Tests:** ring buffer bounds (2 h); matrix assembly from 3 fixture snapshots; the loop is skipped outside market hours.

### M4.7 `orderflow/timescales.py`
```python
def alignment(sym_bars: dict[str, Bars]) -> dict:
    """For tf in 5m,15m,60m,1d,1w: trend = last structure event dir (+1/-1/0) from marketlab.structure; vs_value = +1 if close > VWAP
    (intraday) or > SMA20 (1d/1w) else -1; vs_va = +1 above prev VAH, -1 below prev VAL, 0 inside; roc = sign(close/close[-10]-1);
    score = mean of all signs. Returns {rows:[{tf, trend, vs_value, vs_va, roc}], score}."""
```
15m/60m/1w bars are resampled from 5m/1d with numpy (group by floor of the epoch minus the IST offset). **Test:** an all-rising series gives score +1.

### M4.8 `/api/of/*` (`routes_orderflow.py`)
Data per request:
- today: 1m bars from `Sources.yahoo_chart(f"{sym}.NS", "1m", "1d")`; if that fails, `AngelClient.candles(token, "ONE_MINUTE", ...)`;
- past dates: `load_bars(settings, [sym], d, d, interval="1m")`, falling back to `interval="5m"`.

Routes:
- `/api/of/profile?sym&date&kind=volume|tpo`
- `/api/of/delta?sym&date&method=bvc|clv`
- `/api/of/footprint?sym&date&tf=5`
- `/api/of/bigtrades?sym&date`
- `/api/of/heatmap?sym&date`: `{"empty": true, "reason": "depth not captured for this date"}` when absent.
- `/api/of/timescales?sym`
- `/api/of/bars?sym&date&tf=1|5` (candles for the workbench chart, with `vwap`, `bands` and `anchored` optional)

Each response includes `{method, data_source: "yahoo_1m"|"angel_1m"|"stored_1m"|"stored_5m", estimated: true}`.

**Commits:** one per sub-card: `M-4.1: vwap`, `M-4.2: volume and market profile`, `M-4.3: delta and CVD`, `M-4.4: footprint`, `M-4.5: big trades and absorption`, `M-4.6: depth heatmap capture`, `M-4.7: timescales`, `M-4.8: order-flow endpoints`.

---
## PHASE M5 — Strategy library (CPU)

### M5.0 Pre-registration `config/strategy_library.toml` (committed before ANY gated result)
**Depends:** M5.2–M5.5 code merged. The keys must exist, but no `strat-run` has been executed.
**Files:** `config/strategy_library.toml`, `tests/unit/test_strategy_library.py`.

**Format:**
```toml
# Pre-registered on <date> before any gated result (PLAN-M Part 0 rule 13). Never edit; add strategy_library_<YYYYMMDD>.toml instead.
[protocol]
research_end = 2025-09-30
daily_folds = "calendar years 2014..2025 (validation.splits.walk_forward_folds), embargo 5 sessions"
intraday_folds = "calendar months 2024-10..2025-09, training/inner window = all earlier months from 2024-09-13, embargo 1 session"
universe_daily = "u500 point-in-time (panel_universe.u500 on the signal date)"
universe_intraday = "u200 point-in-time with 5m bars present"
capital_grid = [100000, 200000, 500000, 1000000]
reference_capital = 500000
slots = 10
priority = "rs63"           # same-day signals taken by descending 63-day relative strength vs NIFTY 50 (Market Lab)
costs = "stocksense.costs: delivery for 1d, intraday for 5m/60m; slippage table; net_2x = slippage x2"
exit_selection = "walk_forward_inner: for each test fold, exit with best net Sharpe on all earlier data (minus embargo); min 30 inner trades else default_exit"
gate_version = 2
ic_label_daily = "y_c2o_10"
ic_label_intraday = "y_oc"
note = "Exploratory per-stock results in the UI are not evidence; only these trials are."

[strategy.ml_swing_vcp]
name = "VCP tightening break"
source = "marketlab_swing"
tier = "none"
family = "m-swing"
timeframe = "1d"
kind = "trend"
direction = "long"
exits = ["trail3", "ema20", "rr2_20", "rr3_40", "time5", "time10"]
default_exit = "rr3_40"
params = {}                       # Market Lab defaults; no tuning grid
strength = "breakout_atr"         # see strategies/base.py STRENGTHS
hypothesis = "Market Lab's candidate; expected to beat ew500 net of costs if the edge is real."
reference = "market-lab lab/swing.py:194-200"
```
**Content:**
- One section per registered strategy (M5.2–M5.5), generated by `stocksense lab registry-toml > config/strategy_library.toml` from `strategies.base.REGISTRY` (a command added in this card), then reviewed by hand.
- `mr` strategies list `mr5` in `exits`; `trend` ones do not (Market Lab rule).
- Intraday exits are `["rr1","rr15","rr2","rr3","be2","trail","scale","eod"]` with `default_exit = "rr2"`.

**Tests:**
- The file parses.
- Every `[strategy.*]` key exists in `REGISTRY` and vice versa.
- Every exit exists in `exits.EXITS_1D` / `exits.EXITS_INTRA`.
- Each family has ≥ 10 strategies (PBO needs 10; the test prints counts).
- m-opt trials use benchmark 0 (cash), and their option P&L is a return on underlying notional. Strategies with `status = "await_data"` are listed but skipped by `strat-run`. Every `k151_*` section carries `paper_section` and `paper_eq`.
- The SHA256 of the file equals the constant `LIBRARY_SHA` in the test, so editing the file breaks the suite (pattern of `test_gate_frozen`).

**Commit:** `M-5.0: strategy library pre-registration`.

### M5.1 Indicators, exits, strategy base, cost helper, data loader
**Files:** `src/stocksense/strategies/{__init__.py,base.py,indicators.py,exits.py,costs.py,data.py}`, `tests/unit/test_strat_indicators_golden.py`, `tests/unit/test_strat_exits.py`.

**`base.py`:**
```python
Kind = Literal["trend", "mr", "breakout", "flow", "event", "xs_factor", "index_timing", "options"]
@dataclass(frozen=True)
class StrategySpec:
    key: str; name: str
    source: Literal["marketlab_swing", "marketlab_intraday", "tier", "k151", "flow"]
    tier: Literal["F", "C", "B", "none"]
    family: Literal["m-swing", "m-intra", "m-xs", "m-index", "m-opt"]   # m-opt: §2 option combos + dispersion (M5.4)
    timeframe: Literal["1d", "5m", "60m"]
    kind: Kind
    direction: Literal["long", "short", "both"]
    exits: tuple[str, ...]
    default_exit: str
    signal: Callable[[dict], tuple[np.ndarray, np.ndarray]] | None   # per-stock: f -> (sig bool[n], stop float[n]); None for xs/index
    score: Callable[..., pl.DataFrame] | None = None                 # xs_factor: (settings, start, end) -> (date, sid, score)
    timing: Callable[..., pl.DataFrame] | None = None                # index_timing: (settings, start, end) -> (date, position in {0,1})
    strength: str = "signal"                                          # key into STRENGTHS
    forming: Callable[[dict], dict | None] | None = None             # radar FORMING state (VCP only by default)
    description: str = ""; reference: str = ""
REGISTRY: dict[str, StrategySpec] = {}
def register(spec: StrategySpec) -> StrategySpec: ...            # KeyError on duplicate key or duplicate name
def by_name_or_key(s: str) -> StrategySpec: ...                  # accepts Market Lab display names and keys
STRENGTHS: dict[str, Callable[[dict, np.ndarray], np.ndarray]] = {
    "signal": lambda f, sig: sig.astype(float),
    "breakout_atr": lambda f, sig: np.where(sig, (f["c"] - f.get("hh20", f["c"])) / f["atr"], 0.0),
    "dev_sigma": lambda f, sig: np.where(sig, np.abs(f["c"] - f["vwap"]) / np.maximum(f["atr"], 1e-12), 0.0),
    "rs63": lambda f, sig: np.where(sig, np.nan_to_num(f.get("rs63", np.zeros(f["n"]))), 0.0),
}
```
- The registry is filled by importing `strategies.swing`, `intraday`, `tiers`, `k151` and `flowstrats` from `strategies/__init__.py`.
- `strategies.__version__ = "1"`. The code-version hash used in trial specs is the sha1 of the concatenated source text of those modules, first 10 hex characters (`code_hash()`).

**`indicators.py`:**
- `daily_features(d: DailyFrame, nifty: NiftyTable) -> dict[str, np.ndarray]`: a line-by-line port of `lab/swing.py features()` (lines 26-74, quoted in the extraction), using `npx` helpers:
  - `pd.Series(x).rolling(k).mean()` → `npx.roll(x, k, "mean")`;
  - `.std()` → `"std"`;
  - `ema` / `rma` → `npx.ema` / `npx.rma`;
  - `s.pct_change(63)` → `npx.pct_change(c, 63)`.
- `nifty.reindex(date)` → align by date with a dict lookup. `NiftyTable` is built from `index_daily` (`index_name == "NIFTY 50"`): `c, sma200 = roll(c, 200, "mean"), roc63 = pct_change(c, 63)*100`, as `nifty_table()` (77-82).
- `mkt_up = (nx.c > nx.sma200)`, with False when missing.
- `supertrend(h, l, c, atr, mult)` is ported from `lab/features.py` (verbatim loop).
- `intraday_features(bars: Bars, daily: DailyFrame | None, tf_min: int, index_feat: dict | None, sign: int = 1) -> dict`: a line-by-line port of `lab/features.py build()` (63-215, quoted in the extraction), plus `mirror(bars)` and `index_context(f)`.
  - `pd.factorize(date)` → `npx.factorize`;
  - `groupby(day).cumsum()` → `npx.group_transform`;
  - the rvol pivot → `npx.rvol_same_bar` (Part 5.1);
  - the opening-range `transform("max")` on masked values → `group_transform(np.where(k < nb, h, nan), day, "max")`;
  - the previous-day table `dd[col].shift(1).reindex(day)` → per-day aggregates shifted by one day index;
  - the daily context `.shift(1)` then `reindex(date)` → dict by date.
- `DailyFrame` dataclass: `date: np.ndarray (datetime64[D])`, `open, high, low, close, volume` (float). Built by `strategies/data.py`.

**`data.py`:**
```python
def daily_frame(settings: Settings, sym: str, end: date | None = None, adjusted: bool = True) -> DailyFrame | None:
    """Completed sessions of one symbol from panel_daily (adj_open/high/low/close and adj_volume when adjusted; rows with volume 0 dropped)."""
def intraday_frame(settings: Settings, sym: str, start: date, end: date, interval: Literal["5m", "1m"] = "5m") -> Bars | None:
    """load_bars(..., basis="adjusted") for one symbol as Bars."""
def nifty_table(settings: Settings, end: date | None = None) -> NiftyTable: ...
def completed_end(clock: Clock, cal: TradingCalendar, settings: Settings) -> date:
    """Last session whose EOD data is in panel_daily and whose close has passed (drops today's partial bar)."""
```
Adjusted prices remove splits, so Market Lab's `bad` 35% split filter is not needed and is **not** ported.

**`exits.py`:**
- `EXITS_1D = ("trail3","ema20","rr2_20","rr3_40","time5","time10","mr5")`, and `follow_1d(f, ie, stop, mode) -> tuple[int, float, str]`: a port of `lab/swing.py follow()` (222-248) returning `(exit_index, exit_price, reason)`. The reason is one of `gap_stop|stop|target|mr5|ema20|time`. Fills are unchanged: stop checked before target; a gap through the stop fills at the open; target `max(tgt, open)` after the entry day; time exit at the close.
- `EXITS_INTRA = ("rr1","rr15","rr2","rr3","be2","trail","scale","eod")`, and `follow_intra(f, i_e, s, e, r, xe) -> dict[str, float]` (R multiples, plus `"mfe"`): a verbatim port of `lab/engine.py follow()` (57-103), with `entry_window(f) -> (entry_ok, exit_idx)` from `windows()` (49-54) and `trend_mask(f, kind)` (36-46).

**`costs.py`:**
```python
def trade_cost_frac(product: Literal["delivery","intraday"], notional: float, settings: Settings, slip: SlippageTable,
                    turnover_rank: int, vol_21: float, med_turn_20: float, when_in: str, when_out: str, as_of: date | None) -> float:
    """Round-trip cost as a fraction of notional: (trade_cost BUY + SELL totals)/notional + slip_frac(in) + slip_frac(out)."""
```
- It uses `costs.india.trade_cost(...).total` for BUY and SELL at `notional`, plus `costs.slippage.slip_frac(...)` for each side.
- `when_in`: "open" (next-bar or next-day open).
- `when_out`: "now" for stop/target, "open" for gap stops, "close" for time and MA exits.
- When `turnover_rank` is unknown (index, or intraday with no rank), use 200.

**Golden tests:** extend `scripts/golden/run_marketlab.py` to run Market Lab's `lab.swing.features(df, nifty)` on `inputs_1d_stock.json` plus a synthetic NIFTY, and `lab.features.build(df, daily, 5, index_context, 1)` and `build(mirror(df), ...)` on `inputs_5m_stock.json`. Save every array to `golden_features_1d.json` and `golden_features_5m.json`. Our `daily_features` and `intraday_features` must match every key (floats 1e-9, NaN = NaN, booleans exact).

**Exit tests (hand):**
- Long e = 100, stop 95 (r = 5), mode `rr2_20`. Bars: day ie (o 100, h 104, l 99, c 103), day +1 (o 103, h 111, l 102). Exit on day +1 at max(110, 103) = 110, reason target.
- Gap: day +1 opens 94 ≤ stop 95 → exit 94, reason gap_stop.
- Same bar touching both the stop and the target: the stop wins.
- trail3 moves to breakeven after the close peak ≥ e + r.
- Intraday `follow_intra`: the hand chain from `engine.follow` gives rr2 = 2.0 when H reaches e + 2r before L reaches s; eod = (C[-1] − e)/r when neither.

**Commit:** `M-5.1: indicators, exits, strategy base, costs, data`.
**Provides:** `daily_features`, `intraday_features`, `mirror`, `index_context`, `follow_1d`, `follow_intra`, `entry_window`, `trend_mask`, `trade_cost_frac`, `daily_frame`, `intraday_frame`, `nifty_table`, `REGISTRY`, `register`, `STRENGTHS`.

### M5.2 Market Lab's 23 swing + 39 intraday strategies
**Files:** `src/stocksense/strategies/{swing.py,intraday.py}`, `tests/unit/test_strat_ml_golden.py`.

**Swing:** port all 23 functions of `lab/swing.py` (90-206) verbatim in numpy. Register each with its Market Lab name, a key, and its kind from `STRATS`:

| key | name | kind |
|---|---|---|
| ml_swing_breakout20 | 20-day breakout | trend |
| ml_swing_breakout55 | 55-day breakout (Turtle) | trend |
| ml_swing_52wk | 52-week high + volume | trend |
| ml_swing_pullback20 | Pullback to 20 EMA | trend |
| ml_swing_pullback50 | Pullback to 50 SMA | trend |
| ml_swing_rsi2 | RSI(2) dip | mr |
| ml_swing_3down | 3 down days in uptrend | mr |
| ml_swing_bb_low | Close under lower Bollinger | mr |
| ml_swing_squeeze | Volatility squeeze break | trend |
| ml_swing_nr7 | NR7 breakout | trend |
| ml_swing_inside | Inside-day breakout | trend |
| ml_swing_flag | Bull flag | trend |
| ml_swing_ema_cross | EMA20/SMA50 cross | trend |
| ml_swing_golden | Golden cross | trend |
| ml_swing_macd | MACD cross > 0 | trend |
| ml_swing_supertrend | Supertrend flip | trend |
| ml_swing_adx | ADX trend dip-buy | trend |
| ml_swing_engulf | Bullish engulfing @EMA | trend |
| ml_swing_hammer | Hammer @MA | trend |
| ml_swing_pocket | Pocket pivot | trend |
| ml_swing_gap_go | Earnings-style gap & go | trend |
| ml_swing_vcp | VCP tightening break | trend |
| ml_swing_rs_leader | RS leader dip-buy | trend |

- Family `m-swing`, timeframe 1d, direction long.
- Exits: all 7 for mr, 6 (no mr5) for trend.
- Default exit: `rr3_40` for VCP, `trail3` for trend, `mr5` for mr (Market Lab's chart defaults).
- VCP keeps `1.8*atr*2` (= 3.6 ATR) and says so in its description. `forming` for VCP is `swing_live.setup_state` (43-62).

**Intraday:** port all 39 functions of `lab/strategies.py` and their helpers (`cross_up`, `nz`, `mn`, `_first`) verbatim:
- keys `ml_intra_<snake_name>`, e.g. `ml_intra_orb_15m`, `ml_intra_order_block`, `ml_intra_bos_50_pullback`;
- names exactly as the `STRATEGIES` dict keys;
- family `m-intra`, timeframe `5m`, direction `both` (shorts via `mirror`), exits `EXITS_INTRA`, default `rr2`;
- trend filter: Market Lab's `trends=("D","I","DI")`. The gated runner registers **one trial per (strategy, trend filter)**, so that is 39 × 3 = 117 trials in `m-intra`. List them in the toml as `trend_filters = ["D","I","DI"]`. The UI shows trend `I` by default (Market Lab's candidate uses `"I"`).

**Golden:** extend `run_marketlab.py`. For each swing strategy, Market Lab's `fn(f)` on the 1d input gives (sig, stop). For each intraday strategy, `fn(f)` and `fn(fm)` on the 5m input. Save `golden_signals_*.json` (indices of True and stop values at those indices). Ours must match exactly.

**Commit:** `M-5.2: Market Lab strategies`.

### M5.3 Tier-list strategies (`strategies/tiers.py`), exact rules
Each rule goes in `description` (plain words) and here. "f" is the indicator dict.

**Family by timeframe, for every strategy in M5.3–M5.5:**
- 1d per-stock → `m-swing`;
- 5m per-stock → `m-intra`;
- xs_factor → `m-xs`;
- index_timing and options → `m-index`.

Strategies using an extra exit (`vwap_touch`, `va_edge`, `ib_ext`, `poc_touch`, `ll10_exit`, `next_open`, `ma_exit`) get it from `exits.EXITS_EXTRA`. Each extra exit is a function `(f, ie, stop, side) -> (exit_index, price, reason)`, defined and hand-tested in M5.3 itself.

**F-tier (negative controls):**
- `tier_f_trendline` (1d, long, m-swing):
  - swing lows by fractal n = 3, confirmed 3 bars later;
  - line through the last two confirmed swing lows with the later one higher and ≥ 10 bars apart, extended to t: `line_t = l1 + (l2 − l1)·(t − i1)/(i2 − i1)`;
  - signal: `low_t ≤ line_t + 0.3·atr_t` and `close_t > line_t` and `close_t > open_t`;
  - stop `line_t − 0.5·atr_t`; exits = all 1d except mr5; default `rr2_20`.
- `tier_f_rsi_oversold` (1d, long): `rsi14` crosses up through 30 (`rsi14[t−1] < 30 ≤ rsi14[t]`); stop `c − 2.5·atr`; no trend filter; default `time10`.
- `tier_f_ict_unicorn` (5m, both, m-intra):
  1. A breaker: a bullish order block (from `marketlab.smc.order_blocks` on bars up to t) that was `broken` (close through it).
  2. An FVG formed after the break and overlapping the breaker zone.
  3. Signal when price trades back into the overlap and closes outside it in the break direction.
  - Stop beyond the breaker's far edge. Default `rr2`.

**C-tier:**
- `tier_c_vwap_mr` (5m, both):
  - entry window bar time ≥ 10:00 (mins ≥ 600);
  - long when `close < vwap − 2σ` (σ from `orderflow.vwap.session_vwap`) and `rsi2 < 10`;
  - stop `vwap − 3σ`; target = vwap (custom exit `vwap_touch`, added to `EXITS_INTRA_EXTRA` and defined in `exits.py` as exit at the first bar whose high ≥ vwap, else eod);
  - exits `["vwap_touch","rr1","eod"]`, default `vwap_touch`.
- `tier_c_btst` (1d, long; entry at the **close** of t, exit at the open of t+1):
  - conditions: `chg > 2`, `close ≥ low + 0.9·(high − low)`, `v > 1.5·vavg`, NIFTY close > NIFTY SMA50;
  - its own fill rule: entry at the t close with `when_in="close"` cost plus an extra 0.1% (because the decision needs the close: documented as optimistic); exit at the t+1 open (`when_out="open"`);
  - exits `["next_open"]`.
- `tier_c_orb_15`, `tier_c_orb_30` (5m, both): the close crosses above `or15h`/`or30h` with `rvol ≥ 1.2`; stop = the opposite OR side; default `rr2`. They differ from Market Lab's ORB (midpoint stop, no rvol) on purpose; keep both.
- `tier_c_macro_trend` (index_timing, m-index): position 1 for day t+1 if at least 3 of these hold at close t, else 0:
  - USDINR 20d change < 0;
  - BRENT 20d change < 0;
  - US10Y 20d change < 0 (all from `macro_daily`, using `available_ts ≤ close of t`);
  - FII index-futures long ratio 5d change > 0.
- `tier_c_macro_fii` (index_timing): position 1 if `fii_ratio > 0.5` and `fii_ratio_chg5 > 0`.

**B-tier:**
- `tier_b_absorption` (5m, both): an absorption flag on side `bid` (M4.5) at bar b; signal at bar b+1 if `close > high_b`; stop `low_b − 0.1·atr`; target vwap (`vwap_touch`) or `rr2`.
- `tier_b_cvd_divergence` (5m, both): `divergence == +1` (M4.3, BVC) and `close > high[t−1]`; stop = the swing low; default `rr2`.
- `tier_b_str_1d` (xs_factor, m-xs): score = −`ret` (one-day return) at t.
- `tier_b_str_5d` (xs_factor): score = −Σ over 5 days of (ret − industry mean ret) (industry from `panel_daily.industry`).
- `tier_b_smt` (index_timing, 5m, m-index): NIFTY makes a new 20-bar low while BANKNIFTY's low is above its own 20-bar low → long NIFTY at the next bar open, exit at vwap or 15:15. Mirror for shorts. Data: NIFTY/BANKNIFTY 5m from `intraday_bars` if present; **probe whether index 5m bars exist** (symbols `NIFTY 50`/`NIFTY BANK`); if absent, mark the strategy `status="await_data"` in the toml.
- `tier_b_amt_80` (5m, both):
  - the open is outside the previous day's value area (volume profile of the previous session's 5m bars);
  - two consecutive 30-minute TPO periods close inside the VA → enter at the next bar;
  - target the opposite VA edge (exit `va_edge`, added to the extra exits); stop = the VA edge it re-entered ± 0.2%.
- `tier_b_ib_ext` (5m, both): after 10:15, the close beyond the IB high with `rvol ≥ 1`; target IB high + 1×IB range (`ib_ext` exit); stop at the IB midpoint.
- `tier_b_open_drive` (5m, both): `open_type == "open_drive"` (M4.2) in the direction away from the previous value → enter at 09:45; stop at the first-30-minute extreme opposite; exit eod.
- `tier_b_poor_extreme` (5m, short when a poor high, long when a poor low):
  - the previous day had a poor high and today trades above it, then closes back below it → short;
  - stop = today's high + 0.1·atr; target = the previous day's POC (`poc_touch` exit).
- `tier_b_turtle` (1d, long): `c > hh20` (System 1) → enter at the next open; exit at the close below the 10-day low (`ll10_exit`, an extra exit); stop `e − 2·atr` (N = atr20 via rma 20); no pyramiding.

**Tests:** for every rule, a constructed series where it fires exactly at the expected index; its mirror series does not fire (long rules); truncation (the signal at t is unchanged when rows > t are removed).
**Commit:** `M-5.3: tier-list strategies`.

### M5.4 *151 Trading Strategies*, equity-applicable subset (`strategies/k151.py`)
**Status of this card: rewritten 2026-11-23 after the orchestrator read the paper.** The source is the user's local copy `C:\Users\harsh\Downloads\ssrn-3247865.pdf` (SSRN version dated August 17, 2018; 361 PDF pages; book page N = PDF page N+1).

**How to read the paper.** PyMuPDF is installed in the global Python (`C:\Python314`). Run it from a neutral folder:
```powershell
Push-Location $env:TEMP; python -c "import fitz,sys;d=fitz.open(r'C:\Users\harsh\Downloads\ssrn-3247865.pdf');open(sys.argv[1],'w',encoding='utf-8').write(''.join(f'\n=====PAGE {i+1}=====\n'+p.get_text() for i,p in enumerate(d)))" $env:TEMP\k151.txt; Pop-Location
```
Then read `$env:TEMP\k151.txt` with the Read tool. Do not copy the PDF or its text into the repo: it is copyrighted. Notes cite section and equation numbers only.

**Section numbers and formulas below are verified against the paper; they supersede the earlier draft.** The draft had these errors, now fixed:
- KNN is §3.17 and is **single-stock** (not cross-sectional, not k = 50).
- Support/resistance is §3.14 and Channel is §3.15.
- The Channel base rule is **mean reversion at the floor**; breakout is the paper's stated alternative.
- §3.5 sign: buy rising **call** IV.
- MA lengths: §3.12 uses 10/30 and §3.13 uses 3/10/21.
- §4.4 is **IBS**, not the 5-day return.
- §4.3 is **selectivity (1 − R²) × alpha**, not a trend R².
- §4.1 has **no skip month**.
- §7.4 is a **short straddle** conditioned on VIX − realised vol.
- Macro momentum is §19.2, not §18.x.
- Cash-and-carry is §6.2, not §6.1.

Implement each formula as written. Adapt only where NSE data forces it, and record each adaptation in DECISIONS `### M5.4`. Long-only is forced for delivery equity (no overnight cash shorting): every "short the bottom decile" leg is either dropped (per-stock rules) or kept implicitly as a ranking score (xs factors evaluated by IC and the long top-k of `evaluate_model`). Record this once.

**xs_factor (family m-xs; score frame (date, sid, score) over the u500; evaluated via `evaluate_model` in M5.7):**

**Notation used below.** P = adjusted close (`adj_close`); r = daily simple return; "month" = 21 sessions; all windows end at t (data ≤ t); a score is NaN where its window is incomplete. Every xs score is computed per date over the PIT u500 (or `in_fno` where stated), and written as `(date, sid, score)`.

| key | § | score at t (verified formula; eq. numbers from the paper) | adaptation (record) |
|---|---|---|---|
| k151_mom_cum | 3.1 | `R_cum = P(t−21)/P(t−252) − 1` (eq. 267, T = 12 months, S = 1 month skip) | monthly → 21-session months |
| k151_mom_riskadj | 3.1 | monthly returns R(m) for the 12 formation months m = 1..12 (skipping month 0); `R_mean/σ` with σ ddof=1 (eqs. 268–270) | — |
| k151_low_vol | 3.4 | `−σ(r, 252)` ddof=1, no skip | — |
| k151_low_vol_126 | 3.4 | `−σ(r, 126)` (the paper's 6-month option) | — |
| k151_implied_vol | 3.5 | `ΔIV_call,1m − ΔIV_put,1m`: change over 21 sessions of the ATM call IV and ATM put IV of the stock's nearest non-expiry-day monthly options (from M3.4 `flow_options_strikes`). **Positive = buy** (the paper's "difference" variant) | null before 2024-08; F&O stocks only |
| k151_multifactor | 3.6 | demeaned-rank average (eqs. 276–277): `s_i = mean_A [rank(f_Ai) − mean_j rank(f_Aj)]` over A ∈ {k151_mom_cum, k151_low_vol, k151_resid_mom} | value is unavailable, so residual momentum replaces it |
| k151_resid_mom | 3.7 | monthly returns. For each stock: OLS with intercept of R_i(m) on factors over the 36 months ending at month 1 (skip 1) → β; residuals `ε(m) = R_i(m) − Σβ·F(m)` (α excluded) over formation months 1..12; score `mean(ε)/σ(ε)` (eqs. 278–282) | FF3 is not available for India. F = {MKT = `panel_market.ew500_ret` compounded monthly, SMB = mean monthly return of the bottom half minus the top half of the u500 by `med_turn_60`}; HML dropped. Needs ≥ 37 months |
| k151_pairs | 3.8 | for each stock A, partner B = the most correlated stock (252-day daily-return correlation) in the same `industry`, re-chosen monthly; `R = ln P(t)/P(t−5)` for both; `R̃_A = R_A − (R_A + R_B)/2`; score `−R̃_A` (eqs. 285–289) | 5-day window chosen (the paper leaves t1, t2 free) |
| k151_mr_single | 3.9 | `R_i = ln P(t)/P(t−5)`; score `−(R_i − mean_u500 R)` (eqs. 292–297; D_i = −γR̃_i) | the whole u500 as one cluster |
| k151_mr_multi | 3.9.1 | `−(R_i − mean of R over i's industry)` (eq. 309); industries with < 3 members are merged into "OTHER" | clusters = `panel_daily.industry` |
| k151_mr_weighted | 3.10 | Ω = industry dummies (binary, so the intercept is subsumed), z_i = 1/σ²_i(63); `ε = R − Ω Q⁻¹ Ωᵀ Z R`, `Q = Ωᵀ Z Ω`, `R̃ = Z ε` (eqs. 314–317); score `−R̃` | — |
| k151_statarb_opt | 3.18.1 | dollar-neutral mean-variance weights (eq. 358) with a diagonal C = σ²_i(63) and E_i = −R̃_i of k151_mr_multi: `w_i = E_i/σ²_i − (Σ_j E_j/σ²_j)/(Σ_j 1/σ²_j) · 1/σ²_i`; score `w_i` | a diagonal C instead of a factor model (fn. 64): record |
| k151_knn_xs | 3.17 (fn. 60) | the per-stock KNN prediction `Ŷ` (below) used as a cross-sectional score | the paper's footnote 60 variant |
| k151_fut_contrarian | 10.3 | universe `in_fno`; weekly (5-session) return R_i; score `−(R_i − R_m)`, R_m = equal-weight mean (eqs. 469–470). Rebalanced weekly: hold the score constant from each Friday's close | the stock (cash) return proxies the futures return before 2024-07; afterwards use the near-future settle from `flow_oi_daily`. Record both |
| k151_fut_contrarian_act | 10.3.1 | as above, restricted to the stocks in the upper half by `v = ln(V_wk/V_prevwk)` **and** the lower half (within those) by `u = ln(OI_wk/OI_prevwk)` (eqs. 472–473); others NaN | OI from `flow_oi_daily` (2024-07→) |
| k151_fut_trend | 10.4 | `sign(R_252)/σ_252` (eqs. 474–475) | — |
| k151_fut_trend_tanh | 10.4 | `tanh(R_252/κ)/σ_252`, κ = cross-sectional std of R_252 that day (the paper's smoothing) | — |

**Per-stock 1d rules (family m-swing). These are position-state rules: "establish long while condition true, liquidate when the exit condition is true".** The engine enters at the next open after the first true close, and exits at the next open after the exit close (exit key `state_exit`, a new extra exit = next-open after the exit condition). Stops are added only where stated and recorded as additions.

| key | § | long entry condition at close t | liquidation | stop |
|---|---|---|---|---|
| k151_ma1_sma | 3.11 | `P > SMA(50)` (eq. 321) | `P < SMA(50)` | none (`stop = c − 3·atr` is used only for R/risk display; never hit-tested): record |
| k151_ma1_ema | 3.11 | `P > EMA(50)` (eq. 320, λ-form = pandas span EMA) | `P < EMA(50)` | as above |
| k151_ma2 | 3.12 | `SMA(10) > SMA(30)` (eq. 322, the paper's example lengths) | `SMA(10) < SMA(30)` | — |
| k151_ma2_stop | 3.12 | same | the reverse cross **or** `P < (1 − 0.02)·P(t−1)` (eq. 323, Δ = 2%) | — |
| k151_ma3 | 3.13 | `SMA(3) > SMA(10) > SMA(21)` (eq. 324) | `SMA(3) ≤ SMA(10)` | — |
| k151_pivots | 3.14 | `C_t = (H+L+C)/3` of t−1, `R = 2C_t − L(t−1)`, `S = 2C_t − H(t−1)` (eqs. 325–327); entry when `P > C_t` (eq. 328) | target: the first later bar with high ≥ R fills at max(R, open); flip: `P < C_t` (the short signal, which closes the long) → next open | `S` (added protective stop, recorded; the paper's long leg has none) |
| k151_channel_mr | 3.15 | Donchian on the previous T = 20 closes: `B_down = min P(t−20..t−1)`; entry when `low_t ≤ B_down` and `close_t > B_down` (touch and hold the floor; eq. 331 "P = B_down" on daily bars) | `high ≥ B_up` (= max P(t−20..t−1)) → fill at max(B_up, open) | `B_down − 1·atr` (added) |
| k151_channel_bo | 3.15 | the paper's alternative: `close_t > B_up` with volume > 1.5 × 20-day mean ("break … with an increase in the traded volume") | `close < B_down` | `B_down` |
| k151_knn | 3.17 | single-stock KNN (eqs. 332–341). Target `Y(t) = P(t+10)/P(t) − 1` (T = 10). Features X: SMA of P over 5, 10, 20, 50 and SMA of volume over 5, 20, each divided by P(t) (scale-free; record). Training sample for date t = the T* = 504 most recent dates t′ with `t′ + 10 ≤ t` (label known: purge). Min–max normalise each feature over that sample (eq. 337). k = floor(√T*) = 22, Euclidean distance (eq. 338), `Ŷ` = mean of the neighbours' Y (eq. 339). Entry when `Ŷ > z1 = 0.01` | `Ŷ ≤ z2 = 0` (eq. 341) | `c − 2.5·atr` (added) |

KNN cost note: 500 stocks × ~3,000 dates × 504 × 6 floats. Vectorise per stock with `np.einsum` on (dates × 504) blocks of 250 dates. Target < 2 s per stock. Measure it and record.

**index_timing / sector / multi-asset (family m-index).** The data comes from `index_daily` (sector indices: the NIFTY sectoral list the probe finds, expected 12–15: AUTO, BANK, FIN SERVICE, FMCG, IT, MEDIA, METAL, PHARMA, PSU BANK, PVT BANK, REALTY, ENERGY, HEALTHCARE, CONSUMER DURABLES, OIL & GAS; record the exact list). Returns are open-to-open from the session after the decision. Costs: delivery round trip per rebalanced weight change plus 5 bps (ETF proxy; record).

| key | § | rule |
|---|---|---|
| k151_sector_mom | 4.1 | each month-end: `R_cum = P(t)/P(t−252) − 1` (eq. 361, **no skip**); hold the top 3 equal-weight for 1 month ("top decile" of ~13 indices rounds to 1; 3 chosen, recorded) |
| k151_sector_mom_6m | 4.1 | the same with T = 126 |
| k151_sector_mom_ma | 4.1.1 | as k151_sector_mom, but a pick is held only if `P > SMA(200)` of that index (eq. 362); otherwise its slot is cash |
| k151_sector_dual_mom | 4.1.2 | as k151_sector_mom when NIFTY 50 `P > SMA(200)` (eq. 363); otherwise 100% in GOLD-INR (`macro_daily` GOLD × USDINR; record that no gold ETF series is stored) |
| k151_sector_alpha | 4.2 | rank by the intercept α of an OLS of each sector's daily returns on NIFTY 50 daily returns over 252 sessions (FF3 → market only; record); top 3 monthly |
| k151_sector_r2 | 4.3 | the same regression; keep the sectors with R² below the cross-sectional median (higher selectivity 1 − R²); among them hold the top 3 by α (the paper's quintile/sub-quintile double sort, scaled to ~13 assets; record) |
| k151_sector_ibs | 4.4 | `IBS = (C − L)/(H − L)` of the previous day (eq. 370); hold the 3 sector indices with the lowest IBS for 1 session, rebalanced daily (long leg only) |
| k151_multi_asset_trend | 4.6 | assets: NIFTY 50, NIFTY MIDCAP 150, NIFTY 10 YR BENCHMARK G-SEC (if in `index_daily`; probe), GOLD-INR. Monthly: keep the assets with `R_cum(252) > 0` and `P > SMA(200)`; weights `w_i ∝ R_cum,i/σ²_i` (eq. 373, σ over 252), normalised to sum 1; none qualifies → cash |
| k151_vol_target | 6.5 | NIFTY 50 weight `w = min(1, σ*/σ)` with σ* = 0.15 and σ = INDIA VIX/100 (implied, per fn. 120); the rest in cash (0 return; record). Rebalance only when `|Δw|/w > κ = 0.10` |
| k151_macro_mom | 19.2 | single-country adaptation of the 4 state variables (fn. 235), each a 1-year change z-scored on its own trailing 3-year history: international trade = −Δ1y ln USDINR; monetary policy = −Δ1y US10Y (no Indian short rate is stored; record); risk sentiment = NIFTY 50 1-year return; business cycle dropped (no GDP/CPI series; record). Long NIFTY 50 when the mean > 0; decided monthly, minimum hold 3 months (the paper's 3–6 month holding) |
| k151_announcement | 19.5 | long NIFTY 50 open→close on announcement days (RBI MPC decision days and US FOMC decision days; for FOMC the Indian session **after** the US announcement), cash otherwise. Dates go in a new `config/macro_events.toml` (`rbi_mpc = [dates]`, `fomc = [dates]`, each with a `source_url`), filled from the RBI and Federal Reserve calendars via web fetch, 2012→2025-09. A test pins the file's row count |
| k151_cash_carry | 6.2 | **informational, no trial.** `F* = (S − D)·e^{r(T−t)}` with D = 0 (no dividend forecast; record) and r = `derivs.R_FREE`; basis `B = (F − F*)/S` (eqs. 421–422) for NIFTY/BANKNIFTY near futures from `fo_bhavcopy`; shown on `/flow` |

**Options (family m-index). Underlying NIFTY; **monthly** expiries (weekly NIFTY rules changed during the sample, and monthly gives a consistent series; record). Fills at `settle_price` from `fo_bhavcopy`. Window 2024-07→2025-09, about 14 cycles, so low power: stated in RESULTS.** A generic engine `strategies/options_bt.py:run_combo(legs, entry_rule, settings)` takes legs `(type CE|PE|FUT, side ±1, strike_rule, qty)`, where strike_rule picks the listed strike nearest to `spot·(1+x)`.
- Entry is at the settle of the first session after the previous monthly expiry. Held to expiry, settling at intrinsic `(S_T − K)+` / `(K − S_T)+` with S_T = the expiry-day `underlying_price`.
- Expiry-day option rows are not used for entry prices (M3.4 rule).
- The P&L uses each leg's payoff formula from §2, e.g. eq. 1 for a covered call: `f_T = S_T − S_0 − (S_T − K)+ + C`. It is expressed as a fraction of S_0.

| key | § | legs |
|---|---|---|
| k151_covered_call | 2.2 | long FUT (near month; proxy for stock), short CE at K = ATM+2% |
| k151_protective_put | 2.4 | long FUT, long PE at ATM−5% |
| k151_collar | 2.53 | long FUT, long PE ATM−5%, short CE ATM+5% |
| k151_long_straddle | 2.22 | long CE + long PE at ATM |
| k151_short_straddle | 2.25 | short CE + short PE at ATM (eq. 96) |
| k151_short_strangle | 2.26 | short CE ATM+3%, short PE ATM−3% (eq. 101) |
| k151_iron_condor_credit | 2.50 | **the paper's "long iron condor" is the credit trade** (eq. 226): long PE K1, short PE K2, short CE K3, long CE K4, equidistant κ = 2% of spot around ATM (K2 = ATM−1κ, K3 = ATM+1κ) |
| k151_vrp_straddle | 7.4 | k151_short_straddle taken only when `VRP = INDIA VIX(entry) − realised vol` > 0, where realised vol = the annualised % std of NIFTY daily log returns over the previous 21 sessions. The paper measures "since the beginning of the month", which is 0 days at entry; record the adaptation. Otherwise flat |
| k151_vrp_straddle_dh | 7.4.1 | k151_vrp_straddle plus a daily delta hedge with NIFTY near futures at settle. Δ comes from BS on each leg's IV (`derivs.implied_vol` on that day's settle; rows with dte < `MIN_DAYS` reuse the last IV). Futures hedge costs are included |
| k151_risk_reversal | 7.5 | long CE ATM+5%, short PE ATM−5% (§2.12 long combo) |

**Every §2 option strategy is registered** (user instruction of 2026-11-23: "implement each and everything when possible"). They go in their own family `m-opt`, so they do not dilute `m-index`. The family runs on NIFTY and on BANKNIFTY monthly options, giving 2 trials per strategy.

The leg table below is transcribed from the paper; eq. numbers are the payoff tests. Strike conventions, recorded once in DECISIONS:
- ATM = the listed strike nearest to spot.
- One "OTM/ITM step" κ = the listed strike nearest to spot·(1 ± 0.02).
- Ladders and condors use equidistant steps of κ, as the paper requires.
- Ratio spreads use N_L/N_S = 2/1 (backspreads) and 1/2 (ratio spreads), the paper's "typical" values.
- The stock leg is the near-month NIFTY/BANKNIFTY future.
- Calendar and diagonal spreads (2.18–2.21) are short the near monthly and long the next monthly. They are closed at the near expiry, with the long leg marked at that day's settle.

| § | key suffix | legs (side × type @ strike) | payoff eq. |
|---|---|---|---|
| 2.2 | covered_call | +FUT, −CE@ATM+κ | 1 |
| 2.3 | covered_put | −FUT, −PE@ATM−κ | 5 |
| 2.4 | protective_put | +FUT, +PE@ATM | 9 |
| 2.5 | protective_call | −FUT, +CE@ATM | 13 |
| 2.6 | bull_call_spread | +CE@ATM, −CE@ATM+κ | 17 |
| 2.7 | bull_put_spread | +PE@ATM−2κ, −PE@ATM−κ | 21 |
| 2.8 | bear_call_spread | +CE@ATM+2κ, −CE@ATM+κ | 25 |
| 2.9 | bear_put_spread | +PE@ATM, −PE@ATM−κ | 29 |
| 2.10 | long_synth_fwd | +CE@ATM, −PE@ATM | 33 |
| 2.11 | short_synth_fwd | +PE@ATM, −CE@ATM | 37 |
| 2.12 | long_combo | +CE@ATM+κ, −PE@ATM−κ | 41 |
| 2.13 | short_combo | +PE@ATM−κ, −CE@ATM+κ | 47 |
| 2.14 | bull_call_ladder | +CE@ATM, −CE@ATM+κ, −CE@ATM+2κ | 53 |
| 2.15 | bull_put_ladder | −PE@ATM, +PE@ATM−κ, +PE@ATM−2κ | 58 |
| 2.16 | bear_call_ladder | −CE@ATM, +CE@ATM+κ, +CE@ATM+2κ | 63 |
| 2.17 | bear_put_ladder | +PE@ATM, −PE@ATM−κ, −PE@ATM−2κ | 68 |
| 2.18 | calendar_call | −CE@ATM(near), +CE@ATM(next) | 73–74 |
| 2.19 | calendar_put | −PE@ATM(near), +PE@ATM(next) | 75–76 |
| 2.20 | diagonal_call | +CE@ATM−2κ(next, deep ITM), −CE@ATM+κ(near) | 77–78 |
| 2.21 | diagonal_put | +PE@ATM+2κ(next), −PE@ATM−κ(near) | 79–80 |
| 2.22 | long_straddle | +CE@ATM, +PE@ATM | 81 |
| 2.23 | long_strangle | +CE@ATM+κ, +PE@ATM−κ | 86 |
| 2.24 | long_guts | +CE@ATM−κ, +PE@ATM+κ | 91 |
| 2.25 | short_straddle | −CE@ATM, −PE@ATM | 96 |
| 2.26 | short_strangle | −CE@ATM+κ, −PE@ATM−κ | 101 |
| 2.27 | short_guts | −CE@ATM−κ, −PE@ATM+κ | 106 |
| 2.28 | long_call_synth_straddle | −FUT, +2·CE@ATM | 111 |
| 2.29 | long_put_synth_straddle | +FUT, +2·PE@ATM | 116 |
| 2.30 | short_call_synth_straddle | +FUT, −2·CE@ATM | 121 |
| 2.31 | short_put_synth_straddle | −FUT, −2·PE@ATM | 126 |
| 2.32 | covered_short_straddle | +FUT, −CE@ATM, −PE@ATM | 131 |
| 2.33 | covered_short_strangle | +FUT, −CE@ATM+κ, −PE@ATM−κ | 135 |
| 2.34 | strap | +2·CE@ATM, +PE@ATM | 138 |
| 2.35 | strip | +CE@ATM, +2·PE@ATM | 143 |
| 2.36 | call_ratio_backspread | −1·CE@ATM, +2·CE@ATM+κ | 148 |
| 2.37 | put_ratio_backspread | −1·PE@ATM, +2·PE@ATM−κ | 153 |
| 2.38 | ratio_call_spread | +1·CE@ATM−κ, −2·CE@ATM | 158 |
| 2.39 | ratio_put_spread | +1·PE@ATM+κ, −2·PE@ATM | 163 |
| 2.40 | long_call_butterfly | +CE@ATM+κ, −2·CE@ATM, +CE@ATM−κ | 168 |
| 2.40.1 | mod_call_butterfly | +CE@ATM+κ, −2·CE@ATM, +CE@ATM−2κ (K1−K2 < K2−K3) | 173 |
| 2.41 | long_put_butterfly | +PE@ATM−κ, −2·PE@ATM, +PE@ATM+κ | 177 |
| 2.41.1 | mod_put_butterfly | +PE@ATM−2κ, −2·PE@ATM, +PE@ATM+κ (K3−K2 < K2−K1) | 182 |
| 2.42 | short_call_butterfly | −CE@ATM−κ, +2·CE@ATM, −CE@ATM+κ | 186 |
| 2.43 | short_put_butterfly | −PE@ATM+κ, +2·PE@ATM, −PE@ATM−κ | 191 |
| 2.44 | long_iron_butterfly (credit) | +PE@ATM−κ, −PE@ATM, −CE@ATM, +CE@ATM+κ | 196 |
| 2.45 | short_iron_butterfly (debit) | −PE@ATM−κ, +PE@ATM, +CE@ATM, −CE@ATM+κ | 201 |
| 2.46 | long_call_condor | +CE@ATM−1.5κ, −CE@ATM−0.5κ, −CE@ATM+0.5κ, +CE@ATM+1.5κ (nearest listed) | 206 |
| 2.47 | long_put_condor | same strikes with puts, signs (+,−,−,+) | 211 |
| 2.48 | short_call_condor | the 2.46 strikes, signs (−,+,+,−) | 216 |
| 2.49 | short_put_condor | the 2.47 strikes, signs (−,+,+,−) | 221 |
| 2.50 | long_iron_condor (credit) | +PE K1, −PE K2, −CE K3, +CE K4 (K = ATM−1.5κ, −0.5κ, +0.5κ, +1.5κ) | 226 |
| 2.51 | short_iron_condor (debit) | −PE K1, +PE K2, +CE K3, −CE K4 | 231 |
| 2.52 | long_box | +PE@ATM+κ, −PE@ATM−κ, +CE@ATM−κ, −CE@ATM+κ: **informational arbitrage check** (`f_T = K1 − K2 − D`, eq. 236); no trial | 236 |
| 2.53 | collar | +FUT, +PE@ATM−κ, −CE@ATM+κ | 238 |
| 2.54 | bull_short_seagull | −PE@ATM−κ, +CE@ATM, −CE@ATM+κ | 242 |
| 2.55 | bear_long_seagull | +PE@ATM−κ, −CE@ATM, +CE@ATM+κ | 248 |
| 2.56 | bear_short_seagull | −PE@ATM−κ, +PE@ATM, −CE@ATM+κ | 254 |
| 2.57 | bull_long_seagull | +PE@ATM−κ, −PE@ATM, +CE@ATM+κ | 260 |

Keys are `k151_opt_<und>_<suffix>` with und ∈ {nifty, banknifty}, giving 116 trials in `m-opt`.

Every trial is unconditional: entered every cycle, which is what the paper describes. Only §7.4/7.4.1 add a condition.

**Each payoff is unit-tested against its own equation.** Example: for 2.36, eq. 148 at S_T = K2 + 10 with N_L = 2 and N_S = 1 gives `2·10 − (K2 + 10 − K1) − H`.

The engine computes P&L leg by leg from settle prices; the tests check that it equals the closed-form eq. at expiry.

**Option costs:** a new function `strategies/costs.py:options_cost_frac(legs_premia, spot, settings)`, using the current NSE/SEBI schedule: STT on the sell-side premium; STT on the intrinsic value of exercised ITM long legs; exchange transaction charge on premium; stamp duty on the buy premium; SEBI fee; GST; brokerage ₹20 per order. Futures legs use the futures schedule. **Probe** the current rates from the NSE, SEBI and broker pages, and record them with the date and URL. Add `[options]` and `[futures]` tables to a new `config/costs_derivs.toml`. Do **not** edit `config/costs.toml`.

**Strategies from other asset classes, carried over to NSE equity/index data where the mechanism transfers.** The user asked for everything that can be implemented. Each adaptation is recorded in DECISIONS, and the map row says "adapted from §x".

| key | § | family | rule (formula from the paper; the adaptation in brackets) |
|---|---|---|---|
| k151_hp_ma | 8.1 | m-swing | HP filter (eqs. 437–439) on **monthly** adjusted closes of each stock, λ = 100·12² = 14,400 (the paper's monthly convention). Then MA(3) vs MA(12) of the filtered series: long when MA(3) > MA(12), exit when it falls below. **Look-ahead guard:** HP is two-sided, so S*(t) is re-solved at every month-end on data ≤ t, and only the last point is used (one-sided HP). A test asserts that the value at t is unchanged when later months are removed. [FX → stocks] |
| k151_hp_mom | 10.4 fn. 172 | m-xs | 12-month return of the one-sided HP trend series; score = that return / σ_252 [the paper's suggested filter alternative] |
| k151_roll_yield | 9.1 | m-xs | φ = P_near / P_next of each F&O stock's futures settle (eq. 454); score = φ (buy backwardation) [commodity futures → stock futures; 2024-07→] |
| k151_hedging_pressure | 9.2 | m-index | from `participant_oi` index futures: HP = long/(long+short) per participant. Speculators = FII, hedgers = CLIENT (recorded as an assumption). Long NIFTY next session when FII HP is above its trailing 126-session median **and** CLIENT HP is in the bottom quintile of its trailing 126 sessions; 6-month formation, per the paper [COT → NSE participant OI] |
| k151_lt_value | 9.4 | m-xs | `v = mean P over t−1386..t−1134 (5.5 to 4.5 years ago, fn. 155) / P(t)` (eq. 455); score = v [commodity value → stock long-term reversal; needs 5.5 years of history] |
| k151_skew | 9.5 | m-xs | sample skewness of daily returns over 252 sessions (eqs. 456–458, σ with T−1); score = −S (buy low skew) [commodities → stocks] |
| k151_ou | 9.6 | m-xs | fit OU (eq. 459) to X = ln P over 252 sessions by AR(1) OLS: `X_{t+1} = c + b·X_t + e`, giving κ = −ln b, a = c/(1−b), σ_∞ = sd(e)/√(1−b²); score = (a − X_t)/σ_∞ when 0 < b < 1, else NaN [futures term-structure model → stock mean reversion] |
| k151_gold_diversify | 9.3 | informational | passive 80/20 NIFTY 50 / GOLD-INR, monthly rebalance; no trial (allocation, not alpha). The active version is infeasible: no policy-rate series |
| k151_fut_trend_rel | 10.4 | m-xs | η_i = sign(R_i − R_m) over 252 sessions (the paper's de-skewed variant), score η_i/σ_i, F&O universe |
| k151_distress | 15.3 | m-xs | market-only distress proxy. No Indian bankruptcy labels exist to fit the paper's logistic model, so this is the mean of the z-scores of {SIGMA = 63-day daily σ·√252; −EXRETAVG = −(geometrically weighted, weight 2^{−k/3}, mean of the last 12 monthly log excess returns vs ew500); −RSIZE = −ln(med_turn_60 / Σ med_turn_60 of the u500) (turnover proxies size); −ln P}. Score = −distress (buy the healthiest). Recorded as "not a bankruptcy probability" |
| k151_distress_voltgt | 15.3.1 | m-xs | the same ranking, with the daily return of its S1 long book scaled by `min(σ_target/σ̂, 1)`, σ_target = 0.12 and σ̂ = 252-day realised vol of that book (eq. 519 with the no-leverage prefactor of fn. 203). Evaluated on the scaled return series |
| k151_ann_15m | 18.2 | m-intra | ANN of eqs. 521–537 on **15-minute bars** (resampled from 5m) of the u200 stocks and NIFTY. Inputs: R̂(t) with T1 = 100 bars; EMA and EMSD of R̂ at τ ∈ {2, 4, 12, 24} bars (30 min, 1 h, 3 h, 6 h, as in the paper) with λ = (τ−1)/(τ+1) (fn. 224); RSI at τ′ ∈ {12, 24, 48}. Output: K = 5 quantiles of the next bar's R̂ (eq. 530). Network: 2 hidden ReLU layers of 32, softmax output, cross-entropy (eq. 536), SGD/Adam, seed 1, torch on CPU. Retrained each walk-forward month on all earlier months, with fn. 225's warm-up exclusion. Rule (eq. 537): buy when argmax = top quantile, hold 1 bar (15 min), flat by 15:15. [BTC → NSE; ~14 months of 5m data, so low power, stated] |
| k151_nb_news | 18.3 | m-xs | Bernoulli naive Bayes (eqs. 538–546) on headline keywords per stock → P(up next day). **status = "await_data"** until 250 sessions of stored headlines exist. `marketlab/news.py` must append every fetched headline to a new raw dataset `news_headlines` (`ts, query, symbol, title, source, link`) from the first day the server runs |
| k151_appA_d0_mr | App. A | m-intra (daily bars) | Appendix A's delay-0 strategy, re-implemented in numpy (the R code is copyrighted, so do not copy it). The expected return is the overnight log return `R_i(t) = ln(P^AO(t)/P^AC(t−1))` (eqs. 555–557). Holdings `w = −(mean-variance optimum with E^eff_i = sign(E)·max(|E| − τ_i, 0))` (eqs. 550–552), with the bound `|H_i| ≤ 0.01·ADDV_i` (eq. 548) and dollar neutrality. Universe: top 2,000 → **top 200 by 21-day ADDV**, recomputed every 21 sessions. Risk model: the statistical PCA model below, refit every 21 sessions. Fill at the open and exit at the close (open→close). **The paper itself calls this "borderline in-sample"**: run it as written, plus a variant `k151_appA_d0_mr_5m` that fills at the 09:20 5m bar open (2024-09→), and record both. Long history from daily bars, 2011→ |
| k151_appA_d1_mom | App. A | m-intra (daily bars) | the delay-1 variant: E_i = −ln(C(t−1)/O(t−1)), with the code's sign convention preserved (hold = −optimum). Trade the open→close of t. A test pins the sign convention against a 3-stock hand case |

**Shared utility: `strategies/riskmodel.py` (paper §6.3.1, eqs. 425–427; also used by App. A and §3.18):**
- `stat_risk_model(returns: np.ndarray[N×T], k: int | None) -> (xi2, lam, V)`: the correlation matrix ρ; principal components λ(A), V(A); K = eRank (Roy & Vetterli 2007) = round(exp(−Σ p_a ln p_a)), with p_a = λ_a/Σλ over the positive eigenvalues; `ψ = diag(ξ²) + Σ_{A≤K} λ(A) V V^T`, `ξ² = 1 − Σ λ(A) V(A)_i²`.
- `inv_cov(...)` via Woodbury.
- `mv_weights(E, invC, lower, upper, dollar_neutral=True)`: eq. 358 for the unbounded case; with bounds, iterate clipping until the bounds hold, then re-impose Σw = 0 (≤ 50 iterations; record that this approximates the paper's `bopt.calc.opt`).
- Tests: ψ has unit diagonal; K = 1 for a rank-1 synthetic; the weights satisfy Σw = 0 and the bounds.

Two extra trials use it:
- `k151_statarb_opt_pca` (§3.18.1 with the PCA covariance instead of the diagonal one);
- `k151_dispersion` (§6.3/6.3.1). On each monthly cycle from 2024-07, sell the NIFTY ATM straddle and buy ATM straddles on the N* = 20 NIFTY 50 constituents with the lowest specific variance `w_i²σ_i²ξ_i²` (eq. 427). n_i follows eq. 424, with free-float weights replaced by equal weights; NSE index weights are not stored, so this is recorded. Family m-opt; about 14 cycles, low power, stated.

**Fixed income via NSE G-Sec indices (§5.2–5.4, 5.13); informational only.** If the M5.4 probe finds NSE G-Sec maturity-bucket indices in `index_daily` (for example "NIFTY 4-8 YR G-SEC", "NIFTY 8-13 YR G-SEC", "NIFTY 15 YR AND ABOVE G-SEC", "NIFTY 10 YR BENCHMARK G-SEC"), build:
- bullet = one bucket;
- barbell = the shortest and the longest bucket, with dollar weights that solve eqs. 390–391 for the middle bucket's duration (durations proxied by mid-maturity, recorded);
- ladder = equal weight across all buckets (eq. 395);
- a flattener/steepener (eq. 416) as the long-minus-short bucket return, conditioned on the 3-month change of US10Y.

These show as allocation backtests with no trial (not equity alpha). If no such indices exist, the map row reads "no bond index data".

**Not implemented. Every one gets a map row with this exact reason:**

| § | reason |
|---|---|
| 3.2 | EPS is not parsed (`results_index` has no values); possible after T8.6 |
| 3.3 | no book value |
| 3.16 | no M&A deal terms data |
| 3.19 | market making needs resting orders and queue position: the paper-only rule forbids order code |
| 3.20 | **deferred, not dropped**: the alpha-combo procedure of eqs. 360 + steps 1–11 runs over the finished m-xs trials' daily returns. It is registered after M8.1 in `strategy_library_<date>.toml`, together with **8.4's min-variance combo** (eqs. 445–452) of k151_mom_cum and k151_low_vol |
| 4.5 | no leveraged/inverse ETFs on NSE |
| 5.5–5.12, 5.14, 5.15 | need bond-level prices, credit ratings, CDS or swap curves |
| 6.4 | needs tick bid/ask of two ETFs on one index |
| 7.2, 7.3, 7.3.1 | no VIX futures or VIX ETNs on NSE |
| 7.6 | no variance swaps |
| 8.2, 8.2.1, 8.3, 8.5 | no multi-currency forward or interest-rate data; FX is not in A.6 |
| 10.1, 10.1.1, 10.1.2 | hedging recipes without an alpha rule |
| 10.2 | no trading rule in the paper; NIFTY near/next spread shown informationally on `/flow` |
| 11.2–11.7 | no CDO/MBS market in our data |
| 12.1, 12.2 | no convertible bond data |
| 13.1, 13.2, 13.2.1 | tax structures not applicable (no muni bonds; India has no dividend imputation) |
| 14.1–14.4 | no inflation swaps, TIPS, weather or power futures data |
| 15.1, 15.2, 15.2.1–15.2.3 | distressed-debt control strategies are not exchange-traded |
| 16.2–16.6 | NSE REITs are too few (< 5) for 16.4. 16.2/16.5 informational only if the REIT series exist in bhavcopy (probe series "RR"); 16.3.x and 16.6 are physical property |
| 17.2, 17.6 | illegal; excluded on principle |
| 17.3–17.5 | cash management, not tradable alpha |
| 19.3 | no CPI series |
| 19.4 | no multi-country bond data |
| 20 | infrastructure is not a separate tradable on our data |

**Coverage of the paper (2026-11-23).**
- Read in full by the orchestrator: pp. 1–134 (all strategy chapters, Appendix A code and Appendix B), the glossary, the acronyms, the math notation and the index.
- The 144-page reference list (about 1,740 entries) was scanned programmatically, because it does not fit in context. That scan found three India-specific references, which `docs/learn/15-151-strategies-map.md` and the matching notes must cite:
  - Birari & Rode (2014), "Edge Ratio of Nifty for Last 15 Years on Donchian Channel" (for §3.15);
  - Gunasekarage & Power (2001), moving-average rules in South Asian markets, Emerging Markets Review 2(1) (for §3.11–3.13);
  - Vipul (2009), box-spread arbitrage efficiency of Nifty index options, Journal of Futures Markets (for §2.52's informational box check).

**Map completeness (replaces the `>= 140` test in M1.1).** The paper's contents have **173 headings labelled "Strategy:" plus §3.10** (a strategy whose heading lacks the label), which makes 174 strategy sections, plus 2 Appendix A strategies. `15-151-strategies-map.md` must have exactly one row per section number in a pinned list `K151_SECTIONS` (176 entries, written in the test from the contents: 2.2–2.57 with 2.40.1 and 2.41.1; 3.1–3.20 with 3.9.1, 3.10 and 3.18.1 as a sub-row of 3.18; …; "A.d0" and "A.d1"). Each row's status ∈ {implemented, adapted, informational, await_data, deferred, infeasible}. Implemented and adapted rows name a key that exists in `REGISTRY`. The test fails on any missing, duplicate or unknown section.

**Tests (hand numbers):**
- Momentum: P = 100 at t−252, 120 at t−21 → R_cum = 0.20.
- Low vol: r = (0.01, −0.01, 0.01, −0.01) → σ = 0.011547 (ddof 1).
- Multifactor: 3 stocks with factor A values (1, 2, 3) and B values (3, 1, 2): demeaned ranks A (−1, 0, 1), B (1, −1, 0) → s = (0, −0.5, 0.5).
- Multi-cluster: R = (0.02, 0.00 | 0.05, 0.01, 0.03) in industries (X, X | Y, Y, Y) → R̃ = (0.01, −0.01 | 0.02, −0.02, 0.00).
- Weighted regression: with equal z the result equals the multi-cluster result × z; with z = (1, 4) in one cluster of 2 and R = (0.02, 0.00), the weighted mean is 0.004, ε = (0.016, −0.004) and R̃ = Zε = (0.016, −0.016), which sums to 0 (eq. 318).
- statarb_opt: Σw = 0 within 1e-12.
- Pivots: H 110, L 100, C 105 → C = 105, R = 110, S = 100.
- MA rules: on a 40-day constructed series, entries and exits at the hand-listed indices.
- Channel MR: a floor touch then a ceiling touch gives one trade.
- KNN: the toy series where Y repeats every 20 days gives Ŷ equal to the true Y when k = 1. The purge property: no neighbour t′ has t′ + 10 > t.
- 4.1: the top 3 on a 5-index toy table.
- 4.4 IBS: C 105, L 100, H 110 → 0.5.
- 4.6 weights: R_cum (0.2, 0.1), σ (0.2, 0.1) → raw (5, 10) → (1/3, 2/3).
- vol_target: VIX 30 → w = 0.5; VIX 10 → w = 1 (capped); no rebalance when |Δw|/w = 0.05.
- Straddle P&L: premia 100 + 90, expiry spot 150 away from K → 190 − 150 = 40 before costs.
- Credit iron condor: eq. 226 at S_T between K2 and K3 gives +C; beyond K4 it gives C − κ.
- Covered call eq. 1 at S_T > K gives K − S_0 + C.
- Truncation for every xs score and every timing position (the value at t is unchanged when rows > t are removed).
- Every key in this card exists in `REGISTRY` with its § in `reference`.

**Commit:** `M-5.4: 151 Trading Strategies subset`.

### M5.5 Flow strategies (`strategies/flowstrats.py`), the user's main thesis
- `flow_oi_long_buildup` (1d, m-swing): `flow_oi_daily.quad_label == "long buildup"` with `oi_chg_pct > 5` and `price_chg_pct > 1` at t → enter at the t+1 open; exits `rr2_20`, `time5`, `time10`, `trail3`; stop `c − 2·atr`. Data from 2024-07.
- `flow_short_covering_rev` (1d): "short covering" at t after the 10-day return < −8% → long; stop at the 10-day low − 0.3·atr; exits `time5`, `rr2_20`.
- `flow_deliv_spike` (1d): `deliv_spike` and `close > sma50` → long; stop `c − 2.5·atr`; exits `time10`, `rr2_20`, `trail3`. Data from 2019-08.
- `flow_fii_index` (index_timing, m-index): `fii_ratio_chg5 > 0.05` → long NIFTY for the next 5 sessions.
- `flow_gex_regime` (index_timing, 5m, m-index; NIFTY 5m bars needed, the same probe as `tier_b_smt`):
  - day t's regime is from the EOD GEX of t−1;
  - negative → follow the 09:15–10:15 direction from 10:15 to 15:15;
  - positive → fade a move beyond VWAP ± 1σ back to VWAP.
- `flow_preopen_imb` (5m, m-intra): |preopen imb| > 0.3 and the IEP gap has the same sign → follow at 09:20, exit 15:15. `status = "await_data"` until ≥ 120 sessions of pre-open exist (not gated now; listed in the toml with a status key).
- `flow_cvd_trend` (5m, m-intra): the CVD (BVC) slope over 09:15–10:15 in the top decile of the u200 → long at 10:15, exit 15:15. Bottom decile → short.

**Tests** as in M5.3.
**Commit:** `M-5.5: flow strategies`.

### M5.6 Exploratory engine, radar scan, stock profile (`strategies/explore.py`, `scan.py`, `profile.py`)
**Spec, `explore.py`:**
```python
@dataclass(frozen=True)
class Trade:
    sym: str; strat: str; exit: str; side: int; t_sig: int; t_in: int; p_in: float; stop: float; target: float | None
    t_out: int | None; p_out: float | None; reason: str; R: float | None; gross_pct: float | None; cost_pct: float; net_pct: float | None
    days: int | None; status: Literal["CLOSED", "OPEN", "SIGNAL"]; mkt_up: bool; rs63: float | None

def run_symbol_1d(sym: str, spec: StrategySpec, exit_key: str, settings: Settings, clock: Clock, *,
                  warmup: int = 260, oos_start: date = date(2024, 7, 1), notional: float = 100_000.0) -> list[Trade]:
    """Market Lab trades_for logic on our completed adjusted daily bars: signal at close i -> entry open i+1; one position at a time
    (busy); skip if risk <= 0 or risk/entry > 0.25; exits via follow_1d; costs via trade_cost_frac(delivery, notional, rank/vol/turnover
    at the signal date); SIGNAL if the signal fired on the last completed close; OPEN if the exit is the last bar by time-out not reached."""
def run_symbol_intra(sym: str, spec: StrategySpec, exit_key: str, trend: str, settings: Settings, clock: Clock, *,
                     start: date, end: date, notional: float = 250_000.0) -> list[Trade]:
    """lab/engine.run_symbol on 5m bars for one strategy/exit/trend: max 1 trade per day; entry open of bar i+1 same day; exits via
    follow_intra; R converted to price p_out = e + side*R*r; costs via trade_cost_frac(intraday, ...)."""
def summarize(trades: list[Trade], oos_start: date) -> dict:
    """{n, win, avgR, totR, pct, avg_hold, pf (gross profit/gross loss), max_consec_losses, is{n, avgR}, oos{n, avgR}}; closed trades only."""
def expected_max_sharpe_null(n_tests: int, n_obs: int) -> float:
    """Bailey & López de Prado (2014) expected maximum of n_tests iid Sharpe ratios under H0 (annualised, with n_obs daily observations):
    E[max] ≈ sqrt(252/n_obs) * ((1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))), γ = 0.5772156649; returns 0.0 for N <= 1."""
def trades_for(sym: str, strat: str, exit_key: str | None, settings: Settings, clock: Clock, tf: str = "1d") -> dict:
    """Market Lab /api/lab/trades shape (lab/swing_live.py:65-112) from run_symbol_*; adds exploratory, costs_model, gated, cost_pct."""
def matrix(sym: str, settings: Settings, clock: Clock, tf: str = "1d") -> dict: ...
```
- Cache per (sym, strat, exit, trend, last_date, `code_hash()`) as JSON in `data_store/lab/explore_cache/<sym>/<hash>.json`.
- **Timing target:** one symbol × one strategy × one exit in < 150 ms. The matrix for one symbol (≈ 40 daily strategies × ≤ 7 exits) in < 10 s.

**`scan.py`:**
- `run_scan(settings, asof)`: for every symbol in the u500 ∪ in_fno at `asof` and every per-stock **1d** strategy with its default exit, `run_symbol_1d`.
- Collect SIGNAL (fired at `asof`), OPEN (with `mkt_up`), recent (closed with t_in ≥ asof − 15 calendar days) and FORMING (`spec.forming(f)` where defined).
- Runs with `concurrent.futures.ProcessPoolExecutor(max_workers=6)`, chunked by symbol. Worker functions must be top-level module functions (Windows spawn).
- Writes `data_store/lab/scan/<asof>.parquet` with columns `sym, strat, state, t_in, p_in, stop, target, rs63, dist_pct, risk_pct, trigger, close`.
- Peak RAM must stay < 2.5 GB: measure it and record it.
- `summary(settings) -> dict`: Market Lab's `/api/lab/swing` key tree. Build:
  - `strategies` (from the gated eval.json files if present, else the exploratory full-universe stats computed once per scan date and cached);
  - `profile`, `best_stocks`;
  - `candidate` (the best gated strategy if any passes, else Market Lab's VCP labelled "exploratory candidate");
  - `live` (the radar);
  - `list`, `exits`.

**`profile.py`:**
- `stock_profile(trades, classification)` with Market Lab's buckets: sector; market cap ≥ 50k cr, 15–50k, <15k; turnover `val_cr` <10, 10–30, 30–100, 100+; base width risk% <6, 6–9, 9–13, >13; rs63 <0, 0–15, 15–30, 30+.
- Minimum counts 25 for sector and 40 for the others.
- `best_stocks`: ≥ 4 trades, top 40 by average.

**CLI:**
- `stocksense lab scan [--asof YYYY-MM-DD]`
- `stocksense lab explore --sym X --strat K [--exit E] [--tf 1d]`
- `stocksense lab registry-toml`

**Tests:**
- Constructed daily series → trades equal the hand list (entry at the next open; target 2R; gap stop).
- With a FixedClock at 11:00 on a session, today's partial bar is excluded (`completed_end`).
- Costs: delivery round trip on ₹1,00,000 with rank 100 equals `trade_cost` totals / 1e5 plus 2 × `slip_frac` (computed in the test from the same functions).
- `expected_max_sharpe_null(1, 252) == 0` (within 1e-9) and it increases in N.
- Scan on the synthetic market writes the parquet with the expected columns.
- The scan uses only data ≤ asof (run with a truncated panel and with the full panel: identical rows).

**Commit:** `M-5.6: exploratory engine, radar scan, stock profile`.

### M5.7 Gated walk-forward runner (`validation/strategy_eval.py`, `strategies/portfolio.py`) + `research strat-run`
**Spec, `portfolio.py`:**
```python
@dataclass(frozen=True)
class SimConfig:
    capital: float; slots: int = 10; product: Literal["delivery", "intraday"] = "delivery"; slip_scale: float = 1.0
def simulate_daily(trades_by_signal_date: dict[date, list[Candidate]], panel: PanelView, cfg: SimConfig, settings: Settings,
                   calendar: TradingCalendar, start: date, end: date) -> pl.DataFrame:
    """k-slot book: on each session d, first process exits scheduled on d (fills at the follow_1d price with exit-side cost and slippage),
    then entries: candidates whose signal date is the previous session, sorted by priority desc, while free slots; qty = floor(NAV_{d-1}/slots
    / (open_d*(1+slip_in))); buy cost via trade_cost; then mark every position at close_d. Returns columns date, net (NAV_d/NAV_{d-1}-1),
    gross (sum position value at close / NAV_d), n_pos, turnover, costs_frac."""
def simulate_intraday(trades: list[Trade], cfg: SimConfig, ...) -> pl.DataFrame:
    """Per session: up to `slots` trades in entry-time order, each sized NAV_{d-1}/slots; P&L from p_in/p_out with intraday costs; all flat
    by 15:15; net = sum pnl / NAV_{d-1}; gross = max concurrent notional / NAV_{d-1}."""
def simulate_timing(position: pl.DataFrame, index_bars: pl.DataFrame, cfg: SimConfig) -> pl.DataFrame:
    """position at close t applies from open t+1: day return = pos*(open->close) on entry day, pos*(close->close) while held; cost on
    each change = delivery round trip on NAV via trade_cost/NAV + 2 bps x slip_scale; gross = pos."""
```
`Candidate(sym, sid, t_sig, spec_key, exit_key, priority)`. The simulator re-runs `follow_1d` with the actual entry fill, to keep stop/target logic identical to explore.

**Spec, `strategy_eval.py`:**
```python
def run_strategy(key: str, settings: Settings, state: State, calendar: TradingCalendar, clock: Clock, *, max_hours: float = 6.0) -> dict:
```
Steps:
1. `spec_dict = library_section(key)` from the toml, plus `{"code_hash": code_hash()}`. Then `trial_id = register_trial(state, family, target, spec_dict)`, where target = `"y_c2o_10"` (1d), `"y_oc"` (intraday) or `"index"` (timing). **Register before evaluation.** Skip the strategy if `reports/m/strategies/<trial_id>/eval.json` exists (resume).
2. Folds:
   - 1d: `walk_forward_folds(settings, calendar)` (years 2014..2025, the last ending 2025-09-30);
   - intraday: months 2024-10..2025-09 (built by `_month_folds(calendar)`, same `Fold` type, embargo 1).
3. For each fold:
   - generate every candidate trade per exit on [start of data .. fold.test_end];
   - pick the exit: best net Sharpe of `simulate_*` at the reference capital on `[first trade, fold.train_end − embargo]`, if ≥ 30 inner trades, else `default_exit`;
   - keep the test-fold trades of the chosen exit;
   - record `chosen_exit[fold.name]`.
4. Concatenate the test folds, then `simulate_*` at every capital in the grid with `slip_scale=1`, and once at the reference capital with `slip_scale=2`.
5. Benchmark = `panel_market.ew500_ret` (1d and intraday) or NIFTY 50 `close/prev − 1` (timing).
6. `daily` = `date, net, net_2x, benchmark, gross` at ₹5,00,000.
7. IC: the strength frame over the universe (0 for no signal), with label `y_c2o_10` (1d) or `y_oc` (intraday) from `load_labels(settings, "research", ...)`. Call `ic_stats(frame.rename({"strength": "score"}), label_col)`. For timing strategies, IC = Spearman of position vs the next-day NIFTY return, computed by date blocks of 21 days (record).
8. `finish_trial(state, trial_id, {"sharpe_daily": ..., ...metrics}, returns_path)`, with returns_path = `reports/m/strategies/<trial_id>/daily.parquet`.
9. `cand = CandidateResult(trial_id, family, target, daily, ic={"mean_ic","ic_tstat"}, trial_ids=(trial_id,))`, then `save_candidate_result(cand, settings)` and `report = validate_candidate(trial_id, settings, state, gate_version=2)`.
10. Write `eval.json`: trades, win, avgR, net Sharpe per capital, 2× Sharpe, max DD, per-year returns, chosen exits, gate lines (from `report.to_dict()`), verdict.

For `xs_factor` strategies:
- write scores to `preds` with `sub=trial_id` via `io.upsert(settings, "preds", df, sub=trial_id)` (columns `date, sid, score(Float32), made_at, mode="research"`);
- call `g2_eval.evaluate_model(trial_id, settings, state, calendar, base_id="s1-gbm-y_c2o_10-e0f7276025")`;
- read in `validation/g2_eval.py` where its S1 backtest daily series is stored (the `g2-s1` trial's `returns_path` in the trials table) and build the CandidateResult from it, with `ic` from the eval dict;
- then validate as above.
- `k151_knn` runs `research wf-run` with its model spec first (fold-wise fitting), then takes the same path.

Further rules:
- Wall cap: stop after the current strategy when `max_hours` is exceeded.
- CLI: `research strat-run --family m-swing|m-intra|m-xs|m-index [--keys a,b] [--max-hours 6]` and `research strat-table` (prints every eval.json row).

**Tests (synthetic market):**
- A strategy whose signal = the planted alpha > 1 (`make_market(planted_ic=0.08)`) gets IC > 0 and net > random.
- A random-signal strategy gets |IC| < 0.02.
- Exit selection never reads test-fold returns: instrument `simulate_*` to record the date ranges it sees and assert max(inner) < fold.test_start − embargo.
- The trial is registered before eval.json is written (compare the registry `created_at` with the file mtime).
- Resume skips finished strategies.
- `simulate_daily` hand case: capital 100,000, 2 slots, one trade entering at 100 and exiting at 110, costs 0 (zero-cost Settings copy, slippage scaled 0) → NAV +5% on the exit day.

**Commit:** `M-5.7: gated strategy walk-forward`.

### M5.8 Append-only lab journal (`strategies/journal.py`)
- Migration (the next free version; Part 3.1):
  ```sql
  CREATE TABLE lab_journal(id INTEGER PRIMARY KEY AUTOINCREMENT, created_ts TEXT NOT NULL, trade_date TEXT NOT NULL, strat TEXT NOT NULL,
    sym TEXT NOT NULL, side INTEGER NOT NULL, ref_price REAL NOT NULL, stop REAL, target REAL, status TEXT NOT NULL
    CHECK(status IN('planned','open','closed','expired')), closed_ts TEXT, close_price REAL, reason TEXT,
    UNIQUE(trade_date, strat, sym));
  CREATE TRIGGER lab_journal_immutable BEFORE UPDATE ON lab_journal
    WHEN NEW.created_ts IS NOT OLD.created_ts OR NEW.trade_date IS NOT OLD.trade_date OR NEW.strat IS NOT OLD.strat
      OR NEW.sym IS NOT OLD.sym OR NEW.side IS NOT OLD.side OR NEW.ref_price IS NOT OLD.ref_price
      OR NEW.stop IS NOT OLD.stop OR NEW.target IS NOT OLD.target
    BEGIN SELECT RAISE(ABORT, 'lab_journal rows are append-only'); END;
  ```
- After the scan, `record_signals(state, scan_rows, clock)` inserts SIGNAL rows as `planned`, with `INSERT OR IGNORE` (idempotent by `(trade_date, strat, sym)`).
- `update_marks(state, settings, asof)` moves `planned → open` at the next open, and `open → closed` when the exit rule hits (using `follow_1d` on completed bars). Only the status/close fields change.
- **Paper record only. No broker.**
- **Tests:** an UPDATE of `ref_price` raises `sqlite3.IntegrityError`; the insert is idempotent; the status transitions.

**Commit:** `M-5.8: append-only lab journal`.

### M5.9 Daily Picks: today's stocks from gate-passing strategies (`strategies/picks.py`, `/api/picks`, page `/picks`)
**Depends:** M5.6 (scan), M5.7 (eval.json + gate reports), M5.8 (journal), M8.1 (verdicts exist; the card can merge before M8.1, since it handles "no passes").
**Why:** this is the user's end goal: a daily list of stocks chosen by strategies whose backtests survived the gate.

**Spec, `picks.py`:**
```python
@dataclass(frozen=True)
class Pick:
    sym: str; side: int; entry_ref: float; stop: float; target: float | None
    strategies: tuple[str, ...]                  # gate-passing strategy keys firing on sym at asof
    evidence: float                              # sum over firing strategies of w_s (below)
    best_trial: str; net_sharpe_5l: float; dsr: float; pbo: float; trades_oos: int
    qty: int; notional_inr: float; risk_inr: float   # paper sizing; 0 when capital unavailable
    reason: str                                  # one line: "VCP break + delivery spike; 2 gated strategies"

def build_picks(settings: Settings, state: State, asof: date, capital_inr: float | None,
                max_picks: int = 10, risk_per_trade: float = 0.01) -> dict:
    """Daily Picks at the close of asof. Returns {asof, gated: [Pick], exploratory: [Pick], n_strategies_passing,
    n_trials_total, note}. Gated list uses ONLY strategies whose latest gate v2 report passes all 9 lines."""
```
**Rules:**
1. **Gated set.** Strategies whose `reports/m/strategies/<trial>/eval.json` verdict is PASS on all nine v2 lines. Families: m-swing and m-xs, whose holding fits a daily pick. m-intra and m-opt picks show in a separate "intraday/options" tab with the same rule.
2. **Signals.** Only from that day's scan (`data_store/lab/scan/<asof>.parquet`, state SIGNAL, completed bars only). For m-xs strategies: the top 10 names by that day's score in the universe at asof.
3. **Strategy weight.** `w_s = DSR_s × min(1, trades_oos_s/100)`, where DSR is the gate's deflated-Sharpe probability, already penalised by the total trial count. `evidence(sym) = Σ w_s` over the gated strategies firing on sym. Rank by evidence; ties go to the higher `net_sharpe_5l` of the best strategy.
4. **Entry, stop and target.** From the best strategy's spec (entry ref = the next open estimate = asof close; stop/target per its exit rule). Side is long only for delivery.
5. **Sizing (CLAUDE.md rule 6).** `capital_inr` is read from the broker at decision time by the caller (`AngelClient` funds via the existing live job path), never a constant. `qty = floor(min(capital·risk_per_trade/(entry − stop), capital/max_picks/entry))`. If capital is unavailable, `qty = 0` and `note` says so.
6. **Exploratory list.** The top 10 by exploratory full-universe stats (M5.6 summary), **always labelled** "EXPLORATORY — not validated". Never merged into the gated list.
7. **No passes.** When no strategy passes, `gated = []` and `note = "No strategy has passed the gate yet (n_trials_total = N). Exploratory candidates below are not evidence."`
8. **Journal.** Every gated pick is written via `journal.record_signals` with `strat = "picks:" + best_trial`. Paper only.
9. **Idempotent by (asof).** Written to `data_store/lab/picks/<asof>.json` and recomputed only if missing or forced (CLAUDE.md rule 7).
10. **Catch-up.** The M2.7 catch-up adds step `"picks"` after `"scan"` and `"flow_eod"`.

**API.** `GET /api/picks?asof=` returns the dict above. Capital and broker details never appear: only `qty` and `notional_inr` (paper). `GET /api/picks/history?days=30` returns past pick files plus their realised outcome from the journal (`closed` rows: R, net %).

**Web, `/picks`** (the first item in the nav; Market Lab sector palette):
- A header strip with the as-of date, strategies passing / trials total, and the note.
- Gated picks as cards: symbol, firing strategies as chips, evidence bar, entry/stop/target, paper qty, best trial's net Sharpe/DSR/PBO, and an "open chart" button (`LabActions.openChart`).
- An "Exploratory" section below, with a grey border and the label.
- A "Track record" table from `/api/picks/history`.
- The footer line: "Backtests decide which strategies may pick; past results do not guarantee future returns."

**Tests:**
- No passing strategy gives `gated == []` and the note mentions n_trials.
- Two passing strategies firing on the same symbol add their weights.
- A failing strategy's signal never appears in gated.
- An exploratory pick never appears in gated.
- Sizing: capital 1,00,000, risk 1%, entry 100, stop 95 → `min(1000/5, 100000/10/100) = min(200, 100) = 100`.
- Capital None → qty 0 and the note says so.
- Idempotent file.
- The journal row is inserted once.
- The API never contains keys matching `/(capital|funds|client|token)/i`.
- Web: the gated and exploratory sections render from the fixture, and the exploratory cards carry the label.

**Commit:** `M-5.9: daily picks from gate-passing strategies`.

---
## PHASE M6 — Frontend: Next.js rebuild (components can start on fixtures after M2.3; live data after M2.6)

**Global rules for M6:**
- Only `web/` is edited; never `src/`.
- Exact dependency versions only (no `^` or `~`). `package-lock.json` is committed.
- Every pure helper ported from Market Lab JS gets a Vitest test. The expected values are computed by hand, or by running the original JS snippet in Node once (copy the function text into a scratch script; record that in a comment).
- API fixtures: the JSON responses saved by the M2.x tests, copied to `web/src/test/fixtures/*.json` (sm_live.json, sm_chart.json, horizon.json, sector.json, stock.json, research.json, lab_trades.json, lab_swing.json, flow_market.json, ...). Each M2/M3 card that adds a route also writes its fixture with `pytest tests/unit/test_api_* --write-web-fixtures` (an option added in M2.0's conftest: when set, the API tests dump responses to `web/src/test/fixtures/`).

### M6.0 Scaffold
**Steps:**
1. Find the version, then scaffold:
   ```
   npm view next versions --json
   ```
   Pick the newest stable that was published ≥ 14 days ago (`npm view next time --json`). Then:
   ```
   npx create-next-app@<that version> web --ts --eslint --app --src-dir --no-tailwind --import-alias "@/*" --use-npm --no-turbopack
   ```
2. Pin these exactly:
   - `react` and `react-dom` (the versions create-next-app chose);
   - `lightweight-charts@4.2.0`;
   - `vitest`, `@vitejs/plugin-react`, `jsdom`, `@testing-library/react`, `@testing-library/jest-dom`;
   - `postcss-prefix-selector`;
   - `react-markdown`, `remark-gfm` (for M6.9).
   For each, choose the latest version published ≥ 14 days ago. Record every version in DECISIONS `### M6.0`.
3. `web/next.config.ts`:
   ```ts
   import type { NextConfig } from "next";
   const config: NextConfig = { output: "export", trailingSlash: true, images: { unoptimized: true }, reactStrictMode: true };
   export default config;
   ```
4. `web/postcss.config.mjs` as in Part 5.7.
5. `web/vitest.config.ts`: jsdom environment, `setupFiles: ["src/test/setup.ts"]` (imports `@testing-library/jest-dom/vitest`), alias `@` → `src`.
6. `package.json` scripts:
   ```
   "dev": "next dev -p 3000",
   "build": "node scripts/copy-learn.mjs && next build",
   "lint": "next lint",
   "test": "vitest"
   ```
   `scripts/copy-learn.mjs` copies `../docs/learn/*.md` to `content/learn/` and writes `content/learn/index.json` (`[{slug, title}]`, title = the first `# ` line).
7. `src/lib/api.ts`:
   ```ts
   export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";
   export async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
     const r = await fetch(API_BASE + path, { cache: "no-store", signal });
     if (!r.ok) throw new Error(`${path}: ${r.status}`);
     return r.json() as Promise<T>;
   }
   export function usePoll<T>(path: string | null, ms: number, opts?: { enabled?: boolean }): { data: T | null; error: Error | null; refresh: () => void } {
     // setInterval while enabled && document.visibilityState === "visible"; AbortController per request; cleanup on unmount/path change
   }
   ```
8. `src/lib/types.ts`: TypeScript interfaces for every route in `docs/API-M.md`, transcribed from the key trees in M2.3–M3.7.
9. `src/lib/ml/format.ts`: the ports of `esc`, `pc`, `cls`, `rs`, `sg`, `rf`, `rsg` (Part A4 of the extraction):
   - `pc(x, d=2)` gives `+1.23%` or `–`;
   - `rs(x)` gives `₹` + en-IN with 2 decimals.
10. `src/app/layout.tsx`: the Google Fonts link (Part 5.7), `globals.css` (index.html's `html,body{margin:0;background:#edf0f3}`, `#disc`, `.divider`, `#boot`), the disclaimer bar text verbatim from Market Lab `index.html`, and a top nav: `Daily Picks · Smart Money · Sector Map · Swing Lab · Money Flow · Order Flow · Strategies · Learn` (Daily Picks is the `/picks` page of M5.9).

**Tests:**
- `usePoll`: fake timers; it stops on unmount and pauses when `document.visibilityState = "hidden"`.
- `format.ts`: `pc(1.234)` = `"+1.23%"`; `pc(null)` = `"–"`; `rs(1234.5)` = `"₹1,234.50"`; `esc('<a "b">')` = `"&lt;a &quot;b&quot;&gt;"`.

**Commit:** `M-6.0: Next.js scaffold`.

### M6.1 Styles, shell and section scaffolds
- `src/styles/ml-smart.css`, `ml-sector.css`, `ml-lab.css`: **verbatim** copies of the `<style>` blocks of `smart.html`, `sector.html` and `lab.html`, including the `:root` variables.
- `src/app/page.tsx`: three sections stacked like `index.html`, separated by `.divider`:
  `<section className="ml-smart" id="smart">`, `<section className="ml-sector" id="sector">`, `<section className="ml-lab" id="lab">`.
- `LabActionsProvider` (Part 5.6) wraps the page. The single `ChartModal` (M6.2) is mounted once at page level, inside `.ml-smart` styling. Market Lab has two chart-modal variants (smart and sector), with identical JS but different CSS: use smart's.
- Each section renders a placeholder until its card lands.

**Tests:** render `page.tsx` with mocked `usePoll`; the three section ids exist; the nav links exist; the build succeeds (`npm run build` produces `out/index.html`).
**Commit:** `M-6.1: styles and shell`.

### M6.2 Smart Money section (port of `smart.html`)
**Read `C:\Users\harsh\Downloads\market-lab\web\smart.html` fully.** Port every function into `src/lib/ml/smart/*.ts`:
- `stage.ts`: `draw`, the line paths, waterfall, brackets, `fxLoop`;
- `tiles.ts`: `tileHTML`, `tiles`, `COLS=[[13,20],[8,12],[4,7],[1,3]]`, `ladder`, stars `Math.max(1, Math.min(5, Math.ceil(Math.abs(disp)/0.8)))`;
- `beads.ts`: `beads`;
- `tip.ts`: `setH`, `tip`;
- `chart.ts`: `openChart`/`loadChart`, `liveCandle`, `layers`, `over`, `levels`, `price`;
- `trades.ts`: `tbtn`, `tlevels`, `tscale`, `tload`, `tsel`, `tmarks`, `tdraw`, `rrbar`, `tstate`, `tpanel`;
- `research.ts`: `rload`, `ring`, `qchart`, `rdraw`.

**React components** (`src/components/smart/`): `IndexCards`, `Stage` (canvases `#bg`, `#fx`, `#beads`, labels, counts, chips, tiles), `Tooltip`, `ChartModal` (a lightweight-charts 4.2.0 instance, created once with the same options as Market Lab, plus the overlay canvas `#mo` redrawn each animation frame), `TradesPanel`, `ResearchNote`. They use the DOM ids and classes of Part A2 so the copied CSS applies.

**Polling:**
- `/api/sm/live` every 4000 ms, with `paths=1` every 30 s or more, else `paths=0`;
- `/api/sm/quote?sym=` every 3000 ms while the modal is open;
- `/api/lab/trades` every 60000 ms while the modal is open;
- `/api/research?sym=` on open.

**Additions over Market Lab:**
1. Strategy dropdown `#tstrat` lists `/api/lab/strategies` grouped by `<optgroup>` (Market Lab swing, Market Lab intraday, Tier F/C/B, 151, Flow).
2. The `#tpanel` header shows a badge `EXPLORATORY · net of StockSense costs`, and a gated chip: `GATE PASS 9/9` (green), `GATE FAIL n/9` (red), `NOT YET TESTED` (grey). The data comes from `trades.gated`.
3. Each stock tooltip and tile detail shows the `flow` mini-card when present: OI quad chip, rvol, delivery z, flow score.
4. Do **not** port the `window` "rt" listener (dead hook).

**Tests:**
- `ladder` scaling on the 36-bar fixture: the bar heights match the JS original's output (record the values by running the original function in Node once).
- Tile column assignment for ranks 1..20.
- Stars: disp 0.5 → 1, 0.9 → 2, 4.1 → 5.
- Bead x for from_open ±4% maps to the edges.
- `esc` applied: render a stock named `<script>x</script>`; the DOM contains no script element.
- `Stage` renders 20 bull and 20 bear tiles from `sm_live.json`.
- `ChartModal` opens on a tile click (mock lightweight-charts with `vi.mock`).

**Commit:** `M-6.2: Smart Money UI`.

### M6.3 Sector Map section (port of `sector.html`)
- Port `route`, `go`, `load`, `drawHorizon` (bubbles: `COLORS` ramp, RUNNING ring, tails, replay controls), the sector view (hub, flows `.flow`, nodes, fcards), the stock view (peers, fcards, side `.scard`), `lineChart` (bottom), `tick`/`onScreen`/`flash`, search (180 ms debounce), and the watchlist (`localStorage` key `sm_watch`, wrapped in try/catch).
- Hash routes become query routes: `/?view=sector&name=Energy&range=today`, `/?view=stock&sym=RELIANCE`. Use `useSearchParams` inside a `<Suspense>` (required for static export).
- Polling:
  - `/api/tick?syms=` every 1000 ms (visible symbols only, as `onScreen`);
  - full `load(false)` every 30000 ms when not replaying and the modal is closed;
  - `/api/status` retry every 4000 ms while `!classified`.
- The stock view's "Open chart" uses `LabActions.openChart` (the shared modal from M6.2).

**Tests:** `COLORS(0)`, `COLORS(3)`, `COLORS(-3)` give the exact rgb strings; the search debounce is 180 ms; the horizon renders 20 bubbles from `horizon.json`; the stock view renders peers from `stock.json`.
**Commit:** `M-6.3: Sector Map UI`.

### M6.4 Swing Lab section (port of `lab.html`)
- Port `load`, `radar`, `profile`, `board`, `robust`, `draw` (the equity canvas with CASH bands and hover), plus the 30 s radar refresh with `.flash`.
- **Additions:**
  - a filter row `Source: all|Market Lab swing|Tier|151|Flow` and `Tier: all|F|C|B`;
  - a "Gate" column in the leaderboard;
  - a multiple-testing note: "N strategy×exit combinations shown; the best of N random strategies would show Sharpe ≈ X by luck (expected_max_sharpe_null)", using `/api/lab/matrix`'s field or computed client-side with the same formula.
- Rows call `LabActions.labShow(sym, strat, "1d")`.

**Tests:** the leaderboard sorts by excess; the radar lists render from `lab_swing.json`; the CASH band geometry for a regime array `[1,1,0,0,1]` gives one band from index 2 to 4.
**Commit:** `M-6.4: Swing Lab UI`.

### M6.5 Money Flow page `/flow`
**Layout** (new design in Market Lab's sector palette; `className="ml-sector"`):
1. **Market strip** (cards in a grid of 6):
   - FII and DII cash today and 5d;
   - participant index-futures long ratios with 60-day sparklines (SVG polyline);
   - NIFTY and BANKNIFTY gamma cards: regime badge `POSITIVE GAMMA · range / mean-reverting` or `NEGATIVE GAMMA · trending / volatile`, spot, zero-gamma (+ distance %), call wall, put wall, max pain, PCR;
   - VIX, breadth.
2. **Pre-open board:**
   - 09:00–09:15: a live table (IEP gap %, imbalance bar, total qty), sorted by |imb|;
   - after 09:15: "pre-open vs first 15 minutes", with a follow-through tick per row.
3. **Flow table:** every F&O stock, with columns sym, sector, ltp, %chg, disp, rvol, OI quad chip (green long buildup, amber short covering, orange long unwinding, red short buildup), OI Δ%, delivery z (prev), depth imbalance, buy/sell ratio, CVD slope, flow score.
   - Sortable; filters for quad and sector; a row click opens `ChartModal`.
4. **Sector flow grid:** a tile per sector, with area ∝ turnover share, colour by `flow_dir` (sector `COLORS` ramp with the value ×3), a rotation arrow ↑ in / ↓ out, and counts of quads.
5. **OI quadrant scatter** (SVG): x = price Δ%, y = OI Δ%, one dot per F&O stock, quadrant labels, hover name.
6. **Deals list** (last 5 days).

- A footer line on every card: "Descriptive: shows where money moved; whether it predicts returns is tested on the Strategies page (m-flow)."
- Polling: market 30 s; stocks 15 s; pre-open 15 s inside 09:00–09:15; sectors 30 s; deals 300 s.

**Tests:** quad chip colours; scatter axes for a fixture; pre-open phase switching by the fixture `phase`.
**Commit:** `M-6.5: Money Flow page`.

### M6.6 Options positioning page `/flow/options?und=NIFTY`
- A strike ladder, as horizontal mirrored bars (SVG): call OI to the right, put OI to the left, ΔOI overlay as thin bars, spot line, max pain, walls, zero-gamma marker.
- A per-strike GEX bar chart and the cumulative GEX line (SVG).
- The IV smile (call_iv/put_iv by strike).
- An expiry selector.
- History: total GEX and PCR (lightweight-charts line series, 2024-07→).

**Tests:** bar scaling; the zero-gamma marker position.
**Commit:** `M-6.6: options positioning page`.

### M6.7 Order-flow workbench `/of?sym=RELIANCE&date=YYYY-MM-DD`
**Panels:**
1. A candle chart (lightweight-charts; 1m/5m toggle) with VWAP and ±1/2σ bands, plus an anchored VWAP (click a candle to anchor; `/api/of/bars?anchored=<ts>`).
2. A volume profile histogram docked right (canvas), with POC/VAH/VAL lines.
3. TPO view: a canvas grid of letters per price bin, IB bracket, single prints shaded.
4. A CVD pane (lightweight-charts second chart; BVC default, CLV toggle) with divergence markers.
5. A footprint view (canvas): columns = bars, rows = price bins, cell text `sell×buy`, background by delta (green/red alpha by |delta|/max), stacked imbalances outlined, POC per bar boxed.
6. A big trades / absorption list: time, side, volume multiple, reaction +5/+15/+30.
7. A heatmap (canvas; x time, y price, intensity = qty; viridis-like ramp with 5 fixed stops), or the `reason` text.
8. A timescales table and the day-type / open-type badges.

- Every panel title carries its `method` label, e.g. "Delta (BVC estimate from 1-minute bars)".
- Each panel has an "ⓘ" link to `/learn?topic=<slug>`.

**Tests:** the footprint cell colour function; profile scaling; the TPO letter grid for a fixture.
**Commit:** `M-6.7: order-flow workbench`.

### M6.8 Strategies page `/strategies`
1. **Library table** (`/api/lab/strategies`): name, source, tier, kind, timeframe, gate verdict (lines passed, net Sharpe at ₹5L, 2×, trades, positive years), a link to the eval JSON (opened via `/api/lab/strategies?key=` detail), and a filter.
2. **"Tier list vs evidence":** for F/C/B, the median net Sharpe and the count passing the gate, with a bar per tier. Text: "Claimed order F < C < B; measured order …".
3. **Per-stock matrix:** a symbol input with search; then `/api/lab/matrix?sym=` as a heat table (strategies × exits, cell = avg net R, colour ramp), with the multiple-testing banner.

**Commit:** `M-6.8: strategies page`.

### M6.9 Learn pages `/learn` and `/learn?topic=<slug>`
- Render `content/learn/*.md` with `react-markdown` + `remark-gfm` (exact versions). The index comes from `content/learn/index.json`.
- Every dashboard "ⓘ" links here.

**Test:** a fixture markdown renders its headings and its table.
**Commit:** `M-6.9: learn pages`.

---

## PHASE M7 — Integration, parity, live check

### M7.1 Integration smoke
- `tests/integration/test_web_served.py`, marked `integration` and skipped if `web/out/index.html` is missing:
  - `create_app(settings, start_feeds=False, store=<store seeded from fixtures>, web_out=REPO/"web"/"out")`;
  - GET `/`, `/flow/`, `/flow/options/`, `/of/`, `/strategies/`, `/learn/` → 200 HTML;
  - GET `/api/sm/live` → JSON with `stocks`.
- A clean-clone build check, run by the orchestrator:
  ```powershell
  git clone . $env:TEMP\ss-clean
  cd $env:TEMP\ss-clean
  cmd /c mklink /J .venv C:\Users\harsh\Desktop\HARSH\WEB-DEV-PROJECTS\stocksense\.venv
  npm --prefix web ci
  npm --prefix web run build
  ```
  `web\out\index.html` must exist. Delete the temp folder after.

**Commit:** `M-7.1: integration smoke`.

### M7.2 Parity with Market Lab (user-started, market hours)
1. The orchestrator asks the user to start Market Lab (`run.bat` in the clone) and to say "go".
2. The orchestrator starts `stocksense serve` in the background.
3. Run `scripts/m_parity.py`.
4. With the Chrome tools, take screenshots of `http://localhost:8810` and `http://127.0.0.1:8765` at the same moment: the Smart Money stage, the Sector Map horizon, Swing Lab, and a chart modal with trades.
5. Record the table and the visual differences in `docs/RESULTS-M.md` section 1. Fix any visual mismatch in a follow-up card `M-7.2a: visual parity fixes`.
6. Stop our server (`taskkill /F /T /PID`). Ask the user to close Market Lab.

### M7.3 Five-session live dry run (user-started)
On each of 5 sessions, the user runs `run-lab.bat` before 09:00. The orchestrator checks:
- the pre-open board fills from 09:00:30;
- the tape updates about every 4 s;
- every feed's `last_ok` is under 2 minutes in `/api/status`;
- RSS < 800 MB;
- after 15:45 the catch-up runs `panel` and `scan`;
- the next morning, the radar shows the previous session's signals and `lab_journal` has the planned rows.

One line per session goes in RESULTS-M section 1.

---

## PHASE M8 — Gated evaluation and the honest write-up

### M8.1 Run the library (one family at a time, overnight; server stopped)
Free RAM must be ≥ 4 GB with no other heavy job. In this order, each in the background:
```bash
timeout 21600 .venv/Scripts/python.exe -m stocksense research strat-run --family m-swing --max-hours 6 > data_store/logs/m_M8_m-swing.log 2>&1
```
Then `m-xs`, `m-index`, `m-intra`. Each is resumable (re-run the same command).

### M8.2 Does following the money pay? (`m-flow-composite`, family m-xs)
1. The historical flow score at 10:15 for the u200 F&O names, from the components computable in history:
   - signed OI quad of t−1 EOD;
   - delivery z of t−1;
   - disp at 10:15 = (close of the 10:10 bar / open − 1) / range_pct;
   - rvol at 10:15 (5m history);
   - BVC CVD slope 09:15–10:15.
   Each component is z-scored by date, clipped at ±3, and averaged.
2. A new intraday label `y_1015_close = close(15:25 bar) / close(10:10 bar) − 1` from 5m bars. Add it as a function in `flow/composite.py`; it is **not** in labels v1. The label end is the same session.
3. Register the trial **before** computing. Compute:
   - IC vs `y_1015_close` and vs `y_c2o_10`;
   - orth_ic vs the GBM;
   - the k-slot intraday simulation: long the top 10 at 10:20, flat at 15:15, intraday costs.
4. Gate v2.
5. RESULTS-M section 3: "Is following the money profitable after costs?" Record whatever it shows.

### M8.3 `docs/RESULTS-M.md`
Sections:
1. Parity and live dry run.
2. The strategy table: every strategy, family, tier, trades, win %, avg net R, net Sharpe at ₹1L and ₹5L, 2× slip Sharpe, max DD, positive years, gate lines passed, verdict.
3. The flow composite.
4. "Does the tier list hold on NSE?": the median net Sharpe per tier, the Spearman between the claimed tier order (F = 0, C = 1, B = 2) and the measured Sharpe, and the gate passes per tier.
5. *151 Trading Strategies*: implemented, worked, failed, infeasible.
6. Caveats:
   - intraday history is about 12 months;
   - the total trial count (from `n_trials`);
   - survivorship handled through PIT universes;
   - costs;
   - the lockbox stays closed.

Every number must be traceable to `reports/m/strategies/*/eval.json` and the registry.

### M8.4 Verdicts in the UI
- `/api/lab/strategies` and `/api/lab/trades` read `eval.json` into `gated`: `{verdict, lines_passed, net_sharpe_5l, sharpe_2x, trial_id}`.
- A strategy passing all nine v2 lines shows `PASS (lockbox closed)`.
- Opening the lockbox follows PLAN-G2 D12, for one pre-chosen candidate, only after the user approves.

**Commit:** `M-8.4: gated verdicts in the dashboard`.

---

# PART 7 — Order, parallelism, schedule

1. **Sequential start:**
   - M0.0 → M0.1 → M0.2. The IPCA run takes hours in the background. While it runs, **only docs cards** may proceed: M0.3 and M1.1.
   - Then M0.2 finishes (eval, docs, merge).
2. **Then at most two executors at once:**
   - **Lane A (backend):** M2.0 → M2.1 → M2.2 → M2.3 → M2.4 → M5.1 → M5.2 → M2.5 → M5.6 → M2.6 → M2.7 → M3.1 → M3.2 → M3.3 → M3.4 → M3.5 → M3.6 → M3.7 → M4.1 … M4.8 → M5.3 → M5.4 → M5.5 → **M5.0** (pre-registration; after all keys exist and before any `strat-run`) → M5.7 → M5.8 → M5.9 → M2.8. (M5.4 is the largest card. Split it into M5.4a (stocks §3 + App. A + riskmodel), M5.4b (sector/macro §4, 6, 7.4–7.5, 9, 10, 15, 19), M5.4c (options §2 + dispersion) and M5.4d (§8.1 HP, §18.2 ANN, §18.3 news storage). One commit each: `M-5.4a: …` and so on.)
   - **Lane B (frontend):** M6.0 (after M2.0) → M6.1 → M6.2 (after M2.3 fixtures) → M6.3 (after M2.4) → M6.4 (after M2.6) → M6.5 (after M3.7) → M6.6 (after M3.4/M3.7) → M6.7 (after M4.8) → M6.8 (after M2.6; verdicts later) → M6.9 (after M1.1).
   - Lane B never edits `src/`, and Lane A never edits `web/`. Shared docs conflicts: keep both sides.
3. **Then:** M7.1 → M2.9 and M7.2 (user) → M8.1 (nights; one family per night; server stopped) → M8.2 → M8.3 → M8.4 → M7.3 (user, 5 sessions).
4. **After Program M:** resume G2:
   - D1 MASTER (needs ≥ 6 GB free; server stopped), D2–D6, D8–D10, D12;
   - E8 (now fed by `depth_snaps` and the flow datasets), E9, F1;
   - A9 (1m to 2022) on a quiet night. It increases intraday power; re-running the intraday families after it creates **new** trials (`strategy_library_<date>.toml`).
5. **Orchestrator duties after every card:**
   - re-run the acceptance command and the full suite (and web lint/test/build) on main;
   - check processes and RAM;
   - tick Part 10 of `docs/PLAN-M.md`;
   - give the user a 3-line summary.

---

# PART 8 — Failure table (Program M)

| Problem | Where | Built-in response |
|---|---|---|
| NSE website JSON 401/403 after warm-ups | M2.1+, M3 | `NseClient` raises `DataNotAvailable`. The feed marks itself `blocked` for 300 s. The tape falls back to broker quotes (`QuoteHub`) for ltp/open/prev when Angel secrets exist. Record the incident in DECISIONS |
| NextApi path changed | M2.1 | Legacy path fallback (`/api/equity-stockIndices`). Record it |
| Yahoo 429 / format change | tape, charts | 120 s back-off. ltp from NSE market-watch. Charts fall back to stored 5m + Angel candles |
| Golden mismatch vs Market Lab | M2.2–M5.2 | Find the line. If it is a deliberate fix, document it and override only that field in the golden comparison, with a test pinning our behaviour. Never loosen tolerances globally |
| A pandas behaviour that numpy can't match exactly | npx | Write a targeted golden for that call, implement the exact semantics, and document them |
| Next.js build breaks | M6 | Exact pins and `npm ci`. Never upgrade mid-program |
| RAM pressure | any | Stop the server during M8 and G2 jobs. Catch-up defers on low memory |
| Option OI units / live chain unit mismatch | M3.4 | Probe and record. Tests pin the conversion |
| An index has no 5m bars | M5.3/M5.5 | Strategy `status="await_data"` in the toml (not run, listed as such in RESULTS) |
| A strategy cannot be implemented faithfully | M5.3–M5.5 | Closest faithful version with a documented difference, or infeasible with the reason. Never silently change rules |
| PBO family < 10 trials | M5.0 | The test enforces ≥ 10 per family at registration |
| Feed thread exception | server | Top-of-loop catch, log, count. `/api/status` shows it |
| Market Lab clone venv missing | M2.2 | Stop and report. Do not install into it |
| Usage limit / agent dies mid-card | any | Long jobs are resumable. The orchestrator resumes the agent with SendMessage. Never start a second copy of a running job |

---

# PART 9 — Verification (how anyone can tell Program M worked)

1. Python suite green, including:
   - `test_no_orders` (also scanning `web/src`);
   - `test_no_pandas_outside_kronos`, `test_no_naked_now`;
   - `test_gate_frozen`, `test_strategy_library` SHA, `test_feature_truncation`;
   - the API 405 tests and the golden Market Lab tests.

   Web green: `npm --prefix web run lint`, `npm --prefix web run test -- --run`, `npm --prefix web run build`.
2. `run-lab.bat` from a clean clone opens `http://127.0.0.1:8765`. In market hours:
   - the tape updates about every 4 s;
   - `/api/status` shows every feed `last_ok` < 2 min;
   - pre-open fills from 09:00.
3. Parity: `scripts/m_parity.py` meets the thresholds (Jaccard ≥ 0.85, Spearman ≥ 0.98, chip MAD ≤ 0.05 pp, tau ≥ 0.9), or the residuals are explained. The screenshots match the layout.
4. `/flow` shows OI buildup, delivery, participants, FII/DII, options GEX/PCR/walls, pre-open, sector flow and deals, each with its as-of date. `/of?sym=RELIANCE` shows VWAP, the profiles, CVD, footprint, big trades, absorption, timescales, and the heatmap or its reason.
5. Every strategy in `config/strategy_library.toml` has an `eval.json` and a RESULTS-M row, and the trial count is stated. Clicking any stock lets the user run any strategy × exit as an exploratory backtest net of costs, with its gated verdict shown beside it.
6. `docs/learn/` has every note, and the 151 map has ≥ 140 rows.
7. `schtasks /query | findstr /i stocksense` is empty. No order code exists. The server binds 127.0.0.1 only.

---

# PART 10 — Status checklist (the orchestrator ticks it in `docs/PLAN-M.md`)

**Execution log (2026-11-23; update as cards complete):**
- **M0.0 done.**
  - `.vscode/settings.json` unstaged; `xyz.txt` and `.vscode/` added to `.git/info/exclude`.
  - Worktree `stocksense-wt-g2d1` and branch `g2/d1` removed; `data_store` verified intact.
  - Market Lab's own `app.py` (PIDs 25116, 32436, port 8810) was running. It is the user's process and was left alone.
  - Free RAM was only about 2.9 GB.
- **M0.1 partly done.**
  - H5 finished in worktree `stocksense-wt-g2h`: `tests/unit/test_h5_mcp_verify.py` 12 passed, ruff clean, committed as `48212fd` "G2-H5: nightly NSE MCP verification stage".
  - On main: `git merge --no-ff g2/h` made merge commit `4427fd9`. **Local only, not pushed.**
  - **Remaining for M0.1:**
    1. **DONE 2026-11-23:** the full suite on main finished with `done rc=0` (scratch_main.log). The original instruction was: confirm the full suite on main is green. The agent's background run logs to `C:\Users\harsh\Desktop\HARSH\WEB-DEV-PROJECTS\scratch_main.log` and writes `done rc=N` at the end. Check that no stray python of that run is still alive before starting anything else.
    2. `ruff check src tests` on main.
    3. `git push origin main`. A 403 is likely: an earlier push failed because the stored GitHub credential belongs to account `gursimrannkaurr`. Do not work around it; report it and ask the user to fix the credential.
    4. Tick G2 H1, H4 and H5 in `docs/PLAN-G2.md` Part 9, then commit `docs: tick G2 H1 H4 H5`.
    5. Remove worktree `stocksense-wt-g2h`: `cmd /c rmdir` its `.venv` and `data_store` links first, then `git worktree remove --force`, then verify `Test-Path data_store\parquet\bhavcopy_eq`.
  - **Note:** inside a worktree the `.venv` junction resolves `stocksense` to the MAIN repo's `src`. To test worktree code, set `PYTHONPATH=<worktree>\src`.
  - **Note:** `pytest -q` twice suppresses the summary line (pyproject `addopts` already has `-q`). Use plain `pytest tests` when a "N passed" line is needed.

- [x] M0.0 housekeeping · [x] M0.1 G2-H merged and pushed · [ ] M0.2 G2-D7 merged (waiting for 6 GB free RAM; 4.2 GB on 2026-11-23) · [x] M0.3 plan/sources/licence
- [x] M1.1 learning notes (19 files; 151 map covers all 176 sections)
- [x] M2.0 API skeleton · [x] M2.1 sources/clock/store/npx · [x] M2.2 structure/smc golden · [x] M2.3 tape (suite green) · [x] M2.4 sectors · [x] M2.5 research note · [ ] M2.6 lab endpoints · [ ] M2.7 one-command start · [ ] M2.8 parity script · [ ] M2.9 resources
- [ ] M3.1 OI + NFO · [ ] M3.2 delivery/deals · [ ] M3.3 participants · [ ] M3.4 options/GEX · [ ] M3.5 pre-open · [ ] M3.6 sector flow · [ ] M3.7 composite + endpoints
- [ ] M4.1 vwap · [ ] M4.2 profile/TPO · [ ] M4.3 delta/CVD · [ ] M4.4 footprint · [ ] M4.5 big trades/absorption · [ ] M4.6 heatmap · [ ] M4.7 timescales · [ ] M4.8 endpoints
- [ ] M5.0 pre-registration · [x] M5.1 indicators/exits/base · [ ] M5.2 Market Lab strategies · [ ] M5.3 tiers · [ ] M5.4 151 · [ ] M5.5 flow strategies · [ ] M5.6 explore/scan/profile · [ ] M5.7 gated runner · [ ] M5.8 journal · [ ] M5.9 daily picks
- [ ] M6.0 scaffold · [ ] M6.1 shell · [ ] M6.2 Smart Money · [ ] M6.3 Sector Map · [ ] M6.4 Swing Lab · [ ] M6.5 Money Flow · [ ] M6.6 options · [ ] M6.7 order flow · [ ] M6.8 strategies · [ ] M6.9 learn
- [ ] M7.1 smoke · [ ] M7.2 parity (user) · [ ] M7.3 five sessions: ____ ____ ____ ____ ____
- [ ] M8.1 library runs · [ ] M8.2 flow composite · [ ] M8.3 RESULTS-M · [ ] M8.4 verdicts in UI

---

# PART 11 — Sources

- **Market Lab:** `github.com/heysanjeevv/market-lab` (MIT), commit `ba3f771`.
- **Kakushadze, Z. & Serur, J. A. (2018).** *151 Trading Strategies*. Palgrave Macmillan; SSRN 3247865; arXiv 1912.04492.
- **Order flow:**
  - Easley, López de Prado & O'Hara (2012), RFS 25(5) (BVC; VPIN);
  - Lee & Ready (1991), J. Finance 46(2);
  - Cont, Kukanov & Stoikov (2014), J. Financial Econometrics 12(1);
  - Harris (2003), *Trading and Exchanges*;
  - Bouchaud, Bonart, Donier & Gould (2018), *Trades, Quotes and Prices*.
- **Auction market theory:** Dalton, Jones & Dalton (2013), *Mind Over Markets*; Steidlmayer & Koy (1986), *Markets and Market Logic*.
- **Gamma exposure:** Barbon & Buraschi (2021), "Gamma Fragility" (SSRN 3725454); Ni, Pearson, Poteshman & White (2021), RFS 34(4).
- **Reversal and momentum:**
  - Jegadeesh (1990), J. Finance 45(3);
  - Lehmann (1990), QJE 105(1);
  - Jegadeesh & Titman (1993);
  - Blitz, Huij & Martens (2011), residual momentum;
  - Avellaneda & Lee (2010).
- **Multiple testing:** Bailey & López de Prado (2014), JPM 40(5); Bailey, Borwein, López de Prado & Zhu (2017).
- **NSE:** pre-open call auction circulars; block-deal window circulars; UDiFF F&O bhavcopy format. The M1.1 executor cites the exact circular numbers.
