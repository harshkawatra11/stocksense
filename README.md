# StockSense

Autonomous NSE swing-trading intelligence system. StockSense runs a multi-layer signal engine every 30 minutes during NSE market hours — combining a LightGBM model, a Kronos time-series forecaster, a Qwen2.5 macro/news layer, and a Claude synthesis pass — and presents actionable BUY signals as:

```
BUY RELIANCE @ ₹1291 → ₹1352 (+4.7%) in ~2.5 days  [conf: 83%]
```

Signals flow through a full lifecycle: **SUGGESTED → RATED → BOUGHT → REANALYZED** — with capital tracking, position re-analysis against original forecasts, and EOD Claude reviews that write learnings back into the next cycle.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Signal Pipeline](#signal-pipeline)
3. [Directory Structure](#directory-structure)
4. [Database Schema](#database-schema)
5. [API Reference](#api-reference)
6. [Frontend](#frontend)
7. [Configuration](#configuration)
8. [Prerequisites](#prerequisites)
9. [Running Locally](#running-locally)
10. [Deployment](#deployment)
11. [Technical Notes](#technical-notes)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  Angel One getCandleData  ──►  TimescaleDB (ohlcv_daily)        │
│  NSE Bhavcopy (fallback)  ──►  2,300+ NSE-EQ tickers           │
│  Angel One LTP (live)     ──►  Nifty 50 / Sensex index feed    │
│  Moneycontrol + ET RSS    ──►  macro_context (30-min cache)     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                      SIGNAL ENGINE                              │
│                                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │ LightGBM    │   │   Kronos     │   │  Qwen2.5 Macro Layer │ │
│  │ 40+ features│ + │ 1D–5D paths  │ + │  14-sector sentiment │ │
│  │ SHAP reason │   │ ETA from path│   │  ±0.10 conf nudge    │ │
│  └──────┬──────┘   └──────┬───────┘   └──────────┬───────────┘ │
│         └────────┬─────────┘                      │             │
│                  ▼                                 │             │
│         Weighted Combine ◄─────────────────────────┘            │
│                  │                                               │
│          Confidence Gate (≥0.55)                                │
│                  │                                               │
│         BUY-only filter · Affordability · ATR stops             │
│                  │                                               │
│         Claude Sonnet Synthesis (top-N + learnings)             │
└──────────────────┬──────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────┐
│                    ACTIVITY LIFECYCLE                           │
│  SUGGESTED → RATED (like/dislike) → BOUGHT → REANALYZED        │
│  Capital ledger · Position re-analysis · EOD Claude review      │
└─────────────────────────────────────────────────────────────────┘
```

### Component Stack

| Component | Technology | Role |
|---|---|---|
| Database | TimescaleDB (PostgreSQL 16) | OHLCV hypertables, signals, portfolio, learnings |
| Backend | Python 3.11 + FastAPI | REST API + SSE streaming, 35+ endpoints |
| Signal Engine | LightGBM + Kronos + Ollama + Claude CLI | 4-layer inference pipeline |
| Scheduler | APScheduler | Every 30 min during market hours + EOD jobs |
| Frontend | React 18 + Vite + Tailwind CSS | 8-tab dashboard with real-time SSE terminals |
| Data Feed | Angel One SmartAPI | Primary OHLCV + live LTP; NSE Bhavcopy fallback |
| Macro Layer | Qwen2.5-7B via Ollama | News RSS → 14-sector sentiment scores |
| Synthesis | Claude Sonnet (Claude Code CLI) | Final signal ranking + learnings injection |

---

## Signal Pipeline

### Multi-Timeframe Flow

`run_pipeline_multi()` fires every 30 minutes, processes up to 200 active tickers concurrently (semaphore=8), and emits one signal per active timeframe per ticker.

```
For each ticker:
  1. Load 300 daily OHLCV candles from TimescaleDB
  2. Compute 40+ features (RSI, MACD, Bollinger, SMAs, F&O OI/PCR)

  3. LightGBM inference  →  (signal, confidence, top-5 SHAP features)

  4. Kronos forecast     →  (forecast path [1..N closes])
     For each timeframe (1D / 2D / 3D / 4D / 5D):
       - Extract peak close from forecast path as target price
       - Interpolate fractional day when path first crosses target (e.g. 2.5d)

  5. Weighted combine    →  avg(ml_conf, kronos_conf) + macro nudge
     - Macro nudge: sector_score ∈ [-1,+1] × 0.10 applied to confidence
     - Gate: drop if final_confidence < 0.55

  6. Per-signal annotations:
     - target_eta_days    (float, e.g. 2.5)
     - expected_move_pct  (float, e.g. +4.7)
     - predicted_path     (JSON array of N closes)
     - affordable         (bool: deployable_cash / price ≥ 1 share)
     - shares_affordable  (int)
     - horizon_stops      (ATR × 1.5 × √hold_days, clamped 0.5%–8%)
     - macro_sector_score (float ∈ [-1,+1])

7. Claude Synthesis (once per cycle, top-N signals):
   - Best signal per ticker + recent DB learnings injected into prompt
   - Claude adjusts final_confidence, can demote signals to HOLD/REJECT
   - Updates signals.claude_confidence + signals.final_confidence in DB

8. Persist to signals table + log SUGGESTED event to activity_log
```

### EOD Learning Loop

At 15:45 IST, `task_eod_review()` runs Claude on all signals from the day, compares predicted vs. actual closes, and writes structured learnings to the `learnings` table. These learnings feed the next day's Claude synthesis prompt — a compounding feedback loop that improves signal quality over time.

### Streaming Pipeline (Intelligence Dashboard)

`run_single_ticker_streaming()` yields SSE events per model stage for real-time terminal rendering in the frontend:

```
ml_result  →  kronos_result  →  slm_result  →  claude_enriched
```

---

## Directory Structure

```
stocksense/
├── backend/
│   ├── main.py                  # FastAPI app, CORS, router registration, SSE endpoint
│   └── routers/
│       ├── signals.py           # Recent/today signals, per-signal reasoning
│       ├── live.py              # Multi-timeframe signals, account, activity, positions
│       ├── portfolio.py         # Holdings, P&L
│       ├── accuracy.py          # Model accuracy stats + daily breakdown
│       ├── ohlcv.py             # Candle data endpoint with signal overlay
│       ├── market_data.py       # Live Nifty/Sensex LTP (15s cache, circuit breaker)
│       ├── market_overview.py   # Top movers, sector summary, 7-day stats
│       └── logs.py              # Learnings query, app.log tail by date
│
├── data/
│   ├── db/
│   │   ├── database.py          # SQLAlchemy ORM (9 table models)
│   │   ├── schema.sql           # TimescaleDB DDL (hypertables + indices)
│   │   ├── schema_v2_live.sql   # Adds account, decisions, signal annotation cols
│   │   └── schema_v3_intelligence.sql  # Adds target_eta_days, position_reviews, activity_log
│   └── pipeline/
│       ├── fetch_angel_daily.py # Primary: Angel One getCandleData bulk OHLCV
│       ├── fetch_historical.py  # Fallback: NSE Bhavcopy daily candles
│       ├── fetch_live.py        # Angel One intraday + LTP
│       ├── fetch_f_and_o.py     # NSE Futures & Options (OI, PCR, delivery %)
│       ├── feature_engineering.py  # 40+ technical indicators + F&O features
│       ├── nse_ticker_loader.py # ~200 active NSE-EQ tickers
│       └── sector_map.py        # Ticker → sector map (153 liquid names, 14 sectors)
│
├── intelligence/
│   ├── signal_pipeline.py       # Core: run_pipeline_multi, target_and_eta,
│   │                            #       horizon_stops, apply_macro_nudge
│   ├── macro_context.py         # RSS headlines → Ollama → sector scores (30-min cache)
│   ├── trading_account.py       # Capital ledger, record_decision, get_actionable_signals
│   ├── activity.py              # Audit trail (SUGGESTED/RATED/BOUGHT/SOLD/REANALYZED)
│   ├── position_monitor.py      # Re-analyze positions vs. original forecast (HOLD/EXIT)
│   ├── accuracy_tracker.py      # Rolling 7-day accuracy, per-model breakdown
│   ├── eod_review.py            # EOD Claude review → learnings table
│   ├── claude_cli.py            # Claude subprocess wrapper (stdin, UTF-8, Windows-safe)
│   ├── portfolio_guard.py       # Active position tracking + stop/target monitoring
│   ├── data_freshness.py        # Detect LIVE intraday vs. past EOD data
│   └── trading_mode.py          # PAPER vs. LIVE gate (accuracy threshold)
│
├── models/
│   ├── ml/
│   │   ├── predict.py           # LightGBM inference + SHAP top-5 reasoning
│   │   └── train.py             # Training pipeline (labels from resolved signals)
│   ├── kronos/
│   │   ├── integration.py       # Kronos wrapper: forecast() → price path + reasoning
│   │   ├── combine.py           # ML + Kronos weighted signal fusion
│   │   └── finetune_nse.py      # Fine-tune Kronos on NSE data
│   └── slm/
│       └── infer.py             # Ollama Qwen2.5 inference (graceful fallback)
│
├── scheduler/
│   ├── market_runner.py         # APScheduler daemon — all 9 scheduled jobs
│   └── weekend_job.py           # Off-market deep review + cleanup
│
├── frontend/
│   └── src/
│       ├── App.tsx              # Tab router (8 views)
│       ├── types.ts             # LiveSignal, Account, ActivityEvent, PositionReview
│       ├── hooks/
│       │   ├── useMarketIndices.ts   # Polls /api/market/indices (30s open, 2m closed)
│       │   ├── useMarketStatus.ts    # IST market-hours detection
│       │   └── useBackendHealth.ts   # Backend liveness check
│       └── components/
│           ├── Layout.tsx            # Nav bar + market status badge + index ticker
│           ├── live/Live.tsx         # Signal cards + activity feed + account
│           ├── intelligence/
│           │   ├── IntelligenceDashboard.tsx
│           │   ├── SignalFeed.tsx / SignalCard.tsx / ReasoningDrawer.tsx
│           │   ├── MultiTerminalGrid.tsx
│           │   └── terminals/  (MLTerminal, KronosTerminal, SLMTerminal, ClaudeTerminal)
│           ├── market/MarketOverview.tsx
│           ├── portfolio/Portfolio.tsx
│           └── logs/LogsPanel.tsx
│
├── research/
│   └── edge_by_year.py          # LightGBM edge validation by year (OOS 2025-26)
│
├── config.py                    # Central settings (env-var backed, Pydantic)
├── requirements.txt
├── docker-compose.yml
└── start.ps1                    # One-command Windows startup
```

---

## Database Schema

All time-series tables are TimescaleDB hypertables partitioned by `time`.

### `stocks`
```sql
ticker      TEXT PRIMARY KEY
name        TEXT
sector      TEXT
industry    TEXT
exchange    TEXT DEFAULT 'NSE'
active      BOOLEAN DEFAULT TRUE
```

### `ohlcv_daily` *(hypertable)*
```sql
time        TIMESTAMPTZ NOT NULL   -- UTC midnight = IST trading day
ticker      TEXT NOT NULL
open        NUMERIC(12,2)
high        NUMERIC(12,2)
low         NUMERIC(12,2)
close       NUMERIC(12,2)
volume      BIGINT
adj_close   NUMERIC(12,2)
UNIQUE (time, ticker)
```

### `signals`
```sql
id                  SERIAL PRIMARY KEY
ticker              TEXT
signal_type         TEXT CHECK IN ('BUY','SELL','HOLD')
timeframe           TEXT                  -- '1D','2D','3D','4D','5D'
price_at_signal     NUMERIC(12,2)
target_price        NUMERIC(12,2)
stop_loss           NUMERIC(12,2)
ml_confidence       NUMERIC(5,4)
kronos_confidence   NUMERIC(5,4)
slm_confidence      NUMERIC(5,4)
claude_confidence   NUMERIC(5,4)
final_confidence    NUMERIC(5,4)
status              TEXT DEFAULT 'ACTIVE' -- ACTIVE, CLOSED_WIN, CLOSED_LOSS, EXPIRED
fired_at            TIMESTAMPTZ DEFAULT now()
resolved_at         TIMESTAMPTZ
actual_close        NUMERIC(12,2)
horizon_days        INTEGER
affordable          BOOLEAN
shares_affordable   INTEGER
macro_sector_score  NUMERIC(5,4)
target_eta_days     NUMERIC(6,2)          -- fractional days to target (e.g. 2.5)
expected_move_pct   NUMERIC(7,4)          -- % upside to target
predicted_path      JSONB                 -- array of N forecast closes
```

### `activity_log`
```sql
id          SERIAL PRIMARY KEY
event_type  TEXT   -- SUGGESTED, RATED, BOUGHT, SOLD, REANALYZED, NOTE
ticker      TEXT
signal_id   INTEGER REFERENCES signals(id)
detail      JSONB
created_at  TIMESTAMPTZ DEFAULT now()
```

### `account`
```sql
id              SERIAL PRIMARY KEY
cash_available  NUMERIC(12,2)
cash_reserve    NUMERIC(12,2)
note            TEXT
updated_at      TIMESTAMPTZ DEFAULT now()
```

### `decisions`
```sql
id          SERIAL PRIMARY KEY
action      TEXT    -- BUY, SELL, PASS, SKIP
ticker      TEXT
signal_id   INTEGER REFERENCES signals(id)
quantity    INTEGER
price       NUMERIC(12,2)
cash_after  NUMERIC(12,2)
rationale   TEXT
decided_at  TIMESTAMPTZ DEFAULT now()
```

### `position_reviews`
```sql
id              SERIAL PRIMARY KEY
signal_id       INTEGER REFERENCES signals(id)
ticker          TEXT
review_date     DATE
days_elapsed    INTEGER
eta_days        NUMERIC(6,2)
progress_pct    NUMERIC(7,2)    -- % of expected move achieved
status          TEXT            -- on_track, ahead, behind, expired
verdict         TEXT            -- HOLD, EXIT
reviewed_at     TIMESTAMPTZ DEFAULT now()
```

### `learnings`
```sql
id                SERIAL PRIMARY KEY
learning_date     DATE
learning_type     TEXT    -- eod_review, pattern_success, pattern_failure
ticker            TEXT
signal_id         INTEGER REFERENCES signals(id)
title             TEXT
body              TEXT
tags              TEXT[]
raw_claude_output TEXT
created_at        TIMESTAMPTZ DEFAULT now()
```

### `portfolio`
```sql
id          SERIAL PRIMARY KEY
ticker      TEXT
quantity    INTEGER
avg_price   NUMERIC(12,2)
buy_date    DATE
active      BOOLEAN DEFAULT TRUE
notes       TEXT
```

---

## API Reference

Base URL: `http://localhost:8000`

### Health
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | `{"status":"ok","service":"StockSense"}` |

### Live Signals
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/live/signals` | Actionable BUY signals, affordable-first. Query: `limit`, `only_affordable` |
| GET | `/api/live/account` | Capital state (cash_available, cash_reserve) |
| GET | `/api/live/activity` | Activity feed (SUGGESTED, RATED, BOUGHT, SOLD, REANALYZED) |
| GET | `/api/live/decisions` | Decision ledger (BUY/PASS/SKIP history) |
| GET | `/api/live/positions/reviews` | Position re-analysis (days_elapsed, ETA, progress_pct, HOLD/EXIT) |
| POST | `/api/live/rate` | Rate a signal `{signal_id, rating: "like"/"dislike", reason}` |
| POST | `/api/live/buy` | Record a buy `{signal_id, ticker, quantity, price, rationale}` |
| POST | `/api/live/pass` | Pass on a signal `{ticker, reason}` |
| POST | `/api/live/run` | Trigger an immediate pipeline run |

### Signals
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/signals/recent` | Recent signals (limit ≤200) |
| GET | `/api/signals/today` | Signals fired today, sorted by confidence |
| GET | `/api/signals/{id}/reasoning` | Per-model reasoning for a signal |

### Portfolio
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/portfolio/` | All active positions |
| POST | `/api/portfolio/add` | Add position `{ticker, quantity, avg_price, buy_date?, notes?}` |
| DELETE | `/api/portfolio/{ticker}` | Soft-delete position |
| GET | `/api/portfolio/pnl` | P&L summary (total invested, current value, pnl %) |

### Market
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/market/indices` | Nifty 50 + Sensex LTP (15s cache, graceful fallback) |
| GET | `/api/market/overview` | Top BUY/SELL signals (7d), sector heat map, top movers |

### OHLCV
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ohlcv/{ticker}` | Daily candles. Query: `days` (default 365, max 3650) |
| GET | `/api/ohlcv/{ticker}/signals` | Recent signals for a ticker (chart overlay) |

### Accuracy
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/accuracy/summary` | Overall accuracy (resolved signals, correct %, avg conf) |
| GET | `/api/accuracy/by-model` | Per-model accuracy breakdown |
| GET | `/api/accuracy/daily` | Daily accuracy for last 30 days |

### Logs
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/logs/learnings` | Learnings table (limit ≤200, optional `ticker` filter) |
| GET | `/api/logs/learnings/today` | Today's learnings |
| GET | `/api/logs/files` | App.log tail by date + available log dates |

### Streaming (SSE)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/stream/signals` | Per-stage SSE events as each ticker processes |

SSE event types: `ml_result`, `kronos_result`, `slm_result`, `claude_checking`, `claude_enriched`, `batch_complete`

---

## Frontend

Eight tabs accessible from the persistent navigation bar:

| Tab | Key Features |
|-----|-------------|
| **Live Signals** | Multi-timeframe BUY cards (₹X→₹Y, +Z%, ~Nd), Like/Dislike/Buy/Pass, activity feed, account balance |
| **Intelligence** | 4-terminal live SSE view (ML → Kronos → Qwen2.5 → Claude), signal feed, reasoning drawer, learnings panel |
| **Market Overview** | Top BUY/SELL signals (7d), sector sentiment heat map, top gainers/losers |
| **Portfolio** | Active positions, per-position P&L, total portfolio performance |
| **Charts** | OHLCV candlestick chart with signal overlays per ticker |
| **Watchlist** | Custom ticker tracking |
| **Logs** | Real-time app.log tail, learnings audit trail |
| **Orders** | Reserved for live order execution |

The header shows a market status badge (OPEN / CLOSED / PRE-MARKET), live Nifty 50 + Sensex LTP with % change, and a backend health indicator.

---

## Configuration

All settings are in `config.py`, backed by a `.env` file (gitignored).

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | Async PostgreSQL DSN |
| `DATABASE_DSN` | `postgresql://...` | Sync DSN for asyncpg |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `SLM_MODEL` | `qwen2.5:7b` | Ollama model for macro layer |
| `CONFIDENCE_THRESHOLD` | `0.55` | Minimum confidence to emit a signal |
| `MAX_TICKERS_PER_RUN` | `200` | Tickers processed per pipeline cycle |
| `TOP_SIGNALS_FOR_CLAUDE` | `40` | Signals passed to Claude synthesis per run |
| `BUY_ONLY` | `true` | Emit BUY signals only |
| `CASH_AVAILABLE` | — | Deployable capital (₹) |
| `CASH_RESERVE` | — | Reserved capital — never touched by the engine |
| `CLAUDE_SYNTHESIS_ENABLED` | `true` | Run Claude Sonnet synthesis pass |
| `ANGEL_ONE_API_KEY` | — | Angel One SmartAPI key |
| `ANGEL_ONE_CLIENT_ID` | — | Angel One client ID |
| `ANGEL_ONE_PIN` | — | Angel One PIN |
| `ANGEL_ONE_TOTP_KEY` | — | TOTP secret for OTP generation |

**Active timeframes:**

| Label | Kronos Steps | Hold Days |
|-------|-------------|-----------|
| 1D | 1 | 1 |
| 2D | 2 | 2 |
| 3D | 3 | 3 |
| 4D | 4 | 4 |
| 5D | 5 | 5 |

30m and 2h timeframes are defined but dormant until an intraday feed is connected.

---

## Prerequisites

### Required
- Python 3.11+
- Node.js 18+ and npm
- Docker Desktop (TimescaleDB)
- Angel One SmartAPI account

### Recommended
- **GPU** (4GB+ VRAM) — only if you run the macro layer on local Ollama, or fine-tune Kronos.

---

## Choose your engine

StockSense runs on **your** LLMs — no provider API key is wired in. On first launch the app
shows a setup screen to connect two layers; both degrade gracefully if absent.

**Synthesis layer** (final signal validation) — install and log in to **one** CLI:

| Engine | Install | Log in |
|--------|---------|--------|
| Claude Code | `npm i -g @anthropic-ai/claude-code` | run `claude`, then `/login` |
| Codex (OpenAI) | `npm i -g @openai/codex` | `codex login` |
| Gemini (Google) | `npm i -g @google/gemini-cli` | run `gemini`, then `/auth` |

**Macro / news layer** (sector sentiment) — pick one:
- **Local Ollama** (free, needs a GPU/decent CPU): `ollama pull qwen2.5:7b`
- **Ollama Cloud** (no local GPU): set `OLLAMA_API_KEY`, point `OLLAMA_URL` at the cloud base,
  and set `SLM_MODEL` to a hosted model such as Nemotron 3 Ultra.

The choice is saved in the `app_config` table; secrets stay in `.env` only. You can re-open the
setup screen from the app header to switch engines later.

---

## Running Locally

### 1. Clone and configure
```bash
git clone <repo>
cd stocksense
cp .env.example .env
# Fill in ANGEL_ONE_* and DATABASE_URL
pip install -r requirements.txt
```

### 2. Start everything (Windows)
```powershell
./start.ps1
```

This starts Docker + TimescaleDB, checks Ollama, then opens three windows: backend (:8000), scheduler, and frontend (:5173).

```powershell
./start.ps1 -NoFrontend    # API + scheduler only
./start.ps1 -NoScheduler   # backend + frontend only
```

### 3. Initialize the database
```bash
docker exec -i stocksense-db-1 psql -U postgres -d stocksense < data/db/schema.sql
docker exec -i stocksense-db-1 psql -U postgres -d stocksense < data/db/schema_v2_live.sql
docker exec -i stocksense-db-1 psql -U postgres -d stocksense < data/db/schema_v3_intelligence.sql
docker exec -i stocksense-db-1 psql -U postgres -d stocksense < data/db/schema_v4_brain.sql
docker exec -i stocksense-db-1 psql -U postgres -d stocksense < data/db/schema_v5_providers.sql
```

### 4. Backfill historical data
```bash
# Last 30 days via Angel One
python -m data.pipeline.fetch_angel_daily 30
```

### 5. Trigger a pipeline run
```bash
curl -X POST http://localhost:8000/api/live/run
```

### 6. Open the app
- **App:** http://localhost:5173 → Live Signals tab
- **API docs:** http://localhost:8000/docs

---

## Deployment

### Architecture

| Service | Platform | Notes |
|---------|----------|-------|
| TimescaleDB | Railway Postgres | Persistent, always-on |
| Backend + Scheduler | Railway | Daily OHLCV capture, signal pipeline 24/7 |
| Frontend | Vercel | Standard Vite build |
| Ollama (Qwen2.5) | Local (GPU) | Requires 4GB+ VRAM |
| Claude synthesis | Local (Claude Code CLI) | Requires logged-in session |

The backend degrades gracefully without Ollama and Claude — macro context returns NEUTRAL, synthesis is skipped, ML + Kronos signals still generate normally.

### Railway — Backend + Scheduler
```bash
# Required environment variables
DATABASE_URL=postgresql://...
ANGEL_ONE_API_KEY=...
ANGEL_ONE_CLIENT_ID=...
ANGEL_ONE_PIN=...
ANGEL_ONE_TOTP_KEY=...
CLAUDE_SYNTHESIS_ENABLED=false
```

### Vercel — Frontend
```bash
cd frontend && npm run build
# Set VITE_API_URL=https://your-railway-backend.railway.app
```

---

## Technical Notes

### Scheduler Jobs

| Job | Schedule | Description |
|-----|----------|-------------|
| `task_ticker_sync` | Daily 8:00 IST | Sync NSE ticker list to DB |
| `task_incremental_ohlcv` | Mon–Fri 18:30 IST | Angel One OHLCV (NSE Bhavcopy fallback) |
| `task_incremental_fo` | Mon–Fri 18:45 IST | NSE F&O data |
| `task_signal_pipeline` | Mon–Fri 9:15–15:45 IST, :15/:45 | Core signal engine |
| `task_position_review` | Mon–Fri 9:25–15:55 IST, :25/:55 | Position re-analysis (offset to avoid GPU contention) |
| `task_eod_review` | Mon–Fri 15:45 IST | Claude EOD review → learnings |
| `task_refresh_weights` | Mon–Fri hourly :05 | Update combine model weights from rolling accuracy |
| `task_accuracy_tracker` | Mon–Fri 20:00 IST | Rolling accuracy + weight adjustment |
| `task_weekend_review` | Saturday 9:00 IST | Deep review + cleanup |

### Market Data Reliability

The Angel One LTP endpoint is used for live index quotes only (Nifty 50 / Sensex). The `market_data.py` router implements a **circuit breaker**: after any network failure it backs off for 300 seconds before retrying, preventing log spam and cascading timeouts. The `getCandleData` endpoint (used for OHLCV history) is a separate path and remains available independently.

### Trading Mode Gate

The system runs in **PAPER mode** by default. Unlocking live execution requires:
- 28+ days of signal history
- 50+ resolved signals
- Rolling 7-day accuracy ≥ 55%

All signals are advisory until these conditions are met.

### Claude CLI Integration

Claude synthesis runs via the `claude -p` CLI subprocess. On Windows, the binary is resolved via `shutil.which("claude")` to find the `.CMD` npm shim. Prompts are passed via stdin rather than argv to correctly handle multi-byte characters (₹, →) through the Windows codepage. Uses `--append-system-prompt` for system context injection.
