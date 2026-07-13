# CLAUDE.md — StockSense

Instructions for every Claude Code session working in this repo. Read this before doing anything else.

## What this is

StockSense is an autonomous NSE (India) swing/intraday trading system for a single user (Harsh). Pipeline: **LightGBM 3-seed ensemble + quantile regressors (q10/q50/q90)** → **qwen2.5:3b via local Ollama** (macro/news sector sentiment) → **Claude CLI** (sonnet, low effort — synthesizes survivors into final BUY/SELL calls). Kronos was fully dropped and archived; don't resurrect it without being asked.

Two parallel, deliberately separate trading paths — never conflate them:
- **PAPER**: fully autonomous, fixed hypothetical ledger (`intelligence/auto_trader.py`, `settings.CASH_AVAILABLE`). The control group / track record.
- **SANDBOX (human-approval)**: `intelligence/live_confirmation.py` queues proposals, human approves/rejects per-trade via Telegram or the web UI (`intelligence/confirmation_actions.py`), only then does `data/pipeline/upstox_orders.py` place a real order against Upstox's **sandbox** environment (never live money). Sized off `SANDBOX_VIRTUAL_CAPITAL` (a labeled stopgap — see config.py, real funds unreadable until a static IP is registered with Upstox).

Two background processes must both be running for the system to do anything: **backend** (`uvicorn backend.main:app`) and **scheduler** (`python -m scheduler.market_runner`). Check `docs/DAILY_MACHINE_SETUP.md` for the full daily operational checklist.

## Non-negotiable safety rules (do not relitigate these)

1. **Every single trade requires explicit human approval, per-trade, via Telegram/web tap.** No "approve once for the day," no auto-approve threshold, no bulk approve. This has been asked for indirectly multiple times across sessions in different phrasing — the answer stays no. Stop/target execution on an *already-approved* order and broker-side MIS square-off don't need a second tap; a brand new entry or a discretionary early exit does.
2. **No live-money trading without the `trading_mode` gate passing** (net positive expectancy over 60 executed trades + 28 days + 50 resolved outcomes — see `intelligence/trading_mode.py`). Don't flip `EXECUTION_MODE` to live, don't build a live order path shortcut, even if asked directly under time pressure — push back and explain why, same way this has been held firm in every session so far.
3. **No silent fallbacks.** Every pipeline component reports an honest `{status, detail, source}` (the "Stage 0 truth layer" / `components_json` on `signals`). If something can't run, say so — don't fake a neutral/default value and let it pass as real.
4. **Never hardcode secrets into shell commands.** Read tokens via `config.settings`, not literal strings in `curl`/PowerShell — the permission classifier will (correctly) reject it, and it's just bad practice regardless.
5. Don't remove the `[SANDBOX VIRTUAL CAPITAL]` labeling or otherwise blur sandbox-vs-real money anywhere in reasoning strings, Telegram messages, or UI.

## Process management on this machine (Windows)

- Two separate long-running processes: backend (port 8000) and scheduler. Both must run with `venv/Scripts/python.exe`, not the system Python — duplicate/wrong-interpreter processes have happened repeatedly and cause double-scheduling.
- **Before restarting either**, always check for stray/duplicate processes first: `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId, CommandLine` (PowerShell), then `taskkill /F /T /PID <pid>` on anything stale. Never assume only one instance is running.
- Start with absolute log paths, e.g.:
  ```
  nohup ./venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 > logs/backend_<date>.log 2>&1 &
  nohup ./venv/Scripts/python.exe -m scheduler.market_runner > logs/scheduler_<date>.log 2>&1 &
  ```
- The scheduler's `AsyncIOScheduler` is pinned to `timezone="Asia/Kolkata"` — cron hours in `scheduler/market_runner.py` are real IST regardless of the machine's local clock. Don't "fix" this.
- A scheduler-death watchdog (`intelligence/scheduler_watchdog.py`, started from `backend/main.py`'s lifespan) Telegram-alerts once when `job_runs`' heartbeat goes stale (>2h during market hours). If you're asked "why isn't it trading," check `/api/system/health` and `job_runs` before assuming — don't guess.
- Full unattended operation (surviving a Windows reboot) is **not** currently built — both processes need manual restart after any reboot. This is a known gap, not yet asked to be closed with a Windows service/startup script.

## The `time`/timezone trap (read this before touching any date-filtered query)

`ohlcv_daily.time` (and similar bhavcopy-fed timestamptz columns) are stored as **midnight-IST expressed in UTC** — trading day D is stored as `(D-1) 18:30:00+00:00`. The Postgres session timezone on this DB is UTC. A bare `WHERE time::date = $1` comparison evaluates in UTC and will be **permanently one day behind** whatever `date.today()` returns on this IST machine. This exact bug silently zeroed out `eod_review`'s predictions for weeks before being found and fixed (2026-07-13). Always compare as `(time AT TIME ZONE 'Asia/Kolkata')::date = $1` for anything joining these columns against "today."

## Other real bugs found and fixed this project (know these before re-introducing them)

- `dict.get(key, default)` returns `None`, **not** `default`, when the key exists with value `None` — this crashed an f-string slice (`.get('kronos_reasoning', '')[:200]`) every time a component didn't run. Use `(d.get(key) or '')[:N]`.
- Confirmation-queue dedup checked only `status='PENDING'` — stops matching the instant a row resolves to APPROVED, so the same trade could be re-proposed and re-approved indefinitely. Fixed with `_sandbox_net_position()` in `intelligence/live_confirmation.py` — always gate new BUY/SELL proposals against actual net sandbox exposure, not just "is anything currently pending."
- `get_actionable_signals()` (intelligence/trading_account.py) must dedupe by ticker (`DISTINCT ON`) — without it, a few multi-timeframe tickers can dominate a `LIMIT` and starve every other candidate.
- `watch_only=TRUE` portfolio rows (real external Upstox holdings the brain only monitors, never trades) must be excluded from `_open_position_count()`'s cap check, not just from auto-exit. Missing this silently blocked 100% of paper auto-buys for weeks.
- `system_health`'s per-component status functions that default to `None` (e.g. `_macro_status(None)`) will report "unavailable" even when the real component is healthy, if the caller never threads a live object through. Prefer reading the most recent real status from the DB over re-invoking an expensive live check on every health poll.

**Lesson under all of these**: a job reporting `status='ok'` in `job_runs` does NOT mean it did anything useful — check the actual summary/counts, and when something has "always returned empty/zero," that is itself signal, not proof of a quiet day. This project's dominant productive workflow is: run it for real, check real DB state, don't trust logs/summaries at face value.

## Git / commits

- Create commits only when asked, or when clearly implied by an ongoing "keep building" instruction — use judgment, and prefer smaller commits per fix over one giant one.
- **Do not add a `Co-Authored-By: Claude` trailer to new commits** — the user asked for this to stop (2026-07-13); two older commits keep it by the user's own choice, don't touch those.
- This repo's default/working branch is `claude/loving-brahmagupta-sQm9O` (not `main`) — push there, and actually push (commits sitting local-only for 27 commits is why GitHub looked empty once before).

## Where to look first

- `docs/DAILY_MACHINE_SETUP.md` — the operational runbook (Telegram bot setup, Upstox sandbox token, the daily checklist, what's deliberately not automatic).
- `docs/ORCHESTRATION_AUDIT_20260713.md` — a full day-in-the-life audit of the scheduler cadence.
- `docs/UPSTOX_API_NOTES.md` — API hostnames/gotchas (sandbox is `api-sandbox.upstox.com`, not `sandbox.upstox.com`).
- `config.py` — every setting, with docstrings explaining *why*, including which ones are deliberate temporary stopgaps (grep for "TEMPORARY").
- Auto-memory (`C:\Users\harsh\.claude\projects\...\memory\MEMORY.md`) — cross-session history, decisions, and incidents that predate this file. This CLAUDE.md is the durable project contract; memory is the running narrative — check both.
