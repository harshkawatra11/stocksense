# StockSense runbook

StockSense runs only when you start it by hand. Nothing starts it automatically and no scheduled task is
installed (`scripts/install-task.ps1` and `uninstall-task.ps1` stay in the repo but are NOT USED). The daemon
places no orders anywhere: every fill is a paper fill in `data_store/state/stocksense.sqlite`.

## Start and stop

```powershell
scripts\run-daemon.ps1                             # start (leave the terminal open)
.venv\Scripts\python.exe -m stocksense run         # the same thing without the script
```

Stop with Ctrl+C in that terminal, or find the python PID (`Get-Process python`) and run
`taskkill /F /T /PID <pid>`. A second start prints "already running" and exits. At its next start the daemon
catches up on missed days (data, stale orders, missed evening pipelines). Keep the laptop on and awake from
before 09:15 until the evening pipeline has finished.

## Go-live checklist (manual start, paper only)

Nothing is scheduled. You start the system yourself, in this order. Every fill is a paper fill.

1. Machine: laptop plugged in, lid action "Do nothing", no sleep on AC, internet on, you are logged in.
   Start before 07:30 on a trading day (08:45 at the latest, so the capital read happens).
2. Health check:
   ```powershell
   .venv\Scripts\python.exe -m stocksense doctor
   ```
   Everything except `heartbeat` should be fine (heartbeat is `STALE` while the daemon is off; `upstox: token ok`;
   data fresh to the last completed session). Fix what it names before going on.
3. Start (leave the terminal open):
   ```powershell
   scriptsun-daemon.ps1
   .venv\Scripts\python.exe -m stocksense run        # the same without the script
   ```
4. Within a minute `stocksense doctor` (second terminal) shows `heartbeat: alive`.
5. What you should see, by time of day (IST, trading day):

   | Time | Job | What it means |
   |---|---|---|
   | 07:30-07:45 | `preflight`, `broker_login`, `overnight_data` | catch-up, Angel login, overnight data |
   | 08:45 | `capital_am` | funds read from the broker (the capital sizing uses; `capital_stale=1` = yesterday's) |
   | 08:50 | `plan_open` | S1 exits planned the evening before are moved to the open |
   | 09:00-09:07 | `preopen_capture`, `s2_decide` | pre-open prices; S2 orders go only to the two measurement books |
   | 09:15 | `open_fills`, then `market_loop` every minute | paper fills of the open orders; marks |
   | 15:05 | `capital_pm` | capital read again |
   | 15:10 | `s1_entries` | S1 BUY orders from last evening's plan, re-sized to today's capital and quotes |
   | 15:22 | `close_fills` | paper fills of the close orders |
   | 15:35 | `eod_mark` | books marked to the close |
   | from 18:30 | `eod_ingest`, `eod_pipeline` | data, features, predictions (with the score EMA), tomorrow's S1 plan, live IC, report |

   The report of the day is `data_store/reports/YYYY-MM-DD.md`; it is the thing to read each evening.
6. Stop: Ctrl+C in the daemon terminal (or `Get-Process python`, then `taskkill /F /T /PID <pid>`). A stopped
   daemon places nothing and nothing restarts it; the next start catches up.
7. `degraded` in the heartbeat: the first quote source (Angel) is failing and a fallback (Upstox, then NSE MCP) is
   serving. Orders still work with the fallback quotes; check `broker_login`; it clears itself after the 5-minute breaker.
8. If a job fails, the table "What a `failed` job means" below applies. In short: `broker_login` or `capital_*`
   failed means yesterday's capital and fallback quotes are used (entries are still sized, but check the report);
   `eod_ingest` failed means no data for the day, so no prediction, no plan and no entries tomorrow;
   `eod_pipeline.predict` or `.s1_plan` failed means the next 15:10 re-plans from scratch (or skips if scores are
   missing); `s1_entries` skipped with a failed pre-trade check (see the report: `model_fresh`, `score_sane`,
   `quotes_alive`, `capital_read_today`) means no entries that day and nothing else; `eod_pipeline.nightly_learn`
   `skipped: refit failed` means the old model stays live until it is 7 days old, after which entries are blocked.
9. A model older than 7 days blocks entries (`model_fresh`). The Friday evening refit (or a refit when the model is
   6+ days old) keeps it fresh as long as the daemon ran the evening pipeline.

## Is it alive?

```powershell
.venv\Scripts\python.exe -m stocksense doctor
```

Look at the `heartbeat` line: `alive` means the file `data_store/state/heartbeat.json` was written less than
60 s ago. `STALE` means the daemon is not running (see "Start and stop"). The file also names the job running
now (`current_job`), the last job that finished well (`last_ok_job`) and `degraded` (true while the first quote
source is failing and a fallback is serving).

## Where things are

| What | Where |
|---|---|
| Daily report | `data_store/reports/YYYY-MM-DD.md` and `.json` (a missed day's report is marked "catch-up") |
| Log of the day | `data_store/logs/daemon-YYYY-MM-DD.log` (one JSON object per line; secrets are masked) |
| Job history | SQLite table `job_runs` (job, trade_date, attempt, status, detail) |
| Paper books | SQLite tables `decisions`, `orders`, `fills`, `marks` |
| Single-instance lock | `data_store/state/daemon.lock` |

Quick look at today's jobs:

```powershell
.venv\Scripts\python.exe -c "import sqlite3,datetime as d;c=sqlite3.connect('data_store/state/stocksense.sqlite');[print(r) for r in c.execute('select job,attempt,status,detail from job_runs where trade_date=? order by rowid',(d.date.today().isoformat(),))]"
```

## Manual runs

Before a manual restart after a crash, make sure no stray process is left:
`taskkill /F /T /IM python.exe` kills every Python process, so prefer finding the daemon first
(`Get-Process python`) and killing only that one with `taskkill /F /T /PID <pid>`.

Run something by hand (same recording wrapper as the daemon; the daemon must not be running the same job):

```powershell
.venv\Scripts\python.exe -m stocksense job eod_pipeline --date 2026-11-13
.venv\Scripts\python.exe -m stocksense run --once      # do everything currently due, then exit
```

## After an absence

Nothing to do. When the daemon starts (laptop switched on, you logged in) `preflight` rebuilds the calendar,
catches up all data, moves S1 exit orders that were left pending to the next open (everything else pending
from a past day is cancelled), and runs the evening pipeline for every session that was missed, oldest first.
No orders are made for missed days. Late-start rules for the current day: `s2_decide` after 09:08 and
`open_fills` after 09:25 are recorded as `skipped: late` (pending open orders are cancelled `late_start`, S1
exits roll to the next session's open), `s1_entries` after 15:20 is skipped; the evening jobs always run.

## A book stopped by the drawdown rule

When a book falls `book_dd_stop` below its peak its S1 policy stops opening positions and the flag stays set.
After you have looked at why:

```powershell
.venv\Scripts\python.exe -m stocksense resume --book main
```

## What the machine must do (your standing requirements)

- Laptop on and you logged in (you start the daemon by hand).
- Lid action "Do nothing"; no sleep on AC power (the daemon also asks Windows to stay awake while it runs).
- Internet on from 07:30 to about 23:30.

## Rotating the Upstox token (once a year)

The Upstox Analytics Token lasts a year. Create a new one in the Upstox developer console, put it in `.env` as
`STOCKSENSE_UPSTOX_ANALYTICS_TOKEN=...` (never anywhere else; never paste it into a terminal or a chat), then
restart the daemon. `stocksense doctor` shows `upstox: token ok` when it works. Without a token the system keeps
running on Angel and the NSE MCP fallback.

## Telegram messages (optional)

If `STOCKSENSE_TELEGRAM_BOT_TOKEN` and `STOCKSENSE_TELEGRAM_ALLOWED_CHAT_ID` are set in `.env` you get a message
at `s1_entries`, one after the report (with the NAVs) and one for every failed job. Not set: nothing is sent.

## What a `failed` job means and what to do

`failed` rows whose detail starts with `retry:` are "not yet" (data not published, quotes not there yet); the
daemon retries them and they are not errors. Anything else is a real failure; the detail has the exception.

| Job or symptom | Meaning | Fix |
|---|---|---|
| `eod_ingest` retry: not fresh | Bhavcopy or another required file is not published yet (NSE publishes late some days) | Nothing; it retries every 15 min until 23:30; the next start catches up |
| `eod_ingest` failed all night | NSE blocked or changed a file | `stocksense ingest --all --catch-up`, read the error, fix the ingester; the next preflight reruns it |
| `broker_login` failed | Angel TOTP rejected or `AB1021` false rate limit | The client backs off 60 s then 300 s and retries the TOTP once; if it persists check the Angel secrets in `.env` (`stocksense doctor` shows set/unset only) and the system clock; quotes fall back to Upstox then NSE MCP |
| `capital_am` failed / `capital_stale=1` | Funds could not be read; yesterday's value is used for sizing | Check `broker_login`; orders for the day are sized from the stale capital |
| `open_fills` / `close_fills` retry: orders pending | No usable quote yet, or a circuit-blocked order | Normal for a few minutes; blocked orders are cancelled with a reason at 15:20/15:28 |
| `degraded` true in the heartbeat | First quote source failing; fallback serving | Check Angel login; it self-heals after the 5-minute breaker |
| `eod_pipeline` failed at a stage | The stage row `eod_pipeline.<stage>` has the error | Fix and rerun `stocksense job eod_pipeline --date <d>`; finished stages are not repeated |
| `eod_pipeline.reconcile` retry: no official rows | The panel for the day is not built yet | It follows `build_panel`; rerun the job |
| `promote` and `kronos_features` stages show `skipped: not implemented` | Stubs by design (listed in `docs/DECISIONS.md`) | Expected |
| `skipped: late` | The daemon started after the job's cut-off | Expected after a late start |
| Heartbeat `STALE` | Daemon not running | Start it by hand (`scripts\run-daemon.ps1`); if it exits at once look at the last lines of the day's log |
| "already running" | Another daemon holds `state/daemon.lock` | Nothing; there is already one |
