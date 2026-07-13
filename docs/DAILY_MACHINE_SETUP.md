# Daily Machine Setup — Going Live (Sandbox) Monday Morning

This is the doc you follow at 8:45 AM Monday. It covers the one-time setup (Telegram
bot, Upstox sandbox token, the config flip) and the every-morning checklist.

## 1. What actually happens each day

Every 30 minutes during market hours (09:15–15:45 IST) the pipeline analyzes all
~2,500 tradeable NSE stocks: the LightGBM ensemble + quantile models (with F&O
features) score each stock, qwen2.5:3b (local Ollama) adds macro/news sector
sentiment, and Claude CLI (sonnet, low effort) synthesizes the survivors into final
BUY signals. Qualifying BUY proposals — and EXIT proposals for positions you hold —
are queued for **your** review; each one arrives as a Telegram message with
Approve/Reject. Only trades you explicitly approve are sent, and they go to the
**Upstox SANDBOX** (simulated fills, no real money). Approved positions are then
watched by a fast intraday stop/target monitor (every ~7 seconds against live
quotes), plus a fuller re-analysis every 30 minutes. After close, the evening jobs
pull the official closes (18:30), reconcile them against Upstox (18:35), update F&O
data (18:45), and run the Claude end-of-day review (18:50) with a Telegram summary.

**Two things this system never does on its own:** it never places an order without
your explicit per-trade approval, and it never touches real money — execution is
sandbox-only until you separately and deliberately decide otherwise (see §6).
Unapproved proposals expire automatically after 120 minutes (CONFIRMATION_EXPIRY_MINUTES)
so you never approve a stale price.

(The paper-trading "brain" keeps running fully autonomously in parallel — that's the
control group and is unaffected by any of this.)

## 2. One-time: create your Telegram bot (~5 minutes)

1. Open Telegram and message **@BotFather** (the verified one, blue check).
2. Send `/newbot`.
3. It asks for a display name — anything, e.g. `StockSense`.
4. It asks for a username — must end in `bot` and be unique, e.g.
   `harsh_stocksense_bot`.
5. BotFather replies with an HTTP API **token** that looks like
   `1234567890:AAExampleExampleExampleExample`. Copy it.
6. In the repo root, open `.env` and set:
   ```
   TELEGRAM_BOT_TOKEN=1234567890:AAExampleExampleExampleExample
   ```
7. Now get your chat id: in Telegram, open your new bot (BotFather's reply has a
   `t.me/...` link), press **Start**, and send it any message ("hi").
8. In a browser, visit (paste your real token in place of `<TOKEN>`):
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
9. In the JSON you'll see `"chat":{"id":123456789,...}` — that number is your chat
   id. Copy it into `.env`:
   ```
   TELEGRAM_CHAT_ID=123456789
   ```
   (If the JSON is empty, send the bot another message and refresh.)

The bot only talks to that chat id and ignores everyone else.

## 3. One-time: Upstox sandbox token

1. Log in to the Upstox developer portal (account.upstox.com → Developer / My Apps).
2. Open the **Sandbox** section and generate a sandbox access token.
3. Put it in `.env`:
   ```
   UPSTOX_SANDBOX_TOKEN=<the token>
   ```

Note: this is **separate** from `UPSTOX_ANALYTICS_TOKEN`, which is already configured
and used for live market data (quotes/candles). The sandbox token is used only for
placing simulated orders. Both live in `.env`; don't swap them.

## 4. The flip: enable human-confirmation mode

In `.env`, set:

```
LIVE_CONFIRMATION_ENABLED=true
```

Then restart everything from the repo root:

```powershell
./start.ps1
```

What `start.ps1` actually launches (all of it — you don't need to start the scheduler
separately): the Postgres DB in Docker, then the **backend** (uvicorn, port 8000) in
its own PowerShell window, then the **scheduler** (`python -m scheduler.market_runner`)
in its own window, then the frontend dev server (port 5173). If you previously had any
of these running, close those windows first (kill the old python processes —
`taskkill /F /T` on lingering PIDs) so you don't end up with two schedulers.

Two caveats worth knowing:
- `start.ps1` only **checks** Ollama — it does not start it. If Ollama isn't running,
  start it yourself (`ollama serve`) before or after; the macro layer falls back to
  NEUTRAL without it.
- `-NoScheduler` exists as a flag; never use it on a trading day. Without the
  scheduler window there are **no signals and no Telegram proposals at all** — the
  backend alone only serves the UI, the live price feed, and stop monitoring for
  already-held positions.

Both flag checks live inside the code (the scheduler jobs and the queueing functions
self-gate on `LIVE_CONFIRMATION_ENABLED`), so with the flag false everything is a
harmless no-op — flipping it to true is the entire opt-in.

## 5. Monday-morning checklist (be done by 09:00)

- [ ] **Machine on and plugged in** before 09:00; disable sleep for the day
      (Settings → Power: sleep = Never while plugged in). A sleeping machine skips
      scheduler jobs — they do not catch up when it wakes.
- [ ] **Docker DB up** — `docker ps` shows the stocksense db container healthy
      (start.ps1 does this for you).
- [ ] **Backend alive** — open http://localhost:8000/api/health → `{"status":"ok"}`.
- [ ] **System health green** — http://localhost:8000/api/system/health → check
      `components.scheduler.status` is `ok` (this is the job_runs heartbeat), and
      data freshness looks sane. Same info visually: the **Brain tab** in the app
      (http://localhost:5173) shows every job's last run.
- [ ] **Scheduler window open** — the PowerShell window running
      `scheduler.market_runner` is up and logged "Scheduler started. Jobs: [...]".
- [ ] **Ollama running with qwen2.5:3b** — `ollama list` shows `qwen2.5:3b`;
      `curl http://localhost:11434/api/tags` responds. First macro call of the day is
      slow (model warm-up) — that's normal.
- [ ] **Claude CLI logged in** — run `claude -p "say ok"` in a terminal; if it asks
      to log in, run `claude` then `/login`. (The synthesis layer and EOD review
      silently degrade without it.)
- [ ] **Telegram bot responding** — send your bot a message; expect a reply/ack once
      the bot service is running with the backend.
- [ ] **Watch the price dot** — in the app header, the price-age dot should be green
      shortly after 09:15 (live Upstox feed ticking). If it's red mid-session, the
      live feed is down: fast stop monitoring is degraded to the 30-min cycle —
      restart the backend.

Timeline once the market opens at 09:15: the first `signal_pipeline` run fires at
**09:15** and takes a while (2,500 stocks + cold Ollama), so expect the first
qualifying proposals — and therefore the first **Telegram approval messages** —
somewhere in the **09:15–09:45** window. Position reviews run at :25/:55. Approve or
reject each proposal from Telegram (or the Confirmations panel in the app); anything
you ignore expires in 2 hours.

During the day, glance occasionally at the Brain tab: if the scheduler heartbeat goes
stale (>2h during market hours), the scheduler window has died — nothing else will
alert you yet. Restart it: `./venv/Scripts/python.exe -m scheduler.market_runner`.

After close: evening data jobs run 18:30–18:50; the EOD review + Telegram summary
lands shortly after 18:50. Leave the machine on until at least 20:15 (the 20:00
accuracy/retrain check) if you can.

## 6. What is deliberately NOT automatic

1. **Per-trade approval.** Every single order — BUY and SELL alike — requires your
   explicit approval on that specific proposal. There is no "approve all", no
   auto-approve threshold, and expired proposals are dropped, never executed late.
   (The autonomous behavior you see in the Brain tab is paper trading only.)
2. **Live-money trading.** Approved orders go exclusively to the Upstox **sandbox**.
   Going live with real money is a separate, later decision that would require:
   the `trading_mode` gate to actually pass (a sustained track record per
   `intelligence/trading_mode.py`, currently judged on the paper/sandbox history),
   a live Upstox trading token and order path (not the sandbox one), and an explicit
   code/config change you make deliberately — there is no flag in `.env` today that
   can send real orders. Sandbox results over a few weeks are the evidence for or
   against ever flipping that.
