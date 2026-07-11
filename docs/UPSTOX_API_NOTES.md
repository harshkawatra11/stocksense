# Upstox API — Integration Reference (researched 2026-07-11)

Purpose: source-of-truth for migrating StockSense's live data layer to Upstox
(Stage 1 of WHAT_TO_DO_NEXT.txt). Covers REST/WebSocket data APIs, auth, the
official MCP server, and the official Agent Skill — the last two are directly
relevant to giving the Claude synthesis layer better live-market grounding.

Sources: upstox.com/developer/api-documentation/* (fetched directly),
github.com/upstox/upstox-python, github.com/upstox/upstox-skills.
Chrome/claude-in-chrome could not navigate upstox.com directly — it's
blocked by the extension's financial-site safety restriction — so this is
compiled via WebFetch/WebSearch against the same public docs.

---

## 1. Creating the API app (what the user needs to do first)

1. Go to https://account.upstox.com/developer/apps (requires an active,
   non-dormant Upstox trading account — user has this).
2. Create a new app. You'll register a **redirect URI** (for local dev this
   can be a `localhost` URL StockSense's backend will handle, e.g.
   `http://localhost:8000/api/upstox/callback`).
3. You get back:
   - `client_id` (API key)
   - `client_secret` (API secret)
4. Store both in `.env` as `UPSTOX_CLIENT_ID` / `UPSTOX_CLIENT_SECRET`
   (`.env` is already git-ignored — see .gitignore).

## 2. Authentication flow (OAuth 2.0, daily token)

```
Step 1 — send user to:
  https://api.upstox.com/v2/login/authorization/dialog
    ?response_type=code
    &client_id=<UPSTOX_CLIENT_ID>
    &redirect_uri=<registered redirect URI>
    &state=<optional csrf token>

  User logs in (password + TOTP 2FA supported instead of SMS OTP — prefer
  TOTP, same pattern StockSense already uses for other providers per
  memory: "Claude CLI Windows invocation" and Groww's TOTP flow).

Step 2 — Upstox redirects back to redirect_uri with:
  ?code=<single-use auth code>&state=<echoed>

  The code is single-use regardless of whether the next step succeeds.

Step 3 — backend exchanges the code server-side:
  POST https://api.upstox.com/v2/login/authorization/token
    code=<code>
    client_id=<UPSTOX_CLIENT_ID>
    client_secret=<UPSTOX_CLIENT_SECRET>
    redirect_uri=<same URI>
    grant_type=authorization_code

  Returns an access_token. Docs do not publish an exact TTL, but the
  ecosystem-wide behavior (confirmed by both the MCP server and the Agent
  Skill docs) is: **tokens expire daily and must be re-generated.** Plan for
  a daily re-auth step, same shape as the existing Groww TOTP daily-login
  pattern already in data/pipeline/fetch_groww.py — reuse that pattern here
  rather than inventing a new one.

Alternative: tokens can also be generated manually, one-off, from the
developer dashboard — useful for local dev/testing without wiring the full
OAuth redirect dance immediately.
```

No IP whitelisting is required for any of this (data APIs). IP whitelisting
only applies to the **Order API**, and only once SEBI's Apr-2026 circular is
enforced for algo order flow — irrelevant while StockSense stays paper-only.

## 2b. Analytics Token — CORRECTION, confirmed 2026-07-11 from official docs screenshots

The daily-OAuth-token design above is not the only option, and for StockSense's
data-only needs it's not even the preferred one. Upstox publishes a distinct
**Analytics Token**:

- Generated **once** from the developer portal — no daily re-authorization.
- Specifically powers the **Market Data** and **Realtime & Streaming APIs**
  (exactly what `upstox_feed.py` and `get_historical_candles` need).
- Also unlocks read-only **Portfolio** and **Account & Funds** APIs, but only
  when called from a registered static IP (not relevant to StockSense today).
- curl shape: `Authorization: Bearer {analytics_token}` — no other change to
  request shape vs. the daily token.

**Implementation**: `data/pipeline/upstox_client.get_data_token()` prefers
`settings.UPSTOX_ANALYTICS_TOKEN` (env var `UPSTOX_ANALYTICS_TOKEN`) and
falls back to the daily OAuth token (`get_valid_token()`) if unset. Generate
one from the developer portal's "Learn how to generate an analytics token"
link and drop it in `.env` — this removes the daily-re-auth requirement for
the entire live-data path.

## 2c. Sandbox — risk-free order-API rehearsal

`sandbox.upstox.com/v2` emulates the real API (place/modify/cancel orders)
with no time restrictions and a separate sandbox access token — no real
money, no real market impact. This is the natural place to rehearse the
confirmation-gated execution path (`pending_trade_confirmations` table,
`backend/routers/confirmations.py`) end-to-end before any live-order wiring
is even considered: build and test the approve/reject flow against sandbox
orders first, entirely decoupled from the "should this ever go live"
decision.

## 3. REST — Historical & intraday candles (V3)

```
GET https://api.upstox.com/v3/historical-candle/
        {instrument_key}/{unit}/{interval}/{to_date}/{from_date}

Auth: Authorization: Bearer <access_token>

instrument_key format: EXCHANGE_SEGMENT|ID
  e.g. NSE_EQ|INE848E01016   (equity, by ISIN)
       NSE_INDEX|Nifty 50    (index)

unit:      minutes | hours | days | weeks | months
interval:  minutes: 1-300 | hours: 1-5 | days/weeks/months: 1

Lookback / max retrieval window per request:
  minutes (1-15):   available since Jan 2022, max 1 month per request
  minutes (>15):    available since Jan 2022, max 1 quarter per request
  hours:            available since Jan 2022, max 1 quarter per request
  days:             available since Jan 2000, max 1 decade per request
  weeks / months:   available since Jan 2000, no limit

Response: array of [timestamp, open, high, low, close, volume, open_interest]
```

Implication for StockSense: 1-minute intraday only goes back to Jan 2022 and
only 1 month per call — fine for live/recent intraday features, NOT a
replacement for the deep multi-decade daily history. Keep NSE Bhavcopy as
the authoritative EOD/daily historical source (per WHAT_TO_DO_NEXT.txt
Phase 1) and use Upstox only for live/intraday + recent-minute data, with a
nightly Upstox-close vs Bhavcopy-close reconciliation check.

## 4. WebSocket — Market Data Feed V3 (the live tick source)

```
wss://<redirected-authorized-url>/feed/market-data-feed
  (connect via wss:, expect an automatic redirect to an authorized URL —
   client must follow it, standard WS reconnect/redirect handling)

Headers:
  Authorization: Bearer <access_token>
  Accept: */*

Subscribe (send as BINARY, not text):
{
  "guid": "<unique-id>",
  "method": "sub" | "change_mode" | "unsub",
  "data": {
    "mode": "ltpc" | "option_greeks" | "full" | "full_d30",
    "instrumentKeys": ["NSE_EQ|INE848E01016", "NSE_INDEX|Nifty 50"]
  }
}

Modes:
  ltpc          — latest traded price + close price only (this is what
                  StockSense needs for the live quote_cache / price ticker)
  option_greeks — greeks only
  full          — ltpc + 5 depth levels + extended metadata + greeks
  full_d30      — ltpc + 30 depth levels (Upstox Plus tier only)

Message decoding: PROTOBUF, using the official MarketDataFeed.proto schema
(get it from the community-pinned proto file or upstox-python SDK repo).
Do not attempt naive JSON parsing of the WS payload.

Response sequence on connect:
  1. Market status message (per-segment: NSE_EQ, NSE_FO, BSE_EQ open/closed)
  2. A full snapshot of current state for subscribed instruments
  3. Continuous live delta updates thereafter

Heartbeat: server sends WS ping frames when idle; most WS libraries
auto-respond with pong — no custom keepalive logic needed.

Subscription limits (Standard/free tier — what the user has today):
  Connections per user: 2
  LTPC:          5,000 individual keys / 2,000 when combined with other modes
  option_greeks: 3,000 individual / 2,000 combined
  full:          2,000 individual / 1,500 combined

  (Upstox Plus, paid tier, unlocks full_d30 + more connections — not needed
   for StockSense's ~200-ticker universe; the free LTPC tier at 5,000 keys
   comfortably covers it with huge headroom.)
```

Recommended StockSense usage: subscribe **ltpc mode** for the active
ticker universe (indices + watchlist + open-position tickers + top-N
signal candidates) — this is the cheapest mode and is exactly what's needed
to kill the fake 5-minute Groww snapshot and replace it with sub-second
ticks (Phase 1, Stage 1 in the roadmap).

## 5. Rate limits (REST)

```
Order Placement API:
  Regular (unregistered) algo: 10 req/s, 500 req/min, 2,000 req/30min
  SEBI-registered algo:        50 req/s, 500 req/min, 2,000 req/30min
  (irrelevant until live order execution is built — Phase 5+, not in scope now)

Standard APIs (holdings, positions, funds, historical candles, quotes):
  50 req/s, 500 req/min, 2,000 req/30min

Payout APIs: not relevant to StockSense.
```

No published daily cap. Enforcement is per-API, per-user; exceeding causes
temporary suspension, not a hard ban — build in basic backoff regardless.

## 6. Official Python SDK

- `pip install upstox-python-sdk`
- GitHub: https://github.com/upstox/upstox-python
- Docs: https://upstox.github.io/upstox-python/
- Ships `MarketDataStreamerV3` — a ready-made WS client:
  ```python
  from upstox_client.feeder.market_data_streamer_v3 import MarketDataStreamerV3

  streamer = MarketDataStreamerV3(
      api_client, instrumentKeys=["NSE_INDEX|Nifty 50", "NSE_INDEX|Nifty Bank"],
      mode="ltpc"
  )
  streamer.auto_reconnect(True, interval=10, retry_count=3)
  streamer.on("message", handle_tick)
  streamer.connect()
  ```
  It already decodes protobuf to a dict/JSON-like structure for you — prefer
  this over hand-rolling protobuf decoding in
  `data/pipeline/upstox_feed.py` (Phase 1 plan). Auto-reconnect with
  interval/retry is built in, satisfying the "reconnect/backoff" requirement
  from the roadmap without custom code.

---

## 7. Upstox MCP Server — read-only, NOT for order execution

**URL:** `https://mcp.upstox.com/mcp` (hosted, OAuth-authenticated, no local
process required for HTTP-based clients like Claude Code).

**Install in Claude Code:**
```
/plugin marketplace add upstox/upstox-plugin-marketplace
/plugin install upstox-mcp@upstox-plugins-official
```
Verify with `/mcp`. First tool call triggers a browser OAuth consent screen;
Upstox explicitly requires **daily re-authorization** — the token/session
expires every day, matching the REST/WS access-token behavior in section 2.

**What it exposes (confirmed read-only):**
- Holdings, orders, positions
- Mutual funds, account funds, user profile
- Portfolio P&L, account margins
- Market quotes and historical trading data

**Explicitly cannot place orders, modify positions, or execute trades** —
Upstox's own docs state this is deliberate, for security. This makes it
useful for interactive/ad-hoc analysis sessions (e.g. you asking Claude Code
"how is my portfolio doing" mid-development) but it is **not** infrastructure
StockSense's autonomous pipeline should depend on, since (a) it requires
daily human-in-the-loop browser re-auth, incompatible with a scheduler that
runs unattended, and (b) StockSense already has its own DB-backed portfolio
state — this MCP server would be a redundant, weaker read path for the app
itself. It IS worth having installed for you personally, for quick manual
portfolio checks from within a Claude Code session, separate from the app.

## 8. Upstox Agent Skill — the one the user asked about (order execution capable)

**Verified 2026-07-11 against the official page** (upstox.com/developer/api-documentation/agent-skills) via user-provided screenshots — every detail below matches what was already researched via WebFetch; nothing changed. Additional confirmed details from that pass:
- Prerequisites: Node.js (for the `skills` CLI), Python 3.8+ with `pip install upstox-python-sdk`, an active Upstox account with API access, an access token from the developer portal, and one of Claude Code or Codex.
- `/plugin marketplace add` takes a **repository** name (`upstox-plugin-marketplace`); `/plugin install` takes a **marketplace** name (`@upstox-plugins-official`) — easy to transpose, the official docs call this out explicitly. The same marketplace also offers a separate `upstox-mcp` plugin for read-only account access (Section 7 above).
- Config file alternative to the env var: copy `skills/upstox/config.json.example` to `skills/upstox/config.json` and fill in `access_token` — this file is git-ignored by the skill's own `.gitignore`, so a token placed there is never committed.
- Official framing, verbatim: *"Your agent stays a knowledgeable assistant, not an autonomous trader."* — this is Upstox's own design intent, and it matches the human-approve/reject model StockSense has adopted (Stage 1's `pending_trade_confirmations` scaffolding).
- **This stage (Stage 1.5) deliberately did NOT run any `/plugin install` or `npx skills add` command** — doing so would install order-execution capability into whichever Claude Code session runs it, which is the same "unattended real-money action" risk already ruled out for the app itself. This section stays documentation-only; see the recommendation below.

**GitHub:** https://github.com/upstox/upstox-skills
**Standard:** open `SKILL.md` format (same family of thing as the Skills
already available in this Claude Code environment — see the skill list at
session start).

**Install options:**
```bash
# Claude Code plugin marketplace
/plugin marketplace add upstox/upstox-plugin-marketplace
/plugin install upstox-skill@upstox-plugins-official

# or via npx (works for Claude Code or Codex)
npx skills add upstox/upstox-skills --skill upstox

# or manual
git clone https://github.com/upstox/upstox-skills
pip install upstox-python-sdk
```

**What it actually does — this is the "agentic trading" capability:**
wraps the official Python SDK with pre-flight validation so an agent (you,
talking to Claude Code, OR StockSense's own Claude-CLI synthesis layer if
wired in) can execute real trading workflows via natural language:

- Orders: place/modify/cancel across NSE/BSE/MCX, GTT conditional orders,
  multi-leg option strategies (bull call spread, short strangle, etc.)
- Market feeds: live data + order book depth via `MarketDataStreamerV3`
- Portfolio: holdings with live P&L, funds, margin, position conversion

**8 built-in safety guardrails** (directly relevant to StockSense's own
`intelligence/portfolio_guard.py` — these are worth mirroring even if you
don't adopt the skill wholesale):
1. Confirmation required — full human-readable order preview before any
   place/modify/cancel action executes.
2. LIMIT-by-default — never fires a MARKET order unless explicitly asked.
3. Quantity default of 1 share/lot when unspecified (fail-safe-small).
4. Lot-size validation — rejects F&O orders with invalid multiples.
5. Market Price Protection — auto-bounds market orders via Upstox's native
   MPP feature.
6. Kill switch — can halt all trading in a segment programmatically
   (`UserApi.update_kill_switch(...)`).
7. Sandbox/paper mode encouraged before live execution.
8. No hardcoded secrets — tokens only from env vars or git-ignored config.

**Auth for the skill:** `export UPSTOX_ACCESS_TOKEN="..."` or a git-ignored
`skills/upstox/config.json`. Same daily-token-expiry caveat as everywhere
else in the Upstox ecosystem.

### Should StockSense adopt this skill directly?

**Recommendation: not yet, and not as a drop-in.** Reasoning:
- The skill is designed for a human-in-the-loop chat session confirming
  each order — StockSense's auto_trader is designed to run unattended on a
  scheduler. Wiring the confirmation-gated skill into an autonomous loop
  defeats its own safety model (either you disable confirmation, which
  removes the guardrail, or you block the scheduler waiting on a human).
- StockSense is still paper-only (WHAT_TO_DO_NEXT.txt Section 5, "don't
  chase live order execution yet") — there's no live order path to wire
  this into today.
- What DOES transfer directly, right now: the **8 guardrails as a design
  reference** for hardening `intelligence/portfolio_guard.py` and
  `intelligence/auto_trader.py` once live execution is eventually built
  (WHAT_TO_DO_NEXT.txt Section 5, item 1) — particularly LIMIT-by-default,
  lot-size validation, and a kill switch, none of which StockSense currently
  has.
- Separately, install the skill for **your own personal use** in Claude
  Code (outside the app) if you want to place manual trades or explore
  strategies conversationally — that's a legitimate, low-risk use today
  since it's you confirming each order, not an unattended agent.

A new file, `intelligence/skills/SKILL_upstox.md` (see companion file),
gives StockSense's own Claude-synthesis layer live-market-awareness rules
distilled from this research — that's a read-oriented complement to this
skill, not a replacement for portfolio_guard.py's future hardening.

---

## 9. Summary — what Phase 1 (WHAT_TO_DO_NEXT.txt) should actually build

| Need | Upstox mechanism | New StockSense file |
|---|---|---|
| Live tick feed | WS V3, `ltpc` mode, via `MarketDataStreamerV3` | `data/pipeline/upstox_feed.py` |
| Auth (daily token) | OAuth code flow, reuse Groww-style daily-login pattern | `data/pipeline/upstox_client.py` |
| Recent intraday candles | REST V3 historical-candle, minutes unit | `data/pipeline/upstox_client.py` |
| In-memory quote store | n/a (own code) | `backend/services/quote_cache.py` |
| Backend → frontend live push | n/a (own code, WS fan-out) | `backend/routers/ws_prices.py` |
| Personal ad-hoc portfolio checks | Upstox MCP server (read-only) | N/A — install as a Claude Code plugin, not app code |
| Future live-order guardrail design reference | Upstox Agent Skill's 8 rules | inform future `portfolio_guard.py` hardening (not now) |
