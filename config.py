"""
Central configuration for StockSense.
All values read from environment variables (with sensible defaults).
Load a .env file via python-dotenv before importing this module,
or set variables in the shell / docker-compose environment block.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ------------------------------------------------------------------ #
    # Database                                                             #
    # ------------------------------------------------------------------ #
    @property
    def DATABASE_URL(self) -> str:
        return os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://stocksense:stocksense@localhost:5432/stocksense",
        )

    @property
    def DATABASE_DSN(self) -> str:
        return os.getenv(
            "DATABASE_DSN",
            "postgresql://stocksense:stocksense@localhost:5432/stocksense",
        )

    # ------------------------------------------------------------------ #
    # Ollama / SLM                                                         #
    # ------------------------------------------------------------------ #
    @property
    def OLLAMA_URL(self) -> str:
        return os.getenv("OLLAMA_URL", "http://localhost:11434")

    @property
    def SLM_MODEL(self) -> str:
        return os.getenv("SLM_MODEL", "qwen2.5:7b")

    @property
    def OLLAMA_API_KEY(self) -> str:
        """Set this to use Ollama Cloud (e.g. Nemotron 3 Ultra). Empty = local Ollama."""
        return os.getenv("OLLAMA_API_KEY", "")

    # ------------------------------------------------------------------ #
    # Pipeline tuning                                                      #
    # ------------------------------------------------------------------ #
    @property
    def PIPELINE_INTERVAL_MINUTES(self) -> int:
        return int(os.getenv("PIPELINE_INTERVAL_MINUTES", "30"))

    @property
    def MAX_TICKERS_PER_RUN(self) -> int:
        return int(os.getenv("MAX_TICKERS_PER_RUN", "200"))

    @property
    def TOP_SIGNALS_FOR_CLAUDE(self) -> int:
        return int(os.getenv("TOP_SIGNALS_FOR_CLAUDE", "40"))

    @property
    def CONFIDENCE_THRESHOLD(self) -> float:
        return float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))

    # ------------------------------------------------------------------ #
    # Signal horizons (concurrent, config-driven)                          #
    # ------------------------------------------------------------------ #
    # Each horizon produces its own signal per ticker per run. `forecast_steps`
    # is the forward-looking window used when deriving target/stop from the
    # quantile regressors; `hold_days` is how long a signal stays active
    # before it's resolved/expired; `requires_intraday` marks sub-day
    # horizons that stay dormant until a live intraday feed exists.
    # Add/remove a dict here to add/remove a timeframe — nothing else hardcodes them.
    # Fractional precision (e.g. "touches target in ~2.5 days") is delivered
    # per-signal by target_eta_days, NOT by separate half-day buckets — there's
    # no half-day candle to forecast.
    TIMEFRAMES: list[dict] = [
        {"label": "1D", "forecast_steps": 1, "hold_days": 1, "requires_intraday": False},
        {"label": "2D", "forecast_steps": 2, "hold_days": 2, "requires_intraday": False},
        {"label": "3D", "forecast_steps": 3, "hold_days": 3, "requires_intraday": False},
        {"label": "4D", "forecast_steps": 4, "hold_days": 4, "requires_intraday": False},
        {"label": "5D", "forecast_steps": 5, "hold_days": 5, "requires_intraday": False},
        # Sub-day horizons — enabled automatically once intraday data is present.
        {"label": "30m", "forecast_steps": 1, "hold_days": 0, "requires_intraday": True},
        {"label": "2h", "forecast_steps": 1, "hold_days": 0, "requires_intraday": True},
    ]

    @property
    def ACTIVE_TIMEFRAMES(self) -> list[dict]:
        """Timeframes we actually emit now — sub-day ones are skipped until a feed exists."""
        return [tf for tf in self.TIMEFRAMES if not tf["requires_intraday"]]

    # ------------------------------------------------------------------ #
    # Trading policy                                                       #
    # ------------------------------------------------------------------ #
    @property
    def BUY_ONLY(self) -> bool:
        """When true, the pipeline only surfaces BUY signals (initial phase)."""
        return os.getenv("BUY_ONLY", "true").lower() in ("1", "true", "yes")

    @property
    def CASH_AVAILABLE(self) -> float:
        """Deployable capital (₹) — used to annotate affordability of buy signals.
        Raised from the original ₹500 (mathematically insolvent — most signals
        resolved to 0 affordable shares, see WHAT_TO_DO_NEXT.txt 2.3) to ₹100,000
        as part of the "epoch 2" funding reset. See intelligence/trading_account.py
        get_current_epoch_start() — stats computed before this reset are excluded
        from the live PAPER->LIVE gate."""
        return float(os.getenv("CASH_AVAILABLE", "100000"))

    @property
    def CASH_RESERVE(self) -> float:
        """Reserve capital (₹) held back, not deployed. ~10% of CASH_AVAILABLE by
        default, leaving ~90% deployable."""
        return float(os.getenv("CASH_RESERVE", "10000"))

    @property
    def CLAUDE_SYNTHESIS_ENABLED(self) -> bool:
        """Whether to run the Claude CLI final-synthesis stage on top signals."""
        return os.getenv("CLAUDE_SYNTHESIS_ENABLED", "true").lower() in ("1", "true", "yes")

    # ------------------------------------------------------------------ #
    # Anthropic / Claude                                                   #
    # ------------------------------------------------------------------ #
    @property
    def ANTHROPIC_API_KEY(self) -> str:
        return os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def CLAUDE_SONNET_MODEL(self) -> str:
        return os.getenv("CLAUDE_SONNET_MODEL", "claude-sonnet-4-6")

    @property
    def CLAUDE_OPUS_MODEL(self) -> str:
        return os.getenv("CLAUDE_OPUS_MODEL", "claude-opus-4-8")

    @property
    def CLAUDE_CLI_EFFORT(self) -> str:
        """--effort passed to every `claude -p` call (low|medium|high|xhigh|max).
        Default low: this runs dozens of times/day (every 30-min pipeline cycle,
        EOD review, nightly calibration) — token cost compounds fast, and none
        of these calls need max reasoning depth to do a bounded JSON task."""
        return os.getenv("CLAUDE_CLI_EFFORT", "low")

    # ------------------------------------------------------------------ #
    # Synthesis layer — pluggable CLI backend (claude | codex | gemini)   #
    # Defaults below are fallbacks; the live choice is stored in the       #
    # app_config table and read via intelligence/provider_config.py.       #
    # ------------------------------------------------------------------ #
    @property
    def LLM_SYNTH_BACKEND(self) -> str:
        return os.getenv("LLM_SYNTH_BACKEND", "claude").lower()

    # codex CLI (OpenAI). Empty model = let codex use its configured default.
    @property
    def CODEX_FAST_MODEL(self) -> str:
        return os.getenv("CODEX_FAST_MODEL", "")

    @property
    def CODEX_DEEP_MODEL(self) -> str:
        return os.getenv("CODEX_DEEP_MODEL", "")

    # gemini CLI (Google).
    @property
    def GEMINI_FAST_MODEL(self) -> str:
        return os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash")

    @property
    def GEMINI_DEEP_MODEL(self) -> str:
        return os.getenv("GEMINI_DEEP_MODEL", "gemini-2.5-pro")

    # ------------------------------------------------------------------ #
    # Groww (primary live feed — TOTP auth, no IP whitelisting)            #
    # ------------------------------------------------------------------ #
    @property
    def GROWW_API_KEY(self) -> str:
        return os.getenv("GROWW_API_KEY", "")

    @property
    def GROWW_API_SECRET(self) -> str:
        return os.getenv("GROWW_API_SECRET", "")

    @property
    def GROWW_TOTP_SECRET(self) -> str:
        """TOTP secret from the Groww API page — preferred: no daily approval, no IP binding."""
        return os.getenv("GROWW_TOTP_SECRET", "")

    # ------------------------------------------------------------------ #
    # Upstox (live data layer — REST candles + WS V3 ltpc feed)            #
    # ------------------------------------------------------------------ #
    @property
    def UPSTOX_CLIENT_ID(self) -> str:
        return os.getenv("UPSTOX_CLIENT_ID", "")

    @property
    def UPSTOX_CLIENT_SECRET(self) -> str:
        return os.getenv("UPSTOX_CLIENT_SECRET", "")

    @property
    def UPSTOX_REDIRECT_URI(self) -> str:
        return os.getenv("UPSTOX_REDIRECT_URI", "http://localhost:8000/api/upstox/callback")

    @property
    def UPSTOX_ANALYTICS_TOKEN(self) -> str:
        """Generated once from the Upstox developer portal, no daily re-auth needed.
        Powers Market Data + Realtime & Streaming APIs — preferred over the
        daily OAuth token for everything this app's data layer needs. See
        data/pipeline/upstox_client.get_data_token()."""
        return os.getenv("UPSTOX_ANALYTICS_TOKEN", "")

    @property
    def UPSTOX_SANDBOX_TOKEN(self) -> str:
        """Separate access token for api-sandbox.upstox.com/v2 — risk-free order
        rehearsal, no real money, no market impact. Generated from a
        dedicated 'Sandbox' section of the developer portal, distinct from
        both the live OAuth token and UPSTOX_ANALYTICS_TOKEN above. Required
        for backend/routers/confirmations.py's approve() to actually place
        an order; without it, approve() records the decision but the order
        placement step reports unavailable rather than silently no-op'ing."""
        return os.getenv("UPSTOX_SANDBOX_TOKEN", "")

    @property
    def LIVE_CONFIRMATION_ENABLED(self) -> bool:
        """Opt-in flag: when true, intelligence/live_confirmation.py queues
        qualifying fresh signals into pending_trade_confirmations for human
        approve/reject. Default OFF — nothing queues for your review until
        you explicitly turn this on."""
        return os.getenv("LIVE_CONFIRMATION_ENABLED", "false").lower() in ("1", "true", "yes")

    @property
    def EXECUTION_MODE(self) -> str:
        """"paper" | "sandbox" | "live" — makes the existing implicit modes
        explicit. Setting this to "live" does NOT by itself unlock real-money
        orders: place_live_order() (data/pipeline/upstox_orders.py) still
        hard-checks intelligence.trading_mode.get_trading_mode()'s expectancy
        gate (60 executed trades, 28 days, 50 resolved outcomes) and the kill
        switch before ever calling the real order API — this setting only
        selects which code path CAN run if that gate has separately been
        earned. Default "sandbox" (current behavior, unchanged)."""
        return os.getenv("EXECUTION_MODE", "sandbox").lower()

    @property
    def UPSTOX_LIVE_ORDER_TOKEN(self) -> str:
        """Real-money order-capable access token — deliberately separate from
        UPSTOX_ANALYTICS_TOKEN (data-only) and UPSTOX_SANDBOX_TOKEN (fake
        money). Empty until the user deliberately generates and sets one;
        place_live_order() refuses to run without it regardless of
        EXECUTION_MODE or the expectancy gate."""
        return os.getenv("UPSTOX_LIVE_ORDER_TOKEN", "")

    @property
    def MAX_SANDBOX_ORDER_VALUE(self) -> float:
        """Sanity ceiling (₹, quantity * price) enforced in place_sandbox_order()
        before any order is sent to Upstox sandbox. This is fake money and a
        rejected sandbox order costs nothing real — but a runaway loop or a bad
        `shares_affordable` calculation placing a 10,000-share order is still a
        real bug worth catching loudly instead of silently rehearsing it.
        Default ₹50,000 is well above any single-position size CASH_AVAILABLE
        (₹100,000 total capital) should ever produce under normal position
        sizing; deliberately not "unlimited" even in sandbox."""
        return float(os.getenv("MAX_SANDBOX_ORDER_VALUE", "50000"))

    @property
    def SANDBOX_MAX_POSITION_PCT(self) -> float:
        """Fraction of sandbox funds committed to a single trade — deliberately
        SEPARATE from brain_params' max_position_pct (~21%, adaptive, tuned for
        the ₹100,000 PAPER ledger's ~5-8-position diversification). Reusing
        that value here was a real bug (found live 2026-07-20): on a ₹1,000
        real account it capped every trade's budget at ~₹210, silently
        excluding every stock priced above that regardless of signal
        quality — the observed "only ever getting sub-₹50 stocks" symptom
        was a sizing bug, not a model bias.

        Default 0.40 (not 0.21, and NOT 1.00 — the user explicitly does not
        want everything concentrated into one position either): on ₹1,000
        that's a ~₹400 per-trade budget, wide enough to afford real
        mid-priced stocks while still leaving room for 2 concurrent
        positions (queue_fresh_signals already decrements remaining_budget
        per candidate within a batch, and check_position_limit-style
        capping should gate total concurrent count — see the sandbox
        position-cap TODO). As real funds grow, this stays a genuine
        percentage, so the same logic scales up without redeploying."""
        return float(os.getenv("SANDBOX_MAX_POSITION_PCT", "0.40"))

    @property
    def SANDBOX_MAX_OPEN_POSITIONS(self) -> int:
        """Hard cap on concurrent sandbox-held tickers (net position > 0).
        Previously unlimited — nothing stopped queue_fresh_signals from
        proposing a new position every single cycle regardless of how many
        were already open. At SANDBOX_MAX_POSITION_PCT=0.40 this caps total
        deployed sandbox exposure at roughly 2-3x the per-trade budget, a
        genuinely diversified handful of real positions rather than either
        one all-in trade or an unbounded pile of tiny ones."""
        return int(os.getenv("SANDBOX_MAX_OPEN_POSITIONS", "3"))

    @property
    def UPSTOX_FUNDS_CACHE_SECONDS(self) -> int:
        """How long a fetched Upstox available-funds figure may be reused
        within a single queue_fresh_signals() batch (intelligence/
        live_confirmation.py) before re-fetching. Short by design — this is
        real account money sizing real (sandbox) orders, so staleness should
        be measured in seconds, not minutes; the point of the cache is only
        to avoid hitting the funds endpoint once per candidate ticker within
        the same run, not to avoid hitting it across runs. Default 30s."""
        return int(os.getenv("UPSTOX_FUNDS_CACHE_SECONDS", "30"))

    @property
    def SANDBOX_VIRTUAL_CAPITAL(self) -> float:
        """TEMPORARY STOPGAP — DELETE THIS SETTING ONCE REAL FUNDS ARE READABLE.

        get_available_funds() (data/pipeline/upstox_orders.py) currently
        cannot read your real Upstox balance: sandbox has no funds/margin
        endpoint at all, and the live Analytics-Token funds endpoint returns
        UDAPI1221 ("permitted only from the static IP configured in your
        account") because no static IP is registered. Until one is set up
        (SEBI allows changing it once/week — see docs/UPSTOX_API_NOTES.md),
        intelligence/live_confirmation.py falls back to sizing sandbox
        rehearsal trades against THIS number instead of skipping queueing
        entirely. It is NOT your real balance — every reasoning string and
        Telegram message built from this fallback is labeled
        "[SANDBOX VIRTUAL CAPITAL]" so it can never be mistaken for a real
        funds read. Once a static IP is registered and get_available_funds()
        returns status "ok", this setting becomes dead code — remove it,
        the SANDBOX_VIRTUAL_CAPITAL branch in queue_fresh_signals(), and this
        docstring together."""
        return float(os.getenv("SANDBOX_VIRTUAL_CAPITAL", "100000"))

    @property
    def CONFIRMATION_EXPIRY_MINUTES(self) -> int:
        """How long a PENDING row in pending_trade_confirmations stays valid
        before its quoted price is considered stale. Default 120 minutes (2h)
        — long enough to review a trade without being glued to the screen,
        short enough that the quoted price/target/ETA reasoning hasn't drifted
        far from reality. Enforced twice: a scheduler job flips old PENDING
        rows to EXPIRED (see scheduler/market_runner.py task_expire_confirmations),
        and approve() independently re-checks row age at click time in case the
        human's browser tab was open with a stale list."""
        return int(os.getenv("CONFIRMATION_EXPIRY_MINUTES", "120"))

    # ------------------------------------------------------------------ #
    # Telegram (approve/reject control surface for the confirmation queue) #
    # ------------------------------------------------------------------ #
    @property
    def TELEGRAM_BOT_TOKEN(self) -> str:
        """Bot token from @BotFather. Empty = Telegram surface disabled
        (backend/services/telegram_bot.py becomes a clean no-op)."""
        return os.getenv("TELEGRAM_BOT_TOKEN", "")

    @property
    def TELEGRAM_CHAT_ID(self) -> str:
        """The ONE chat id the bot talks to and listens to. Updates from any
        other chat/user/group are silently ignored (logged, never answered)
        — trade data must never leak to an unverified chat."""
        return os.getenv("TELEGRAM_CHAT_ID", "")

    # ------------------------------------------------------------------ #
    # Angel One (Phase 2 — optional live trading)                          #
    # ------------------------------------------------------------------ #
    @property
    def ANGEL_ONE_API_KEY(self) -> str:
        return os.getenv("ANGEL_ONE_API_KEY", "")

    @property
    def ANGEL_ONE_CLIENT_ID(self) -> str:
        return os.getenv("ANGEL_ONE_CLIENT_ID", "")

    @property
    def ANGEL_ONE_PIN(self) -> str:
        return os.getenv("ANGEL_ONE_PIN", "")

    @property
    def ANGEL_ONE_TOTP_KEY(self) -> str:
        return os.getenv("ANGEL_ONE_TOTP_KEY", "")

    # ------------------------------------------------------------------ #
    # Filesystem paths                                                     #
    # ------------------------------------------------------------------ #
    @property
    def DATA_DIR(self) -> Path:
        return Path(os.getenv("DATA_DIR", "./data"))

    @property
    def MODELS_DIR(self) -> Path:
        return Path(os.getenv("MODELS_DIR", "./models"))

    @property
    def NSE_DATA_DIR(self) -> Path:
        return Path(os.getenv("NSE_DATA_DIR", "./data/nse_raw"))


settings = Settings()
