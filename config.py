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
    # Each horizon produces its own signal per ticker per run. `kronos_steps`
    # is the forecast horizon fed to Kronos; `hold_days` is how long a signal
    # stays active before it's resolved/expired; `requires_intraday` marks
    # sub-day horizons that stay dormant until a live intraday feed exists.
    # Add/remove a dict here to add/remove a timeframe — nothing else hardcodes them.
    # Integer-day horizons are Kronos-backed (daily candles). Fractional precision
    # (e.g. "touches target in ~2.5 days") is delivered per-signal by target_eta_days,
    # NOT by separate half-day buckets — there's no half-day candle to forecast.
    TIMEFRAMES: list[dict] = [
        {"label": "1D", "kronos_steps": 1, "hold_days": 1, "requires_intraday": False},
        {"label": "2D", "kronos_steps": 2, "hold_days": 2, "requires_intraday": False},
        {"label": "3D", "kronos_steps": 3, "hold_days": 3, "requires_intraday": False},
        {"label": "4D", "kronos_steps": 4, "hold_days": 4, "requires_intraday": False},
        {"label": "5D", "kronos_steps": 5, "hold_days": 5, "requires_intraday": False},
        # Sub-day horizons — enabled automatically once intraday data is present.
        {"label": "30m", "kronos_steps": 1, "hold_days": 0, "requires_intraday": True},
        {"label": "2h", "kronos_steps": 1, "hold_days": 0, "requires_intraday": True},
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
    # Kronos forecast — ARCHIVED (see WHAT_TO_DO_NEXT.txt Section 2.1/2.2 and
    # Stage 3 of quiet-giggling-bumblebee.md). Kronos never ran fine-tuned on
    # NSE data; what was live was either off-domain pretrained weights or a
    # random-walk mock. Dropped in favor of a 3-seed LightGBM ensemble +
    # quantile regressors (models/ml/train.py, models/ml/predict.py). Code is
    # left intact (not deleted) behind this flag, same archiving convention
    # as the Groww retirement — default OFF. Flip KRONOS_ENABLED=true only to
    # A/B a resurrected, properly fine-tuned Kronos against the ensemble in
    # the walk-forward harness.
    # ------------------------------------------------------------------ #
    @property
    def KRONOS_ENABLED(self) -> bool:
        return os.getenv("KRONOS_ENABLED", "false").lower() in ("1", "true", "yes")

    @property
    def KRONOS_FORECAST_STEPS(self) -> int:
        return int(os.getenv("KRONOS_FORECAST_STEPS", "5"))

    @property
    def KRONOS_MIN_CANDLES(self) -> int:
        return int(os.getenv("KRONOS_MIN_CANDLES", "30"))

    @property
    def KRONOS_MODEL_SIZE(self) -> str:
        return os.getenv("KRONOS_MODEL_SIZE", "mini")

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
        """Separate access token for sandbox.upstox.com/v2 — risk-free order
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
        you explicitly turn this on, same pattern as KRONOS_ENABLED."""
        return os.getenv("LIVE_CONFIRMATION_ENABLED", "false").lower() in ("1", "true", "yes")

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
