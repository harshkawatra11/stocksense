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
        """Deployable capital (₹) — used to annotate affordability of buy signals."""
        return float(os.getenv("CASH_AVAILABLE", "500"))

    @property
    def CASH_RESERVE(self) -> float:
        """Reserve capital (₹) held back, not deployed."""
        return float(os.getenv("CASH_RESERVE", "500"))

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

    # ------------------------------------------------------------------ #
    # Kronos forecast                                                      #
    # ------------------------------------------------------------------ #
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
