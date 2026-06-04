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
    # Anthropic / Claude                                                   #
    # ------------------------------------------------------------------ #
    @property
    def ANTHROPIC_API_KEY(self) -> str:
        return os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def CLAUDE_SONNET_MODEL(self) -> str:
        return os.getenv("CLAUDE_SONNET_MODEL", "claude-sonnet-4-5")

    @property
    def CLAUDE_OPUS_MODEL(self) -> str:
        return os.getenv("CLAUDE_OPUS_MODEL", "claude-opus-4-5")

    # ------------------------------------------------------------------ #
    # Angel One (Phase 2 — optional live trading)                          #
    # ------------------------------------------------------------------ #
    @property
    def ANGEL_ONE_API_KEY(self) -> str:
        return os.getenv("ANGEL_ONE_API_KEY", "")

    @property
    def ANGEL_ONE_CLIENT_ID(self) -> str:
        return os.getenv("ANGEL_ONE_CLIENT_ID", "")

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
