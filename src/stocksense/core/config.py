"""Typed settings, loaded from .env.

Every credential lives in .env (gitignored) and is read through here -- no module
reaches for os.environ directly. `redacted()` exists because these values get
serialised into LLM prompts and log lines, and a leaked TOTP secret is a leaked
brokerage account.
"""

from __future__ import annotations

import functools
from datetime import time
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]

# Names whose VALUES must never appear in a log line, an LLM prompt, or an API
# response. Checked by tests/unit/test_config.py against the real field list.
_SECRET_FIELDS = frozenset(
    {
        "upstox_api_secret",
        "upstox_access_token",
        "angel_api_key",
        "angel_password",
        "angel_totp_secret",
        "angel_client_code",
    }
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="STOCKSENSE_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- storage ----
    data_store: Path = REPO_ROOT / "data_store"
    duckdb_path: Path = REPO_ROOT / "data_store" / "stocksense.duckdb"
    parquet_root: Path = REPO_ROOT / "data_store" / "parquet"

    # ---- market data ----
    upstox_api_key: str | None = None
    upstox_api_secret: str | None = None
    upstox_access_token: str | None = None

    # ---- broker ----
    angel_api_key: str | None = None
    angel_client_code: str | None = None
    angel_password: str | None = None
    angel_totp_secret: str | None = None

    # ---- session (IST). 15:10 square-off is deliberately ahead of the broker's
    # ~15:20 MIS auto-square-off, so we close on our own terms, not theirs. ----
    market_open: time = time(9, 15)
    market_close: time = time(15, 30)
    squareoff_time: time = time(15, 10)
    first_signal_time: time = time(9, 30)  # 09:15-09:30 is deliberately silent

    # ---- capital and risk. Defaults are the user's real numbers; every one of
    # these is an input to simulation/sizing.py, never an output of it. ----
    equity_inr: float = 17_500.0
    max_leverage: float = 5.0
    max_open_positions: int = 2
    max_orders_per_day: int = 8
    daily_loss_limit_inr: float = 700.0  # HARD stop. Re-derived from Q5 once measured.

    # ---- LLM ----
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "qwen2.5"
    ollama_embed_model: str = "nomic-embed-text"
    obsidian_vault: Path = Path.home() / "Documents" / "Obsidian Vault"

    # ---- compute ----
    search_workers: int = 10  # of 12 threads; 2 reserved for the UI and OS
    gpu_vram_budget_mb: int = 2500  # leave headroom for the display and Ollama

    log_level: str = Field(default="INFO")

    def redacted(self) -> dict[str, object]:
        """Settings as a dict with every secret replaced by a presence marker.

        Use this -- never `model_dump()` -- anywhere the result can reach a log,
        an LLM prompt, or an HTTP response.
        """
        out: dict[str, object] = {}
        for name, value in self.model_dump().items():
            if name in _SECRET_FIELDS:
                out[name] = "<set>" if value else "<unset>"
            else:
                out[name] = value
        return out


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
