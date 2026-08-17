"""
Central configuration. Every path, threshold, and cost assumption used by
Phase 0 lives here — nothing hardcoded downstream, so a sweep can vary any
of it without touching pipeline code.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_STORE = REPO_ROOT / "data_store"
RESEARCH_DIR = REPO_ROOT / "research"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STOCKSENSE_", env_file=".env", extra="ignore")

    # storage
    duckdb_path: Path = DATA_STORE / "stocksense.duckdb"
    parquet_dir: Path = DATA_STORE / "parquet"

    # universe / data
    min_history_days: int = 300  # ticker must have at least this many rows to be used
    min_price_inr: float = 5.0  # exclude penny stocks below this price
    min_avg_daily_turnover_inr: float = 5_000_000.0  # ~50 lakh/day liquidity floor

    # Phase D2: which price source and universe the pipeline trains/backtests
    # against. Defaults preserve the Phase 0 path unchanged (candles=yfinance,
    # fixed 98-symbol PHASE0_UNIVERSE via quarantine_symbols) so the old
    # numbers stay reproducible for direct A/B comparison against the
    # bhavcopy re-run (research/preregistration_bhavcopy_rerun.md).
    price_source: str = "candles"  # "candles" (yfinance) | "bhavcopy" (NSE, point-in-time)
    return_basis: str = "price"    # "price" (splits/bonuses only) | "total" (also dividends)
    use_point_in_time_universe: bool = False  # only meaningful when price_source="bhavcopy"

    # Upstox (Phase 0 uses yfinance; these are read only if present)
    upstox_api_key: str | None = None
    upstox_api_secret: str | None = None
    upstox_access_token: str | None = None

    # cost sweep grid (round-trip, in basis points)
    cost_grid_bps: tuple[float, ...] = (10.0, 15.0, 25.0, 35.0)

    # horizon sweep grid (in trading bars)
    horizon_grid: tuple[int, ...] = (1, 3, 5, 10, 20)

    # selectivity sweep grid (top-N by cross-sectional rank)
    top_n_grid: tuple[int, ...] = (10, 20, 50, 100)

    random_seed: int = 42

    def ensure_dirs(self) -> None:
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
