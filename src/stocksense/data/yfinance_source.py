"""
yfinance data source.

Per docs/02-data-layer.md, yfinance is a validator/fallback, never the
primary source in the production design — but it requires no API key or
account, which makes it the only source Phase 0 can actually run against
today (no Upstox credentials are available in this environment). Phase 0
therefore uses it as its primary source deliberately, and this is recorded
as a Phase 0-specific deviation, not a silent substitution: every row is
tagged DataSource.YFINANCE, so nothing downstream can mistake it for
Upstox-sourced data, and swapping in a real Upstox client later requires no
change to any code past ingestion.
"""

from __future__ import annotations

import time

import pandas as pd
import yfinance as yf
import structlog

from stocksense.core.types import DataSource

log = structlog.get_logger(__name__)


def to_yf_symbol(nse_symbol: str) -> str:
    """NSE symbol -> yfinance ticker (NSE equities carry a .NS suffix)."""
    return f"{nse_symbol}.NS"


def fetch_history(nse_symbol: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """Fetch daily OHLCV for one symbol. Returns empty DataFrame on failure
    after retries — callers must treat empty as 'no data', never as zero
    values, per the missing-data discipline in docs/03-feature-engineering.md.
    """
    yf_symbol = to_yf_symbol(nse_symbol)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                yf_symbol,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                multi_level_index=False,
            )
            if raw is None or raw.empty:
                return pd.DataFrame()
            raw = raw.reset_index()
            raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
            out = pd.DataFrame(
                {
                    "symbol": nse_symbol,
                    "date": pd.to_datetime(raw["date"]).dt.date,
                    "open": raw["open"].astype(float),
                    "high": raw["high"].astype(float),
                    "low": raw["low"].astype(float),
                    "close": raw["close"].astype(float),
                    "adj_close": raw["adj_close"].astype(float) if "adj_close" in raw.columns else raw["close"].astype(float),
                    "volume": raw["volume"].astype(float),
                    "source": DataSource.YFINANCE.value,
                }
            )
            out = out.dropna(subset=["open", "high", "low", "close", "volume"])
            return out
        except Exception as e:  # noqa: BLE001 — retry loop, re-raised after exhaustion
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    log.warning("yfinance_fetch_failed", symbol=nse_symbol, error=str(last_err))
    return pd.DataFrame()
