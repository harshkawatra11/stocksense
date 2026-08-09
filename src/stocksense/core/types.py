"""Typed domain objects shared across modules."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class DataSource(str, Enum):
    """Provenance tag. Every ingested field records which of these produced
    it (docs/02-data-layer.md), so a per-source bias is traceable rather
    than absorbed silently into the store."""

    UPSTOX = "upstox"
    NSE_ARCHIVE = "nse_archive"
    YFINANCE = "yfinance"


class OHLCVBar(BaseModel):
    """One instrument, one bar. `adj_close` is corporate-action-adjusted
    (used for feature/label computation); `close` is what actually printed
    (used for display and cost calculation). Conflating the two is a
    classic source of silent training-data corruption — see
    docs/02-data-layer.md's note on adjusted vs unadjusted closes."""

    symbol: str
    date: str  # ISO date, daily resolution for Phase 0
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float
    source: DataSource
