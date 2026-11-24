"""Load daily and intraday bars of one symbol from our stored data as plain-numpy frames.

Daily bars are the split-adjusted panel (`adj_*`), completed sessions only, so Market Lab's forward-looking
35% split filter is not needed and today's unfinished bar can never produce a signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np
import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.data import io
from stocksense.data.ingest_intraday import load_bars
from stocksense.marketlab import npx
from stocksense.marketlab.bars import Bars


@dataclass(frozen=True)
class DailyFrame:
    """Daily OHLCV in ascending date order (dates are datetime64[D])."""

    date: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return int(len(self.date))

    def mirrored(self) -> DailyFrame:
        """Upside-down copy: a short setup on this frame is a long setup on the mirror."""
        return DailyFrame(self.date, -self.open, -self.low, -self.high, -self.close, self.volume)


@dataclass(frozen=True)
class NiftyTable:
    """Nifty 50 daily close, its 200-day mean and 63-day rate of change in percent, by date."""

    date: np.ndarray
    c: np.ndarray
    sma200: np.ndarray
    roc63: np.ndarray

    @staticmethod
    def from_close(dates: np.ndarray, close: np.ndarray) -> NiftyTable:
        """Build the table from a dated close series (duplicate dates keep the last)."""
        keep = np.r_[dates[1:] != dates[:-1], True] if len(dates) else np.array([], bool)
        d, c = dates[keep], np.asarray(close, dtype=float)[keep]
        return NiftyTable(d, c, npx.roll(c, 200, "mean"), npx.pct_change(c, 63) * 100)

    def reindex(self, dates: np.ndarray) -> dict[str, np.ndarray]:
        """The table's columns on exactly `dates` (NaN where the date is missing)."""
        pos = np.minimum(np.searchsorted(self.date, dates), max(len(self.date) - 1, 0))
        ok = (self.date[pos] == dates) if len(self.date) else np.zeros(len(dates), bool)
        return {k: np.where(ok, getattr(self, k)[pos], np.nan) for k in ("c", "sma200", "roc63")}


def daily_frame(
    settings: Settings, sym: str, end: date | None = None, adjusted: bool = True
) -> DailyFrame | None:
    """Completed sessions of one symbol from panel_daily (rows with no volume dropped); None when absent."""
    cols = ["date", "series", "volume", "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]
    raw = ["open", "high", "low", "close"]
    lf = io.read(settings, "panel_daily", start=None, end=end, columns=[*cols, *raw, "symbol"])
    df = lf.filter((pl.col("symbol") == sym) & (pl.col("series") == "EQ")).sort("date").collect()
    df = df.unique(subset=["date"], keep="last", maintain_order=True)
    if not df.height:
        return None
    pre = "adj_" if adjusted else ""
    vol = pl.col("adj_volume") if adjusted else pl.col("volume").cast(pl.Float64)
    df = df.filter(vol > 0).drop_nulls([f"{pre}close"])
    if not df.height:
        return None

    def f(k: str) -> np.ndarray:
        return df[f"{pre}{k}"].to_numpy().astype(float)

    v = df["adj_volume"].to_numpy().astype(float) if adjusted else df["volume"].to_numpy().astype(float)
    return DailyFrame(
        np.array(df["date"].to_list(), dtype="datetime64[D]"), f("open"), f("high"), f("low"), f("close"), v
    )


def intraday_frame(
    settings: Settings, sym: str, start: date, end: date, interval: Literal["5m", "1m"] = "5m"
) -> Bars | None:
    """Adjusted intraday bars of one symbol between two dates; None when there are none."""
    df = load_bars(settings, [sym], start, end, interval=interval, basis="adjusted")
    return Bars.from_polars(df.sort("ts")) if df.height else None


def nifty_table(settings: Settings, end: date | None = None) -> NiftyTable:
    """The Nifty 50 table from index_daily (completed sessions only)."""
    df = (
        io.read(settings, "index_daily", start=None, end=end, columns=["index_name", "date", "close"])
        .filter(pl.col("index_name") == "NIFTY 50")
        .sort("date")
        .collect()
    )
    return NiftyTable.from_close(
        np.array(df["date"].to_list(), dtype="datetime64[D]"), df["close"].to_numpy()
    )


def completed_end(clock: Clock, cal: TradingCalendar, settings: Settings) -> date:
    """Last session that has closed and whose bars are in panel_daily (never today's partial bar)."""
    last = cal.last_completed_session(clock.now())
    got = (
        io.read(settings, "panel_daily", start=last - timedelta(days=15), end=last, columns=["date"])
        .select(pl.col("date").max())
        .collect()
        .item()
    )
    return got if got is not None else last
