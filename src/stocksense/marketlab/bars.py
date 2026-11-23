"""A plain-numpy OHLCV bar series used by the Market Lab ports (no pandas)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

IST_OFFSET_S = 19800  # IST is UTC+05:30 all year


@dataclass(frozen=True)
class Bars:
    """Bars in ascending time order; `ts` is epoch seconds (UTC); prices and volume are float arrays."""

    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return int(len(self.ts))

    @staticmethod
    def from_arrays(
        ts: object, open_: object, high: object, low: object, close: object, volume: object | None = None
    ) -> Bars:
        """Build from array-likes; a missing volume becomes zeros."""
        t = np.asarray(ts, dtype=np.int64)
        v = np.zeros(len(t)) if volume is None else np.asarray(volume, dtype=float)
        return Bars(
            t,
            np.asarray(open_, dtype=float),
            np.asarray(high, dtype=float),
            np.asarray(low, dtype=float),
            np.asarray(close, dtype=float),
            v,
        )

    @staticmethod
    def from_polars(df: pl.DataFrame) -> Bars:
        """From a frame with `ts` (Datetime, or epoch seconds as ints) plus open, high, low, close, volume."""
        ts = df["ts"]
        epoch = (
            ts.dt.epoch("s")
            if ts.dtype in (pl.Datetime, pl.Date) or str(ts.dtype).startswith("Datetime")
            else ts
        )
        return Bars.from_arrays(
            epoch.to_numpy(),
            df["open"].to_numpy(),
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            df["close"].to_numpy(),
            df["volume"].to_numpy() if "volume" in df.columns else None,
        )

    def day_index(self) -> np.ndarray:
        """IST calendar day number (days since 1970-01-01) of every bar."""
        return (self.ts + IST_OFFSET_S) // 86400

    def minute_of_day(self) -> np.ndarray:
        """Minutes since IST midnight of every bar."""
        return ((self.ts + IST_OFFSET_S) % 86400) // 60

    def take(self, sel: np.ndarray | slice) -> Bars:
        """Rows picked by a boolean mask, index array or slice."""
        return Bars(
            self.ts[sel], self.open[sel], self.high[sel], self.low[sel], self.close[sel], self.volume[sel]
        )

    def tail(self, n: int) -> Bars:
        """The last n bars."""
        return self.take(slice(-n, None)) if n < len(self) else self


def day_to_date(day_number: int) -> date:
    """Calendar date of an IST day number."""
    return date(1970, 1, 1) + timedelta(days=int(day_number))


def weekday(day_number: int) -> int:
    """Monday=0 .. Sunday=6 of an IST day number (1970-01-01 was a Thursday)."""
    return int((int(day_number) + 3) % 7)


def hm(ts_epoch_s: int) -> str:
    """'HH:MM' of an epoch second in IST."""
    m = int(((int(ts_epoch_s) + IST_OFFSET_S) % 86400) // 60)
    return f"{m // 60:02d}:{m % 60:02d}"
