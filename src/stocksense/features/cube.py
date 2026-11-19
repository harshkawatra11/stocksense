"""The feature cube (G2 card D0.2): the whole feature panel once, as a float16 array for window lookups.

`data[date_idx, slot, f]` holds feature `columns[f]` of the name in `slot` on day `date_idx`; the sids of
each day are kept sorted in `sid_at[date_idx]` (padded with a sentinel), so a slot is found by binary
search. Cells that do not exist are NaN. `window(date_idx, sids, T)` returns the last T sessions ending at
`date_idx` (inclusive) for each sid, float32, zero-filled with a boolean mask where the name has no row on
that session (a NaN feature value inside an existing row stays NaN). Memory: 3,950 dates x 620 slots x 138
features x 2 bytes is about 0.68 GB.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import polars as pl

from stocksense.core.config import Settings
from stocksense.data import io
from stocksense.models.seqnet import date_to_days
from stocksense.validation.lockbox import guard

PAD_SID = np.iinfo(np.int32).max
_CHUNK = 200_000


class FeatureCube:
    """Features of feature store `version` (columns `columns`, purpose-guarded) as a dense cube."""

    def __init__(
        self,
        settings: Settings,
        version: int,
        columns: Sequence[str],
        purpose: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> None:
        self.columns = list(columns)
        lf = io.read(settings, "features", start=start, end=end, columns=["date", "sid", *self.columns],
                     sub=f"v{version}")  # fmt: skip
        df = guard(lf, "date", purpose, settings).collect().sort("date", "sid")
        self._fill(df)

    def _fill(self, df: pl.DataFrame) -> None:
        dates = df["date"].unique().sort()
        self.dates: np.ndarray = dates.to_numpy().astype("datetime64[D]")
        self.days: np.ndarray = date_to_days(self.dates)
        n_d, n_f = len(self.dates), len(self.columns)
        di = np.searchsorted(self.dates, df["date"].to_numpy().astype("datetime64[D]"))
        slot = df.select(pl.int_range(pl.len()).over("date").cast(pl.Int64))["literal"].to_numpy()
        self.n_sids_max = int(slot.max()) + 1 if len(slot) else 0
        self.data = np.full((n_d, self.n_sids_max, n_f), np.nan, dtype=np.float16)
        self.sid_at = np.full((n_d, self.n_sids_max), PAD_SID, dtype=np.int32)
        self.sid_at[di, slot] = df["sid"].to_numpy().astype(np.int32)
        for a in range(0, df.height, _CHUNK):
            part = df.slice(a, _CHUNK).select([pl.col(c).cast(pl.Float32) for c in self.columns])
            self.data[di[a : a + _CHUNK], slot[a : a + _CHUNK]] = part.to_numpy().astype(np.float16)

    # ---- index ---------------------------------------------------------------------------------
    def date_idx(self, dates: np.ndarray) -> np.ndarray:
        """Cube index of each date (datetime64 or date objects); -1 for a date that is not in the cube."""
        q = np.asarray(dates).astype("datetime64[D]")
        pos = np.minimum(np.searchsorted(self.dates, q), max(len(self.dates) - 1, 0))
        ok = (self.dates[pos] == q) if len(self.dates) else np.zeros(len(q), dtype=bool)
        return np.where(ok, pos, -1).astype(np.int64)

    def day_idx(self, days: np.ndarray) -> np.ndarray:
        """Cube index of each day-since-epoch value (as `add_keys` writes them); -1 if absent."""
        q = np.asarray(days, dtype=np.int64)
        pos = np.minimum(np.searchsorted(self.days, q), max(len(self.days) - 1, 0))
        ok = (self.days[pos] == q) if len(self.days) else np.zeros(len(q), dtype=bool)
        return np.where(ok, pos, -1).astype(np.int64)

    def slot(self, date_idx: np.ndarray, sids: np.ndarray) -> np.ndarray:
        """Slot of each sid on each day (-1 where the name has no row, or the day is outside the cube)."""
        di = np.asarray(date_idx, dtype=np.int64)
        sids = np.asarray(sids, dtype=np.int64)
        out = np.full(len(sids), -1, dtype=np.int64)
        inside = (di >= 0) & (di < len(self.dates))
        for d in np.unique(di[inside]):
            sel = np.flatnonzero(inside & (di == d))
            row = self.sid_at[d]
            pos = np.minimum(np.searchsorted(row, sids[sel]), max(self.n_sids_max - 1, 0))
            hit = row[pos] == sids[sel]
            out[sel[hit]] = pos[hit]
        return out

    # ---- windows -------------------------------------------------------------------------------
    def window(self, date_idx: np.ndarray | int, sids: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
        """(x float32 [n, T, F], mask bool [n, T]): the T sessions ending at `date_idx` (scalar or one per
        sid), oldest first. Where a name has no row (or the session is before the cube starts) x is 0 and
        the mask False."""
        sids = np.asarray(sids, dtype=np.int64)
        n = len(sids)
        di = np.broadcast_to(np.asarray(date_idx, dtype=np.int64), (n,))
        x = np.zeros((n, T, len(self.columns)), dtype=np.float32)
        mask = np.zeros((n, T), dtype=bool)
        for k in range(T):
            dk = di - (T - 1 - k)
            sl = self.slot(dk, sids)
            ok = sl >= 0
            if ok.any():
                x[ok, k] = self.data[dk[ok], sl[ok]].astype(np.float32)
                mask[ok, k] = True
        return x, mask
