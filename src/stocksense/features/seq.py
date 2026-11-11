"""Sequence tensors for the project's own network (PLAN T7.3): 60 sessions x 12 channels per (date, sid).

Channels (all causal, from the panel only):
 0 la          ln(1 + tret)                      total return, close to close
 1 lco         ln(1 + ret_co)                    overnight gap
 2 loc         ln(1 + ret_oc)                    open to close
 3 range       ln(adj_high / adj_low)
 4 close_loc   (adj_close - adj_low) / (adj_high - adj_low), 0.5 on a flat bar
 5 turn_shock  ln(turnover / mean of the sid's last 20 turnovers)
 6 deliv_z     delivery % z-scored on the sid's last 60 rows (0 if null)
 7 mkt         ln(1 + ew500_ret)
 8 mkt_vol_21  std of ew500_ret over the last 21 sessions
 9 vix_chg_1   ln(vix_t / vix_{t-1}) (0 when no VIX file)
10 rel         la - mkt
11 vol_shock   ln(adj_volume / mean of the sid's last 60 adj_volume)

Return-like channels (0, 1, 2, 3, 10) are divided by the stock's own vol_63 (std of la over 63 rows) as of
the row 60 rows before the window end, then everything is clipped to +-10. A window is dropped when it has
fewer than 60 rows, spans more than ceil(1.25 * 60) + 2 sessions (the wide-span rule), or has no scale.
Stored per year as float16 `features/seq/v<N>/X_<YYYY>.npy` [n, 60, 12] plus `index_<YYYY>.parquet`
(row, date, sid).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

from stocksense.core.config import Settings
from stocksense.core.paths import Paths
from stocksense.data import io

WINDOW = 60
N_CH = 12
RETURN_LIKE = (0, 1, 2, 3, 10)
CLIP = 10.0
MAX_SPAN = math.ceil(1.25 * WINDOW) + 2
CHANNELS = (
    "la", "lco", "loc", "range", "close_loc", "turn_shock", "deliv_z", "mkt", "mkt_vol_21", "vix_chg_1",
    "rel", "vol_shock",
)  # fmt: skip


def compute_channels(
    daily: pl.DataFrame, market: pl.DataFrame, vix: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Per (sid, date): sx (session index), the 12 channels, `vol63` and (date, sid). Sorted by sid, date.

    `daily` needs sid, date, tret, ret_co, ret_oc, adj_high, adj_low, adj_close, adj_volume, turnover_inr,
    deliv_pct; `market` needs date, ew500_ret; `vix` (optional) date, close. Every value at date t uses
    rows at or before t only.
    """
    sessions = daily["date"].unique().sort()
    sx = pl.DataFrame({"date": sessions, "sx": np.arange(len(sessions), dtype=np.int64)})
    m = market.select("date", "ew500_ret").sort("date")
    m = m.with_columns(
        pl.col("ew500_ret").rolling_std(21, min_samples=10).alias("mkt_vol_21"),
        (1.0 + pl.col("ew500_ret")).log().alias("mkt"),
    )
    if vix is not None and vix.height:
        v = vix.select("date", pl.col("close")).sort("date")
        v = v.select("date", (pl.col("close") / pl.col("close").shift(1)).log().alias("vix_chg_1"))
        m = m.join(v, on="date", how="left")
    else:
        m = m.with_columns(pl.lit(None, dtype=pl.Float64).alias("vix_chg_1"))
    df = (
        daily.sort("sid", "date")
        .join(sx, on="date", how="left")
        .join(m.select("date", "mkt", "mkt_vol_21", "vix_chg_1"), on="date", how="left")
    )
    hl = pl.col("adj_high") - pl.col("adj_low")
    la = (1.0 + pl.col("tret")).log()
    df = df.with_columns(
        la.alias("la"),
        (1.0 + pl.col("ret_co")).log().alias("lco"),
        (1.0 + pl.col("ret_oc")).log().alias("loc"),
        (pl.col("adj_high") / pl.col("adj_low")).log().alias("range"),
        pl.when(hl > 0)
        .then((pl.col("adj_close") - pl.col("adj_low")) / hl)
        .otherwise(0.5)
        .alias("close_loc"),
        (
            pl.col("turnover_inr")
            / pl.col("turnover_inr").rolling_mean(20, min_samples=5).over("sid").clip(lower_bound=1.0)
        )
        .clip(lower_bound=1e-6)
        .log()
        .alias("turn_shock"),
        (
            (pl.col("deliv_pct") - pl.col("deliv_pct").rolling_mean(60, min_samples=20).over("sid"))
            / pl.col("deliv_pct").rolling_std(60, min_samples=20).over("sid")
        )
        .fill_nan(None)
        .fill_null(0.0)
        .alias("deliv_z"),
        (la - pl.col("mkt")).alias("rel"),
        (
            pl.col("adj_volume")
            / pl.col("adj_volume").rolling_mean(60, min_samples=20).over("sid").clip(lower_bound=1.0)
        )
        .clip(lower_bound=1e-6)
        .log()
        .alias("vol_shock"),
        la.rolling_std(63, min_samples=40).over("sid").alias("vol63"),
    )
    cols = ["sid", "date", "sx", *CHANNELS, "vol63"]
    return df.select(cols).with_columns(pl.col("vix_chg_1").fill_null(0.0))


def window_rows(sid: np.ndarray, sx: np.ndarray, vol63: np.ndarray) -> np.ndarray:
    """Indices (into the sid-sorted frame) of rows that end a valid window: 60 rows of the same sid, span
    within MAX_SPAN, and a vol_63 scale at the row 60 rows earlier."""
    n = len(sid)
    i = np.arange(n)
    ok = i >= WINDOW
    j = np.maximum(i - (WINDOW - 1), 0)
    ok &= sid[j] == sid
    ok &= (sx - sx[j] + 1) <= MAX_SPAN
    k = np.maximum(i - WINDOW, 0)
    ok &= (sid[k] == sid) & np.isfinite(vol63[k]) & (vol63[k] > 0)
    return np.flatnonzero(ok)


def make_windows(chan: np.ndarray, vol63: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """float16 [n, 60, 12] windows ending at `rows`: return-like channels scaled, clipped, NaN -> 0."""
    idx = rows[:, None] - (WINDOW - 1) + np.arange(WINDOW)[None, :]
    w = chan[idx].astype(np.float32)  # [n, 60, 12]
    scale = vol63[rows - WINDOW].astype(np.float32)
    for c in RETURN_LIKE:
        w[:, :, c] /= scale[:, None]
    w = np.nan_to_num(w, nan=0.0, posinf=CLIP, neginf=-CLIP)
    return np.clip(w, -CLIP, CLIP).astype(np.float16)


def seq_root(settings: Settings, version: int) -> Path:
    return Paths(settings).features / "seq" / f"v{version}"


def build_sequence_tensors(settings: Settings, version: int = 1) -> dict[str, int]:
    """Write the per-year tensors for every u500 (date, sid) with a valid window; returns rows per year."""
    daily = io.read(settings, "panel_daily").collect()
    market = io.read(settings, "panel_market").collect()
    try:
        vix: pl.DataFrame | None = io.read(settings, "india_vix").collect()
    except Exception:  # no VIX file yet: the channel is 0  # noqa: BLE001
        vix = None
    uni = io.read(settings, "panel_universe", columns=["date", "sid", "u500"]).collect()
    return build_from_frames(
        settings, version, daily, market, vix, uni.filter(pl.col("u500")).select("date", "sid")
    )


def build_from_frames(
    settings: Settings,
    version: int,
    daily: pl.DataFrame,
    market: pl.DataFrame,
    vix: pl.DataFrame | None,
    members: pl.DataFrame,
) -> dict[str, int]:
    ch = compute_channels(daily, market, vix)
    sid = ch["sid"].to_numpy()
    sx = ch["sx"].to_numpy()
    vol63 = ch["vol63"].to_numpy().astype(np.float64)
    chan = ch.select(list(CHANNELS)).to_numpy().astype(np.float64)
    rows = window_rows(sid, sx, vol63)
    keep = (
        ch.select("date", "sid")
        .with_row_index("__r")
        .join(members.with_columns(pl.lit(True).alias("__m")), on=["date", "sid"], how="left")
    )
    member = keep.sort("__r")["__m"].fill_null(False).to_numpy()
    rows = rows[member[rows]]
    dates = ch["date"].to_numpy()[rows]
    years = ch["date"].dt.year().to_numpy()[rows]
    root = seq_root(settings, version)
    root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for y in np.unique(years):
        sel = rows[years == y]
        out = np.lib.format.open_memmap(
            root / f"X_{y}.npy", mode="w+", dtype=np.float16, shape=(len(sel), WINDOW, N_CH)
        )
        for a in range(0, len(sel), 20000):
            out[a : a + 20000] = make_windows(chan, vol63, sel[a : a + 20000])
        out.flush()
        del out
        pl.DataFrame(
            {
                "row": np.arange(len(sel), dtype=np.int64),
                "date": dates[years == y],
                "sid": sid[sel].astype(np.int32),
            }
        ).write_parquet(root / f"index_{y}.parquet")
        counts[str(y)] = len(sel)
    return counts
