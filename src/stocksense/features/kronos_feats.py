# ruff: noqa: E501  (long human-readable feature labels)
"""Kronos zero-shot features (PLAN T7.5c): seven `kr_*` features from sampled 5-session price paths.

The expensive part (`compute_kronos_day`) writes one file per decision date,
`features/kronos/<tag>/YYYY-MM-DD.parquet`, from data on or before that date only. The registered
features just read those files (null where a date has no file) through the as_of gate, so they pass the
truncation test by construction. `compute_kronos_range` is resumable: dates whose file exists are skipped.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.log import get_logger
from stocksense.core.paths import Paths
from stocksense.data import io
from stocksense.features.base import FeatureContext, feature
from stocksense.models import kronos_model as km

log = get_logger(__name__)

LOOKBACK = 256
PRED_LEN = 5
N_PATHS = 20
TEMPERATURE = 1.0
TOP_P = 0.9
VOL_WINDOW = 20
TAG_ENV = "STOCKSENSE_KRONOS_TAG"
DEFAULT_TAG = "small-zs"
NAMES = (
    "kr_ret5_mean",
    "kr_ret5_std",
    "kr_up_prob",
    "kr_oc1_mean",
    "kr_co1_mean",
    "kr_range5",
    "kr_volchg5",
)
_PANEL_COLS = [
    "date",
    "sid",
    "sx",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adj_volume",
    "turnover_inr",
]
_SPAN_MAX = math.ceil(1.25 * LOOKBACK) + 2  # the wide-span rule of features.base.span_ok


def default_tag() -> str:
    return os.environ.get(TAG_ENV, DEFAULT_TAG)


def _dir(settings, tag: str) -> Path:
    return Paths(settings).features / "kronos" / tag


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema={"date": pl.Date, "sid": pl.Int32, **{n: pl.Float64 for n in NAMES}})


# ---- paths -> features ---------------------------------------------------------------------------


def features_from_paths(
    paths: np.ndarray, last_close: np.ndarray, last_vol20: np.ndarray
) -> dict[str, np.ndarray]:
    """The seven features from price-unit paths [n, n_paths, 5, 6] (open, high, low, close, volume, amount).

    `last_close` is the last actual close, `last_vol20` the mean of the last 20 actual volumes.
    Non-finite results become NaN. Standard deviation is the population one (ddof 0).
    """
    p = np.asarray(paths, dtype=np.float64)
    lc = np.asarray(last_close, dtype=np.float64)[:, None]
    o, h, lo, c, v = p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4]
    with np.errstate(all="ignore"):
        ret5 = c[:, :, 4] / lc - 1.0
        oc1 = c[:, :, 0] / o[:, :, 0] - 1.0
        co1 = o[:, :, 0] / lc - 1.0
        top = np.maximum(np.maximum(h, o), c)
        bot = np.minimum(np.minimum(lo, o), c)
        rng = np.log(top / bot)
        out = {
            "kr_ret5_mean": ret5.mean(axis=1),
            "kr_ret5_std": ret5.std(axis=1),
            "kr_up_prob": (c[:, :, 4] > lc).mean(axis=1),
            "kr_oc1_mean": oc1.mean(axis=1),
            "kr_co1_mean": co1.mean(axis=1),
            "kr_range5": rng.reshape(len(p), -1).mean(axis=1),
            "kr_volchg5": v.reshape(len(p), -1).mean(axis=1) / np.asarray(last_vol20, dtype=np.float64),
        }
    return {k: np.where(np.isfinite(a), a, np.nan) for k, a in out.items()}


# ---- compute ---------------------------------------------------------------------------------


def _holidays(settings) -> list[date]:
    try:
        col = (
            io.read(settings, "holidays", columns=["date"]).select(pl.col("date").unique()).collect()["date"]
        )
    except Exception:  # no holidays dataset: every future weekday is a session
        return []
    return sorted(d for d in col.to_list() if d is not None)


def _future_sessions(settings, sessions_to_d: list[date], holidays: list[date]) -> list[date]:
    cal = TradingCalendar.from_dates(sessions_to_d, holidays, settings.schedule.special_sessions)
    out, cur = [], sessions_to_d[-1]
    for _ in range(PRED_LEN):
        cur = cal.next_session(cur)
        out.append(cur)
    return out


class _Store:
    """Panel, u500 flags and calendar loaded once up to `as_of` (the last date of a run)."""

    def __init__(self, settings, as_of: date) -> None:
        ctx = FeatureContext(settings, as_of=as_of)
        self.settings = settings
        self.panel = ctx.panel().select(_PANEL_COLS).collect()
        self.u500 = ctx.universe().filter(pl.col("u500")).select("date", "sid").collect()
        self.sessions: list[date] = ctx.sessions().collect()["date"].to_list()
        self.holidays = _holidays(settings)


def _day_frame(
    store: _Store, d: date, bundle: km.KronosBundle, batch_series: int, autocast: bool
) -> pl.DataFrame:
    sess = [x for x in store.sessions if x <= d]
    if not sess or sess[-1] != d:
        raise ValueError(f"{d} is not a session of the store")
    sx_d = len(sess) - 1
    sids = store.u500.filter(pl.col("date") == d)["sid"]
    win = (
        store.panel.filter(
            pl.col("sid").is_in(sids.implode())
            & (pl.col("date") <= d)
            & (pl.col("sx") >= sx_d - _SPAN_MAX - 2)
        )
        .sort("sid", "date")
        .group_by("sid", maintain_order=True)
        .tail(LOOKBACK)
    )
    future = _future_sessions(store.settings, sess, store.holidays)
    frames: list[km.KronosInput] = []
    keep: list[int] = []
    last_close: list[float] = []
    vol20: list[float] = []
    for (sid,), g in win.partition_by("sid", maintain_order=True, as_dict=True).items():
        if g.height < LOOKBACK or g["date"][-1] != d:
            continue
        if g["sx"][-1] - g["sx"][0] + 1 > _SPAN_MAX:
            continue
        vals = g.select(
            "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume", "turnover_inr"
        ).to_numpy()
        if not np.isfinite(vals).all() or (vals[:, :4] <= 0).any():
            continue
        frames.append(km.KronosInput(values=vals, dates=g["date"].to_list(), future_dates=future))
        keep.append(int(sid))
        last_close.append(float(vals[-1, 3]))
        vol20.append(float(vals[-VOL_WINDOW:, 4].mean()))
    if not frames:
        return _empty()
    paths = km.forecast_panel(
        frames, bundle, PRED_LEN, N_PATHS, T=TEMPERATURE, top_p=TOP_P, seed=int(d.strftime("%Y%m%d")),
        batch_series=batch_series, autocast=autocast,
    )  # fmt: skip
    feats = features_from_paths(paths, np.array(last_close), np.array(vol20))
    return pl.DataFrame(
        {"date": [d] * len(keep), "sid": pl.Series(keep, dtype=pl.Int32), **{n: feats[n] for n in NAMES}},
        schema={"date": pl.Date, "sid": pl.Int32, **{n: pl.Float64 for n in NAMES}},
    ).sort("sid")


def _write(settings, tag: str, d: date, df: pl.DataFrame) -> None:
    folder = _dir(settings, tag)
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / f"{d.isoformat()}.parquet"
    tmp = folder / f"{d.isoformat()}.parquet.tmp"
    df.write_parquet(tmp)
    os.replace(tmp, final)


def compute_kronos_day(
    settings, d: date, tag: str, bundle: km.KronosBundle, *, batch_series: int = 8, autocast: bool = False
) -> pl.DataFrame:
    """The kr_* rows for decision date `d` (u500 sids with a full 256-row window), from data <= d only.

    Returns the frame; does not write it (use `compute_kronos_range` for the resumable file store).
    """
    return _day_frame(_Store(settings, d), d, bundle, batch_series, autocast)


def compute_kronos_range(
    settings,
    dates: Sequence[date],
    tag: str,
    bundle: km.KronosBundle | None = None,
    *,
    batch_series: int = 8,
    autocast: bool = False,
) -> list[date]:
    """Compute and store every date without a file yet; returns the dates computed in this call."""
    todo = sorted(d for d in set(dates) if not (_dir(settings, tag) / f"{d.isoformat()}.parquet").is_file())
    if not todo:
        return []
    b = bundle if bundle is not None else km.load_bundle("small")
    store = _Store(settings, todo[-1])
    done: list[date] = []
    for d in todo:
        df = _day_frame(store, d, b, batch_series, autocast)
        _write(settings, tag, d, df)
        done.append(d)
        log.info("kronos_day_done", date=d.isoformat(), tag=tag, rows=df.height)
    return done


# ---- registered features (read the stored files) ---------------------------------------------------


def _read(ctx: FeatureContext, col: str) -> pl.LazyFrame:
    folder = _dir(ctx.settings, default_tag())
    parts = []
    for f in sorted(folder.glob("*.parquet")) if folder.is_dir() else []:
        try:
            d = date.fromisoformat(f.stem)
        except ValueError:
            continue
        if ctx.as_of is not None and d > ctx.as_of:
            continue
        parts.append(pl.read_parquet(f, columns=["date", "sid", col]))
    df = pl.concat(parts) if parts else _empty().select("date", "sid", col)
    if ctx.as_of is not None:
        df = df.filter(pl.col("date") <= ctx.as_of)
    return df.lazy()


_LABELS = {
    "kr_ret5_mean": "What the Kronos forecast model expects the stock to return over the next 5 sessions (average of 20 simulated paths).",
    "kr_ret5_std": "How much the 20 Kronos simulated 5-session paths disagree about the return.",
    "kr_up_prob": "Share of Kronos simulated paths that end the 5 sessions above today's close.",
    "kr_oc1_mean": "Kronos expected open-to-close move of the next session.",
    "kr_co1_mean": "Kronos expected overnight gap (close to next open).",
    "kr_range5": "Kronos expected average daily high-low range over the next 5 sessions (log).",
    "kr_volchg5": "Kronos expected volume over the next 5 sessions relative to the last 20 sessions.",
}


def _register(name: str) -> None:
    @feature(name, group="kronos", kind="xs", window=LOOKBACK, label=_LABELS[name])
    def _fn(ctx: FeatureContext, _n: str = name) -> pl.LazyFrame:
        return _read(ctx, _n)

    _fn.__name__ = name


for _name in NAMES:
    _register(_name)

__all__ = ["NAMES", "compute_kronos_day", "compute_kronos_range", "features_from_paths", "default_tag"]
