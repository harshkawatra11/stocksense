"""Ensembles of out-of-fold scores (PLAN T7.6): equal blend, causal online blend, expected-return map.

Inputs are `{model_name: frame[date, sid, score]}` (a list is named m0, m1, ...). Scores are ranked per
date to [-0.5, 0.5]; a blend only covers (date, sid) pairs every model scored. The label frame carries
`date, sid`, the raw target column `<target>` (a fraction) and its end-date column (`end_5`, `end_oc`...):
a label is usable at decision date t only when its end date is strictly before t, and because every
row of one date has its own end date, a date counts as known once the LATEST end on that date has passed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
import polars as pl

from stocksense.validation.walkforward import END_COLUMN

MIN_NAMES = 10


def _named(pred_frames: Mapping[str, pl.DataFrame] | Sequence[pl.DataFrame]) -> dict[str, pl.DataFrame]:
    if isinstance(pred_frames, Mapping):
        return dict(pred_frames)
    return {f"m{i}": f for i, f in enumerate(pred_frames)}


def rank_scores(frame: pl.DataFrame) -> pl.DataFrame:
    """(date, sid, score) with score replaced by its per-date rank in [-0.5, 0.5] (ties averaged)."""
    n = pl.len().over("date")
    r = (pl.col("score").rank("average").over("date") - 1.0) / pl.max_horizontal(n - 1, pl.lit(1)).cast(
        pl.Float64
    )
    return frame.select("date", "sid", (r - 0.5).alias("score"))


def _ranked_wide(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """date, sid and one rank column per model (inner join, sorted by date, sid)."""
    out: pl.DataFrame | None = None
    for name, f in frames.items():
        r = rank_scores(f.select("date", "sid", "score")).rename({"score": name})
        out = r if out is None else out.join(r, on=["date", "sid"], how="inner")
    assert out is not None
    return out.sort("date", "sid")


def blend_equal(pred_frames: Mapping[str, pl.DataFrame] | Sequence[pl.DataFrame]) -> pl.DataFrame:
    """Equal-weight mean of the per-date ranks: date, sid, score in [-0.5, 0.5]."""
    frames = _named(pred_frames)
    wide = _ranked_wide(frames)
    return wide.select("date", "sid", pl.mean_horizontal(list(frames)).alias("score"))


def _label_info(label_frame: pl.DataFrame, target: str, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(known_end per date as int days, aligned with `dates`): the latest label end among the date's rows."""
    end_col = END_COLUMN[target]
    e = label_frame.group_by("date").agg(pl.col(end_col).max().alias("e"))
    m = dict(zip(e["date"].to_list(), e["e"].to_list(), strict=True))
    far = date(2100, 1, 1)
    ends = [m.get(d, far) or far for d in dates.tolist()]
    return np.array([x.toordinal() for x in ends], dtype=np.int64), np.array(
        [d.toordinal() for d in dates.tolist()], dtype=np.int64
    )


def _daily_ic(wide: pl.DataFrame, names: list[str], label_frame: pl.DataFrame, target: str) -> np.ndarray:
    """[n_dates, n_models] Spearman IC of each model's rank vs the raw label (nan where < MIN_NAMES names)."""
    j = wide.join(label_frame.select("date", "sid", target), on=["date", "sid"], how="left")
    j = j.filter(pl.col(target).is_not_null() & pl.col(target).is_finite())
    ry = pl.col(target).rank("average").over("date")
    ic = (
        j.with_columns(ry.alias("__ry"), pl.len().over("date").alias("__n"))
        .filter(pl.col("__n") >= MIN_NAMES)
        .group_by("date")
        .agg([pl.corr(n, "__ry").alias(n) for n in names])
    )
    dates = wide["date"].unique().sort()
    return (
        pl.DataFrame({"date": dates})
        .join(ic, on="date", how="left")
        .select(names)
        .to_numpy()
        .astype(np.float64)
    )


def _floored(w: np.ndarray, floor: float) -> np.ndarray:
    """Weights summing to 1 with every entry >= floor: entries below the floor are fixed at it and the rest
    are rescaled proportionally to what remains (repeated until stable)."""
    m = len(w)
    if floor * m >= 1.0:
        return np.full(m, 1.0 / m)
    w = w / w.sum()
    fixed = np.zeros(m, dtype=bool)
    for _ in range(m):
        under = (w < floor) & ~fixed
        if not under.any():
            break
        fixed |= under
        free = ~fixed
        rest = 1.0 - floor * fixed.sum()
        base = w[free] / w[free].sum()
        w = np.where(fixed, floor, 0.0)
        w[free] = base * rest
    return w


def online_weights(
    ic: np.ndarray, known_end: np.ndarray, day: np.ndarray, eta: float, halflife: float, floor: float
) -> np.ndarray:
    """[n_dates, n_models] weights. For date index t only dates s with known_end[s] < day[t] contribute,
    each with decay 0.5 ** ((t - s) / halflife) in session positions."""
    T, M = ic.shape
    out = np.full((T, M), 1.0 / M)
    pos = np.arange(T)
    valid = np.isfinite(ic)
    ic0 = np.where(valid, ic, 0.0)
    for t in range(T):
        elig = known_end < day[t]
        elig &= pos < t
        if not elig.any():
            continue
        dec = 0.5 ** ((t - pos[elig]) / halflife)
        v = valid[elig]
        num = (dec[:, None] * ic0[elig]).sum(0)
        den = (dec[:, None] * v).sum(0)
        s = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
        z = eta * s
        raw = np.exp(z - z.max())
        out[t] = _floored(raw, floor)
    return out


def blend_online(
    pred_frames: Mapping[str, pl.DataFrame] | Sequence[pl.DataFrame],
    label_frame: pl.DataFrame,
    target: str,
    eta: float = 20.0,
    halflife: float = 60,
    floor: float = 0.05,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(blend, weights_by_date[date, <model>...]); the weights at t use only labels ended before t."""
    frames = _named(pred_frames)
    names = list(frames)
    wide = _ranked_wide(frames)
    dates = wide["date"].unique().sort().to_numpy()
    ic = _daily_ic(wide, names, label_frame, target)
    known_end, day = _label_info(label_frame, target, dates)
    w = online_weights(ic, known_end, day, eta, float(halflife), floor)
    wdf = pl.DataFrame({"date": dates, **{n: w[:, i] for i, n in enumerate(names)}})
    j = wide.join(wdf.rename({n: f"__w_{n}" for n in names}), on="date")
    score = sum((pl.col(n) * pl.col(f"__w_{n}") for n in names), pl.lit(0.0)).alias("score")
    return j.select("date", "sid", score), wdf


def expected_return_map(
    blend: pl.DataFrame, label_frame: pl.DataFrame, target: str, window: int = 504
) -> pl.DataFrame:
    """date, sid, score, exp_ret: exp_ret = b(t) * score with b(t) the through-the-origin slope of the raw
    label on the blended score over the trailing `window` dates with labels ended before t (b >= 0)."""
    j = blend.join(label_frame.select("date", "sid", target), on=["date", "sid"], how="left")
    jj = j.filter(pl.col(target).is_not_null() & pl.col(target).is_finite())
    st = jj.group_by("date").agg(
        (pl.col("score") * pl.col(target)).sum().alias("sxy"), (pl.col("score") ** 2).sum().alias("sxx")
    )
    dates = blend["date"].unique().sort().to_numpy()
    d = pl.DataFrame({"date": dates}).join(st, on="date", how="left")
    sxy, sxx = d["sxy"].fill_null(0.0).to_numpy(), d["sxx"].fill_null(0.0).to_numpy()
    known_end, day = _label_info(label_frame, target, dates)
    T = len(dates)
    b = np.zeros(T)
    for t in range(T):
        idx = np.flatnonzero(known_end[:t] < day[t])[-window:]
        if len(idx) == 0:
            continue
        den = sxx[idx].sum()
        b[t] = max(0.0, float(sxy[idx].sum() / den)) if den > 0 else 0.0
    bdf = pl.DataFrame({"date": dates, "__b": b})
    return (
        blend.join(bdf, on="date")
        .select("date", "sid", "score", (pl.col("__b") * pl.col("score")).alias("exp_ret"))
        .sort("date", "sid")
    )
