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


# ---- G2-C2: score smoothing and the convex expected-return map ---------------------------------------

EMA_RESTART_GAP = 5  # a sid missing for more than this many sessions restarts its EMA


def _ema_scores(frame: pl.DataFrame, span: int) -> pl.DataFrame:
    """date, sid, score with the score replaced by its per-sid EMA (adjust=False) over consecutive session
    rows. A session is a distinct date of `frame`; a gap of more than EMA_RESTART_GAP sessions between two
    rows of a sid restarts the average. The value at t uses only rows <= t."""
    f = frame.select("date", "sid", "score").sort("date", "sid")
    dates = f["date"].unique().sort()
    idx = pl.DataFrame({"date": dates, "_i": np.arange(len(dates), dtype=np.int64)})
    f = f.join(idx, on="date").sort("sid", "date")
    seg = (pl.col("_i").diff().over("sid") > EMA_RESTART_GAP).fill_null(False).cum_sum().over("sid")
    f = f.with_columns(seg.alias("_seg"))
    ema = pl.col("score").ewm_mean(span=span, adjust=False).over(["sid", "_seg"])
    return f.select("date", "sid", ema.alias("score")).sort("date", "sid")


def smooth_scores(frame: pl.DataFrame, span: int) -> pl.DataFrame:
    """Causal EMA of every sid's score (see `_ema_scores`), re-ranked per date to [-0.5, 0.5]."""
    if span < 1:
        raise ValueError("span must be >= 1")
    return rank_scores(_ema_scores(frame, span)).sort("date", "sid")


def james_stein_shrink(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Empirical-Bayes (James-Stein style) shrinkage of bin means `m` with sampling variances `v`.

    mu = mean(m); tau2 = max(0, var(m, ddof=1) - mean(v)); m*_j = mu + (1 - v_j / (v_j + tau2)) (m_j - mu).
    tau2 = 0 gives mu for every bin; v -> 0 gives m. Bins with a non-finite mean or variance stay NaN and
    take no part in mu or tau2 (fewer than two usable bins: returned unchanged)."""
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(m) & np.isfinite(v)
    out = np.full(m.shape, np.nan)
    if ok.sum() < 2:
        out[ok] = m[ok]
        return out
    mu = float(m[ok].mean())
    tau2 = max(0.0, float(np.var(m[ok], ddof=1)) - float(v[ok].mean()))
    denom = v[ok] + tau2
    b = np.where(denom > 0, v[ok] / np.where(denom > 0, denom, 1.0), 0.0)
    out[ok] = mu + (1.0 - b) * (m[ok] - mu)
    return out


def expected_return_map_decile(
    blend: pl.DataFrame,
    label_frame: pl.DataFrame,
    target: str,
    window: int = 504,
    n_bins: int = 10,
    shrink_k: float = 200,
    shrink: str = "k",
) -> pl.DataFrame:
    """date, sid, score, exp_ret from a monotone decile map instead of a straight line.

    At date t the trailing `window` dates s with label end < t are used (same rule as
    `expected_return_map`). Per date the names are cut into `n_bins` equal-count bins by score; the mean raw
    label of each bin over the window is shrunk toward the all-bin mean with weight n_j / (n_j + shrink_k),
    then made non-decreasing across bins by isotonic regression weighted by n_j. A name's `exp_ret` is its
    bin's value (no floor: a poor bin can be negative). With no known label yet the map is 0.

    `shrink="james_stein"` (G2-E2) replaces the fixed-k shrinkage by `james_stein_shrink` of the window's bin
    means with sampling variance s_j^2 / n_j (s_j^2 the within-bin label variance over the window), then the
    same isotonic step."""
    from sklearn.isotonic import IsotonicRegression

    if shrink not in ("k", "james_stein"):
        raise ValueError(f"unknown shrink {shrink!r}")

    n = pl.len().over("date")
    bin_ = (
        ((pl.col("score").rank("average").over("date") - 1.0) / n * n_bins).floor().clip(0, n_bins - 1)
    ).cast(pl.Int64)
    j = blend.select("date", "sid", "score").with_columns(bin_.alias("_bin"))
    j = j.join(label_frame.select("date", "sid", target), on=["date", "sid"], how="left")
    dates = j["date"].unique().sort().to_numpy()
    pos = {d: i for i, d in enumerate(dates.tolist())}
    T = len(dates)
    lab = j.filter(pl.col(target).is_not_null() & pl.col(target).is_finite())
    g = lab.group_by("date", "_bin").agg(
        pl.col(target).sum().alias("s"), (pl.col(target) ** 2).sum().alias("ss"), pl.len().alias("n")
    )
    S, N, SS = np.zeros((T, n_bins)), np.zeros((T, n_bins)), np.zeros((T, n_bins))
    ti = np.array([pos[d] for d in g["date"].to_list()], dtype=np.int64)
    bi = g["_bin"].to_numpy().astype(np.int64)
    S[ti, bi] = g["s"].to_numpy()
    N[ti, bi] = g["n"].to_numpy()
    SS[ti, bi] = g["ss"].to_numpy()
    known_end, day = _label_info(label_frame, target, dates)
    values = np.zeros((T, n_bins))
    xs = np.arange(n_bins, dtype=float)
    for t in range(T):
        idx = np.flatnonzero(known_end[:t] < day[t])[-window:]
        if len(idx) == 0:
            continue
        s, c = S[idx].sum(0), N[idx].sum(0)
        tot = c.sum()
        if tot <= 0:
            continue
        grand = s.sum() / tot
        have = c > 0
        m = np.where(have, s / np.where(have, c, 1.0), grand)
        if shrink == "james_stein":
            ss = SS[idx].sum(0)
            var = np.where(c > 1, np.maximum(ss - c * m * m, 0.0) / np.where(c > 1, c - 1.0, 1.0), 0.0)
            js = james_stein_shrink(
                np.where(have, m, np.nan), np.where(have, var / np.where(have, c, 1.0), np.nan)
            )
            shrunk = np.where(have, js, grand)
        else:
            shrunk = c / (c + shrink_k) * m + (shrink_k / (c + shrink_k) * grand if shrink_k > 0 else 0.0)
        if have.sum() < 2:
            values[t] = grand
            continue
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(xs[have], shrunk[have], sample_weight=c[have])
        values[t] = iso.predict(xs)
    vdf = pl.DataFrame({"date": dates, **{f"_v{b}": values[:, b] for b in range(n_bins)}})
    out = j.join(vdf, on="date").select(
        "date",
        "sid",
        "score",
        pl.concat_list([f"_v{b}" for b in range(n_bins)]).list.get(pl.col("_bin")).alias("exp_ret"),
    )
    return out.sort("date", "sid")
