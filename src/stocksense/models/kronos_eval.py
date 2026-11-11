"""Evaluation helpers for the Kronos zero-shot test (PLAN T7.5d): daily rank IC, paired t-test and an
expanding walk-forward ridge fitted with and without the kr_* features on the same rows."""

from __future__ import annotations

import numpy as np
import polars as pl


def daily_rank_ic(df: pl.DataFrame, pred: str, y: str) -> pl.DataFrame:
    """Per-date Spearman correlation between `pred` and `y` (dates with < 5 valid rows are dropped)."""
    d = df.select("date", pred, y).drop_nulls()
    d = d.with_columns(
        pl.col(pred).rank("average").over("date").alias("_a"),
        pl.col(y).rank("average").over("date").alias("_b"),
    )
    out = d.group_by("date").agg(pl.corr("_a", "_b").alias("ic"), pl.len().alias("n"))
    return out.filter((pl.col("n") >= 5) & pl.col("ic").is_not_null() & pl.col("ic").is_finite()).sort("date")


def paired_t(diffs: np.ndarray) -> tuple[float, float, int]:
    """(mean, t, n) of a vector of paired differences (t = mean / (sd / sqrt(n)), ddof 1)."""
    x = np.asarray(diffs, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan"), float("nan"), n
    sd = x.std(ddof=1)
    return float(x.mean()), float(x.mean() / (sd / np.sqrt(n))) if sd > 0 else float("nan"), n


def make_folds(dates: list, train_frac: float = 0.4, n_folds: int = 4) -> list[tuple[list, list]]:
    """Expanding walk-forward: the first `train_frac` of dates are train-only; the rest is cut into
    `n_folds` consecutive blocks. Returns [(train_dates_before_block, test_block_dates)]."""
    ds = sorted(dates)
    start = int(train_frac * len(ds))
    rest = ds[start:]
    bounds = np.linspace(0, len(rest), n_folds + 1).round().astype(int)
    return [
        (ds[: start + int(a)], rest[int(a) : int(b)]) for a, b in zip(bounds[:-1], bounds[1:], strict=True)
    ]


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    xm, ym = x.mean(0), y.mean()
    xc = x - xm
    w = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ (y - ym))
    return w, ym - xm @ w


def incremental_walk_forward(
    df: pl.DataFrame,
    base_cols: list[str],
    kr_cols: list[str],
    y_col: str,
    end_col: str,
    *,
    target_col: str | None = None,
    alpha: float = 100.0,
    train_frac: float = 0.4,
    n_folds: int = 4,
    return_detail: bool = False,
) -> dict:
    """Fit ridge(base) and ridge(base + kr) on identical rows with a purged expanding walk-forward and
    compare per-date rank ICs (prediction vs raw `y_col`). The fit target is `target_col` (default y_col).
    A training row needs `end_col` (label end date) strictly before the first test date of its fold.
    Feature columns must have no nulls (the caller fills them)."""
    tgt = target_col or y_col
    data = df.drop_nulls([*base_cols, *kr_cols, y_col, tgt, end_col]).sort("date", "sid")
    folds = make_folds(data["date"].unique().to_list(), train_frac, n_folds)
    parts, detail = [], []
    for train_dates, test_dates in folds:
        t0 = test_dates[0]
        tr = data.filter(pl.col("date").is_in(train_dates) & (pl.col(end_col) < t0))
        te = data.filter(pl.col("date").is_in(test_dates))
        if tr.height < 50 or te.is_empty():
            continue
        detail.append({"test_start": t0, "train_end_max": tr[end_col].max(), "n_train": tr.height})
        ytr = tr[tgt].to_numpy().astype(float)
        preds = {}
        for arm, cols in (("without", base_cols), ("with", base_cols + kr_cols)):
            w, b = _ridge(tr.select(cols).to_numpy().astype(float), ytr, alpha)
            preds[arm] = te.select(cols).to_numpy().astype(float) @ w + b
        parts.append(te.select("date", "sid", y_col).with_columns(
            pl.Series("p_without", preds["without"]), pl.Series("p_with", preds["with"])))  # fmt: skip
    if not parts:
        raise ValueError("no usable fold")
    te_all = pl.concat(parts)
    a = daily_rank_ic(te_all, "p_without", y_col).rename({"ic": "ic_without"})
    b = daily_rank_ic(te_all, "p_with", y_col).rename({"ic": "ic_with"})
    j = a.select("date", "ic_without").join(b.select("date", "ic_with"), on="date").sort("date")
    j = j.with_columns((pl.col("ic_with") - pl.col("ic_without")).alias("diff"))
    mean, t, n = paired_t(j["diff"].to_numpy())
    res = {
        "mean_ic_without": float(j["ic_without"].mean()),
        "mean_ic_with": float(j["ic_with"].mean()),
        "mean_diff": mean,
        "t": t,
        "n_dates": n,
        "n_test_rows": te_all.height,
        "per_date": j,
    }
    if return_detail:
        res["detail"] = pl.DataFrame(detail)
    return res
