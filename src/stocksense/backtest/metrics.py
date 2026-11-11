from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
import polars as pl

_MIN_IC_PAIRS = 50
_N_DECILES = 10


def newey_west_tstat(x: np.ndarray, lags: int = 5) -> float:
    """t-statistic of mean(x) with a Bartlett-weighted Newey-West long-run variance (unitless)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    d = x - x.mean()
    lrv = float(d @ d) / n
    for lag in range(1, min(lags, n - 1) + 1):
        lrv += 2.0 * (1.0 - lag / (lags + 1.0)) * float(d[lag:] @ d[:-lag]) / n
    if not lrv > 0.0:
        return float("nan")
    return float(x.mean() / math.sqrt(lrv / n))


def _moments(r: np.ndarray) -> tuple[float, float]:
    d = r - r.mean()
    m2 = float(np.mean(d**2))
    if m2 <= 0.0:
        return 0.0, 3.0
    return float(np.mean(d**3) / m2**1.5), float(np.mean(d**4) / m2**2)


def _frac_positive_years(r: np.ndarray, dates: Sequence[date]) -> float:
    years: dict[int, float] = {}
    for x, d in zip(r, dates, strict=True):
        years[d.year] = years.get(d.year, 1.0) * (1.0 + float(x))
    return sum(1 for v in years.values() if v > 1.0) / len(years)


def perf_stats(
    daily_ret: np.ndarray,
    benchmark: np.ndarray | None,
    periods: int = 252,
    dates: Sequence[date] | None = None,
) -> dict[str, Any]:
    """Performance statistics of daily returns (fractions); skew/kurtosis are population, non-excess."""
    r = np.asarray(daily_ret, dtype=float)
    n = len(r)
    curve = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    cagr = float(curve[-1] ** (periods / n) - 1.0) if n and curve[-1] > 0 else float("nan")
    sd = float(np.std(r, ddof=1)) if n > 1 else 0.0
    mean = float(r.mean()) if n else 0.0
    sharpe = mean / sd * math.sqrt(periods) if sd > 0 else 0.0
    dd_dev = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2))) if n else 0.0
    sortino = mean / dd_dev * math.sqrt(periods) if dd_dev > 0 else 0.0
    peak = np.maximum.accumulate(curve)
    dd = 1.0 - curve / peak
    trough = int(np.argmax(dd))
    max_dd = float(dd[trough])
    peak_idx = int(np.argmax(curve[: trough + 1]))
    rec_idx: int | None = None
    if max_dd > 0:
        after = np.nonzero(curve[trough + 1 :] >= curve[peak_idx])[0]
        if len(after):
            rec_idx = trough + 1 + int(after[0])

    def to_date(k: int | None) -> date | None:
        if k is None or dates is None:
            return None
        return dates[max(k - 1, 0)]

    skew, kurt = _moments(r) if n else (0.0, 3.0)
    out: dict[str, Any] = {
        "n": n,
        "cagr": cagr,
        "vol": sd * math.sqrt(periods),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "dd_start": to_date(peak_idx),
        "dd_trough": to_date(trough),
        "dd_recovery": to_date(rec_idx),
        "calmar": cagr / max_dd if max_dd > 0 else float("nan"),
        "hit_rate_days": float(np.mean(r > 0)) if n else float("nan"),
        "skew": skew,
        "kurtosis": kurt,
        "excess_tstat": newey_west_tstat(r - (0.0 if benchmark is None else np.asarray(benchmark)), 5),
    }
    if dates is not None and n:
        out["frac_positive_years"] = _frac_positive_years(r, dates)
    return out


def ic_stats(pred_frame: pl.DataFrame, label_col: str) -> dict[str, Any]:
    """Per-date Spearman IC of `score` vs the label (dates with < 50 pairs skipped) plus decile table."""
    df = (
        pred_frame.select("date", "score", pl.col(label_col).alias("label"))
        .drop_nulls()
        .with_columns(
            pl.col("score").rank("average").over("date").alias("rs"),
            pl.col("label").rank("average").over("date").alias("rl"),
            pl.len().over("date").alias("n"),
        )
        .filter(pl.col("n") >= _MIN_IC_PAIRS)
    )
    if df.is_empty():
        return {
            "n_dates": 0, "mean_ic": float("nan"), "ic_std": float("nan"), "ic_tstat": float("nan"),
            "ic_by_year": {}, "decile_mean_label": [], "monotonicity": float("nan"),
        }  # fmt: skip
    ics = df.group_by("date").agg(pl.corr("rs", "rl").alias("ic")).sort("date").drop_nulls()
    v = ics["ic"].to_numpy()
    n = len(v)
    sd = float(np.std(v, ddof=1)) if n > 1 else float("nan")
    mean = float(v.mean())
    tstat = mean / sd * math.sqrt(n) if sd and sd > 0 else float("nan")
    by_year = (
        ics.with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("year")
        .agg(pl.col("ic").mean())
        .sort("year")
    )
    dec = (
        df.with_columns(((pl.col("rs") - 1) * _N_DECILES / pl.col("n")).floor().cast(pl.Int64).alias("dec"))
        .group_by("date", "dec")
        .agg(pl.col("label").mean().alias("m"))
        .group_by("dec")
        .agg(pl.col("m").mean())
        .sort("dec")
    )
    means = dec["m"].to_list()
    mono = (
        float(np.corrcoef(_rank(np.arange(len(means))), _rank(np.array(means)))[0, 1])
        if len(means) > 2
        else float("nan")
    )
    return {
        "n_dates": n,
        "mean_ic": mean,
        "ic_std": sd,
        "ic_tstat": tstat,
        "ic_by_year": dict(zip(by_year["year"].to_list(), by_year["ic"].to_list(), strict=True)),
        "decile_mean_label": means,
        "monotonicity": mono,
    }


def _rank(a: np.ndarray) -> np.ndarray:
    return pl.Series(a).rank("average").to_numpy()


def trade_stats(trades: pl.DataFrame, rejects: pl.DataFrame | None = None) -> dict[str, Any]:
    """Trade statistics from columns ret (net fraction), hold_sessions, cost_frac; rejects has `reason`."""
    rej: dict[str, int] = {}
    if rejects is not None and not rejects.is_empty():
        rej = {k: int(c) for k, c in rejects["reason"].value_counts().iter_rows()}
    n = trades.height
    if n == 0:
        return {
            "count": 0, "win_rate": float("nan"), "avg_win": float("nan"), "avg_loss": float("nan"),
            "profit_factor": float("nan"), "avg_hold_sessions": float("nan"),
            "avg_cost_frac": float("nan"), "rejects_by_reason": rej,
        }  # fmt: skip
    ret = trades["ret"]
    wins = ret.filter(ret > 0)
    losses = ret.filter(ret < 0)
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
    return {
        "count": n,
        "win_rate": len(wins) / n,
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "profit_factor": float(wins.sum()) / gross_loss if gross_loss > 0 else float("inf"),
        "avg_hold_sessions": float(trades["hold_sessions"].mean()),
        "avg_cost_frac": float(trades["cost_frac"].mean()),
        "rejects_by_reason": rej,
    }


def yearly_table(daily: pl.DataFrame, periods: int = 252) -> pl.DataFrame:
    """Per calendar year: n days, compounded return, Sharpe and max drawdown of columns date, ret."""
    rows = []
    for (year,), g in daily.sort("date").group_by(pl.col("date").dt.year(), maintain_order=True):
        r = g["ret"].to_numpy()
        s = perf_stats(r, None, periods)
        rows.append(
            {
                "year": int(year),
                "n": len(r),
                "ret": float(np.prod(1.0 + r) - 1.0),
                "sharpe": s["sharpe"],
                "max_drawdown": s["max_drawdown"],
            }
        )
    return pl.DataFrame(rows).sort("year")
