"""Extreme-value stops (G2-E6): generalised-Pareto peaks-over-threshold expected shortfall.

For every date the daily return of a stock is scaled by the stock's causal 21-day volatility
(`z = ret / vol_21`, the engine's `vol_21`, which uses returns up to the previous day) and the loss is
`-z`. Names are pooled by liquidity tercile (rank among the 500 most liquid at the last day before the
month) and, once per calendar month, the last `WINDOW` rows of every pooled name are used: the threshold
`u` is the pooled 95th percentile of the loss, the excesses over it are fitted with `scipy.stats.genpareto`
(location 0, maximum likelihood), and the 99% expected shortfall in vol units is
    VaR_q = u + beta / xi * (((1 - q) / p_u) ** -xi - 1)          (u + beta ln(p_u / (1 - q)) when xi = 0)
    ES_q  = VaR_q / (1 - xi) + (beta - xi u) / (1 - xi)           (VaR_q + beta when xi = 0)
with `p_u` the fraction of pooled losses above `u`. `xi` is clipped to 0.9 (ES is infinite from 1). A
stock's 99% daily ES is its vol times this multiplier; the S1 stop (`stop_model = "evt"`) then sits at
`max(stop_min, es_mult * vol_at_entry * sqrt(h))` below the entry price instead of the fixed
`max(stop_min, stop_vol_mult * vol_at_entry * sqrt(h))`.

The table of month M only reads rows dated before M, so it is causal. Only research worlds use it; no live
code path reads it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

WINDOW = 750  # rows per stock
THRESHOLD_Q = 0.95
ES_Q = 0.99
XI_MAX = 0.9
MIN_OBS = 2000  # pooled losses needed for a fit
MIN_EXCESS = 40
N_TERCILES = 3


@dataclass(frozen=True)
class EsFit:
    es: float  # 99% expected shortfall of the vol-scaled loss
    xi: float
    beta: float
    u: float
    p_u: float
    n: int


def fit_gpd(excess: np.ndarray) -> tuple[float, float]:
    """(xi, beta) of a generalised Pareto with location 0 fitted by maximum likelihood to the excesses."""
    x = np.asarray(excess, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    c, _, scale = stats.genpareto.fit(x, floc=0.0)
    return float(c), float(scale)


def gpd_es(u: float, xi: float, beta: float, p_u: float, q: float = ES_Q) -> float:
    """Expected shortfall at level q of a tail modelled as P(L > u) = p_u times a GPD(xi, beta) survival."""
    if xi >= 1.0:
        return math.inf
    ratio = (1.0 - q) / p_u
    if abs(xi) < 1e-9:
        var = u + beta * math.log(1.0 / ratio)
        return var + beta
    var = u + beta / xi * (ratio**-xi - 1.0)
    return var / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)


def pooled_es(z: np.ndarray, min_obs: int | None = None) -> EsFit | None:
    """The POT fit of the pooled vol-scaled RETURNS `z` (a loss is -z); None with too few usable values."""
    loss = -np.asarray(z, dtype=float)
    loss = loss[np.isfinite(loss)]
    if len(loss) < (MIN_OBS if min_obs is None else min_obs):
        return None
    u = float(np.quantile(loss, THRESHOLD_Q))
    exc = loss[loss > u] - u
    if len(exc) < MIN_EXCESS:
        return None
    xi, beta = fit_gpd(exc)
    xi = min(xi, XI_MAX)
    p_u = len(exc) / len(loss)
    return EsFit(gpd_es(u, xi, beta, p_u), xi, beta, u, p_u, len(loss))


def _month_starts(lo: date, hi: date) -> list[date]:
    out, y, m = [], lo.year, lo.month
    while date(y, m, 1) <= hi:
        out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def es_mult_table(
    dates: Sequence[date],
    sid: np.ndarray,
    ret: np.ndarray,
    vol: np.ndarray,
    rank: np.ndarray,
    *,
    from_date: date,
    to_date: date | None = None,
    universe_rank: int = 500,
    window: int | None = None,
    min_obs: int | None = None,
) -> pl.DataFrame:
    """(ym, sid, es_mult): for every calendar month from that of `from_date` the ES multiplier of each name
    that was among the `universe_rank` most liquid on its last row before the month, from rows dated before
    the month only. ym = year * 12 + month - 1. Terciles are by that last rank (1 = most liquid)."""
    w = WINDOW if window is None else window
    df = pl.DataFrame(
        {"sid": sid.astype(np.int64), "date": list(dates), "ret": ret, "vol": vol, "rank": rank}
    )
    df = df.with_columns(
        pl.when(pl.col("vol") > 0).then(pl.col("ret") / pl.col("vol")).alias("z"),
    ).sort("sid", "date")
    hi = to_date or df["date"].max()
    parts: list[pl.DataFrame] = []
    per = max(1, math.ceil(universe_rank / N_TERCILES))
    for s in _month_starts(from_date, hi):
        before = df.filter(pl.col("date") < s)
        if before.is_empty():
            continue
        last = before.group_by("sid").agg(pl.col("rank").last().alias("rank"))
        members = last.filter(pl.col("rank") <= universe_rank).with_columns(
            ((pl.col("rank") - 1) // per).clip(upper_bound=N_TERCILES - 1).alias("ter")
        )
        if members.is_empty():
            continue
        tail = (
            before.filter(pl.col("sid").is_in(members["sid"].implode()))
            .group_by("sid", maintain_order=True)
            .tail(w)
        )
        tail = tail.join(members.select("sid", "ter"), on="sid")
        mult: dict[int, float] = {}
        for ter in range(N_TERCILES):
            fit = pooled_es(tail.filter(pl.col("ter") == ter)["z"].drop_nulls().to_numpy(), min_obs)
            if fit is not None and math.isfinite(fit.es) and fit.es > 0:
                mult[ter] = fit.es
        if not mult:
            continue
        ym = s.year * 12 + s.month - 1
        rows = [
            (ym, int(r["sid"]), mult[r["ter"]]) for r in members.iter_rows(named=True) if r["ter"] in mult
        ]
        parts.append(
            pl.DataFrame(rows, schema={"ym": pl.Int64, "sid": pl.Int64, "es_mult": pl.Float64}, orient="row")
        )
    if not parts:
        return pl.DataFrame(schema={"ym": pl.Int64, "sid": pl.Int64, "es_mult": pl.Float64})
    return pl.concat(parts)


def align_es_mult(table: pl.DataFrame, sid: np.ndarray, dates: Sequence[date]) -> np.ndarray:
    """The ES multiplier of the (sid, date) panel rows (NaN where the name has none that month)."""
    n = len(sid)
    left = pl.DataFrame(
        {
            "sid": sid.astype(np.int64),
            "ym": pl.Series([d.year * 12 + d.month - 1 for d in dates], dtype=pl.Int64),
            "_i": np.arange(n),
        }
    )
    j = left.join(table, on=["ym", "sid"], how="left").sort("_i")
    return j["es_mult"].fill_null(float("nan")).to_numpy()


def cache_path(root: Path, end: date, start: date, n: int) -> Path:
    return root / "cache" / "evt" / f"es_{end:%Y%m%d}_{start:%Y%m%d}_{n}.parquet"
