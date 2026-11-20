"""Volatility forecasts (G2-E1): HAR-RV (Corsi 2009) and GARCH(1,1), as features and as S1 inputs.

Daily realised volatility `rv`: the square root of the summed squared 5-minute log returns where that
session has bars (`intraday_rv`, 2024-09 on), else the one-day Garman-Klass estimate (full history), never
below `RV_FLOOR`. HAR is a pooled log-RV regression
    ln RV_{t+1} = b0 + b1 ln RV_t + b2 mean(ln RV_{t-4..t}) + b3 mean(ln RV_{t-21..t}) + e
refitted every January on rows whose target date is <= the previous 31 December, then iterated h steps (the
lagged window is fed its own forecasts). The forecast is `scale * exp(f + s2/2)`: `s2` the residual variance
(lognormal mean) and `scale` = sqrt(sum la^2 / sum rv^2) on the fit sample, which puts the range-based RV on
the close-to-close scale of `vol_21`. `har_vol_10d` is the 10-day vol: sqrt of the sum of the ten iterated
daily variances. GARCH(1,1) (zero mean, normal) is fitted per stock at every month end on the last 1,000
returns; its 10-step variance sum is held fixed for the next calendar month: `garch_vol_10d`.

Both are causal by construction (a row on date t uses data <= t and fits ending before t's month/year), so
the truncation test holds. The module registers its features when imported; it is deliberately NOT in
`features.registry.FEATURE_MODULES` (E9 adds it once, for feature version 2), so the nightly feature build
and the live system are unchanged. The S1 backtests read `vol_panel` (cached) instead.
"""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from stocksense.core.config import Settings
from stocksense.core.paths import Paths
from stocksense.features.base import FeatureContext, FeatureSpec, feature

LN2 = math.log(2.0)
RV_FLOOR = 0.002  # daily vol floor (0.2 %): keeps ln RV out of the far left tail of a degenerate range
HAR_LAGS = 22
HORIZON = 10
MIN_HAR_ROWS = 20_000  # a January fit needs at least this many (stock, day) rows
GARCH_WINDOW = 1000
MIN_GARCH_OBS = 500
OUTPUTS = ("har_vol_1d", "har_vol_10d", "garch_vol_10d")


# ---- daily realised volatility ---------------------------------------------------------------------


def rv_daily(panel: pl.DataFrame, rv5: pl.DataFrame | None = None) -> pl.DataFrame:
    """sid, date, sx, rv, la from a panel with sid, date, sx, symbol, adj_open/high/low/close, ret.
    `rv5` (symbol, date, intraday_rv) overrides the Garman-Klass value where present and positive."""
    ok = (pl.col("adj_low") > 0) & (pl.col("adj_high") >= pl.col("adj_low")) & (pl.col("adj_open") > 0)
    ok = ok & (pl.col("adj_close") > 0)
    hl = (pl.col("adj_high") / pl.col("adj_low")).log()
    co = (pl.col("adj_close") / pl.col("adj_open")).log()
    gk = pl.when(ok).then(0.5 * hl**2 - (2.0 * LN2 - 1.0) * co**2)
    df = panel.select("sid", "date", "sx", "symbol", "ret", gk.alias("_gk"))
    if rv5 is not None and rv5.height:
        df = df.join(rv5.select("symbol", "date", "intraday_rv"), on=["symbol", "date"], how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("intraday_rv"))
    rv = (
        pl.when(pl.col("intraday_rv") > 0)
        .then(pl.col("intraday_rv"))
        .otherwise(pl.col("_gk").clip(lower_bound=0.0).sqrt())
    )
    return (
        df.with_columns(
            rv.clip(lower_bound=RV_FLOOR).alias("rv"),
            pl.when(pl.col("ret") > -1.0).then((1.0 + pl.col("ret")).log()).alias("la"),
        )
        .filter(pl.col("rv").is_not_null() & pl.col("rv").is_finite())
        .select("sid", "date", "sx", "rv", "la")
        .sort("sid", "date")
    )


# ---- HAR-RV --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HarFit:
    b: tuple[float, float, float, float]
    sigma2: float  # residual variance of the log regression
    scale: float  # close-to-close / range-based vol ratio on the fit sample
    n: int


def har_design(df: pl.DataFrame) -> pl.DataFrame:
    """Regressors per (sid, date): x1 = ln RV, x2 = mean ln RV over 5 rows, x3 = over 22 rows (all on
    consecutive sessions; `ok` marks complete windows), the target y = next row's ln RV (consecutive
    sessions) and its date. Needs sid, date, sx, rv, la."""
    d = df.sort("sid", "date").with_columns(pl.col("rv").log().alias("_l"))
    contiguous = (pl.col("sx") - pl.col("sx").shift(HAR_LAGS - 1).over("sid")) == HAR_LAGS - 1
    nxt = pl.col("sx").shift(-1).over("sid") - pl.col("sx") == 1
    return d.select(
        "sid",
        "date",
        pl.col("date").shift(-1).over("sid").alias("date_next"),
        pl.col("_l").alias("x1"),
        pl.col("_l").rolling_mean(5, min_samples=5).over("sid").alias("x2"),
        pl.col("_l").rolling_mean(HAR_LAGS, min_samples=HAR_LAGS).over("sid").alias("x3"),
        pl.when(nxt).then(pl.col("_l").shift(-1).over("sid")).alias("y"),
        contiguous.fill_null(False).alias("ok"),
        "la",
        "rv",
    )


def fit_har(design: pl.DataFrame, cutoff: date, min_rows: int | None = None) -> HarFit | None:
    """Pooled OLS on the rows whose target date is <= cutoff (None: too few rows)."""
    need = MIN_HAR_ROWS if min_rows is None else min_rows
    d = design.filter(
        pl.col("ok")
        & pl.col("y").is_not_null()
        & (pl.col("date_next") <= cutoff)
        & pl.col("la").is_not_null()
    ).drop_nulls(["x1", "x2", "x3", "y"])
    if d.height < max(need, 5):
        return None
    x = np.column_stack([np.ones(d.height), d["x1"].to_numpy(), d["x2"].to_numpy(), d["x3"].to_numpy()])
    y = d["y"].to_numpy()
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    res = y - x @ coef
    sigma2 = float(res @ res / max(d.height - 4, 1))
    la, rv = d["la"].to_numpy(), d["rv"].to_numpy()
    scale = math.sqrt(float(np.sum(la * la)) / float(np.sum(rv * rv)))
    return HarFit(tuple(float(c) for c in coef), sigma2, scale, d.height)  # type: ignore[arg-type]


def har_forecast(window: np.ndarray, fit: HarFit, horizon: int = HORIZON) -> tuple[np.ndarray, np.ndarray]:
    """(1-day vol, `horizon`-day vol) from windows of the last 22 ln RV values (newest first, [n, 22]).
    The recursion is iterated on the log scale; level = scale * exp(forecast + sigma2 / 2)."""
    w = np.array(window, dtype=float, copy=True)
    b0, b1, b2, b3 = fit.b
    level = fit.scale * math.exp(fit.sigma2 / 2.0)
    first = np.empty(len(w))
    var_sum = np.zeros(len(w))
    for j in range(horizon):
        f = b0 + b1 * w[:, 0] + b2 * w[:, :5].mean(1) + b3 * w[:, :HAR_LAGS].mean(1)
        vol = level * np.exp(f)
        if j == 0:
            first = vol
        var_sum += vol * vol
        w = np.concatenate([f[:, None], w[:, :-1]], axis=1)
    return first, np.sqrt(var_sum)


def har_series(df: pl.DataFrame, min_rows: int | None = None) -> pl.DataFrame:
    """date, sid, har_vol_1d, har_vol_10d for every row with a complete 22-row window in a year that has a
    fit (the January refit uses targets dated <= the previous 31 December). Needs sid, date, sx, rv, la."""
    design = har_design(df)
    base = df.sort("sid", "date")
    years = sorted({d.year for d in base["date"].unique().to_list()})
    parts: list[pl.DataFrame] = []
    for y in years:
        fit = fit_har(design, date(y - 1, 12, 31), min_rows)
        if fit is None:
            continue
        sub = base.filter(
            (pl.col("date") >= date(y, 1, 1) - timedelta(days=60)) & (pl.col("date") <= date(y, 12, 31))
        ).with_columns(pl.col("rv").log().alias("_l"))
        lags = [pl.col("_l").shift(k).over("sid").alias(f"_k{k}") for k in range(HAR_LAGS)]
        span = (pl.col("sx") - pl.col("sx").shift(HAR_LAGS - 1).over("sid")) == HAR_LAGS - 1
        sub = (
            sub.with_columns(*lags, span.fill_null(False).alias("_span"))
            .filter((pl.col("date") >= date(y, 1, 1)) & pl.col("_span"))
            .drop_nulls([f"_k{k}" for k in range(HAR_LAGS)])
        )
        if not sub.height:
            continue
        w = sub.select([f"_k{k}" for k in range(HAR_LAGS)]).to_numpy()
        v1, vh = har_forecast(w, fit)
        parts.append(
            pl.DataFrame({"date": sub["date"], "sid": sub["sid"], "har_vol_1d": v1, "har_vol_10d": vh})
        )
    if not parts:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "sid": df.schema["sid"],
                "har_vol_1d": pl.Float64,
                "har_vol_10d": pl.Float64,
            }
        )
    return pl.concat(parts).sort("date", "sid")


# ---- GARCH(1,1) ----------------------------------------------------------------------------------


def garch_vol_h(returns: np.ndarray, horizon: int = HORIZON) -> float:
    """sqrt of the summed `horizon`-step GARCH(1,1) variance forecasts, in return units (NaN when the fit
    fails). Zero-mean normal GARCH on the last returns (scaled by 100 for the optimiser)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 50 or not np.any(r != 0.0) or float(np.std(r)) <= 0.0:
        return math.nan
    from arch import arch_model

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(r * 100.0, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(
                disp="off", show_warning=False, options={"maxiter": 200}
            )
            if res.convergence_flag != 0:
                return math.nan
            var = res.forecast(horizon=horizon, reindex=False).variance.to_numpy()[-1]
    except Exception:  # noqa: BLE001  (a failed fit is a missing forecast, not an error)
        return math.nan
    total = float(np.sum(var))
    return math.sqrt(total) / 100.0 if np.isfinite(total) and total > 0 else math.nan


def _garch_task(args: tuple[int, np.ndarray, list[int]]) -> tuple[int, list[tuple[int, float]]]:
    sid, la, ends = args
    out = []
    for e in ends:
        win = la[max(0, e - GARCH_WINDOW + 1) : e + 1]
        win = win[np.isfinite(win)]
        out.append((e, garch_vol_h(win) if len(win) >= MIN_GARCH_OBS else math.nan))
    return sid, out


def garch_series(
    df: pl.DataFrame, *, n_jobs: int | None = None, fit_from: date | None = None
) -> pl.DataFrame:
    """date, sid, garch_vol_10d: per stock, the month-end fit's forecast applies to every row of the next
    calendar month. Needs sid, date, la. `fit_from` skips month ends before that date (history still feeds
    the window)."""
    d = df.sort("sid", "date").select("sid", "date", "la")
    d = d.with_columns((pl.col("date").dt.year() * 12 + pl.col("date").dt.month()).alias("_m"))
    tasks: list[tuple[int, np.ndarray, list[int]]] = []
    months: dict[int, np.ndarray] = {}
    dates: dict[int, list[date]] = {}
    for key, g in d.group_by("sid", maintain_order=True):
        sid = int(key[0])
        m = g["_m"].to_numpy()
        ends = [i for i in range(len(m) - 1) if m[i + 1] != m[i]]
        if fit_from is not None:
            gd = g["date"].to_list()
            ends = [i for i in ends if gd[i] >= fit_from]
        la = g["la"].fill_null(float("nan")).to_numpy()
        months[sid], dates[sid] = m, g["date"].to_list()
        if ends:
            tasks.append((sid, la, ends))
    n_fits = sum(len(t[2]) for t in tasks)
    jobs = (
        n_jobs if n_jobs is not None else (min(6, max(1, (os.cpu_count() or 2) - 2)) if n_fits > 300 else 1)
    )
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_garch_task, tasks, chunksize=8))
    else:
        results = [_garch_task(t) for t in tasks]
    sids: list[int] = []
    dts: list[date] = []
    vals: list[float] = []
    for sid, res in results:
        m = months[sid]
        by_next = {int(m[e]) + 1: v for e, v in res if math.isfinite(v)}
        for i, mi in enumerate(m):
            v = by_next.get(int(mi))
            if v is not None:
                sids.append(sid)
                dts.append(dates[sid][i])
                vals.append(v)
    return pl.DataFrame(
        {"date": dts, "sid": pl.Series(sids, dtype=df.schema["sid"]), "garch_vol_10d": vals},
        schema={"date": pl.Date, "sid": df.schema["sid"], "garch_vol_10d": pl.Float64},
    ).sort("date", "sid")


# ---- features ------------------------------------------------------------------------------------


def _rv5(ctx: FeatureContext) -> pl.DataFrame | None:
    from stocksense.core.errors import DataNotAvailable
    from stocksense.features.intraday import _session_stats

    try:
        return _session_stats(ctx).select("symbol", "date", "intraday_rv")
    except (DataNotAvailable, FileNotFoundError, KeyError, pl.exceptions.PolarsError):
        return None


def _rv_frame(ctx: FeatureContext) -> pl.DataFrame:
    cache = ctx.__dict__.setdefault("_volforecast_rv", {})
    if "rv" not in cache:
        panel = (
            ctx.panel()
            .select("sid", "date", "sx", "symbol", "adj_open", "adj_high", "adj_low", "adj_close", "ret")
            .collect()
        )
        cache["rv"] = (rv_daily(panel, _rv5(ctx)), panel.select("date", "sid"))
    return cache["rv"]


def _har_feature(ctx: FeatureContext) -> pl.LazyFrame:
    rv, keys = _rv_frame(ctx)
    return keys.join(har_series(rv), on=["date", "sid"], how="left").lazy()


def _garch_feature(ctx: FeatureContext) -> pl.LazyFrame:
    rv, keys = _rv_frame(ctx)
    return keys.join(garch_series(rv), on=["date", "sid"], how="left").lazy()


feature(
    "har_vol",
    group="risk",
    window=252,
    version=2,
    label="Forecast of the stock's daily and 10-day volatility from a HAR-RV regression (yearly refit).",
    outputs=("har_vol_1d", "har_vol_10d"),
)(_har_feature)
feature(
    "garch_vol",
    group="risk",
    window=1000,
    version=2,
    label="Forecast of the stock's 10-day volatility from a GARCH(1,1) fitted at the last month end.",
    outputs=("garch_vol_10d",),
)(_garch_feature)

from stocksense.features.registry import REGISTRY  # noqa: E402

SPECS: tuple[FeatureSpec, ...] = (REGISTRY["har_vol"], REGISTRY["garch_vol"])


# ---- the backtest loader (research worlds) ---------------------------------------------------------


def align_daily_vol(
    sid: np.ndarray, dates: Sequence[date], frame: pl.DataFrame, horizon: int = HORIZON
) -> tuple[np.ndarray, np.ndarray]:
    """(har, garch) daily-equivalent vol aligned to panel rows: the 10-day vols divided by sqrt(10) (the
    units of `vol_21`); NaN where no forecast exists."""
    n = len(sid)
    left = pl.DataFrame({"sid": sid.astype(np.int64), "date": list(dates), "_i": np.arange(n)})
    j = left.join(frame.with_columns(pl.col("sid").cast(pl.Int64)), on=["date", "sid"], how="left").sort("_i")
    root = math.sqrt(horizon)
    har = (j["har_vol_10d"].cast(pl.Float64) / root).fill_null(float("nan")).to_numpy()
    garch = (j["garch_vol_10d"].cast(pl.Float64) / root).fill_null(float("nan")).to_numpy()
    return har, garch


def cache_path(settings: Settings, end: date, fit_from: date | None, n_sids: int) -> Path:
    tag = f"{end:%Y%m%d}_{'all' if fit_from is None else f'{fit_from:%Y%m%d}'}_{n_sids}"
    return Paths(settings).root / "cache" / "volforecast" / f"vol_{tag}.parquet"


def vol_panel(
    settings: Settings,
    end: date,
    sids: Sequence[int] | None = None,
    *,
    fit_from: date | None = None,
    n_jobs: int | None = None,
    use_cache: bool = True,
) -> pl.DataFrame:
    """date, sid, har_vol_1d, har_vol_10d, garch_vol_10d over the whole history up to `end` (every number
    causal), optionally for a set of sids only. Cached under data_store/cache/volforecast."""
    path = cache_path(settings, end, fit_from, len(sids) if sids is not None else 0)
    if use_cache and path.exists():
        return pl.read_parquet(path)
    ctx = FeatureContext(settings, as_of=end)
    panel = ctx.panel().select(
        "sid", "date", "sx", "symbol", "adj_open", "adj_high", "adj_low", "adj_close", "ret"
    )
    if sids is not None:
        panel = panel.filter(pl.col("sid").is_in(list(sids)))
    panel = panel.collect()
    rv = rv_daily(panel, _rv5(ctx))
    out = har_series(rv).join(
        garch_series(rv, n_jobs=n_jobs, fit_from=fit_from), on=["date", "sid"], how="full", coalesce=True
    )
    out = out.select("date", "sid", "har_vol_1d", "har_vol_10d", "garch_vol_10d").sort("date", "sid")
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(path)
    return out
