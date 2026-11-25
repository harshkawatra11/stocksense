"""151 Trading Strategies, Appendix A: delay-0 mean reversion and delay-1 momentum, open to close.

PLAN-M M5.4 a. The paper's R code is copyrighted and is not copied: this is a numpy re-implementation of
the procedure the appendix describes (eqs. 548-557). Every month the universe (top `n_top` by 21-day
average daily dollar volume) and the statistical risk model are refit on past data only. Each day the
expected return E_i is cut by the linear trading cost (eq. 551), the mean-variance weights are bounded by 1%
of ADDV (eq. 548), the book is dollar-neutral with gross exposure 1, and the position is opened at the
open and closed at the close.

Delay 0 trades the overnight gap of the same day (E = ln(open_t / close_{t-1}), held with the opposite sign:
mean reversion); the paper itself calls it "borderline in-sample" because the gap is only known once the open
has printed. Delay 1 trades yesterday's intraday move (E = -ln(close_{t-1}/open_{t-1}), held with the opposite
sign of the optimum: momentum). The sign conventions are pinned by tests/unit/test_k151_app.py.
"""

from __future__ import annotations

from datetime import date, time
from typing import Literal

import numpy as np
import polars as pl

from stocksense.core.config import Settings
from stocksense.costs.india import round_trip_cost_frac
from stocksense.data.ingest_intraday import load_bars
from stocksense.strategies import riskmodel
from stocksense.strategies.base import StrategySpec, register
from stocksense.strategies.k151 import Wide, load_wide, month_starts, roll_mean, roll_std, sh2

Delay = Literal[0, 1]
D_R = 21  # risk-model lookback and re-fit interval (the paper's d.r)
D_ADDV = 21
N_TOP = 200  # the paper's top 2,000 by ADDV scaled to NSE
BNDS = 0.01  # eq. 548: |H_i| <= 0.01 ADDV_i
MEAN_COST = 1e-3  # the paper's heuristic: average linear trading cost per dollar 10 bps
FIVE_MIN_FILL = time(9, 20)


def expected_return(delay: Delay, open_: np.ndarray, close: np.ndarray, t: int) -> np.ndarray:
    """E_i on day t (adjusted prices): d0 = ln(open_t/close_{t-1}), d1 = -ln(close_{t-1}/open_{t-1})."""
    with np.errstate(invalid="ignore", divide="ignore"):
        if delay == 0:
            return np.log(open_[t] / close[t - 1])
        return -np.log(close[t - 1] / open_[t - 1])


def linear_cost(sigma: np.ndarray, addv: np.ndarray, mean_cost: float = MEAN_COST) -> np.ndarray:
    """tau_i = zeta sigma_i / ADDV_i with zeta set so that mean(tau) = mean_cost (eqs. 553-554)."""
    r = sigma / addv
    return mean_cost * r / r.mean()


def effective_return(e: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Eq. 551: E_eff = sign(E) max(|E| - tau, 0)."""
    return np.sign(e) * np.maximum(np.abs(e) - tau, 0.0)


def holdings(
    e_eff: np.ndarray, inv_c: np.ndarray, bound: np.ndarray | float, sign: float = -1.0
) -> np.ndarray:
    """Dollar-neutral holdings as fractions of the investment level (sum |h| = 1), `sign` times the optimum.

    The mean-variance optimum (eq. 358) is first scaled to gross 1 so that the bound |h_i| <= bound_i (a
    fraction of the investment level) is comparable; it is solved with the bound, then the book is re-scaled
    to gross 1 as the paper's code does. sign = -1 is the paper's `des.hold <- -des.hold` for both delays.
    """
    n = len(e_eff)
    b = np.broadcast_to(np.asarray(bound, dtype=float), (n,))
    free = riskmodel.mv_weights(e_eff, inv_c, np.full(n, -np.inf), np.full(n, np.inf))
    gross = np.abs(free).sum()
    if not np.isfinite(gross) or gross <= 0:
        return np.zeros(n)
    w = riskmodel.mv_weights(e_eff / gross, inv_c, -b, b)
    g = np.abs(w).sum()
    return sign * w / g if g > 0 else np.zeros(n)


def five_minute_open(settings: Settings, w: Wide, start: date, end: date) -> np.ndarray:
    """Open of the 09:20 5-minute bar per date and stock (T x N, NaN where there is no bar): the 5m fill."""
    out = np.full(w.close.shape, np.nan)
    df = load_bars(settings, w.symbols.astype(str).tolist(), start, end, interval="5m", basis="adjusted")
    if not df.height:
        return out
    m = df.with_columns(
        pl.col("ts").dt.date().alias("d"),
        (pl.col("ts").dt.hour().cast(pl.Int64) * 60 + pl.col("ts").dt.minute().cast(pl.Int64)).alias("mod"),
    ).filter(pl.col("mod") == FIVE_MIN_FILL.hour * 60 + FIVE_MIN_FILL.minute)
    pos = {s: i for i, s in enumerate(w.symbols.astype(str).tolist())}
    for sym, d, op in zip(m["symbol"].to_list(), m["d"].to_list(), m["open"].to_list(), strict=True):
        r = int(np.searchsorted(w.dates, np.datetime64(d)))
        if r < len(w.dates) and w.dates[r] == np.datetime64(d) and sym in pos:
            out[r, pos[sym]] = op
    return out


def appa_run(
    w: Wide,
    delay: Delay,
    capital_inr: float,
    settings: Settings,
    *,
    entry_open: np.ndarray | None = None,
    n_top: int = N_TOP,
    d_r: int = D_R,
    d_addv: int = D_ADDV,
    bnds: float = BNDS,
) -> pl.DataFrame:
    """Daily series of an Appendix A strategy: (date, gross, ret_gross, net, net_2x, n_names).

    `capital_inr` sets the position bound of eq. 548 (a bigger book hits the 1%-of-ADDV bounds sooner). `net`
    deducts the paper's linear cost tau and the statutory intraday round trip; `net_2x` doubles both.
    `entry_open` replaces the adjusted open as the entry price (the 5-minute variant).
    """
    o, c = w.open, w.close
    fill = o if entry_open is None else entry_open
    with np.errstate(invalid="ignore", divide="ignore"):
        lc = np.log(c / sh2(c, 1))
    addv = sh2(roll_mean(w.turnover, d_addv), 1)  # days before t only
    sig = sh2(roll_std(lc, d_r), 1)
    first = month_starts(w.dates)
    t_n = len(w.dates)
    model: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    rows: list[tuple[date, float, float, float, float, int]] = []
    for t in range(max(d_r, d_addv) + 2, t_n):
        if first[t] or model is None:
            win = lc[t - d_r : t]
            ok = np.isfinite(win).all(axis=0) & np.isfinite(addv[t])
            cand = np.flatnonzero(ok)
            if len(cand) < 20:
                model = None
                continue
            take = cand[np.argsort(-addv[t, cand])[:n_top]]
            sd = np.maximum(win[:, take].std(axis=0, ddof=1), 1e-6)
            xi2, lam, v = riskmodel.stat_risk_model(win[:, take].T, None)
            model = (take, sd, xi2, lam, v)
        take, sd, xi2, lam, v = model
        e = expected_return(delay, o, c, t)[take]
        px_in = fill[t, take]
        ok = np.isfinite(e) & np.isfinite(px_in) & np.isfinite(c[t, take]) & np.isfinite(sig[t, take])
        ok &= np.isfinite(addv[t, take]) & (px_in > 0)
        if ok.sum() < 20:
            continue
        sub = np.flatnonzero(ok)
        names = take[sub]
        tau = linear_cost(sig[t, names], addv[t, names])
        e_eff = effective_return(e[sub], tau)
        inv_c = riskmodel.inv_cov(xi2[sub], lam, v[sub], sd[sub])
        h = holdings(e_eff, inv_c, bnds * addv[t, names] / capital_inr)
        gross = float(np.abs(h).sum())
        if gross <= 0:
            continue
        r_i = c[t, names] / px_in[sub] - 1.0
        ret_gross = float(h @ r_i)
        cost_tau = float(np.abs(h) @ tau)
        per_name = capital_inr * gross / max(len(names), 1)
        stat = gross * round_trip_cost_frac("intraday", per_name, settings.costs)
        rows.append(
            (
                date.fromisoformat(str(w.dates[t])),
                gross,
                ret_gross,
                ret_gross - cost_tau - stat,
                ret_gross - 2 * (cost_tau + stat),
                len(names),
            )
        )
    schema = {
        "date": pl.Date,
        "gross": pl.Float64,
        "ret_gross": pl.Float64,
        "net": pl.Float64,
        "net_2x": pl.Float64,
        "n_names": pl.Int32,
    }
    return pl.DataFrame(rows, schema=schema, orient="row")


def _timing(delay: Delay, five_min: bool = False):
    def run(settings: Settings, start: date, end: date, capital_inr: float | None = None) -> pl.DataFrame:
        w = load_wide(settings, start, end, pad_days=120)
        cap = settings.gate.research.reference_capital_inr if capital_inr is None else capital_inr
        entry = five_minute_open(settings, w, start, end) if five_min else None
        out = appa_run(w, delay, cap, settings, entry_open=entry)
        return out.filter((pl.col("date") >= start) & (pl.col("date") <= end))

    return run


for _key, _name, _delay, _five, _desc in (
    (
        "k151_appA_d0_mr",
        "151 Appendix A delay-0 mean reversion",
        0,
        False,
        "Overnight gap reversed, open to close.",
    ),
    (
        "k151_appA_d0_mr_5m",
        "151 Appendix A delay-0 (5m fill)",
        0,
        True,
        "Same, entered at the 09:20 5-minute open.",
    ),
    ("k151_appA_d1_mom", "151 Appendix A delay-1 momentum", 1, False, "Yesterday's intraday move continued."),
):
    register(
        StrategySpec(
            key=_key,
            name=_name,
            source="k151",
            tier="none",
            family="m-intra",
            timeframe="1d",
            kind="event",
            direction="both",
            exits=("eod",),
            default_exit="eod",
            signal=None,
            timing=_timing(_delay, _five),  # type: ignore[arg-type]
            description=_desc + " Dollar-neutral mean-variance book, top 200 by ADDV, PCA risk model.",
            reference="151 Trading Strategies sec A (eqs. 548-557)",
        )
    )
