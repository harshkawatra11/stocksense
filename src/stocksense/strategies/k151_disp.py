"""151 Trading Strategies sections 6.3 and 6.3.1: dispersion trading (PLAN-M M5.4, family m-opt).

Each monthly cycle sells the NIFTY ATM straddle and buys ATM straddles on the `N_STAR` = 20 index constituents
with the lowest specific variance w_i^2 sigma_i^2 xi_i^2 (eq. 427, `riskmodel.stat_risk_model` on the previous
252 daily returns). NSE index weights are not stored, so the constituents are proxied by the 50 most traded
F&O stocks on the day before entry, with equal weights, and each selected straddle carries an equal share of
the index straddle's notional (the equal-weight form of eq. 424). Docs/DECISIONS.md `### M5.0b` records this.

Returns are a fraction of the index notional, like every m-opt timing: `index_short + mean(stock_long)`.
The index leg is sized as 150 lakh INR and each stock leg as 7.5 lakh INR (one stock lot), so the flat
brokerage per order is charged in the right proportion. About 27 monthly cycles exist (2024-07 onward): low
power, stated.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from stocksense.core.config import Settings
from stocksense.core.log import get_logger
from stocksense.data import io
from stocksense.strategies import riskmodel
from stocksense.strategies.base import StrategySpec, register
from stocksense.strategies.k151 import load_wide
from stocksense.strategies.options_bt import ComboResult, EntryRule, Leg, run_combo

log = get_logger(__name__)

N_STAR = 20
N_PROXY = 50
LOOKBACK = 252
STOCK_LOT_INR = 750_000.0
INDEX_NOTIONAL_INR = STOCK_LOT_INR * N_STAR
START = date(2024, 7, 1)
SHORT_STRADDLE = (Leg("CE", -1), Leg("PE", -1))
LONG_STRADDLE = (Leg("CE", 1), Leg("PE", 1))
_TO_INDEX_TYPE = {"STF": "IDF", "STO": "IDO"}


def lowest_specific_variance(returns: np.ndarray, weights: np.ndarray, n_star: int) -> np.ndarray:
    """Row indices (ascending) of the `n_star` names with the lowest w_i^2 sigma_i^2 xi_i^2 (eq. 427).

    `returns` is N names x T periods, `weights` the N index weights, sigma_i the sample std (ddof 1)
    and xi_i^2 the model's specific variance of the correlation matrix. Ties break to the lower row.
    """
    x = np.asarray(returns, dtype=float)
    xi2, _, _ = riskmodel.stat_risk_model(x, None)
    spec = np.asarray(weights, dtype=float) ** 2 * x.var(axis=1, ddof=1) * xi2
    return np.sort(np.argsort(spec, kind="stable")[: min(n_star, len(spec))])


def pick_constituents(
    close: np.ndarray,
    dates: np.ndarray,
    symbols: np.ndarray,
    med_turn: np.ndarray,
    fno: np.ndarray,
    entry: date,
    *,
    n_proxy: int = N_PROXY,
    n_star: int = N_STAR,
    lookback: int = LOOKBACK,
) -> list[str]:
    """Symbols of the straddles to buy for the cycle entered on `entry`, decided on the previous close.

    Proxy index = the `n_proxy` F&O names with the highest 60-day median turnover on the last session before
    `entry`; names without `lookback` finite returns are dropped; empty when fewer than `n_star` remain.
    """
    t = int(np.searchsorted(dates, np.datetime64(entry))) - 1
    if t < lookback:
        return []
    ok = np.flatnonzero(fno[t] & np.isfinite(med_turn[t]))
    top = ok[np.argsort(-med_turn[t, ok], kind="stable")[:n_proxy]]
    win = np.log(close[t - lookback + 1 : t + 1, top] / close[t - lookback : t, top])
    keep = np.isfinite(win).all(axis=0)
    top, win = top[keep], win[:, keep]
    if len(top) < n_star:
        return []
    w = np.full(len(top), 1.0 / len(top))
    sel = lowest_specific_variance(win.T, w, n_star)
    return [str(symbols[top[i]]) for i in sel]


def dispersion_daily(
    index: ComboResult, stocks: dict[str, ComboResult], picks: dict[date, list[str]]
) -> pl.DataFrame:
    """`(date, ret, cost)` of the dispersion book.

    `index` is the short ATM straddle run, `stocks` the long ATM straddle runs per symbol and `picks`
    the names chosen at each index cycle's entry date. Inside a cycle `ret` is the index leg plus the
    mean of the stock legs; a cycle with no picks is flat (zero). `cost` is the round-trip charge already
    inside `ret`, booked on the exit date: the index cycle cost plus the mean of the stock cycle costs.
    """
    daily = index.daily.sort("date")
    ret = dict(zip(daily["date"].to_list(), daily["ret"].to_list(), strict=True))
    out_ret = dict.fromkeys(ret, 0.0)
    out_cost = dict.fromkeys(ret, 0.0)
    stock_ret = {
        s: dict(zip(r.daily["date"].to_list(), r.daily["ret"].to_list(), strict=True))
        for s, r in stocks.items()
    }
    stock_cost = {
        s: list(zip(r.cycles["exit_date"].to_list(), r.cycles["cost"].to_list(), strict=True))
        for s, r in stocks.items()
    }
    for c in index.cycles.iter_rows(named=True):
        names = [n for n in picks.get(c["entry_date"], []) if n in stocks]
        if not names:
            continue
        lo, hi = c["entry_date"], c["exit_date"]
        for d in (d for d in ret if lo <= d <= hi):
            out_ret[d] = ret[d] + sum(stock_ret[n].get(d, 0.0) for n in names) / len(names)
        out_cost[hi] = c["cost"] + sum(x for n in names for e, x in stock_cost[n] if lo <= e <= hi) / len(
            names
        )
    dates = sorted(out_ret)
    return pl.DataFrame(
        {"date": dates, "ret": [out_ret[d] for d in dates], "cost": [out_cost[d] for d in dates]},
        schema={"date": pl.Date, "ret": pl.Float64, "cost": pl.Float64},
    )


def _fo_symbol(settings: Settings, symbol: str, types: list[str]) -> pl.DataFrame:
    return (
        io.read(settings, "fo_bhavcopy")
        .filter((pl.col("symbol") == symbol) & pl.col("instrument_type").is_in(types))
        .collect()
    )


def _lot(fo: pl.DataFrame, notional: float) -> float:
    spot = fo["underlying_price"].drop_nulls()
    return float(notional / spot.median()) if spot.len() else 1.0  # type: ignore[arg-type]


def dispersion_with_cost(
    settings: Settings, start: date | None = None, end: date | None = None
) -> pl.DataFrame:
    """Real-data dispersion book: `(date, ret, cost)` over [start, end] (default from 2024-07)."""
    idx_fo = _fo_symbol(settings, "NIFTY", ["IDF", "IDO"])
    if not idx_fo.height:
        return pl.DataFrame(schema={"date": pl.Date, "ret": pl.Float64, "cost": pl.Float64})
    last = idx_fo["date"].max()
    index = run_combo(
        SHORT_STRADDLE, EntryRule(), settings, "NIFTY", fo=idx_fo, lot=_lot(idx_fo, INDEX_NOTIONAL_INR)
    )
    w = load_wide(settings, START, end or last, pad_days=700)  # type: ignore[arg-type]
    picks: dict[date, list[str]] = {}
    for d in index.cycles["entry_date"].to_list():
        picks[d] = pick_constituents(w.close, w.dates, w.symbols, w.med_turn, w.fno, d)
    stocks: dict[str, ComboResult] = {}
    for sym in sorted({n for v in picks.values() for n in v}):
        fo = _fo_symbol(settings, sym, ["STF", "STO"]).with_columns(
            pl.col("instrument_type").replace(_TO_INDEX_TYPE)
        )
        if fo.height:
            stocks[sym] = run_combo(
                LONG_STRADDLE, EntryRule(), settings, sym, fo=fo, lot=_lot(fo, STOCK_LOT_INR)
            )
    log.info("dispersion_built", cycles=index.cycles.height, names=len(stocks))
    out = dispersion_daily(index, stocks, picks)
    if start is not None:
        out = out.filter(pl.col("date") >= start)
    if end is not None:
        out = out.filter(pl.col("date") <= end)
    return out


def dispersion_returns(
    settings: Settings, start: date | None = None, end: date | None = None
) -> pl.DataFrame:
    """Spec `timing` of k151_dispersion: `(date, ret)` net of option charges, a fraction of index notional."""
    return dispersion_with_cost(settings, start, end).select("date", "ret")


dispersion_returns.with_cost = dispersion_with_cost  # type: ignore[attr-defined]

register(
    StrategySpec(
        key="k151_dispersion",
        name="151 s6.3 dispersion trading (NIFTY)",
        source="k151",
        tier="none",
        family="m-opt",
        timeframe="1d",
        kind="options",
        direction="both",
        exits=("expiry",),
        default_exit="expiry",
        signal=None,
        timing=dispersion_returns,
        description="Short NIFTY ATM straddle vs long straddles on the 20 lowest specific-variance names.",
        reference="Kakushadze and Serur (2018) s6.3, s6.3.1, eqs. 424-427",
    )
)
