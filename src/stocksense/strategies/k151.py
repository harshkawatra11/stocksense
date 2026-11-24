"""151 Trading Strategies (Kakushadze and Serur, SSRN 3247865): stock rules and cross-sectional scores.

PLAN-M M5.4 a. Each rule is written from the paper's formulas (section and equation numbers in the
`reference` of every spec); NSE data forces some adaptations, recorded in docs/DECISIONS.md `### M5.4a`.

Cross-sectional scores are functions of a `Wide` panel (dates x stocks matrices of completed sessions) and are
past-only by construction: the value at row t reads rows <= t. Per-stock rules are
`f -> (signal, stop)` on the daily feature dict; their exit is a condition on the closes (`STATE_RULES`,
exit key `state_exit`).
Long-only: "short the bottom decile" legs are not built; the score ranks and the evaluator takes the top.
"""

from __future__ import annotations

# ruff: noqa: E741
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, timedelta
from functools import cache
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import rankdata

from stocksense.core.config import Settings
from stocksense.data import io
from stocksense.marketlab import npx
from stocksense.strategies import riskmodel
from stocksense.strategies.base import StrategySpec, register
from stocksense.strategies.exits import EXIT_STATE, follow_state

Feat = dict[str, Any]
Out = tuple[np.ndarray, np.ndarray]
sh = npx.shift
MONTH = 21  # sessions in a "month" (PLAN-M M5.4 notation)
YEAR = 252
HP_LAMBDA = 100.0 * 12.0**2  # the paper's monthly convention, lambda = 100 n^2 with n = 12 (section 8.1)
HP_MAX_MONTHS = 120
HP_MIN_MONTHS = 36


# ---- matrix helpers (all along axis 0 = time) -------------------------------------------------------------


def sh2(x: np.ndarray, k: int) -> np.ndarray:
    """x shifted down by k rows (k > 0) with NaN in the first k rows; works for 1-D and 2-D float arrays."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    if k == 0:
        return x.copy()
    if abs(k) < len(x):
        if k > 0:
            out[k:] = x[:-k]
        else:
            out[:k] = x[-k:]
    return out


def _window_sums(x: np.ndarray, n: int, powers: tuple[int, ...]) -> tuple[list[np.ndarray], np.ndarray]:
    """Rolling sums of x**p over n rows and a bool 'window is complete' mask; prefix-only (no look-ahead)."""
    x = np.asarray(x, dtype=float)
    if len(x) < n:
        return [np.full(x.shape, np.nan) for _ in powers], np.zeros(x.shape, dtype=bool)
    fin = np.isfinite(x)
    z = np.where(fin, x, 0.0)
    pad = np.zeros((1, *x.shape[1:]))
    cnt = np.concatenate([pad, np.cumsum(fin, axis=0)])
    full = np.zeros(x.shape, dtype=bool)
    sums = []
    full[n - 1 :] = (cnt[n:] - cnt[:-n]) == n
    for p in powers:
        cs = np.concatenate([pad, np.cumsum(z**p, axis=0)])
        s = np.full(x.shape, np.nan)
        s[n - 1 :] = cs[n:] - cs[:-n]
        sums.append(np.where(full, s, np.nan))
    return sums, full


def roll_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Mean of the last n rows (NaN until n finite rows)."""
    (s1,), _ = _window_sums(x, n, (1,))
    return s1 / n


def roll_sum(x: np.ndarray, n: int) -> np.ndarray:
    """Sum of the last n rows (NaN until n finite rows)."""
    (s1,), _ = _window_sums(x, n, (1,))
    return s1


def roll_std(x: np.ndarray, n: int) -> np.ndarray:
    """Sample standard deviation (ddof = 1) of the last n rows."""
    (s1, s2), _ = _window_sums(x, n, (1, 2))
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.maximum((s2 - s1 * s1 / n) / (n - 1), 0.0))


def roll_skew(x: np.ndarray, n: int) -> np.ndarray:
    """Eq. 456: (1/(sigma^3 n)) sum (x - mean)^3 with sigma^2 of eq. 458 (ddof = 1)."""
    (s1, s2, s3), _ = _window_sums(x, n, (1, 2, 3))
    mu = s1 / n
    m3 = s3 / n - 3 * mu * s2 / n + 2 * mu**3
    var = np.maximum((s2 - s1 * s1 / n) / (n - 1), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return m3 / var**1.5


def row_demean_rank(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per row: average rank (1 = smallest) minus the mean rank over the masked finite entries (else NaN)."""
    out = np.full(x.shape, np.nan)
    ok = mask & np.isfinite(x)
    for t in range(x.shape[0]):
        idx = np.flatnonzero(ok[t])
        if len(idx) > 1:
            r = rankdata(x[t, idx], method="average")
            out[t, idx] = r - r.mean()
    return out


def _zscore_rows(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-row z-score (ddof 1) over the masked finite entries."""
    out = np.full(x.shape, np.nan)
    ok = mask & np.isfinite(x)
    for t in range(x.shape[0]):
        idx = np.flatnonzero(ok[t])
        if len(idx) > 2:
            v = x[t, idx]
            sd = v.std(ddof=1)
            if sd > 0:
                out[t, idx] = (v - v.mean()) / sd
    return out


def month_starts(dates: np.ndarray) -> np.ndarray:
    """True on the first session of each calendar month in the data (and on row 0)."""
    m = dates.astype("datetime64[M]")
    return np.r_[True, m[1:] != m[:-1]]


def completed_month_end_rows(dates: np.ndarray) -> np.ndarray:
    """Rows r whose NEXT row is in a later month, i.e. months known to be complete when row r+1 exists."""
    m = dates.astype("datetime64[M]")
    return np.flatnonzero(m[1:] != m[:-1])


def last_completed_month(dates: np.ndarray, me: np.ndarray) -> np.ndarray:
    """For each row t, the index k of the latest month-end row me[k] < t (-1 when none): known by t."""
    return np.searchsorted(me, np.arange(len(dates)), side="left") - 1


# ---- the panel --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Wide:
    """Completed daily sessions as time x stock matrices (adjusted prices; NaN where a stock has no row)."""

    dates: np.ndarray  # datetime64[D], T
    sids: np.ndarray  # int64, N
    symbols: np.ndarray  # object, N
    close: np.ndarray  # adj_close
    open: np.ndarray  # adj_open
    volume: np.ndarray  # adj_volume
    raw_close: np.ndarray
    turnover: np.ndarray  # traded value in INR
    med_turn: np.ndarray  # panel_universe.med_turn_60
    u500: np.ndarray  # bool
    fno: np.ndarray  # bool (panel_universe.in_fno)
    industry: np.ndarray  # int N (-1 unknown)
    mkt: np.ndarray  # T, panel_market.ew500_ret (a fraction)
    oi: np.ndarray | None = None  # near-future open interest, T x N (flow_oi_daily), NaN before 2024-07
    phi: np.ndarray | None = None  # near/next stock-future settle ratio, T x N (fo_bhavcopy)
    call_iv: np.ndarray | None = None  # ATM call IV of the nearest monthly option, T x N
    put_iv: np.ndarray | None = None

    @staticmethod
    def make(
        close: np.ndarray,
        dates: np.ndarray | None = None,
        *,
        u500: np.ndarray | None = None,
        fno: np.ndarray | None = None,
        industry: np.ndarray | None = None,
        mkt: np.ndarray | None = None,
        med_turn: np.ndarray | None = None,
        open_: np.ndarray | None = None,
        volume: np.ndarray | None = None,
        raw_close: np.ndarray | None = None,
        oi: np.ndarray | None = None,
        phi: np.ndarray | None = None,
        call_iv: np.ndarray | None = None,
        put_iv: np.ndarray | None = None,
    ) -> Wide:
        """A Wide with defaults for everything not given (used by tests and by the loader)."""
        c = np.asarray(close, dtype=float)
        t, n = c.shape
        if dates is None:
            dates = (np.datetime64("2015-01-01") + np.arange(t)).astype("datetime64[D]")
        ones = np.ones((t, n), dtype=bool)
        vol = np.full((t, n), 1e6) if volume is None else np.asarray(volume, dtype=float)
        return Wide(
            dates=np.asarray(dates).astype("datetime64[D]"),
            sids=np.arange(n, dtype=np.int64),
            symbols=np.array([f"S{i:03d}" for i in range(n)], dtype=object),
            close=c,
            open=c.copy() if open_ is None else np.asarray(open_, dtype=float),
            volume=vol,
            raw_close=c.copy() if raw_close is None else np.asarray(raw_close, dtype=float),
            turnover=vol * c,
            med_turn=np.where(np.isfinite(c), np.arange(1, n + 1, dtype=float)[None, :], np.nan)
            if med_turn is None
            else np.asarray(med_turn, dtype=float),
            u500=ones if u500 is None else np.asarray(u500, dtype=bool),
            fno=ones if fno is None else np.asarray(fno, dtype=bool),
            industry=np.zeros(n, dtype=np.int64)
            if industry is None
            else np.asarray(industry, dtype=np.int64),
            mkt=np.zeros(t) if mkt is None else np.asarray(mkt, dtype=float),
            oi=oi,
            phi=phi,
            call_iv=call_iv,
            put_iv=put_iv,
        )

    def truncate(self, rows: int) -> Wide:
        """The same panel with only the first `rows` sessions (the truncation test)."""
        kw: dict[str, Any] = {}
        for name in (
            "dates",
            "close",
            "open",
            "volume",
            "raw_close",
            "turnover",
            "med_turn",
            "u500",
            "fno",
            "mkt",
            "oi",
            "phi",
            "call_iv",
            "put_iv",
        ):
            v = getattr(self, name)
            kw[name] = None if v is None else v[:rows]
        return replace(self, **kw)

    @property
    def ret(self) -> np.ndarray:
        """Daily simple returns of the adjusted close (fractions)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.close / sh2(self.close, 1) - 1.0


def _pivot(dates: np.ndarray, sids: np.ndarray, df: pl.DataFrame, col: str, dtype: Any = float) -> np.ndarray:
    """df(date, sid, col) as a time x stock matrix (NaN / False where absent)."""
    fill = np.nan if dtype is float else False
    out = np.full((len(dates), len(sids)), fill, dtype=dtype)
    ri = np.searchsorted(dates, df["date"].to_numpy().astype("datetime64[D]"))
    ci = np.searchsorted(sids, df["sid"].to_numpy().astype(np.int64))
    v = df[col].to_numpy()
    out[ri, ci] = np.nan_to_num(v, nan=0).astype(bool) if dtype is bool else v.astype(float)
    return out


def load_wide(settings: Settings, start: date, end: date, pad_days: int = 2300) -> Wide:
    """The panel for sessions in [start - pad_days, end] of the stocks that were ever u500 or F&O."""
    lo = start - timedelta(days=pad_days)
    uni = io.read(settings, "panel_universe", start=lo, end=end).collect()
    keep = uni.filter(pl.col("u500") | pl.col("in_fno"))["sid"].unique().sort().to_numpy().astype(np.int64)
    cols = ["sid", "date", "symbol", "series", "close", "adj_open", "adj_close", "adj_volume", "turnover_inr"]
    px = (
        io.read(settings, "panel_daily", start=lo, end=end, columns=[*cols, "industry"])
        .filter((pl.col("series") == "EQ") & pl.col("sid").is_in(keep.tolist()))
        .unique(subset=["sid", "date"], keep="last")
        .collect()
    )
    mk = io.read(settings, "panel_market", start=lo, end=end, columns=["date", "ew500_ret"]).collect()
    dates = np.unique(px["date"].to_numpy().astype("datetime64[D]"))
    sids = np.unique(px["sid"].to_numpy().astype(np.int64))
    uni = uni.filter(pl.col("sid").is_in(sids.tolist()))
    last = (
        px.sort("date").group_by("sid").agg(pl.col("symbol").last(), pl.col("industry").drop_nulls().last())
    )
    last = last.sort("sid")
    ind_names = sorted({s for s in last["industry"].to_list() if s})
    code = {s: i for i, s in enumerate(ind_names)}
    ind = np.array([code.get(s, -1) if s else -1 for s in last["industry"].to_list()], dtype=np.int64)
    m = np.full(len(dates), np.nan)
    if mk.height:
        m[np.searchsorted(dates, mk["date"].to_numpy().astype("datetime64[D]")).clip(0, len(dates) - 1)] = mk[
            "ew500_ret"
        ].to_numpy()
    pv = {
        c: _pivot(dates, sids, px, c)
        for c in ("adj_close", "adj_open", "adj_volume", "close", "turnover_inr")
    }
    w = Wide(
        dates=dates,
        sids=sids,
        symbols=np.array(last["symbol"].to_list(), dtype=object),
        close=pv["adj_close"],
        open=pv["adj_open"],
        volume=pv["adj_volume"],
        raw_close=pv["close"],
        turnover=pv["turnover_inr"],
        med_turn=_pivot(dates, sids, uni, "med_turn_60"),
        u500=_pivot(dates, sids, uni, "u500", bool),
        fno=_pivot(dates, sids, uni, "in_fno", bool),
        industry=ind,
        mkt=np.nan_to_num(m),
    )
    return attach_derivs(w, settings, lo, end)


def near_next_ratio(fo: pl.DataFrame, min_days: int = 3) -> pl.DataFrame:
    """phi = near/next settle of stock futures (eq. 454): rows (date, symbol, phi).

    `fo` has date, symbol, expiry_date, settle_price (STF rows). near = the nearest expiry with at least
    `min_days` calendar days left (derivs.MIN_DAYS), next = the following expiry.
    """
    g = fo.filter(pl.col("settle_price") > 0).with_columns(
        (pl.col("expiry_date") - pl.col("date")).dt.total_days().alias("dte")
    )
    g = g.filter(pl.col("dte") >= min_days).sort(["date", "symbol", "expiry_date"])
    g = g.with_columns(pl.col("settle_price").cum_count().over(["date", "symbol"]).alias("k"))
    near = g.filter(pl.col("k") == 1).select("date", "symbol", pl.col("settle_price").alias("p1"))
    nxt = g.filter(pl.col("k") == 2).select("date", "symbol", pl.col("settle_price").alias("p2"))
    return near.join(nxt, on=["date", "symbol"]).select(
        "date", "symbol", (pl.col("p1") / pl.col("p2")).alias("phi")
    )


def _by_symbol(w_dates: np.ndarray, w_symbols: np.ndarray, df: pl.DataFrame, col: str) -> np.ndarray:
    """df(date, symbol, col) as a time x stock matrix aligned to the panel."""
    out = np.full((len(w_dates), len(w_symbols)), np.nan)
    if not df.height:
        return out
    pos = {s: i for i, s in enumerate(w_symbols.tolist())}
    d = df.filter(pl.col("symbol").is_in(list(pos)))
    ri = np.searchsorted(w_dates, d["date"].to_numpy().astype("datetime64[D]")).clip(0, len(w_dates) - 1)
    ci = np.array([pos[s] for s in d["symbol"].to_list()], dtype=int)
    ok = w_dates[ri] == d["date"].to_numpy().astype("datetime64[D]")
    out[ri[ok], ci[ok]] = d[col].to_numpy().astype(float)[ok]
    return out


def attach_derivs(w: Wide, settings: Settings, lo: date, end: date) -> Wide:
    """Add OI, futures-curve and option-IV matrices when their stored datasets have rows (else stay None)."""
    oi = phi = civ = piv = None
    d = io.read(settings, "flow_oi_daily", start=lo, end=end, columns=["date", "symbol", "fut_oi"]).collect()
    if d.height:
        oi = _by_symbol(w.dates, w.symbols, d, "fut_oi")
    fo = (
        io.read(
            settings,
            "fo_bhavcopy",
            start=lo,
            end=end,
            columns=["date", "symbol", "instrument_type", "expiry_date", "settle_price"],
        )
        .filter(pl.col("instrument_type") == "STF")
        .collect()
    )
    if fo.height:
        phi = _by_symbol(w.dates, w.symbols, near_next_ratio(fo), "phi")
    st = io.read(
        settings,
        "flow_options_strikes",
        start=lo,
        end=end,
        columns=["date", "underlying", "expiry", "strike", "call_iv", "put_iv"],
    ).collect()
    if st.height:
        civ, piv = _atm_ivs(w, st)
    return replace(w, oi=oi, phi=phi, call_iv=civ, put_iv=piv)


def _atm_ivs(w: Wide, st: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """ATM call and put IV of each stock's nearest listed expiry, strike nearest the cash close (T x N)."""
    spot = pl.DataFrame(
        {
            "date": np.repeat(w.dates, len(w.symbols)),
            "underlying": np.tile(w.symbols.astype(str), len(w.dates)),
            "spot": w.raw_close.ravel(),
        }
    ).filter(pl.col("spot").is_not_nan())
    d = st.join(spot, on=["date", "underlying"]).with_columns(
        pl.col("expiry").min().over(["date", "underlying"]).alias("e0")
    )
    d = d.filter(pl.col("expiry") == pl.col("e0"))
    d = d.with_columns((pl.col("strike") - pl.col("spot")).abs().alias("dist")).sort("dist")
    d = d.unique(subset=["date", "underlying"], keep="first").rename({"underlying": "symbol"})
    return (
        _by_symbol(w.dates, w.symbols, d, "call_iv"),
        _by_symbol(w.dates, w.symbols, d, "put_iv"),
    )


# ---- cross-sectional scores: one function per key, Wide -> (T x N) matrix ----------------------------------


def f_mom_cum(w: Wide) -> np.ndarray:
    """Sec 3.1 eq. 267: P(t-21)/P(t-252) - 1 (12 months, skipping the latest month)."""
    p = w.close
    with np.errstate(invalid="ignore", divide="ignore"):
        return sh2(p, MONTH) / sh2(p, YEAR) - 1.0


def f_mom_riskadj(w: Wide) -> np.ndarray:
    """Sec 3.1 eqs. 268-270: mean over the 12 formation months of the monthly return / its sample std."""
    p = w.close
    s1 = np.zeros(p.shape)
    s2 = np.zeros(p.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        for m in range(1, 13):
            r = sh2(p, MONTH * m) / sh2(p, MONTH * (m + 1)) - 1.0
            s1 += r
            s2 += r * r
        sd = np.sqrt(np.maximum((s2 - s1 * s1 / 12) / 11, 0.0))
        return (s1 / 12) / sd


def f_low_vol(w: Wide) -> np.ndarray:
    """Sec 3.4: minus the 252-session standard deviation of daily returns (ddof 1)."""
    return -roll_std(w.ret, YEAR)


def f_low_vol_126(w: Wide) -> np.ndarray:
    """Sec 3.4, the 6-month option: minus the 126-session standard deviation."""
    return -roll_std(w.ret, 126)


def delta_iv_score(call_iv: np.ndarray, put_iv: np.ndarray, k: int = MONTH) -> np.ndarray:
    """Sec 3.5: change over k sessions of the ATM call IV minus the change of the ATM put IV."""
    return (call_iv - sh2(call_iv, k)) - (put_iv - sh2(put_iv, k))


def f_implied_vol(w: Wide) -> np.ndarray:
    """Sec 3.5 (F&O stocks, from 2024-07; all NaN while flow_options_strikes is empty)."""
    if w.call_iv is None or w.put_iv is None:
        return np.full(w.close.shape, np.nan)
    return np.where(w.fno, delta_iv_score(w.call_iv, w.put_iv), np.nan)


def demeaned_rank_mean(factors: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
    """Sec 3.6 eqs. 276-277: mean over factors of (rank - mean rank) per date, names with every factor."""
    ok = mask.copy()
    for f in factors:
        ok &= np.isfinite(f)
    acc = np.zeros(mask.shape)
    for f in factors:
        acc += row_demean_rank(f, ok)
    return np.where(ok, acc / len(factors), np.nan)


def f_resid_mom(w: Wide) -> np.ndarray:
    """Sec 3.7 eqs. 278-282: mean/std of the market+size regression residuals of the 12 formation months.

    Monthly (21-session) returns, OLS with intercept over months 1..36 on MKT (compounded ew500) and SMB
    (small half minus big half of the universe by med_turn_60), alpha excluded from the residual.
    """
    p = w.close
    t, n = p.shape
    out = np.full((t, n), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r21 = p / sh2(p, MONTH) - 1.0
        mk21 = np.exp(roll_sum(np.log1p(w.mkt), MONTH)) - 1.0
    smb = np.full(t, np.nan)
    for i in range(MONTH, t):
        ok = w.u500[i] & np.isfinite(r21[i]) & np.isfinite(w.med_turn[i])
        idx = np.flatnonzero(ok)
        if len(idx) >= 4:
            small = w.med_turn[i, idx] <= np.median(w.med_turn[i, idx])
            smb[i] = r21[i, idx[small]].mean() - r21[i, idx[~small]].mean()
    months = MONTH * np.arange(1, 37)
    for i in range(int(months[-1]) + MONTH, t):
        rows = i - months
        x = np.column_stack([np.ones(36), mk21[rows], smb[rows]])
        if not np.isfinite(x).all():
            continue
        y = r21[rows]  # 36 x N
        good = np.isfinite(y).all(axis=0)
        if not good.any():
            continue
        beta = np.linalg.pinv(x) @ y[:, good]
        eps = y[:12, good] - x[:12, 1:] @ beta[1:]
        sd = eps.std(axis=0, ddof=1)
        out[i, good] = np.where(sd > 0, eps.mean(axis=0) / sd, np.nan)
    return out


def f_multifactor(w: Wide) -> np.ndarray:
    """Sec 3.6: demeaned-rank average of momentum, low volatility and residual momentum."""
    return demeaned_rank_mean([f_mom_cum(w), f_low_vol(w), f_resid_mom(w)], w.u500)


def partner_map(w: Wide, window: int = YEAR) -> np.ndarray:
    """Pairs partners (T x N int, -1 = none): at each month's first session the most correlated same-industry
    stock over the previous `window` daily returns, held until the next month's first session."""
    ret = w.ret
    t, n = ret.shape
    out = np.full((t, n), -1, dtype=np.int64)
    cur = np.full(n, -1, dtype=np.int64)
    first = month_starts(w.dates)
    for i in range(t):
        if first[i] and i >= window:
            win = ret[i - window + 1 : i + 1]
            ok = np.isfinite(win).all(axis=0) & w.u500[i] & (w.industry >= 0)
            idx = np.flatnonzero(ok)
            cur = np.full(n, -1, dtype=np.int64)
            if len(idx) > 1:
                z = win[:, idx] - win[:, idx].mean(axis=0)
                z = z / np.maximum(np.sqrt((z * z).sum(axis=0)), 1e-12)
                c = z.T @ z
                same = w.industry[idx][:, None] == w.industry[idx][None, :]
                c = np.where(same & ~np.eye(len(idx), dtype=bool), c, -np.inf)
                j = c.argmax(axis=1)
                good = np.isfinite(c[np.arange(len(idx)), j])
                cur[idx[good]] = idx[j[good]]
        out[i] = cur
    return out


def f_pairs(w: Wide) -> np.ndarray:
    """Sec 3.8 eqs. 285-289: minus (R_A - (R_A + R_B)/2), R the 5-day log return, B the monthly partner."""
    lp = np.log(w.close)
    r5 = lp - sh2(lp, 5)
    part = partner_map(w)
    rb = np.take_along_axis(r5, np.maximum(part, 0), axis=1)
    return np.where(part >= 0, -(r5 - (r5 + rb) / 2.0), np.nan)


def _r5(w: Wide) -> np.ndarray:
    lp = np.log(w.close)
    return lp - sh2(lp, 5)


def f_mr_single(w: Wide) -> np.ndarray:
    """Sec 3.9 eqs. 292-297: minus the 5-day log return less its mean over the whole universe."""
    r = np.where(w.u500, _r5(w), np.nan)
    return -(r - np.nanmean(np.where(np.isfinite(r), r, np.nan), axis=1, keepdims=True))


def cluster_wmean(
    r: np.ndarray, wgt: np.ndarray | None, mask: np.ndarray, grp: np.ndarray, min_size: int = 3
) -> np.ndarray:
    """Weighted mean of r over each name's cluster per date (T x N); clusters with fewer than `min_size`
    members that day (and names with grp < 0) are merged into one 'OTHER' cluster."""
    t, n = r.shape
    wt = np.ones_like(r) if wgt is None else wgt
    ok = mask & np.isfinite(r) & np.isfinite(wt)
    g = int(grp.max()) + 1 if len(grp) and grp.max() >= 0 else 0
    onehot = np.zeros((n, g + 1))
    gi = np.where(grp >= 0, grp, g)
    onehot[np.arange(n), gi] = 1.0
    cnt = ok.astype(float) @ onehot
    small = (cnt < min_size) | (np.arange(g + 1)[None, :] == g)
    sw = np.where(ok, wt, 0.0) @ onehot
    swr = np.where(ok, wt * np.nan_to_num(r), 0.0) @ onehot
    with np.errstate(invalid="ignore", divide="ignore"):
        own = swr / sw
        o_sw = np.where(small, sw, 0.0).sum(axis=1)
        o_swr = np.where(small, swr, 0.0).sum(axis=1)
        other = (o_swr / o_sw)[:, None]
    gm = np.where(small, other, own)
    return np.where(ok, gm[:, gi], np.nan)


def f_mr_multi(w: Wide) -> np.ndarray:
    """Sec 3.9.1 eq. 309: minus (R_i - mean of R over i's industry), small industries merged into OTHER."""
    r = _r5(w)
    return -(np.where(w.u500, r, np.nan) - cluster_wmean(r, None, w.u500, w.industry))


def _var63(w: Wide) -> np.ndarray:
    return roll_std(w.ret, 63) ** 2


def f_mr_weighted(w: Wide) -> np.ndarray:
    """Sec 3.10 eqs. 314-317: eps = R - weighted industry mean (z = 1/var_63), score = -z * eps."""
    r = _r5(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = 1.0 / _var63(w)
    z = np.where(np.isfinite(z) & (z < 1e12), z, np.nan)
    eps = np.where(w.u500, r, np.nan) - cluster_wmean(r, z, w.u500, w.industry)
    return -z * eps


def statarb_weights(e: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Sec 3.18.1 eq. 358 with a diagonal C: w_i = E_i/v_i - (sum E_j/v_j)/(sum 1/v_j)/v_i for one date."""
    a = e / var
    return a - a.sum() / (1.0 / var).sum() / var


def f_statarb_opt(w: Wide) -> np.ndarray:
    """Sec 3.18.1: dollar-neutral mean-variance weights, diagonal C = var_63, E = the k151_mr_multi score."""
    e, var = f_mr_multi(w), _var63(w)
    out = np.full(e.shape, np.nan)
    for t in range(e.shape[0]):
        ok = w.u500[t] & np.isfinite(e[t]) & np.isfinite(var[t]) & (var[t] > 0)
        if ok.sum() > 2:
            out[t, ok] = statarb_weights(e[t, ok], var[t, ok])
    return out


def f_statarb_opt_pca(w: Wide) -> np.ndarray:
    """Sec 3.18.1 with the statistical risk model of sec 6.3.1: C = diag(sigma) Psi diag(sigma), Psi refit at
    each month's first session on the previous 252 daily returns; weights by eq. 358 (unbounded)."""
    e, ret = f_mr_multi(w), w.ret
    sig = roll_std(ret, 63)
    t, n = e.shape
    out = np.full((t, n), np.nan)
    first = month_starts(w.dates)
    model: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    for i in range(t):
        if first[i] and i >= YEAR:
            win = ret[i - YEAR + 1 : i + 1]
            idx = np.flatnonzero(np.isfinite(win).all(axis=0) & w.u500[i])
            model = (idx, *riskmodel.stat_risk_model(win[:, idx].T, None)) if len(idx) > 5 else None
        if model is None:
            continue
        idx, xi2, lam, v = model
        ok = w.u500[i, idx] & np.isfinite(e[i, idx]) & np.isfinite(sig[i, idx]) & (sig[i, idx] > 0)
        if ok.sum() < 6:
            continue
        sub = np.flatnonzero(ok)
        inv_c = riskmodel.inv_cov(xi2[sub], lam, v[sub], sig[i, idx[sub]])
        wt = riskmodel.mv_weights(
            e[i, idx[sub]], inv_c, np.full(len(sub), -np.inf), np.full(len(sub), np.inf)
        )
        out[i, idx[sub]] = wt
    return out


# -- KNN (sec 3.17) --

KNN_P = (5, 10, 20, 50)
KNN_V = (5, 20)


def knn_features(
    p: np.ndarray, v: np.ndarray, p_w: tuple[int, ...] = KNN_P, v_w: tuple[int, ...] = KNN_V
) -> np.ndarray:
    """Eq. 332-style features of one stock: SMAs of price and of volume, each divided by P(t) (T x F)."""
    cols = [npx.roll(p, k, "mean") / p for k in p_w] + [npx.roll(v, k, "mean") / p for k in v_w]
    return np.column_stack(cols)


def knn_predict(
    p: np.ndarray,
    v: np.ndarray,
    horizon: int = 10,
    t_star: int = 504,
    k: int | None = None,
    p_w: tuple[int, ...] = KNN_P,
    v_w: tuple[int, ...] = KNN_V,
    block: int = 250,
    return_idx: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Single-stock KNN forecast Y_hat(t) of P(t+horizon)/P(t) - 1 (eqs. 332-341), one value per date.

    The training sample of date t is the t_star most recent dates t' with t' + horizon <= t (their label is
    known by t: the purge). Each feature is min-max normalised over that sample (eq. 337), k defaults to
    floor(sqrt(t_star)), distance is Euclidean (eq. 338), Y_hat is the mean label of the k nearest (eq. 339).
    NaN where the sample or the query has a missing value. `return_idx` also returns the neighbours' row
    indices (T x k, -1 where NaN), used by the purge test.
    """
    p = np.asarray(p, dtype=float)
    t = len(p)
    kk = int(math.isqrt(t_star)) if k is None else int(k)
    x = knn_features(p, np.asarray(v, dtype=float), p_w, v_w)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = sh2(p, -horizon) / p - 1.0
    yhat = np.full(t, np.nan)
    nb = np.full((t, kk), -1, dtype=np.int64)
    bad = ~np.isfinite(x).all(axis=1) | ~np.isfinite(y)
    cs = np.r_[0, np.cumsum(bad)]
    first_t = horizon + t_star - 1
    if t > first_t:
        xw = np.lib.stride_tricks.sliding_window_view(x, t_star, axis=0)  # (T-t_star+1, F, t_star)
        yw = np.lib.stride_tricks.sliding_window_view(y, t_star)
        for a in range(first_t, t, block):
            ts = np.arange(a, min(a + block, t))
            s = ts - horizon - t_star + 1
            clean = ((cs[s + t_star] - cs[s]) == 0) & np.isfinite(x[ts]).all(axis=1)
            if not clean.any():
                continue
            ts, s = ts[clean], s[clean]
            xs = xw[s]  # (B, F, t_star)
            ys = yw[s]  # (B, t_star)
            lo, hi = xs.min(axis=2), xs.max(axis=2)
            den = np.where(hi > lo, hi - lo, 1.0)
            xn = (xs - lo[:, :, None]) / den[:, :, None]
            q = (x[ts] - lo) / den
            d2 = ((xn - q[:, :, None]) ** 2).sum(axis=1)  # (B, t_star)
            idx = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
            yhat[ts] = np.take_along_axis(ys, idx, axis=1).mean(axis=1)
            nb[ts] = s[:, None] + idx
    return (yhat, nb) if return_idx else yhat


def f_knn_xs(w: Wide) -> np.ndarray:
    """Sec 3.17 fn. 60: the single-stock KNN forecast used as a cross-sectional score."""
    out = np.full(w.close.shape, np.nan)
    for j in range(w.close.shape[1]):
        c = w.close[:, j]
        if np.isfinite(c).sum() > 560:
            out[:, j] = knn_predict(c, w.volume[:, j])
    return out


# -- weekly futures-style scores (sec 10.3-10.4) --


def friday_anchor_rows(dates: np.ndarray) -> np.ndarray:
    """For each row t the row of the last session on or before the most recent Friday <= date(t): a weekly
    rebalance that needs no knowledge of the future (a holiday Friday falls back to the earlier session)."""
    d = dates.astype("datetime64[D]")
    dow = (d.astype("int64") + 3) % 7  # Monday = 0
    friday = d - ((dow - 4) % 7).astype("timedelta64[D]")
    return np.searchsorted(d, friday, side="right") - 1


def _weekly_hold(raw: np.ndarray, dates: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Hold each Friday's score through the following week (value at t = value at its anchor row)."""
    a = friday_anchor_rows(dates)
    held = np.where(a[:, None] >= 0, raw[np.maximum(a, 0)], np.nan)
    return np.where(mask, held, np.nan)


def f_fut_contrarian(w: Wide) -> np.ndarray:
    """Sec 10.3 eqs. 469-470: minus (R_i - R_m), 5-session returns vs the F&O mean, held from each Friday."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(w.fno, w.close / sh2(w.close, 5) - 1.0, np.nan)
        raw = -(r - np.nanmean(r, axis=1, keepdims=True))
    return _weekly_hold(raw, w.dates, w.fno)


def f_fut_contrarian_act(w: Wide) -> np.ndarray:
    """Sec 10.3.1 eqs. 472-473: the same, only for names in the upper half by volume growth and, within those,
    the lower half by open-interest growth (week = 5 sessions); needs open interest (flow_oi_daily)."""
    if w.oi is None:
        return np.full(w.close.shape, np.nan)
    base = f_fut_contrarian(w)
    with np.errstate(invalid="ignore", divide="ignore"):
        vw = roll_sum(w.volume, 5)
        v = np.log(vw / sh2(vw, 5))
        u = np.log(w.oi / sh2(w.oi, 5))
    raw = np.full(base.shape, np.nan)
    for t in range(base.shape[0]):
        ok = w.fno[t] & np.isfinite(v[t]) & np.isfinite(u[t]) & np.isfinite(base[t])
        idx = np.flatnonzero(ok)
        if len(idx) < 4:
            continue
        up = idx[v[t, idx] >= np.median(v[t, idx])]
        low = up[u[t, up] <= np.median(u[t, up])]
        raw[t, low] = base[t, low]
    return _weekly_hold(raw, w.dates, w.fno)


def _r252_sig(w: Wide) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(w.fno, w.close / sh2(w.close, YEAR) - 1.0, np.nan)
    return r, roll_std(w.ret, YEAR)


def f_fut_trend(w: Wide) -> np.ndarray:
    """Sec 10.4 eqs. 474-475: sign(R_252)/sigma_252 over the F&O universe."""
    r, s = _r252_sig(w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sign(r) / s


def f_fut_trend_tanh(w: Wide) -> np.ndarray:
    """Sec 10.4: tanh(R_252/kappa)/sigma_252, kappa = the day's cross-sectional std of R_252."""
    r, s = _r252_sig(w)
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = np.nanstd(r, axis=1, ddof=1, keepdims=True)
        return np.tanh(r / kappa) / s


def f_fut_trend_rel(w: Wide) -> np.ndarray:
    """Sec 10.4 (de-skewed variant): sign(R_i - R_m)/sigma_i over 252 sessions, R_m the F&O mean."""
    r, s = _r252_sig(w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sign(r - np.nanmean(r, axis=1, keepdims=True)) / s


# -- Hodrick-Prescott (sec 8.1, eqs. 437-439), one-sided --


@cache
def hp_last_weights(m: int, lam: float = HP_LAMBDA) -> np.ndarray:
    """Weights w with S*(m) = w . y for the HP filter of m points: the last row of (I + lam K'K)^-1."""
    k = np.zeros((m - 2, m))
    for i in range(m - 2):
        k[i, i : i + 3] = (1.0, -2.0, 1.0)
    a = np.eye(m) + lam * k.T @ k
    e = np.zeros(m)
    e[-1] = 1.0
    return np.linalg.solve(a, e)


def hp_oneside_series(
    y: np.ndarray, lam: float = HP_LAMBDA, max_m: int = HP_MAX_MONTHS, min_m: int = HP_MIN_MONTHS
) -> np.ndarray:
    """One-sided HP trend: element k is the LAST point of the HP trend solved on the data up to k only.

    The window is the stock's own history up to k, capped at max_m months (the filter's kernel is a few years
    wide, so older months do not move the end value). NaN until min_m months are available.
    """
    y = np.asarray(y, dtype=float)
    out = np.full(len(y), np.nan)
    fin = np.flatnonzero(np.isfinite(y))
    if not len(fin):
        return out
    v0 = int(fin[0])
    for k in range(v0 + min_m - 1, len(y)):
        m = min(k - v0 + 1, max_m)
        win = y[k - m + 1 : k + 1]
        if np.isfinite(win).all():
            out[k] = hp_last_weights(m, lam) @ win
    return out


def hp_ma_state(dates: np.ndarray, close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sec 8.1 per stock: (long state, valid) per day. MA(3) vs MA(12) of the one-sided HP trend of the
    monthly closes; the value on day t uses only calendar months that ended before t."""
    me = completed_month_end_rows(dates)
    s = hp_oneside_series(close[me])
    ma3, ma12 = npx.roll(s, 3, "mean"), npx.roll(s, 12, "mean")
    k = last_completed_month(dates, me)
    ok = k >= 0
    kk = np.maximum(k, 0)
    valid = ok & np.isfinite(ma3[kk]) & np.isfinite(ma12[kk]) if len(me) else np.zeros(len(dates), bool)
    state = valid & (ma3[kk] > ma12[kk]) if len(me) else valid
    return state, valid


def f_hp_mom(w: Wide) -> np.ndarray:
    """Sec 10.4 fn. 172: 12-month return of the one-sided HP trend of the monthly closes, over sigma_252."""
    me = completed_month_end_rows(w.dates)
    k = last_completed_month(w.dates, me)
    sig = roll_std(w.ret, YEAR)
    out = np.full(w.close.shape, np.nan)
    if not len(me):
        return out
    for j in range(w.close.shape[1]):
        s = hp_oneside_series(w.close[me, j])
        with np.errstate(invalid="ignore", divide="ignore"):
            mom = s / sh2(s, 12) - 1.0
        ok = k >= 0
        out[ok, j] = mom[k[ok]]
    with np.errstate(invalid="ignore", divide="ignore"):
        return out / sig


# -- single-stock characteristics from other asset classes --


def f_roll_yield(w: Wide) -> np.ndarray:
    """Sec 9.1 eq. 454: phi = P_near/P_next of the stock future (buy backwardation); from 2024-07."""
    return np.full(w.close.shape, np.nan) if w.phi is None else np.where(w.fno, w.phi, np.nan)


def f_lt_value(w: Wide) -> np.ndarray:
    """Sec 9.4 eq. 455: mean P over 5.5 to 4.5 years ago (rows t-1386..t-1134) divided by P(t)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return sh2(roll_mean(w.close, 253), 1134) / w.close


def f_skew(w: Wide) -> np.ndarray:
    """Sec 9.5 eqs. 456-458: minus the skewness of 252 daily returns (buy low skew)."""
    return -roll_skew(w.ret, YEAR)


def ou_score(logp: np.ndarray, n: int = YEAR) -> np.ndarray:
    """Sec 9.6 eq. 459: AR(1) fit X(t+1) = c + b X(t) + e over n sessions of X = ln P; (a - X_t)/sigma_inf for
    0 < b < 1 (kappa = -ln b, a = c/(1-b), sigma_inf = sd(e)/sqrt(1-b^2)), else NaN."""
    x = np.asarray(logp, dtype=float)
    x = (
        x - x[np.isfinite(x).argmax(axis=0), np.arange(x.shape[1])]
        if x.ndim == 2
        else x - x[np.isfinite(x).argmax()]
    )
    a, b = sh2(x, 1), x
    m = n - 1  # pairs in the window
    (sa, sb, saa, sab, sbb), _ = _window_sums_pairs(a, b, m)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = sa / m, sb / m
        cxx, cxy, cyy = saa - m * ma * ma, sab - m * ma * mb, sbb - m * mb * mb
        slope = cxy / cxx
        c = mb - slope * ma
        sse = np.maximum(cyy - slope * cxy, 0.0)
        sd = np.sqrt(sse / (m - 2))
        sinf = sd / np.sqrt(1.0 - slope**2)
        mean_level = c / (1.0 - slope)
        z = (mean_level - x) / sinf
    return np.where((slope > 0) & (slope < 1), z, np.nan)


def _window_sums_pairs(a: np.ndarray, b: np.ndarray, m: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Rolling sums over m rows of a, b, a*a, a*b, b*b where the pair is finite throughout the window."""
    if len(a) < m:
        return [np.full(a.shape, np.nan) for _ in range(5)], np.zeros(a.shape, dtype=bool)
    fin = np.isfinite(a) & np.isfinite(b)
    aa, bb = np.where(fin, a, 0.0), np.where(fin, b, 0.0)
    pad = np.zeros((1, *a.shape[1:]))
    cnt = np.concatenate([pad, np.cumsum(fin, axis=0)])
    full = np.zeros(a.shape, dtype=bool)
    full[m - 1 :] = (cnt[m:] - cnt[:-m]) == m
    sums = []
    for z in (aa, bb, aa * aa, aa * bb, bb * bb):
        cs = np.concatenate([pad, np.cumsum(z, axis=0)])
        s = np.full(a.shape, np.nan)
        s[m - 1 :] = cs[m:] - cs[:-m]
        sums.append(np.where(full, s, np.nan))
    return sums, full


def f_ou(w: Wide) -> np.ndarray:
    """Sec 9.6: Ornstein-Uhlenbeck mean-reversion score of ln P over 252 sessions."""
    return ou_score(np.log(w.close))


# -- distress (sec 15.3, 15.3.1) --


def f_distress_raw(w: Wide) -> np.ndarray:
    """Market-only distress proxy: mean z of {SIGMA, -EXRETAVG, -RSIZE, -ln P} over the universe (NOT a
    bankruptcy probability: no labels exist to fit the paper's logistic model)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = roll_std(w.ret, 63) * math.sqrt(252.0)
        r21 = w.close / sh2(w.close, MONTH) - 1.0
        mk21 = np.exp(roll_sum(np.log1p(w.mkt), MONTH)) - 1.0
        wsum, acc = 0.0, np.zeros(w.close.shape)
        for k in range(12):
            wk = 2.0 ** (-k / 3.0)
            ex = np.log1p(sh2(r21, MONTH * k)) - np.log1p(sh2(mk21, MONTH * k))[:, None]
            acc = acc + wk * ex
            wsum += wk
        exret = acc / wsum
        tot = np.nansum(np.where(w.u500, w.med_turn, np.nan), axis=1, keepdims=True)
        rsize = np.log(w.med_turn / tot)
        lnp = np.log(w.raw_close)
    parts = [sigma, -exret, -rsize, -lnp]
    ok = w.u500.copy()
    for p in parts:
        ok &= np.isfinite(p)
    z = sum(_zscore_rows(p, ok) for p in parts) / 4.0
    return np.where(ok, z, np.nan)


def f_distress(w: Wide) -> np.ndarray:
    """Sec 15.3: minus the distress proxy (buy the healthiest)."""
    return -f_distress_raw(w)


VOL_TARGET_BOOKS = {
    "k151_distress_voltgt": 0.12
}  # key -> sigma_target; M5.7 scales the long book's return by it


def voltgt_scale(book_ret: np.ndarray, target: float = 0.12, lookback: int = YEAR) -> np.ndarray:
    """Eq. 519 with the no-leverage prefactor (fn. 203): min(target/sigma_hat, 1) applied to day t's return,
    sigma_hat the annualised std of the book's returns over the `lookback` sessions BEFORE t."""
    sig = roll_std(np.asarray(book_ret, dtype=float), lookback) * math.sqrt(252.0)
    prior = sh2(sig, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.minimum(target / prior, 1.0)


# -- naive Bayes on headlines (sec 18.3): the model is here, the data is not --

NB_MIN_SESSIONS = 250


def bernoulli_nb_fit(
    x: np.ndarray, y: np.ndarray, alpha: float = 1.0
) -> tuple[float, np.ndarray, np.ndarray]:
    """Bernoulli naive Bayes (eqs. 538-546): (P(up), P(word|up), P(word|down)), Laplace smoothing alpha."""
    up = y > 0
    px_up = (x[up].sum(axis=0) + alpha) / (up.sum() + 2 * alpha)
    px_dn = (x[~up].sum(axis=0) + alpha) / ((~up).sum() + 2 * alpha)
    return float(up.mean()), px_up, px_dn


def bernoulli_nb_predict(model: tuple[float, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    """P(up | words present) for each row of the 0/1 keyword matrix x."""
    p_up, a, b = model
    lu = np.log(p_up) + (x * np.log(a) + (1 - x) * np.log(1 - a)).sum(axis=1)
    ld = np.log(1 - p_up) + (x * np.log(b) + (1 - x) * np.log(1 - b)).sum(axis=1)
    return 1.0 / (1.0 + np.exp(ld - lu))


def news_sessions(settings: Settings, end: date) -> int:
    """Distinct dates of stored headlines up to `end` (the dataset `news_headlines`)."""
    d = io.read(settings, "news_headlines", start=None, end=end, columns=["ts"]).collect()
    return int(d["ts"].dt.date().n_unique()) if d.height else 0


# ---- registered cross-sectional scores -------------------------------------------------------------

# key -> (function, universe flag, display name, section, one-line description)
XS: dict[str, tuple[Callable[[Wide], np.ndarray], str, str, str, str]] = {
    "k151_mom_cum": (f_mom_cum, "u500", "151 Momentum (cumulative)", "3.1", "P(t-21)/P(t-252) - 1."),
    "k151_mom_riskadj": (
        f_mom_riskadj,
        "u500",
        "151 Momentum (risk-adjusted)",
        "3.1",
        "Mean/std of the 12 formation months.",
    ),
    "k151_low_vol": (f_low_vol, "u500", "151 Low volatility (252d)", "3.4", "Minus the 252-day return std."),
    "k151_low_vol_126": (
        f_low_vol_126,
        "u500",
        "151 Low volatility (126d)",
        "3.4",
        "Minus the 126-day return std.",
    ),
    "k151_implied_vol": (
        f_implied_vol,
        "fno",
        "151 Implied volatility change",
        "3.5",
        "Change of ATM call IV minus put IV over a month.",
    ),
    "k151_multifactor": (
        f_multifactor,
        "u500",
        "151 Multifactor (ranks)",
        "3.6",
        "Demeaned-rank mean of momentum, low vol, residual momentum.",
    ),
    "k151_resid_mom": (
        f_resid_mom,
        "u500",
        "151 Residual momentum",
        "3.7",
        "Market+size regression residuals, mean/std.",
    ),
    "k151_pairs": (
        f_pairs,
        "u500",
        "151 Pairs trading",
        "3.8",
        "Reversion against the most correlated same-industry partner.",
    ),
    "k151_mr_single": (
        f_mr_single,
        "u500",
        "151 Mean reversion (single cluster)",
        "3.9",
        "Minus the 5-day return less the universe mean.",
    ),
    "k151_mr_multi": (
        f_mr_multi,
        "u500",
        "151 Mean reversion (industries)",
        "3.9.1",
        "Minus the 5-day return less the industry mean.",
    ),
    "k151_mr_weighted": (
        f_mr_weighted,
        "u500",
        "151 Mean reversion (weighted regression)",
        "3.10",
        "Weighted-regression residual, z = 1/var.",
    ),
    "k151_statarb_opt": (
        f_statarb_opt,
        "u500",
        "151 Stat-arb optimisation",
        "3.18.1",
        "Mean-variance weights, diagonal covariance.",
    ),
    "k151_statarb_opt_pca": (
        f_statarb_opt_pca,
        "u500",
        "151 Stat-arb optimisation (PCA risk model)",
        "3.18.1",
        "Mean-variance weights, statistical risk model.",
    ),
    "k151_knn_xs": (
        f_knn_xs,
        "u500",
        "151 KNN forecast (cross-section)",
        "3.17",
        "Single-stock KNN forecast ranked across stocks.",
    ),
    "k151_fut_contrarian": (
        f_fut_contrarian,
        "fno",
        "151 Futures contrarian",
        "10.3",
        "Minus the weekly return less the F&O mean.",
    ),
    "k151_fut_contrarian_act": (
        f_fut_contrarian_act,
        "fno",
        "151 Futures contrarian (activity filter)",
        "10.3.1",
        "Contrarian on rising volume and falling open interest.",
    ),
    "k151_fut_trend": (f_fut_trend, "fno", "151 Futures trend (sign)", "10.4", "sign(R252)/sigma252."),
    "k151_fut_trend_tanh": (
        f_fut_trend_tanh,
        "fno",
        "151 Futures trend (tanh)",
        "10.4",
        "tanh(R252/kappa)/sigma252.",
    ),
    "k151_fut_trend_rel": (
        f_fut_trend_rel,
        "fno",
        "151 Futures trend (relative)",
        "10.4",
        "sign(R - mean R)/sigma252.",
    ),
    "k151_hp_mom": (
        f_hp_mom,
        "u500",
        "151 HP-filter momentum",
        "10.4 fn.172",
        "12-month return of the one-sided HP trend.",
    ),
    "k151_roll_yield": (f_roll_yield, "fno", "151 Roll yield", "9.1", "Near/next stock-future settle ratio."),
    "k151_lt_value": (
        f_lt_value,
        "u500",
        "151 Long-term value",
        "9.4",
        "Mean price 5.5-4.5 years ago over the price now.",
    ),
    "k151_skew": (f_skew, "u500", "151 Skewness premium", "9.5", "Minus the 252-day return skewness."),
    "k151_ou": (
        f_ou,
        "u500",
        "151 Ornstein-Uhlenbeck reversion",
        "9.6",
        "Distance of ln P from its AR(1) mean.",
    ),
    "k151_distress": (
        f_distress,
        "u500",
        "151 Distress risk (market proxy)",
        "15.3",
        "Minus a market-only distress proxy.",
    ),
    "k151_distress_voltgt": (
        f_distress,
        "u500",
        "151 Distress risk (vol-targeted)",
        "15.3.1",
        "Same ranking; book return scaled to a 12% vol target.",
    ),
}


def xs_frame(w: Wide, score: np.ndarray, universe: str, start: date, end: date) -> pl.DataFrame:
    """(date, sid, score) for the sessions in [start, end], the universe members with a finite score."""
    mask = (w.u500 if universe == "u500" else w.fno) & np.isfinite(score)
    inside = (w.dates >= np.datetime64(start)) & (w.dates <= np.datetime64(end))
    mask &= inside[:, None]
    ri, ci = np.nonzero(mask)
    return pl.DataFrame(
        {
            "date": pl.Series(w.dates[ri]),
            "sid": pl.Series(w.sids[ci].astype(np.int32)),
            "score": pl.Series(score[ri, ci].astype(np.float32)),
        }
    ).sort(["date", "sid"])


def _score_fn(key: str) -> Callable[..., pl.DataFrame]:
    fn, uni = XS[key][0], XS[key][1]

    def run(settings: Settings, start: date, end: date) -> pl.DataFrame:
        w = load_wide(settings, start, end)
        return xs_frame(w, fn(w), uni, start, end)

    return run


def _nb_news_score(settings: Settings, start: date, end: date) -> pl.DataFrame:
    """Sec 18.3: empty until `news_headlines` holds NB_MIN_SESSIONS sessions (status await_data)."""
    schema = {"date": pl.Date, "sid": pl.Int32, "score": pl.Float32}
    if news_sessions(settings, end) < NB_MIN_SESSIONS:
        return pl.DataFrame(schema=schema)
    raise NotImplementedError(
        "k151_nb_news: the headline-to-keyword feature builder is written once data exists"
    )


# ---- per-stock rules (family m-swing): signal(f) -> (entry condition, stop), plus a state exit -----------


def X(cond: Any) -> np.ndarray:  # noqa: N802
    """Boolean array: True where cond is positive (NaN counts as False)."""
    return np.nan_to_num(np.asarray(cond, float)) > 0


def _sma(c: np.ndarray, k: int) -> np.ndarray:
    return npx.roll(c, k, "mean")


def s_ma1_sma(f: Feat) -> Out:
    """Sec 3.11 eq. 321: long while P > SMA(50); the stop is display only (never hit-tested)."""
    c = f["c"]
    return X(c > _sma(c, 50)), c - 3 * f["atr"]


def x_ma1_sma(f: Feat) -> np.ndarray:
    """Exit state of k151_ma1_sma: P < SMA(50)."""
    return X(f["c"] < _sma(f["c"], 50))


def s_ma1_ema(f: Feat) -> Out:
    """Sec 3.11 eq. 320: long while P > EMA(50) (lambda-form = span EMA); stop display only."""
    c = f["c"]
    return X(c > npx.ema(c, 50)), c - 3 * f["atr"]


def x_ma1_ema(f: Feat) -> np.ndarray:
    """Exit state of k151_ma1_ema: P < EMA(50)."""
    return X(f["c"] < npx.ema(f["c"], 50))


def s_ma2(f: Feat) -> Out:
    """Sec 3.12 eq. 322: long while SMA(10) > SMA(30)."""
    c = f["c"]
    return X(_sma(c, 10) > _sma(c, 30)), c - 3 * f["atr"]


def x_ma2(f: Feat) -> np.ndarray:
    """Exit state of k151_ma2: SMA(10) < SMA(30)."""
    c = f["c"]
    return X(_sma(c, 10) < _sma(c, 30))


def x_ma2_stop(f: Feat) -> np.ndarray:
    """Exit state of k151_ma2_stop (eq. 323): the reverse cross or a close 2% below the previous close."""
    c = f["c"]
    return x_ma2(f) | X(c < (1 - 0.02) * sh(c))


def s_ma3(f: Feat) -> Out:
    """Sec 3.13 eq. 324: long while SMA(3) > SMA(10) > SMA(21)."""
    c = f["c"]
    return X(_sma(c, 3) > _sma(c, 10)) & X(_sma(c, 10) > _sma(c, 21)), c - 3 * f["atr"]


def x_ma3(f: Feat) -> np.ndarray:
    """Exit state of k151_ma3: SMA(3) <= SMA(10)."""
    c = f["c"]
    a, b = _sma(c, 3), _sma(c, 10)
    return np.isfinite(a) & np.isfinite(b) & (a <= b)


def pivot_levels(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sec 3.14 eqs. 325-327 from the previous bar: (center C, resistance R = 2C - L, support S = 2C - H)."""
    ph, pl_, pc = sh(h), sh(l), sh(c)
    cc = (ph + pl_ + pc) / 3.0
    return cc, 2 * cc - pl_, 2 * cc - ph


def s_pivots(f: Feat) -> Out:
    """Sec 3.14 eq. 328: long while P > the pivot center; protective stop at the support S (an addition)."""
    cc, _, s = pivot_levels(f["h"], f["l"], f["c"])
    return X(f["c"] > cc), s


def x_pivots(f: Feat) -> np.ndarray:
    """Exit state of k151_pivots: the short signal P < C closes the long."""
    cc, _, _ = pivot_levels(f["h"], f["l"], f["c"])
    return X(f["c"] < cc)


def t_pivots(f: Feat) -> np.ndarray:
    """Target level of k151_pivots at each signal day: the resistance R."""
    return pivot_levels(f["h"], f["l"], f["c"])[1]


def donchian(c: np.ndarray, n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Sec 3.15: (B_up, B_down) = max and min of the previous n closes (t-n..t-1)."""
    return sh(npx.roll(c, n, "max")), sh(npx.roll(c, n, "min"))


def s_channel_mr(f: Feat) -> Out:
    """Sec 3.15 eq. 331 on daily bars: the low touches the floor B_down and the close holds above it."""
    up, dn = donchian(f["c"])
    return X(f["l"] <= dn) & X(f["c"] > dn), dn - f["atr"]


def t_channel_mr(f: Feat) -> np.ndarray:
    """Target level of k151_channel_mr at each signal day: the ceiling B_up."""
    return donchian(f["c"])[0]


def x_never(f: Feat) -> np.ndarray:
    """No state exit (the rule ends on its target, stop or time cap)."""
    return np.zeros(f["n"], dtype=bool)


def s_channel_bo(f: Feat) -> Out:
    """Sec 3.15 alternative: close above B_up on volume above 1.5x its prior 20-day mean."""
    up, dn = donchian(f["c"])
    vavg = sh(npx.roll(f["v"], 20, "mean"))
    return X(f["c"] > up) & X(f["v"] > 1.5 * vavg), dn


def x_channel_bo(f: Feat) -> np.ndarray:
    """Exit state of k151_channel_bo: close below B_down."""
    return X(f["c"] < donchian(f["c"])[1])


KNN_Z1, KNN_Z2 = 0.01, 0.0


def s_knn(f: Feat) -> Out:
    """Sec 3.17 eq. 340: long while the KNN forecast exceeds z1 = 1%; stop 2.5 ATR (an addition)."""
    yh = knn_predict(f["c"], f["v"])
    return X(yh > KNN_Z1), f["c"] - 2.5 * f["atr"]


def x_knn(f: Feat) -> np.ndarray:
    """Exit state of k151_knn (eq. 341): the forecast is at or below z2 = 0."""
    yh = knn_predict(f["c"], f["v"])
    return np.isfinite(yh) & (yh <= KNN_Z2)


def s_hp_ma(f: Feat) -> Out:
    """Sec 8.1: long while MA(3) > MA(12) of the one-sided HP trend of the monthly closes."""
    state, _ = hp_ma_state(f["date"], f["c"])
    return state, f["c"] - 3 * f["atr"]


def x_hp_ma(f: Feat) -> np.ndarray:
    """Exit state of k151_hp_ma: MA(3) has fallen below MA(12)."""
    state, valid = hp_ma_state(f["date"], f["c"])
    return valid & ~state


@dataclass(frozen=True)
class StateRule:
    """How a per-stock rule ends: the exit condition, an optional price target, and whether the stop bites."""

    exit_fn: Callable[[Feat], np.ndarray]
    target_fn: Callable[[Feat], np.ndarray] | None = None
    use_stop: bool = True


STATE_RULES: dict[str, StateRule] = {
    "k151_ma1_sma": StateRule(x_ma1_sma, use_stop=False),
    "k151_ma1_ema": StateRule(x_ma1_ema, use_stop=False),
    "k151_ma2": StateRule(x_ma2, use_stop=False),
    "k151_ma2_stop": StateRule(x_ma2_stop, use_stop=False),
    "k151_ma3": StateRule(x_ma3, use_stop=False),
    "k151_pivots": StateRule(x_pivots, t_pivots),
    "k151_channel_mr": StateRule(x_never, t_channel_mr),
    "k151_channel_bo": StateRule(x_channel_bo),
    "k151_knn": StateRule(x_knn),
    "k151_hp_ma": StateRule(x_hp_ma, use_stop=False),
}


def follow_rule(f: Feat, key: str, ie: int, stop: float) -> tuple[int, float, str]:
    """Run a per-stock rule from entry day `ie` (signal at the close of ie-1): (exit day, price, reason)."""
    rule = STATE_RULES[key]
    tgt = np.nan
    if rule.target_fn is not None and ie >= 1:
        tgt = float(rule.target_fn(f)[ie - 1])
    return follow_state(f, ie, stop, rule.exit_fn(f), tgt, rule.use_stop)


# key -> (name, signal, kind, section, description)
STOCK_RULES: dict[str, tuple[str, Callable[[Feat], Out], str, str]] = {
    "k151_ma1_sma": ("151 Single SMA(50)", s_ma1_sma, "trend", "3.11"),
    "k151_ma1_ema": ("151 Single EMA(50)", s_ma1_ema, "trend", "3.11"),
    "k151_ma2": ("151 Two moving averages", s_ma2, "trend", "3.12"),
    "k151_ma2_stop": ("151 Two moving averages + 2% stop", s_ma2, "trend", "3.12"),
    "k151_ma3": ("151 Three moving averages", s_ma3, "trend", "3.13"),
    "k151_pivots": ("151 Support and resistance (pivots)", s_pivots, "trend", "3.14"),
    "k151_channel_mr": ("151 Channel (floor bounce)", s_channel_mr, "mr", "3.15"),
    "k151_channel_bo": ("151 Channel (breakout)", s_channel_bo, "breakout", "3.15"),
    "k151_knn": ("151 KNN (single stock)", s_knn, "trend", "3.17"),
    "k151_hp_ma": ("151 Moving averages on the HP trend", s_hp_ma, "trend", "8.1"),
}


AWAIT_DATA = {"k151_implied_vol", "k151_roll_yield", "k151_fut_contrarian_act"}  # their datasets are empty


def _register_all() -> None:
    for key, (name, fn, kind, sec) in STOCK_RULES.items():
        doc = (STATE_RULES[key].exit_fn.__doc__ or "").strip()
        register(
            StrategySpec(
                key=key,
                name=name,
                source="k151",
                tier="none",
                family="m-swing",
                timeframe="1d",
                kind=kind,  # type: ignore[arg-type]
                direction="long",
                exits=(EXIT_STATE,),
                default_exit=EXIT_STATE,
                signal=fn,
                description=((fn.__doc__ or "").strip() + " " + doc).strip(),
                reference=f"151 Trading Strategies sec {sec}",
            )
        )
    for key, (_, _, name, sec, desc) in XS.items():
        register(
            StrategySpec(
                key=key,
                name=name,
                source="k151",
                tier="none",
                family="m-xs",
                timeframe="1d",
                kind="xs_factor",
                direction="long",
                exits=(),
                default_exit="",
                signal=None,
                score=_score_fn(key),
                status="await_data" if key in AWAIT_DATA else "ready",
                description=desc,
                reference=f"151 Trading Strategies sec {sec}",
            )
        )
    register(
        StrategySpec(
            key="k151_nb_news",
            name="151 Naive Bayes on headlines",
            source="k151",
            tier="none",
            family="m-xs",
            timeframe="1d",
            kind="xs_factor",
            direction="long",
            exits=(),
            default_exit="",
            signal=None,
            score=_nb_news_score,
            status="await_data",
            description="Needs 250 sessions of stored headlines (news_headlines).",
            reference="151 Trading Strategies sec 18.3",
        )
    )


_register_all()
