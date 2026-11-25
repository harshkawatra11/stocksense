"""151 Trading Strategies on NSE indices: sector rotation, multi-asset trend, vol targeting, macro rules.

PLAN-M M5.4 b. Every rule maps a panel of index closes (and a few aux series) to target weights decided at the
close of day t from rows <= t; the weight is acted on at the next open. Monthly decisions are taken on each
calendar month's first session (decidable without knowing the next row). Adaptations: docs/DECISIONS.md M5.4b.

Contract: a `timing` returns long rows (date, asset, weight) for the decision dates in [start, end]; `index_book`
turns weights and opens into the daily series (date, gross, ret_gross, net, net_2x).
"""

# ruff: noqa: E501
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from stocksense.core.config import Settings
from stocksense.costs.india import round_trip_cost_frac
from stocksense.data import io
from stocksense.features import derivs
from stocksense.strategies.base import StrategySpec, register
from stocksense.strategies.k151 import month_starts, roll_mean, roll_std, sh2

SECTORS = (
    "NIFTY AUTO",
    "NIFTY BANK",
    "NIFTY FINANCIAL SERVICES",
    "NIFTY FMCG",
    "NIFTY IT",
    "NIFTY MEDIA",
    "NIFTY METAL",
    "NIFTY PHARMA",
    "NIFTY PSU BANK",
    "NIFTY PRIVATE BANK",
    "NIFTY REALTY",
    "NIFTY ENERGY",
)
MAIN = "NIFTY 50"
MIDCAP = "NIFTY MIDCAP 150"
GSEC = "NIFTY 10 YR BENCHMARK G-SEC"
GOLD = "GOLDINR"
GSEC_BUCKETS = {  # name -> mid-maturity in years used as the duration proxy
    "NIFTY 4-8 YR G-SEC INDEX": 6.0,
    "NIFTY 8-13 YR G-SEC": 10.5,
    "NIFTY 11-15 YR G-SEC INDEX": 13.0,
    "NIFTY 15 YR AND ABOVE G-SEC INDEX": 20.0,
}
ASSETS = (*SECTORS, MAIN, MIDCAP, GSEC, GOLD, *GSEC_BUCKETS)
EVENTS_PATH = Path(__file__).resolve().parents[3] / "config" / "macro_events.toml"
ETF_BPS = 0.0005  # extra cost per complete trade (ETF proxy), PLAN-M M5.4
TOP_N = 3


@dataclass(frozen=True)
class Panel:
    """Index closes/opens by date (T x A, NaN where missing) plus aux series aligned to the same dates."""

    dates: np.ndarray
    names: tuple[str, ...]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    vix: np.ndarray  # INDIA VIX close
    usdinr: np.ndarray
    us10y: np.ndarray

    def col(self, name: str) -> np.ndarray:
        """Close series of one asset."""
        return self.close[:, self.names.index(name)]

    def cols(self, names: tuple[str, ...]) -> np.ndarray:
        """Close matrix of several assets."""
        return self.close[:, [self.names.index(n) for n in names]]


# ---- decision helpers -------------------------------------------------------------------------------------


def carry(target: np.ndarray, decide: np.ndarray) -> np.ndarray:
    """Hold the weights decided on the last decision row until the next one (rows before the first stay 0)."""
    out = np.zeros_like(target)
    cur = np.zeros(target.shape[1])
    for t in range(len(target)):
        if decide[t]:
            cur = np.nan_to_num(target[t])
        out[t] = cur
    return out


def _top_n(score: np.ndarray, n: int = TOP_N, ok: np.ndarray | None = None) -> np.ndarray:
    """Per row, 1/n on the n highest finite scores (rows with fewer than n finite names get 1/n each)."""
    w = np.zeros(score.shape)
    for t in range(score.shape[0]):
        good = np.isfinite(score[t]) & (True if ok is None else ok[t])
        idx = np.flatnonzero(good)
        if len(idx):
            top = idx[np.argsort(-score[t, idx], kind="stable")[:n]]
            w[t, top] = 1.0 / n
    return w


def _full_rows(x: np.ndarray) -> np.ndarray:
    """Rows where every sector has a finite value (a rotation needs the whole opportunity set)."""
    return np.isfinite(x).sum(axis=1) >= TOP_N


def cum_return(p: np.ndarray, n: int) -> np.ndarray:
    """Eq. 361: R_cum = P(t)/P(t-n) - 1 (no skip month)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return p / sh2(p, n) - 1.0


def sector_mom(p: np.ndarray, dates: np.ndarray, n: int = 252) -> np.ndarray:
    """Sec 4.1: each month's first session, the top 3 sectors by R_cum over n sessions, equal weight, held a month."""
    return carry(
        _top_n(np.where(_full_rows(cum_return(p, n))[:, None], cum_return(p, n), np.nan)), month_starts(dates)
    )


def sector_mom_ma(p: np.ndarray, dates: np.ndarray, n: int = 252, ma: int = 200) -> np.ndarray:
    """Sec 4.1.1 eq. 362: as sector_mom, a pick is kept only if P > SMA(200); otherwise its slot is cash."""
    r = cum_return(p, n)
    above = p > roll_mean(p, ma)
    w = _top_n(np.where(_full_rows(r)[:, None], r, np.nan), ok=above)
    return carry(w, month_starts(dates))


def sector_dual_mom(
    p: np.ndarray, bench: np.ndarray, gold_col: np.ndarray, dates: np.ndarray, n: int = 252
) -> np.ndarray:
    """Sec 4.1.2 eq. 363: sector_mom when the benchmark is above its SMA(200), else 100% gold (last column)."""
    base = sector_mom(p, dates, n)
    up = bench > roll_mean(bench, 200)
    on = np.isfinite(roll_mean(bench, 200))
    w = np.zeros((len(dates), p.shape[1] + 1))
    dec = month_starts(dates)
    tgt = np.zeros_like(w)
    tgt[:, :-1] = np.where(up[:, None], base, 0.0)
    tgt[:, -1] = np.where(on & ~up & np.isfinite(gold_col), 1.0, 0.0)
    w = carry(tgt, dec)
    # the sector part is already a carried monthly book; keep the gold switch on the same monthly clock
    return w


def regress_alpha_r2(y: np.ndarray, x: np.ndarray, n: int = 252) -> tuple[np.ndarray, np.ndarray]:
    """Rolling OLS of each column of y on x over n sessions: (alpha, R^2), each T x A (NaN until n rows)."""
    t, a = y.shape
    al = np.full((t, a), np.nan)
    r2 = np.full((t, a), np.nan)
    for i in range(n, t):
        xs, ys = x[i - n + 1 : i + 1], y[i - n + 1 : i + 1]
        if not np.isfinite(xs).all():
            continue
        xc = xs - xs.mean()
        vx = (xc**2).sum()
        for j in range(a):
            if not np.isfinite(ys[:, j]).all():
                continue
            yc = ys[:, j] - ys[:, j].mean()
            b = (xc * yc).sum() / vx
            al[i, j] = ys[:, j].mean() - b * xs.mean()
            vy = (yc**2).sum()
            r2[i, j] = (b * b * vx) / vy if vy > 0 else np.nan
    return al, r2


def _returns(p: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return p / sh2(p, 1) - 1.0


def sector_alpha(p: np.ndarray, bench: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Sec 4.2 eq. 364 (market only): the top 3 sectors by the intercept of their return on the benchmark's."""
    al, _ = regress_alpha_r2(_returns(p), _returns(bench))
    return carry(_top_n(al), month_starts(dates))


def sector_r2(p: np.ndarray, bench: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Sec 4.3: keep the sectors with R^2 below the cross-sectional median (more selective), top 3 by alpha."""
    al, r2 = regress_alpha_r2(_returns(p), _returns(bench))
    med = np.nanmedian(np.where(np.isfinite(r2) & np.isfinite(al), r2, np.nan), axis=1, keepdims=True)
    return carry(_top_n(al, ok=r2 < med), month_starts(dates))


def ibs(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:  # noqa: E741
    """Eq. 370: internal bar strength (C - L)/(H - L); NaN when H == L."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(h > l, (c - l) / (h - l), np.nan)


def sector_ibs(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:  # noqa: E741
    """Sec 4.4: every close, hold the 3 sectors with the lowest IBS of the day just closed for one session."""
    return _top_n(-ibs(h, l, c))


def trend_weights(rcum: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Eq. 373 for one date: w_i proportional to R_cum/sigma^2 over the assets with positive trend, sum 1."""
    raw = np.where(np.isfinite(rcum) & (rcum > 0) & (sigma > 0), rcum / sigma**2, 0.0)
    s = raw.sum()
    return raw / s if s > 0 else raw


def multi_asset_trend(p: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Sec 4.6: monthly; keep assets with R_cum(252) > 0 and P > SMA(200); weights by eq. 373; none -> cash."""
    r = cum_return(p, 252)
    sig = roll_std(_returns(p), 252)
    keep = p > roll_mean(p, 200)
    tgt = np.zeros(p.shape)
    for t in range(len(p)):
        tgt[t] = trend_weights(np.where(keep[t], r[t], np.nan), sig[t])
    return carry(tgt, month_starts(dates))


def vol_target_weight(vix: float, sigma_star: float = 0.15) -> float:
    """Sec 6.5: w = min(1, sigma*/sigma) with sigma = INDIA VIX / 100 (implied, fn. 120)."""
    return min(1.0, sigma_star / (vix / 100.0))


def vol_target(vix: np.ndarray, kappa: float = 0.10) -> np.ndarray:
    """Sec 6.5: NIFTY weight per day, re-set only when |dw|/w exceeds kappa (the rest is cash)."""
    out = np.zeros(len(vix))
    held = np.nan
    for t, v in enumerate(vix):
        if np.isfinite(v) and v > 0:
            w = vol_target_weight(float(v))
            if not np.isfinite(held) or abs(w - held) / held > kappa:
                held = w
        out[t] = 0.0 if not np.isfinite(held) else held
    return out


def zscore_trailing(x: np.ndarray, n: int = 756) -> np.ndarray:
    """z of x(t) against the mean and std (ddof 1) of the previous n values (excluding t)."""
    prev_mean = sh2(roll_mean(x, n), 1)
    prev_sd = sh2(roll_std(x, n), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x - prev_mean) / prev_sd


def macro_mom(nifty: np.ndarray, usdinr: np.ndarray, us10y: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Sec 19.2 (single country): mean of the z-scored 1y changes of -ln USDINR, -US10Y and the NIFTY 1y
    return; long when it is > 0, decided monthly, a change of position only after a 3-month minimum hold."""
    with np.errstate(invalid="ignore", divide="ignore"):
        trade = -(np.log(usdinr) - sh2(np.log(usdinr), 252))
    monetary = -(us10y - sh2(us10y, 252))
    risk = cum_return(nifty, 252)
    score = (zscore_trailing(trade) + zscore_trailing(monetary) + zscore_trailing(risk)) / 3.0
    w = np.zeros(len(dates))
    pos, since = 0.0, 99
    for t in np.flatnonzero(month_starts(dates)):
        since += 1
        if np.isfinite(score[t]):
            want = 1.0 if score[t] > 0 else 0.0
            if want != pos and since >= 3:
                pos, since = want, 0
        w[t:] = pos
    return w


def load_events(path: Path = EVENTS_PATH) -> dict[str, list[date]]:
    """config/macro_events.toml: {"fomc": [...], "rbi_mpc": [...]} as dates."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return {k: [date.fromisoformat(d) for d in v.get("dates", [])] for k, v in raw.items()}


def announcement_days(dates: np.ndarray, events: dict[str, list[date]]) -> np.ndarray:
    """Sec 19.5: bool per session: an RBI day itself, or the first Indian session after a US FOMC date."""
    flag = np.zeros(len(dates), dtype=bool)
    for d in events.get("rbi_mpc", []):
        i = np.searchsorted(dates, np.datetime64(d))
        if i < len(dates) and dates[i] == np.datetime64(d):
            flag[i] = True
    for d in events.get("fomc", []):
        i = np.searchsorted(dates, np.datetime64(d), side="right")  # first session strictly after the US date
        if i < len(dates):
            flag[i] = True
    return flag


def hedging_pressure_long(
    fii_long: np.ndarray, fii_short: np.ndarray, cl_long: np.ndarray, cl_short: np.ndarray, n: int = 126
) -> np.ndarray:
    """Sec 9.2: long when FII hedging pressure L/(L+S) is above its trailing-n median and CLIENT's is in the
    bottom quintile of its trailing n. Inputs are already lagged one session (published after the close)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        hf = fii_long / (fii_long + fii_short)
        hc = cl_long / (cl_long + cl_short)
    out = np.zeros(len(hf), dtype=bool)
    for t in range(n - 1, len(hf)):
        a, b = hf[t - n + 1 : t + 1], hc[t - n + 1 : t + 1]
        if np.isfinite(a).all() and np.isfinite(b).all():
            out[t] = hf[t] > np.median(a) and hc[t] <= np.quantile(b, 0.2)
    return out


def cash_and_carry(
    fut: float, spot: float, days: float, r: float = derivs.R_FREE, div: float = 0.0
) -> tuple[float, float]:
    """Sec 6.2 eqs. 421-422: (F*, basis) with F* = (S - D) exp(r (T - t)) and B = (F - F*)/S."""
    fstar = (spot - div) * float(np.exp(r * days / 365.0))
    return fstar, (fut - fstar) / spot


def barbell_weight_long(t_short: float, t_long: float, t_mid: float) -> float:
    """Eqs. 390-391 for equal yields: the dollar weight on the long bucket so that duration equals the mid's."""
    return (t_mid - t_short) / (t_long - t_short)


def basis_table(settings: Settings, start: date, end: date) -> pl.DataFrame:
    """Sec 6.2 (informational): per date and index, near future settle F, fair value F* and basis B."""
    fo = (
        io.read(
            settings,
            "fo_bhavcopy",
            start=start,
            end=end,
            columns=["date", "symbol", "instrument_type", "expiry_date", "settle_price", "underlying_price"],
        )
        .filter((pl.col("instrument_type") == "IDF") & pl.col("symbol").is_in(["NIFTY", "BANKNIFTY"]))
        .collect()
    )
    if not fo.height:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "symbol": pl.Utf8,
                "fut": pl.Float64,
                "fair": pl.Float64,
                "basis": pl.Float64,
            }
        )
    fo = fo.with_columns((pl.col("expiry_date") - pl.col("date")).dt.total_days().alias("dte"))
    near = (
        fo.filter(pl.col("dte") >= 1)
        .sort("expiry_date")
        .group_by(["date", "symbol"], maintain_order=True)
        .first()
    )
    rows = [
        (
            r["date"],
            r["symbol"],
            r["settle_price"],
            *cash_and_carry(r["settle_price"], r["underlying_price"], r["dte"]),
            r["underlying_price"],
        )
        for r in near.iter_rows(named=True)
    ]
    return pl.DataFrame(
        {
            "date": [r[0] for r in rows],
            "symbol": [r[1] for r in rows],
            "fut": [r[2] for r in rows],
            "fair": [r[3] for r in rows],
            "basis": [r[4] for r in rows],
        },
        schema={
            "date": pl.Date,
            "symbol": pl.Utf8,
            "fut": pl.Float64,
            "fair": pl.Float64,
            "basis": pl.Float64,
        },
    )


def gold_diversify_weights(p: Panel) -> np.ndarray:
    """Sec 9.3 (informational): passive 80% NIFTY 50 / 20% gold in INR, rebalanced on each month's first session."""
    tgt = np.zeros((len(p.dates), len(p.names)))
    tgt[:, p.names.index(MAIN)] = 0.8
    tgt[:, p.names.index(GOLD)] = 0.2
    return carry(tgt, month_starts(p.dates))


def bond_allocations(p: Panel) -> dict[str, np.ndarray]:
    """Secs 5.2-5.4, 5.13 (informational) on the G-Sec bucket indices: weight matrices for the bullet (the
    8-13y bucket), barbell (shortest + longest, duration-matched to the bullet), ladder (equal weight) and
    the flattener/steepener (long-minus-short bucket, flattener when the 3-month change of US10Y is positive)."""
    names = tuple(GSEC_BUCKETS)
    dur = [GSEC_BUCKETS[n] for n in names]
    n = len(p.dates)
    out: dict[str, np.ndarray] = {}
    bullet = np.zeros((n, len(p.names)))
    bullet[:, p.names.index("NIFTY 8-13 YR G-SEC")] = 1.0
    out["bullet"] = bullet
    wl = barbell_weight_long(min(dur), max(dur), GSEC_BUCKETS["NIFTY 8-13 YR G-SEC"])
    bar = np.zeros_like(bullet)
    bar[:, p.names.index(names[dur.index(min(dur))])] = 1 - wl
    bar[:, p.names.index(names[dur.index(max(dur))])] = wl
    out["barbell"] = bar
    lad = np.zeros_like(bullet)
    for nm in names:
        lad[:, p.names.index(nm)] = 1.0 / len(names)
    out["ladder"] = lad
    rising = (p.us10y - sh2(p.us10y, 63)) > 0
    sign = np.where(rising, 1.0, -1.0)  # flattener: long back leg, short front leg
    spread = np.zeros_like(bullet)
    spread[:, p.names.index(names[dur.index(max(dur))])] = sign
    spread[:, p.names.index(names[dur.index(min(dur))])] = -sign
    out["spread"] = spread
    return out


# ---- books --------------------------------------------------------------------------------------------------


def one_way_cost(settings: Settings, notional: float = 100_000.0) -> float:
    """Cost of trading one unit of weight once: half a delivery round trip plus half the 5 bps ETF proxy."""
    return round_trip_cost_frac("delivery", notional, settings.costs) / 2 + ETF_BPS / 2


def index_book(
    open_: np.ndarray,
    weights: np.ndarray,
    dates: np.ndarray,
    cost1: float,
    same_day: bool = False,
    close: np.ndarray | None = None,
) -> pl.DataFrame:
    """Daily series (date, gross, ret_gross, net, net_2x) of a weight book on the assets' opens.

    Weights decided at the close of day t are held from the open of t+1 to the open of t+2 and the return is dated
    t+1. With `same_day`, weights[s] belong to day s itself (known in advance) and earn close/open - 1.
    Turnover pays `cost1` per unit of weight changed (a full in-and-out pays a round trip plus the ETF 5 bps).
    """
    w = np.nan_to_num(weights)
    if same_day:
        assert close is not None
        r = close / open_ - 1.0
        wh = w
    else:
        r = sh2(open_, -1) / open_ - 1.0
        wh = sh2(w, 1)
        wh[0] = 0.0
    wh = np.nan_to_num(wh)
    r = np.nan_to_num(r)
    prev = np.vstack([np.zeros((1, w.shape[1])), wh[:-1]]) if not same_day else np.zeros_like(wh)
    turn = np.abs(wh - prev).sum(axis=1) if not same_day else 2.0 * np.abs(wh).sum(axis=1)
    gross = np.abs(wh).sum(axis=1)
    rg = (wh * r).sum(axis=1)
    c = turn * cost1
    keep = np.arange(len(dates)) < (len(dates) if same_day else len(dates) - 1)
    return pl.DataFrame(
        {
            "date": pl.Series(dates[keep]),
            "gross": gross[keep],
            "ret_gross": rg[keep],
            "net": (rg - c)[keep],
            "net_2x": (rg - 2 * c)[keep],
        }
    )


# ---- data -----------------------------------------------------------------------------------------------


def macro_asof(settings: Settings, series: str, dates: np.ndarray, end: date) -> np.ndarray:
    """Latest macro close known by 15:30 IST of each date (uses `available_ts`, so US closes lag a session)."""
    df = (
        io.read(settings, "macro_daily", start=None, end=end, columns=["series", "close", "available_ts"])
        .filter(pl.col("series") == series)
        .sort("available_ts")
        .collect()
    )
    if not df.height:
        return np.full(len(dates), np.nan)
    avail = (
        df["available_ts"]
        .dt.convert_time_zone("UTC")
        .dt.replace_time_zone(None)
        .to_numpy()
        .astype("datetime64[us]")
    )
    cutoff = dates.astype("datetime64[us]") + np.timedelta64(10 * 3600, "s").astype(
        "timedelta64[us]"
    )  # 15:30 IST
    pos = np.searchsorted(avail, cutoff, side="right") - 1
    vals = df["close"].to_numpy()
    return np.where(pos >= 0, vals[np.maximum(pos, 0)], np.nan)


def load_panel(settings: Settings, start: date, end: date, pad_days: int = 1500) -> Panel:
    """Index panel for [start - pad, end] from index_daily plus VIX and the macro series."""
    from datetime import timedelta

    lo = start - timedelta(days=pad_days)
    ix = io.read(
        settings,
        "index_daily",
        start=lo,
        end=end,
        columns=["index_name", "date", "open", "high", "low", "close"],
    )
    ix = ix.filter(pl.col("index_name").is_in([n for n in ASSETS if n != GOLD])).collect()
    dates = np.unique(ix.filter(pl.col("index_name") == MAIN)["date"].to_numpy().astype("datetime64[D]"))
    names = ASSETS
    mats = {k: np.full((len(dates), len(names)), np.nan) for k in ("open", "high", "low", "close")}
    ri = np.searchsorted(dates, ix["date"].to_numpy().astype("datetime64[D]")).clip(0, len(dates) - 1)
    ok = dates[ri] == ix["date"].to_numpy().astype("datetime64[D]")
    ci = np.array([names.index(n) if n in names else -1 for n in ix["index_name"].to_list()])
    for k in mats:
        mats[k][ri[ok], ci[ok]] = ix[k].to_numpy()[ok]
    usd = macro_asof(settings, "USDINR", dates, end)
    gold = macro_asof(settings, "GOLD", dates, end) * usd
    g = names.index(GOLD)
    mats["close"][:, g] = gold
    mats["open"][:, g] = sh2(gold, 1)  # gold INR as known at the previous close stands in for the open
    mats["high"][:, g] = mats["low"][:, g] = np.nan
    for k in ("open", "high", "low"):  # a zero or missing open/high/low falls back to the previous close
        bad = ~(mats[k] > 0)
        mats[k] = np.where(bad, sh2(mats["close"], 1) if k == "open" else mats["close"], mats[k])
    vix = io.read(settings, "india_vix", start=lo, end=end).collect()
    v = np.full(len(dates), np.nan)
    if vix.height:
        r = np.searchsorted(dates, vix["date"].to_numpy().astype("datetime64[D]")).clip(0, len(dates) - 1)
        m = dates[r] == vix["date"].to_numpy().astype("datetime64[D]")
        v[r[m]] = vix["close"].to_numpy()[m]
    return Panel(
        dates,
        names,
        mats["open"],
        mats["high"],
        mats["low"],
        mats["close"],
        v,
        usd,
        macro_asof(settings, "US10Y", dates, end),
    )


def _participants(settings: Settings, dates: np.ndarray, end: date) -> tuple[np.ndarray, ...]:
    df = io.read(
        settings,
        "participant_oi",
        start=None,
        end=end,
        columns=["date", "client_type", "fut_idx_long", "fut_idx_short"],
    ).collect()
    out = []
    for who in ("FII", "CLIENT"):
        for col in ("fut_idx_long", "fut_idx_short"):
            a = np.full(len(dates), np.nan)
            d = df.filter(pl.col("client_type") == who)
            r = np.searchsorted(dates, d["date"].to_numpy().astype("datetime64[D]")).clip(0, len(dates) - 1)
            m = dates[r] == d["date"].to_numpy().astype("datetime64[D]")
            a[r[m]] = d[col].to_numpy()[m]
            out.append(sh2(a, 1))  # published after the close: usable from the next session
    return tuple(out)


# ---- weight functions by key: (Panel, settings, end) -> T x A weights over `ASSETS` -------------------------


def _embed(panel: Panel, names: tuple[str, ...], w: np.ndarray) -> np.ndarray:
    out = np.zeros((len(panel.dates), len(panel.names)))
    out[:, [panel.names.index(n) for n in names]] = w
    return out


def w_sector_mom(p: Panel, **_: object) -> np.ndarray:
    return _embed(p, SECTORS, sector_mom(p.cols(SECTORS), p.dates, 252))


def w_sector_mom_6m(p: Panel, **_: object) -> np.ndarray:
    return _embed(p, SECTORS, sector_mom(p.cols(SECTORS), p.dates, 126))


def w_sector_mom_ma(p: Panel, **_: object) -> np.ndarray:
    return _embed(p, SECTORS, sector_mom_ma(p.cols(SECTORS), p.dates))


def w_sector_dual_mom(p: Panel, **_: object) -> np.ndarray:
    w = sector_dual_mom(p.cols(SECTORS), p.col(MAIN), p.col(GOLD), p.dates)
    return _embed(p, (*SECTORS, GOLD), w)


def w_sector_alpha(p: Panel, **_: object) -> np.ndarray:
    return _embed(p, SECTORS, sector_alpha(p.cols(SECTORS), p.col(MAIN), p.dates))


def w_sector_r2(p: Panel, **_: object) -> np.ndarray:
    return _embed(p, SECTORS, sector_r2(p.cols(SECTORS), p.col(MAIN), p.dates))


def w_sector_ibs(p: Panel, **_: object) -> np.ndarray:
    i = [p.names.index(n) for n in SECTORS]
    return _embed(p, SECTORS, sector_ibs(p.high[:, i], p.low[:, i], p.close[:, i]))


def w_multi_asset_trend(p: Panel, **_: object) -> np.ndarray:
    names = (MAIN, MIDCAP, GSEC, GOLD)
    return _embed(p, names, multi_asset_trend(p.cols(names), p.dates))


def w_vol_target(p: Panel, **_: object) -> np.ndarray:
    return _embed(p, (MAIN,), vol_target(p.vix)[:, None])


def w_macro_mom(p: Panel, **_: object) -> np.ndarray:
    return _embed(p, (MAIN,), macro_mom(p.col(MAIN), p.usdinr, p.us10y, p.dates)[:, None])


def w_announcement(p: Panel, events: dict[str, list[date]] | None = None, **_: object) -> np.ndarray:
    ev = load_events() if events is None else events
    return _embed(p, (MAIN,), announcement_days(p.dates, ev).astype(float)[:, None])


def w_hedging_pressure(
    p: Panel, settings: Settings | None = None, end: date | None = None, **_: object
) -> np.ndarray:
    assert settings is not None and end is not None
    fl, fs, cl, cs = _participants(settings, p.dates, end)
    on = hedging_pressure_long(fl, fs, cl, cs)
    return _embed(p, (MAIN,), on.astype(float)[:, None])


# key -> (name, weights fn, section, same_day book, description)
INDEX_RULES = {
    "k151_sector_mom": (
        "151 Sector momentum (12m)",
        w_sector_mom,
        "4.1",
        False,
        "Top 3 sector indices by 252-day return, monthly.",
    ),
    "k151_sector_mom_6m": (
        "151 Sector momentum (6m)",
        w_sector_mom_6m,
        "4.1",
        False,
        "Top 3 by 126-day return, monthly.",
    ),
    "k151_sector_mom_ma": (
        "151 Sector momentum + MA filter",
        w_sector_mom_ma,
        "4.1.1",
        False,
        "Picks held only above SMA(200).",
    ),
    "k151_sector_dual_mom": (
        "151 Sector dual momentum",
        w_sector_dual_mom,
        "4.1.2",
        False,
        "Gold in INR when NIFTY 50 is below SMA(200).",
    ),
    "k151_sector_alpha": (
        "151 Sector alpha rotation",
        w_sector_alpha,
        "4.2",
        False,
        "Top 3 by market-model alpha.",
    ),
    "k151_sector_r2": (
        "151 Sector R-squared rotation",
        w_sector_r2,
        "4.3",
        False,
        "Low R-squared half, top 3 by alpha.",
    ),
    "k151_sector_ibs": (
        "151 Sector IBS mean reversion",
        w_sector_ibs,
        "4.4",
        False,
        "3 lowest IBS sectors, one session.",
    ),
    "k151_multi_asset_trend": (
        "151 Multi-asset trend",
        w_multi_asset_trend,
        "4.6",
        False,
        "NIFTY, midcap, G-Sec, gold; R/sigma^2 weights.",
    ),
    "k151_vol_target": (
        "151 Index volatility targeting",
        w_vol_target,
        "6.5",
        False,
        "NIFTY weight 15% / (VIX/100), 10% band.",
    ),
    "k151_macro_mom": (
        "151 Macro momentum",
        w_macro_mom,
        "19.2",
        False,
        "Single-country 3-variable macro momentum.",
    ),
    "k151_announcement": (
        "151 Announcement days",
        w_announcement,
        "19.5",
        True,
        "Long NIFTY open to close on RBI/FOMC days.",
    ),
    "k151_hedging_pressure": (
        "151 Hedging pressure",
        w_hedging_pressure,
        "9.2",
        False,
        "FII vs CLIENT index-futures positioning.",
    ),
}


def run_book(settings: Settings, key: str, start: date, end: date) -> pl.DataFrame:
    """Daily series (date, gross, ret_gross, net, net_2x) of an index rule over [start, end]."""
    _, fn, _, same_day, _ = INDEX_RULES[key]
    p = load_panel(settings, start, end)
    w = fn(p, settings=settings, end=end)
    book = index_book(p.open, w, p.dates, one_way_cost(settings), same_day, p.close)
    return book.filter((pl.col("date") >= start) & (pl.col("date") <= end))


def _timing(key: str):
    def run(settings: Settings, start: date, end: date) -> pl.DataFrame:
        p = load_panel(settings, start, end)
        w = INDEX_RULES[key][1](p, settings=settings, end=end)
        ri, ci = np.nonzero(np.abs(w) > 0)
        keep = (p.dates[ri] >= np.datetime64(start)) & (p.dates[ri] <= np.datetime64(end))
        return pl.DataFrame(
            {
                "date": pl.Series(p.dates[ri][keep]),
                "asset": [p.names[c] for c in ci[keep]],
                "weight": w[ri[keep], ci[keep]],
            },
            schema={"date": pl.Date, "asset": pl.Utf8, "weight": pl.Float64},
        )

    return run


def _register_all() -> None:
    for key, (name, _fn, sec, _sd, desc) in INDEX_RULES.items():
        register(
            StrategySpec(
                key=key,
                name=name,
                source="k151",
                tier="none",
                family="m-index",
                timeframe="1d",
                kind="index_timing",
                direction="long",
                exits=(),
                default_exit="",
                signal=None,
                timing=_timing(key),
                description=desc,
                reference=f"151 Trading Strategies sec {sec}",
            )
        )
    for key, name, sec, desc in (
        (
            "k151_cash_carry",
            "151 Cash-and-carry basis",
            "6.2",
            "Informational: NIFTY/BANKNIFTY futures basis vs fair value.",
        ),
        (
            "k151_gold_diversify",
            "151 Gold diversification (80/20)",
            "9.3",
            "Informational: passive 80/20 NIFTY 50 and gold in INR.",
        ),
    ):
        register(
            StrategySpec(
                key=key,
                name=name,
                source="k151",
                tier="none",
                family="m-index",
                timeframe="1d",
                kind="index_timing",
                direction="long",
                exits=(),
                default_exit="",
                signal=None,
                description=desc + " No trial.",
                reference=f"151 Trading Strategies sec {sec}",
            )
        )


_register_all()
