"""Options positioning from the F&O bhavcopy: PCR, walls, max pain, implied vol and gamma exposure.

Conventions (an approximation; the true dealer inventory is unknown):
- OI is in shares (the bhavcopy value), so gamma exposure needs no lot size.
- Dealers are assumed long calls and short puts (the common SpotGamma-style convention): calls count
  positive, puts negative.
- GEX per strike = gamma x OI x spot^2 x 0.01, in rupees per 1% move of the underlying.
- Positive total GEX: dealers hedge against the move (buy dips, sell rallies), a damping, range regime.
  Negative total GEX: they hedge with the move and amplify it, a trending, volatile regime.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl

from stocksense.core.config import Settings
from stocksense.data import io
from stocksense.features.derivs import MIN_DAYS, R_FREE, implied_vol, max_pain

GRID = np.arange(-0.10, 0.10 + 1e-9, 0.0025)  # zero-gamma search grid, as a fraction of spot
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_OPT_COLS = [
    "expiry_date",
    "strike_price",
    "option_type",
    "settle_price",
    "oi",
    "change_in_oi",
    "underlying_price",
    "iv",
]


def bs_gamma(s: float, k: float, t: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma per rupee of spot: phi(d1) / (s sigma sqrt(t)); 0 if t <= 0 or sigma <= 0."""
    if t <= 0 or sigma <= 0 or s <= 0 or k <= 0:
        return 0.0
    v = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / v
    return math.exp(-0.5 * d1 * d1) / _SQRT_2PI / (s * v)


def _gamma_vec(s: np.ndarray, k: np.ndarray, t: np.ndarray, r: float, sigma: np.ndarray) -> np.ndarray:
    """Vectorised `bs_gamma`; s may be a column (grid) against row vectors k, t, sigma."""
    v = sigma * np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * sigma * sigma) * t) / v
    return np.exp(-0.5 * d1 * d1) / _SQRT_2PI / (s * v)


def chain_frame(fo: pl.DataFrame, d: date, underlying: str, n_expiries: int = 2) -> pl.DataFrame:
    """Option rows (IDO/STO) of one underlying on date d for the nearest expiries strictly after d.

    Expiry-day rows are excluded (settle_price equals the underlying then). Adds dte (calendar days)
    and t = dte / 365. An optional `iv` column (NSE's own implied vol) is carried through.
    """
    cols = [c for c in _OPT_COLS if c in fo.columns]
    ch = fo.filter(
        (pl.col("date") == d)
        & (pl.col("symbol") == underlying)
        & pl.col("instrument_type").is_in(["IDO", "STO"])
        & (pl.col("expiry_date") > d)
    ).select(cols)
    keep = ch["expiry_date"].unique().sort().head(n_expiries).to_list()
    dte = (pl.col("expiry_date") - pl.lit(d)).dt.total_days()
    return (
        ch.filter(pl.col("expiry_date").is_in(keep))
        .with_columns(dte.alias("dte"))
        .with_columns((pl.col("dte") / 365.0).alias("t"))
        .sort("expiry_date", "strike_price", "option_type")
    )


def _zero_gamma(
    spot: float, k: np.ndarray, t: np.ndarray, iv: np.ndarray, w: np.ndarray, oi: np.ndarray, r: float
) -> float | None:
    """Spot level nearest to `spot` where total GEX changes sign on the +-10% grid (IV held fixed)."""
    ok = np.isfinite(iv) & (oi > 0)
    if not ok.any():
        return None
    levels = spot * (1.0 + GRID)
    g = _gamma_vec(levels[:, None], k[ok][None, :], t[ok][None, :], r, iv[ok][None, :])
    tot = (g * (w[ok] * oi[ok])[None, :] * levels[:, None] ** 2 * 0.01).sum(axis=1)
    best: float | None = None
    for i in range(len(levels) - 1):
        a, b = tot[i], tot[i + 1]
        if a == 0.0 or a * b < 0.0:
            x = levels[i] if a == 0.0 else levels[i] + (levels[i + 1] - levels[i]) * a / (a - b)
            if best is None or abs(x - spot) < abs(best - spot):
                best = float(x)
    if best is None and tot[-1] == 0.0:
        best = float(levels[-1])
    return best


def _empty_metrics() -> dict:
    keys = [
        "atm_iv",
        "skew",
        "pcr_oi",
        "pcr_doi",
        "max_pain",
        "max_pain_dist_pct",
        "call_wall",
        "put_wall",
        "gex_total_cr",
        "zero_gamma",
        "zero_gamma_dist_pct",
        "near_expiry",
    ]
    strikes = pl.DataFrame(
        schema={
            "expiry": pl.Date,
            "strike": pl.Float64,
            "call_oi": pl.Int64,
            "put_oi": pl.Int64,
            "call_doi": pl.Int64,
            "put_doi": pl.Int64,
            "call_iv": pl.Float64,
            "put_iv": pl.Float64,
            "gex_cr": pl.Float64,
            "oi_imbalance": pl.Int64,
        }
    )
    return {**dict.fromkeys(keys), "strikes": strikes}


def chain_metrics(ch: pl.DataFrame, spot: float, r: float = R_FREE) -> dict:
    """Positioning metrics of one underlying-day chain (see the module docstring for the GEX convention).

    Implied vol comes from `settle_price` (or the chain's `iv` column when present); rows with dte below
    MIN_DAYS are skipped. Walls, PCR and GEX use every expiry in `ch`; atm_iv, skew and max pain use the
    nearest expiry. The result also carries `strikes`, a per (expiry, strike) frame.
    """
    if ch.height == 0 or not spot or spot <= 0:
        return _empty_metrics()
    k = ch["strike_price"].to_numpy().astype(float)
    t = ch["t"].to_numpy().astype(float)
    dte = ch["dte"].to_numpy()
    oi = ch["oi"].fill_null(0).to_numpy().astype(float)
    price = ch["settle_price"].to_numpy().astype(float)
    exp = ch["expiry_date"].to_list()
    is_call = ch["option_type"].str.to_uppercase().str.starts_with("C").to_numpy()
    given = ch["iv"].to_numpy().astype(float) if "iv" in ch.columns else np.full(len(k), np.nan)
    iv = np.full(len(k), np.nan)
    for i in range(len(k)):
        if dte[i] < MIN_DAYS:
            continue
        if np.isfinite(given[i]) and given[i] > 0:
            iv[i] = given[i]
        else:
            v = implied_vol(float(price[i]), spot, float(k[i]), float(t[i]), bool(is_call[i]), r)
            if v is not None:
                iv[i] = v
    good = np.isfinite(iv)
    sign = np.where(is_call, 1.0, -1.0)
    gamma = np.zeros(len(k))
    gamma[good] = _gamma_vec(np.full(good.sum(), spot), k[good], t[good], r, iv[good])
    gex = gamma * oi * spot * spot * 0.01 * sign
    gex_total = float(gex.sum() / 1e7)
    zg = _zero_gamma(spot, k, t, iv, sign, oi, r)

    near = min(exp)
    at_near = np.array([e == near for e in exp])
    doi = ch["change_in_oi"].fill_null(0).to_numpy().astype(float)
    c_oi, p_oi = oi[is_call].sum(), oi[~is_call].sum()
    c_doi, p_doi = doi[is_call].sum(), doi[~is_call].sum()

    def by_strike(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.unique(k[mask])
        return u, np.array([oi[mask & (k == x)].sum() for x in u])

    def wall(mask: np.ndarray, above: bool) -> float | None:
        u, o = by_strike(mask)
        sel = (u >= spot) if above else (u <= spot)
        if not sel.any() or o[sel].max() <= 0:
            return None
        return float(u[sel][int(np.argmax(o[sel]))])

    ns = np.unique(k[at_near])
    mp = max_pain(
        ns,
        np.array([oi[at_near & is_call & (k == x)].sum() for x in ns]),
        np.array([oi[at_near & ~is_call & (k == x)].sum() for x in ns]),
    )

    def iv_at(mask: np.ndarray, target: float) -> float | None:
        idx = np.flatnonzero(mask & good)
        if len(idx) == 0:
            return None
        return float(iv[idx[int(np.argmin(np.abs(k[idx] - target)))]])

    atm_k = k[at_near][int(np.argmin(np.abs(k[at_near] - spot)))]
    atm = [
        v
        for v in (
            iv_at(at_near & is_call & (k == atm_k), atm_k),
            iv_at(at_near & ~is_call & (k == atm_k), atm_k),
        )
        if v is not None
    ]
    p95, c105 = iv_at(at_near & ~is_call, 0.95 * spot), iv_at(at_near & is_call, 1.05 * spot)

    nan_none = [None if not np.isfinite(x) else float(x) for x in iv]
    strikes = (
        ch.select("expiry_date", "strike_price", "oi", "change_in_oi", "option_type")
        .with_columns(
            pl.Series("_iv", nan_none, dtype=pl.Float64),
            pl.Series("_gex", gex / 1e7, dtype=pl.Float64),
            pl.Series("_call", is_call),
            pl.col("oi").fill_null(0),
            pl.col("change_in_oi").fill_null(0),
        )
        .group_by("expiry_date", "strike_price")
        .agg(
            pl.col("oi").filter(pl.col("_call")).sum().alias("call_oi"),
            pl.col("oi").filter(~pl.col("_call")).sum().alias("put_oi"),
            pl.col("change_in_oi").filter(pl.col("_call")).sum().alias("call_doi"),
            pl.col("change_in_oi").filter(~pl.col("_call")).sum().alias("put_doi"),
            pl.col("_iv").filter(pl.col("_call")).first().alias("call_iv"),
            pl.col("_iv").filter(~pl.col("_call")).first().alias("put_iv"),
            pl.col("_gex").sum().alias("gex_cr"),
        )
        .with_columns((pl.col("put_doi") - pl.col("call_doi")).alias("oi_imbalance"))
        .rename({"expiry_date": "expiry", "strike_price": "strike"})
        .sort("expiry", "strike")
    )
    return {
        "atm_iv": sum(atm) / len(atm) if atm else None,
        "skew": None if p95 is None or c105 is None else p95 - c105,
        "pcr_oi": float(p_oi / c_oi) if c_oi > 0 else None,
        "pcr_doi": float(p_doi / c_doi) if c_doi != 0 else None,
        "max_pain": float(mp),
        "max_pain_dist_pct": (mp - spot) / spot * 100,
        "call_wall": wall(is_call, above=True),
        "put_wall": wall(~is_call, above=False),
        "gex_total_cr": gex_total,
        "zero_gamma": zg,
        "zero_gamma_dist_pct": None if zg is None else (zg - spot) / spot * 100,
        "near_expiry": near,
        "strikes": strikes,
    }


_DAILY_COLS = [
    "spot",
    "atm_iv",
    "skew",
    "pcr_oi",
    "pcr_doi",
    "max_pain",
    "max_pain_dist_pct",
    "call_wall",
    "put_wall",
    "gex_total_cr",
    "zero_gamma",
    "zero_gamma_dist_pct",
]


def build_options_daily(settings: Settings, start: date, end: date) -> int:
    """Compute and upsert flow_options_daily and flow_options_strikes for start..end; returns daily rows."""
    fo = io.read(settings, "fo_bhavcopy", start, end).filter(pl.col("instrument_type").is_in(["IDO", "STO"]))
    fo = fo.collect()
    daily: list[dict] = []
    strikes: list[pl.DataFrame] = []
    for (d, sym), grp in fo.group_by(["date", "symbol"], maintain_order=True):
        ch = chain_frame(grp, d, sym)
        spot = ch["underlying_price"].drop_nulls().mean() if ch.height else None
        if not spot or spot <= 0:
            continue
        m = chain_metrics(ch, float(spot))
        st = m.pop("strikes").with_columns(pl.lit(d).alias("date"), pl.lit(sym).alias("underlying"))
        daily.append({"date": d, "underlying": sym, "spot": float(spot), **m})
        strikes.append(st.drop("oi_imbalance"))
    if not daily:
        return 0
    schema = {"date": pl.Date, "underlying": pl.Utf8, **dict.fromkeys(_DAILY_COLS, pl.Float64)}
    schema["near_expiry"] = pl.Date
    out = pl.DataFrame(daily, schema=schema)
    io.upsert(settings, "flow_options_daily", out)
    io.upsert(settings, "flow_options_strikes", pl.concat(strikes))
    return out.height
