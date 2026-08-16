"""
F&O open-interest features (docs/17-data-spine.md), built from
`bhavcopy_fo` (data/nse_archive.fetch_fo_bhavcopy, 2005+). `docs/03`
explicitly scoped F&O out of Phase 0 for lack of data; that constraint
is gone now that bhavcopy is ingested, but per the plan these features
are admitted only if they beat an ablation, same as delivery.py.

NSE's instrument field distinguishes futures (FUTSTK/FUTIDX) from
options (OPTSTK/OPTIDX in legacy naming; STO/ITO-style codes in UDiFF —
see nse_archive.py's era normalization). option_type is 'CE'/'PE' for
options, a placeholder ('XX') for futures.
"""

from __future__ import annotations

import pandas as pd


def _is_future(instrument: str) -> bool:
    return "FUT" in instrument.upper()


def _is_option(instrument: str, option_type: str) -> bool:
    return option_type.upper() in ("CE", "PE")


def build_oi_features(bhavcopy_fo: pd.DataFrame) -> pd.DataFrame:
    """Aggregates per (symbol, date) across all contracts (all strikes,
    all expiries) for that underlying: total OI, OI change, and the
    long-buildup/short-covering/short-buildup/long-unwinding quadrant
    (the classic F&O read: price direction crossed with OI direction).
    Requires a `underlying_ret` column (the underlying's own price
    return that day) to classify the quadrant -- callers join this in
    from bhavcopy_eq or candles before calling, matching
    delivery.delivery_weighted_return's pattern."""
    fut = bhavcopy_fo[bhavcopy_fo["instrument"].apply(_is_future)].copy()
    agg = fut.groupby(["symbol", "date"]).agg(
        total_oi=("open_interest", "sum"),
        total_chg_in_oi=("chg_in_oi", "sum"),
        futures_close=("close", "mean"),  # mean across contract months as a simple proxy
    ).reset_index()
    agg["oi_pct_change"] = agg["total_chg_in_oi"] / (agg["total_oi"] - agg["total_chg_in_oi"]).replace(0, pd.NA)
    return agg.sort_values(["symbol", "date"])


def classify_oi_quadrant(oi_features: pd.DataFrame, underlying_returns: pd.Series) -> pd.Series:
    """The standard F&O read, computed from price direction x OI
    direction:
    - price up,   OI up   -> 'long_buildup'    (new longs entering)
    - price up,   OI down -> 'short_covering'  (shorts exiting)
    - price down, OI up   -> 'short_buildup'   (new shorts entering)
    - price down, OI down -> 'long_unwinding'  (longs exiting)
    `underlying_returns` must be aligned to oi_features' index."""
    price_up = underlying_returns > 0
    oi_up = oi_features["oi_pct_change"] > 0

    quadrant = pd.Series(index=oi_features.index, dtype=object)
    quadrant[price_up & oi_up] = "long_buildup"
    quadrant[price_up & ~oi_up] = "short_covering"
    quadrant[~price_up & oi_up] = "short_buildup"
    quadrant[~price_up & ~oi_up] = "long_unwinding"
    return quadrant


def build_put_call_ratio(bhavcopy_fo: pd.DataFrame) -> pd.DataFrame:
    """PCR by open interest and by volume-proxy (contracts traded is not
    directly in this schema, so OI is the primary signal here) per
    (symbol, date), aggregated across all strikes/expiries for that
    underlying. PCR > 1 is conventionally read as bearish sentiment
    (more put OI than call OI), < 1 as bullish -- a reference reading,
    not a claim this codebase validates without the ablation."""
    opt = bhavcopy_fo[bhavcopy_fo.apply(lambda r: _is_option(r["instrument"], r["option_type"]), axis=1)].copy()
    pivot = opt.groupby(["symbol", "date", "option_type"])["open_interest"].sum().unstack(fill_value=0.0)

    result = pd.DataFrame(index=pivot.index)
    result["put_oi"] = pivot.get("PE", 0.0)
    result["call_oi"] = pivot.get("CE", 0.0)
    result["pcr_oi"] = result["put_oi"] / result["call_oi"].replace(0, pd.NA)
    return result.reset_index()


def days_to_expiry(bhavcopy_fo: pd.DataFrame) -> pd.Series:
    """Calendar days from `date` to the NEAREST expiry among that row's
    contracts -- expiry-proximity effects (elevated volatility, pinning)
    are a documented F&O phenomenon worth having as a feature candidate,
    same ablation-gated status as everything else here."""
    dates = pd.to_datetime(bhavcopy_fo["date"])
    expiries = pd.to_datetime(bhavcopy_fo["expiry_date"], format="mixed", errors="coerce")
    return (expiries - dates).dt.days
