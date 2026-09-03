"""Point-in-time tradeable universe. THE anti-survivorship control.

The rule that makes it point-in-time: every filter is computed from rows dated
STRICTLY BEFORE the as-of date. A single `<=` here silently leaks tomorrow's
liquidity into today's universe and inflates every downstream result.

This is the single most important control against survivorship bias in the
whole pipeline. Testing a strategy against "today's" liquid names means the
backtest secretly knows which companies did not go bankrupt; a point-in-time
universe includes the names that were liquid THEN, including the ones that
later delisted, went illiquid, or collapsed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from stocksense.simulation.sizing import tradeable_price_band


@dataclass(frozen=True)
class UniverseFilter:
    min_avg_turnover_inr: float = 5_000_000.0  # ~50 lakh/day liquidity floor
    lookback_days: int = 60  # calendar days for the average
    min_price_inr: float | None = None  # None -> derived from tradeable_price_band
    max_price_inr: float | None = None
    series: str = "EQ"
    min_observations: int = 20  # needs real history, not 2 prints


def _resolve_price_bounds(
    flt: UniverseFilter, equity_inr: float | None
) -> tuple[float | None, float | None]:
    """Explicit bounds win; otherwise derive from the account's tradeable band
    when capital is known; otherwise no price filter at all.

    Deliberately does NOT default equity_inr to a stored constant -- per the
    plan's capital rule, a caller with no live capital gets liquidity-only
    filtering rather than a fabricated price band.
    """
    if flt.min_price_inr is not None or flt.max_price_inr is not None:
        return flt.min_price_inr, flt.max_price_inr
    if equity_inr is None:
        return None, None
    band = tradeable_price_band(equity_inr)
    return band.min_price_inr, band.max_price_inr


def universe_as_of(
    reader,
    as_of: date,
    flt: UniverseFilter = UniverseFilter(),
    equity_inr: float | None = None,
) -> list[str]:
    """Symbols tradeable AS OF `as_of`, using only prior data.

    Algorithm -- one SQL pass, no python loop:
      1. window = [as_of - lookback_days, as_of)     <-- END EXCLUSIVE. Critical.
      2. avg turnover, row count and last close per symbol over that window.
      3. keep avg_turnover >= min_avg_turnover_inr AND n_obs >= min_observations.
      4. keep min_price <= last_close <= max_price (if bounds resolved).
      5. return sorted(symbols).
    """
    if not reader.exists("bhavcopy_eq"):
        return []

    window_start = as_of - timedelta(days=flt.lookback_days)
    min_price, max_price = _resolve_price_bounds(flt, equity_inr)

    df = reader.sql(
        """
        SELECT symbol,
               avg(turnover_inr) AS avg_turnover,
               count(*)          AS n_obs,
               arg_max(close, date) AS last_close
        FROM {bhavcopy_eq}
        WHERE series = ? AND date >= ? AND date < ?
        GROUP BY symbol
        """,
        [flt.series, window_start, as_of],
    )
    if df.empty:
        return []

    keep = (df["avg_turnover"] >= flt.min_avg_turnover_inr) & (df["n_obs"] >= flt.min_observations)
    if min_price is not None:
        keep &= df["last_close"] >= min_price
    if max_price is not None:
        keep &= df["last_close"] <= max_price

    return sorted(df.loc[keep, "symbol"].tolist())


def universe_membership(
    reader,
    dates: list[date],
    flt: UniverseFilter = UniverseFilter(),
    equity_inr: float | None = None,
) -> pd.DataFrame:
    """[date, symbol] membership across many dates.

    ONE query per unique date, never per (symbol, date) row -- for a multi-year
    daily frame that is one query per trading day, not one per pair.
    """
    rows = []
    for d in sorted(set(dates)):
        for sym in universe_as_of(reader, d, flt, equity_inr):
            rows.append({"date": pd.Timestamp(d), "symbol": sym})
    if not rows:
        return pd.DataFrame(columns=["date", "symbol"])
    return pd.DataFrame(rows)


def filter_panel(panel: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Inner-join a long [symbol, date, ...] frame to membership, dropping rows
    for a (symbol, date) that was not point-in-time tradeable. Preserves every
    other column unchanged.
    """
    if panel.empty or membership.empty:
        return panel.iloc[0:0]

    left = panel.copy()
    left["date"] = pd.to_datetime(left["date"])
    right = membership.copy()
    right["date"] = pd.to_datetime(right["date"])
    right["_member"] = True

    merged = left.merge(right[["date", "symbol", "_member"]], on=["date", "symbol"], how="left")
    return merged[merged["_member"] == True].drop(columns=["_member"]).reset_index(drop=True)  # noqa: E712
