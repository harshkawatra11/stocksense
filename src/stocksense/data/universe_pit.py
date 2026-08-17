"""
Point-in-time universe (docs/17-data-spine.md). Replaces the hand-picked
98-symbol `PHASE0_UNIVERSE` list with a rule applied AS OF a date,
built from ingested `bhavcopy_eq` rows rather than today's liquid names.

This is the fix for HIGH-4 (survivorship bias) from the original audit:
`universe_as_of(2010-01-04)` must return names that were liquid THEN,
including ones since delisted (DHFL, EDUCOMP, BHUSANSTL) -- and must
never reference a row dated after `d`. That second property is the one
that actually matters and the one most likely to be silently violated by
an innocent-looking bug (survivorship bias is invisible by construction:
a broken point-in-time filter and a correct one can look identical on a
single date, and only differ when you check many dates against history
that later happened).
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def universe_as_of(
    store, d: date, min_turnover_inr: float = 5_000_000.0, min_price_inr: float = 5.0,
    lookback_days: int = 60, series: str = "EQ",
) -> list[str]:
    """Symbols tradeable as of `d`: series == 'EQ' (excludes bonds, ETFs,
    government securities etc. that also appear in bhavcopy), average
    daily turnover over the trailing `lookback_days` calendar days
    (ending on or before `d`, never after) at or above `min_turnover_inr`,
    and last close at or above `min_price_inr` (excludes penny stocks).

    Every row this function reads is constrained to date <= d at the SQL
    level, not filtered in Python after a broader read -- so a caller
    cannot accidentally see post-d data even by future code-reading
    mistake, and the constraint is visible directly in the query.
    """
    lookback_start = (pd.Timestamp(d) - timedelta(days=lookback_days)).date()

    rows = store.con.execute(
        """
        SELECT symbol,
               AVG(turnover_inr) AS avg_turnover,
               LAST(close ORDER BY date) AS last_close
        FROM bhavcopy_eq
        WHERE series = ? AND date >= ? AND date <= ?
        GROUP BY symbol
        HAVING AVG(turnover_inr) >= ? AND LAST(close ORDER BY date) >= ?
        """,
        [series, lookback_start, d, min_turnover_inr, min_price_inr],
    ).fetchdf()

    return sorted(rows["symbol"].tolist())


def filter_to_point_in_time_universe(
    store, df: pd.DataFrame, date_col: str = "date", symbol_col: str = "symbol",
    min_turnover_inr: float = 5_000_000.0, min_price_inr: float = 5.0,
    lookback_days: int = 60, series: str = "EQ",
) -> pd.DataFrame:
    """Drops every (symbol, date) row of `df` not in that date's
    point-in-time universe. This is the actual wiring point that closes
    HIGH-4: universe_as_of/universe_membership_table were correct and
    tested but had exactly one caller (a display CLI command) before
    this -- every model, feature build, and walk-forward fold ran on
    'every symbol present in the source table', which for bhavcopy_eq
    means the full 7,556-symbol history including names that were
    illiquid or unlisted on any given date, not the universe a trader
    could actually have held then.

    One membership query per unique date in `df` (not per row) -- for a
    multi-year daily frame that's one query per trading day, not per
    (symbol, date) pair.
    """
    dates = sorted(pd.to_datetime(df[date_col]).dt.date.unique())
    membership = universe_membership_table(store, dates, min_turnover_inr, min_price_inr, lookback_days, series)
    if membership.empty:
        return df.iloc[0:0]

    out = df.copy()
    out["_d"] = pd.to_datetime(out[date_col]).dt.date
    membership = membership.rename(columns={"symbol": symbol_col, "date": "_d"})[[symbol_col, "_d"]]
    membership["_in_universe"] = True
    merged = out.merge(membership, on=[symbol_col, "_d"], how="left")
    return merged[merged["_in_universe"] == True].drop(columns=["_d", "_in_universe"]).reset_index(drop=True)  # noqa: E712


def universe_membership_table(
    store, dates: list[date], min_turnover_inr: float = 5_000_000.0, min_price_inr: float = 5.0,
    lookback_days: int = 60, series: str = "EQ",
) -> pd.DataFrame:
    """Builds a (symbol, date, is_tradeable) frame across many dates at
    once -- for backtesting, computing universe_as_of per-date with a
    Python loop calling this repeatedly is correct but slow (one query
    per rebalance date); this is the batch equivalent for when a caller
    needs membership across an entire walk-forward run."""
    rows = []
    for d in dates:
        symbols = universe_as_of(store, d, min_turnover_inr, min_price_inr, lookback_days, series)
        for sym in symbols:
            rows.append({"symbol": sym, "date": d, "is_tradeable": True})
    return pd.DataFrame(rows, columns=["symbol", "date", "is_tradeable"])
