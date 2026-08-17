"""
Builds an adjusted price series from raw `bhavcopy_eq` closes and parsed
`corporate_actions` factors (docs/17-data-spine.md, Phase D1).

Two return bases, computed side by side rather than one replacing the
other (the Phase 0 baseline used yfinance's dividend-adjusted
adj_close; NSE bhavcopy is price-only):

- "price": splits/bonuses only. Matches NSE's own convention and what a
  retail trader sees on screen. Understates true return by the
  dividend yield (~1-1.5%/yr for NSE large-caps).
- "total": also folds in dividends, using the ex-date's ADJUSTED close
  (from the same backward pass) as the reference price for the
  percentage-return factor -- so a dividend on an already-split-adjusted
  price doesn't get double-counted against the wrong scale.

Adjustment is applied BACKWARD from the most recent date: the factor is
1.0 at the last available date for a symbol, and each corporate action
multiplies every price strictly before its ex-date by that action's
factor. This is the standard convention (adj_close today == raw close
today) and is what makes the result comparable to yfinance's adj_close
column in the reconciliation check.
"""

from __future__ import annotations

from datetime import date

import pandas as pd


def adjustment_factors(store, symbols: list[str] | None, basis: str) -> pd.DataFrame:
    """Per (symbol, ex_date): the cumulative backward-adjustment factor
    to apply to every raw price strictly BEFORE ex_date, for the given
    basis ('price' or 'total'). Dividends only contribute under
    basis='total'; their per-event factor is computed here using the
    bhavcopy close on the trading day immediately preceding ex_date as
    the reference price (the standard convention: factor = (P - D) / P).
    """
    if basis not in ("price", "total"):
        raise ValueError(f"basis must be 'price' or 'total', got {basis!r}")

    ca = store.read_corporate_actions()
    if symbols is not None:
        ca = ca[ca["symbol"].isin(symbols)]
    ca = ca[ca["parse_status"] == "ok"]

    events = ca[ca["action_type"].isin(["split", "bonus"])][["symbol", "ex_date", "factor_price"]].copy()
    events = events.rename(columns={"factor_price": "event_factor"})

    if basis == "total":
        div = ca[ca["action_type"] == "dividend"][["symbol", "ex_date", "dividend_amount"]].copy()
        if not div.empty:
            div_factors = _dividend_factors(store, div)
            events = pd.concat([events, div_factors[["symbol", "ex_date", "event_factor"]]], ignore_index=True)

    if events.empty:
        return pd.DataFrame(columns=["symbol", "ex_date", "cum_factor"])

    events = events.sort_values(["symbol", "ex_date"])
    # Cumulative product from the LATEST ex_date backward -- so cum_factor
    # at a given ex_date is the product of that event's factor and every
    # later event's factor, which is what a price strictly before this
    # ex_date must be multiplied by.
    events["cum_factor"] = (
        events.iloc[::-1].groupby("symbol")["event_factor"].cumprod().iloc[::-1]
    )
    return events[["symbol", "ex_date", "cum_factor"]]


def _dividend_factors(store, div: pd.DataFrame) -> pd.DataFrame:
    """factor = (prior_close - dividend_amount) / prior_close, where
    prior_close is the raw bhavcopy close on the last trading day
    strictly before ex_date for that symbol."""
    rows = []
    for _, r in div.iterrows():
        prior = store.con.execute(
            "SELECT close FROM bhavcopy_eq WHERE symbol = ? AND series = 'EQ' AND date < ? ORDER BY date DESC LIMIT 1",
            [r["symbol"], r["ex_date"]],
        ).fetchone()
        if prior is None or prior[0] in (None, 0):
            continue
        prior_close = prior[0]
        amount = r["dividend_amount"]
        if amount is None or amount <= 0 or amount >= prior_close:
            continue
        rows.append({"symbol": r["symbol"], "ex_date": r["ex_date"], "event_factor": (prior_close - amount) / prior_close})
    return pd.DataFrame(rows, columns=["symbol", "ex_date", "event_factor"])


def adjusted_prices(store, symbols: list[str], start: date, end: date, basis: str) -> pd.DataFrame:
    """Returns bhavcopy_eq rows for `symbols` in [start, end] with an
    added `adj_close` (and adjusted open/high/low) column, shaped to
    match what features.build_features expects from `candles`."""
    raw = store.con.execute(
        """
        SELECT symbol, date, open, high, low, close, volume, turnover_inr
        FROM bhavcopy_eq WHERE series = 'EQ' AND symbol IN (SELECT UNNEST(?)) AND date BETWEEN ? AND ?
        ORDER BY symbol, date
        """,
        [symbols, start, end],
    ).fetchdf()
    if raw.empty:
        return raw.assign(adj_close=pd.Series(dtype=float))

    factors = adjustment_factors(store, symbols, basis)
    if factors.empty:
        raw["adj_close"] = raw["close"]
        raw["adj_open"] = raw["open"]
        raw["adj_high"] = raw["high"]
        raw["adj_low"] = raw["low"]
        return raw

    out_parts = []
    for sym, g in raw.groupby("symbol"):
        f = factors[factors["symbol"] == sym].sort_values("ex_date")
        g = g.sort_values("date").copy()
        if f.empty:
            g["cum_factor"] = 1.0
        else:
            # side="right": finds the first ex_date STRICTLY GREATER than
            # this price's date. A price ON an ex-date is already quoted
            # at the post-action scale (that's what "ex-date" means) and
            # must NOT be multiplied by that event's own factor -- only
            # by events still ahead of it.
            idx = f["ex_date"].searchsorted(g["date"], side="right")
            factor_arr = f["cum_factor"].to_numpy()
            g["cum_factor"] = [factor_arr[i] if i < len(factor_arr) else 1.0 for i in idx]
        for col, adj_col in [("close", "adj_close"), ("open", "adj_open"), ("high", "adj_high"), ("low", "adj_low")]:
            g[adj_col] = g[col] * g["cum_factor"]
        out_parts.append(g.drop(columns=["cum_factor"]))

    return pd.concat(out_parts, ignore_index=True)
