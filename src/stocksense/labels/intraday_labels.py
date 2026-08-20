"""
Intraday labels (Phase E2). Two genuinely different things, per the
plan's own distinction -- daily labels (labels/forward_return.py) are
close-to-close and PATH-INDEPENDENT, but intraday P&L with a stop is
PATH-DEPENDENT: whether price touched -1.5% before +2% decides the
entire outcome, and a close-to-close return cannot tell you that.

- add_session_forward_return: the path-independent analogue, for
  ranking/IC -- same shape as labels.forward_return, session-bounded.
- first_touch_label: the path-dependent one that actually maps to real
  money, simulated bar-by-bar against the underlying 1-MINUTE path (not
  the coarser research grain) so a stop/target between two 5-minute bars
  is not silently missed.

Per labels/forward_return.py's own rule, restated here: this is the one
place `.shift(-k)` / forward-looking bar iteration is allowed to exist.
Must never be imported by anything in stocksense.features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_session_forward_return(
    bars: pd.DataFrame, horizon_bars: int, label_col: str | None = None,
) -> pd.DataFrame:
    """Forward return `horizon_bars` bars ahead, computed strictly within
    each (symbol, session) group -- a row within `horizon_bars` of a
    session's last bar gets NaN rather than reaching into the next day's
    opening bars, which would fabricate an overnight return for a
    strategy that holds no overnight position."""
    col = label_col or f"fwd_ret_{horizon_bars}b"
    df = bars.sort_values(["symbol", "ts"]).copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df["_session_date"] = df["ts"].dt.date

    # groupby(...).shift() rather than .apply(): vectorized, and avoids a
    # real pandas footgun where .apply() on a Series-returning function
    # produces a transposed DataFrame instead of a concatenated Series
    # when the frame happens to contain exactly one group (hit before in
    # this project's D2 work, in labels/forward_return.py's equivalent).
    shifted_close = df.groupby(["symbol", "_session_date"])["close"].shift(-horizon_bars)
    df[col] = (shifted_close / df["close"]) - 1.0
    return df.drop(columns=["_session_date"])


def first_touch_label(
    bars_1min: pd.DataFrame,
    entries: pd.DataFrame,
    stop_pct: float,
    target_pct: float,
    max_holding_minutes: int = 60,
) -> pd.DataFrame:
    """For each entry (symbol, entry_ts, entry_price), walks forward
    through the SAME session's 1-minute bars and reports which happened
    first: stop, target, or a time-based exit at max_holding_minutes /
    session close, whichever comes first. Never crosses into the next
    session -- an MIS position does not exist overnight, so a stop that
    would only be reached tomorrow is not a stop this label recognizes.

    If a single bar's range touches BOTH stop and target (a real
    possibility at 1-minute granularity around a fast move), the
    conservative assumption is taken: stop is recorded as having
    triggered first. This is the standard conservative convention for
    bar-level (not tick-level) backtesting -- it can only ever
    understate the label's favorability, never overstate it.

    Returns one row per entry: symbol, entry_ts, outcome
    ('stop'|'target'|'time_exit'|'no_data'), exit_ts, exit_price, ret.
    """
    bars = bars_1min.copy()
    bars["ts"] = pd.to_datetime(bars["ts"])
    bars["_session_date"] = bars["ts"].dt.date
    bars = bars.sort_values(["symbol", "ts"])
    sessions = {key: g for key, g in bars.groupby(["symbol", "_session_date"])}

    rows = []
    for _, e in entries.iterrows():
        symbol = e["symbol"]
        entry_ts = pd.Timestamp(e["entry_ts"])
        entry_price = float(e["entry_price"])
        session_bars = sessions.get((symbol, entry_ts.date()))

        if session_bars is None:
            rows.append({
                "symbol": symbol, "entry_ts": entry_ts, "outcome": "no_data",
                "exit_ts": pd.NaT, "exit_price": np.nan, "ret": np.nan,
            })
            continue

        cutoff = entry_ts + pd.Timedelta(minutes=max_holding_minutes)
        future = session_bars[(session_bars["ts"] > entry_ts) & (session_bars["ts"] <= cutoff)]

        target_price = entry_price * (1 + target_pct)
        stop_price = entry_price * (1 - stop_pct)

        outcome, exit_ts, exit_price = None, None, None
        for _, bar in future.iterrows():
            stop_hit = bar["low"] <= stop_price
            target_hit = bar["high"] >= target_price
            if stop_hit:  # conservative: check stop first, covers the both-hit-in-one-bar case too
                outcome, exit_ts, exit_price = "stop", bar["ts"], stop_price
                break
            if target_hit:
                outcome, exit_ts, exit_price = "target", bar["ts"], target_price
                break

        if outcome is None:
            outcome = "time_exit"
            if len(future):
                exit_ts, exit_price = future["ts"].iloc[-1], float(future["close"].iloc[-1])
            else:
                exit_ts, exit_price = entry_ts, entry_price  # no bars left in the session at all

        rows.append({
            "symbol": symbol, "entry_ts": entry_ts, "outcome": outcome,
            "exit_ts": exit_ts, "exit_price": exit_price,
            "ret": (exit_price / entry_price) - 1.0,
        })

    return pd.DataFrame(rows, columns=["symbol", "entry_ts", "outcome", "exit_ts", "exit_price", "ret"])
