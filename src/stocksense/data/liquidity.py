"""
Trading-gap segmentation -- found live during Phase G1's real point-in-
time bhavcopy sweep (research/preregistration_bhavcopy_rerun.md), the
same way the ADANIENT adjustment-factor bug (data/adjust.py) was found:
a suspiciously large gate-passing alpha (10-25x Run 4's own headline
number) turned out to be real, but fabricated by the harness, not the
market.

The full point-in-time universe (thousands of thin, sometimes-
suspended small/micro-caps, unlike the 98 hand-picked, continuously-
liquid large caps `PHASE0_UNIVERSE` used) contains symbols with long
trading HALTS or SUSPENSIONS -- NSE bhavcopy simply has no row for a
symbol on any day it doesn't trade. When trading resumes, the next
available row's `close` can differ from the PRIOR row's `close` by
years of unobserved reality landing in a single bar (measured: 1,087
rows with |1-day return| > 50%, median 225 calendar days since the
symbol's previous print, max 6,011 days -- over 16 years).
`data/adjust.py`'s quarantine_unexplained_jumps does NOT catch this:
these are not adjustment-factor discontinuities (the RAW `close`, not
just `adj_close`, jumps too), they are genuine prints, correctly
recorded -- the bug is treating two rows separated by a long halt as an
ordinary adjacent trading-day pair when computing a bar-sequence return
(features/engine.py and labels/forward_return.py both use
pct_change/shift over the row SEQUENCE, with no awareness of elapsed
calendar time). No real position could have been held through the halt
and sold at the reopening price; a return computed across that gap
fabricates alpha nothing could capture.

Dropping every affected symbol's ENTIRE history outright (the same
coarse policy data/validate.quarantine_symbols and data/adjust.
quarantine_unexplained_jumps use for adjustment bugs) was tried first
and measured too costly here: 2,186 of 3,512 point-in-time symbols
(62%) have at least one gap over 10 calendar days somewhere in their
history, which would drop 66% of all rows -- gutting exactly the mid/
small-cap segment Phase G1 exists to test, most of which are simply
ordinarily illiquid, not actually broken. The fix instead SEGMENTS each
symbol's history at gap boundaries, so bar-sequence return computation
resets at a halt exactly the way it already resets at a genuine new
listing -- real, valid trading history on both sides of a halt is kept,
only the one fabricated cross-halt return is ever excluded.
"""

from __future__ import annotations

import pandas as pd


def flag_stale_reopening_rows(candles: pd.DataFrame, max_gap_calendar_days: int = 10) -> pd.DataFrame:
    """Diagnostic: returns the subset of rows where the gap since that
    SYMBOL's own previous row exceeds `max_gap_calendar_days` -- a proxy
    for "this symbol was halted/suspended and just reopened," not an
    adjustment anomaly. 10 calendar days is deliberately generous: NSE
    trades ~250 days/year, so an ordinary run of weekends plus a
    national holiday or two is usually 3-4 calendar days; 10 comfortably
    covers a long festival cluster without flagging normal trading gaps
    as halts. A symbol's very first row in the frame has no prior row
    and is never flagged (a listing/backfill start, not a reopening)."""
    df = candles.sort_values(["symbol", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    gap_days = df.groupby("symbol")["date"].diff().dt.days
    stale = df[gap_days > max_gap_calendar_days].copy()
    stale["gap_calendar_days"] = gap_days[gap_days > max_gap_calendar_days]
    return stale[["symbol", "date", "close", "gap_calendar_days"]]


def segment_symbols_by_trading_gap(candles: pd.DataFrame, max_gap_calendar_days: int = 10) -> pd.Series:
    """Returns a Series, aligned to `candles`' original index, of each
    row's SEGMENT symbol: the real `symbol` suffixed with a per-symbol
    counter that increments every time the gap since that symbol's own
    previous row exceeds `max_gap_calendar_days`. A symbol with no long
    gap anywhere gets a single segment (`SYM__seg0`) covering its whole
    history -- functionally unchanged from the real symbol, just
    renamed, so this is a no-op in effect for the vast majority of
    liquid names.

    Intended use: temporarily substitute this for the real `symbol`
    column before calling bar-sequence-based feature/label code
    (features.engine.build_features, labels.forward_return.
    add_forward_return_labels), then map the segment symbol back to the
    real symbol on every output frame before anything downstream (the
    ranker, the gate, the portfolio, the prediction ledger) ever sees a
    symbol name -- see data/loader.py's load_features_and_labels for the
    actual wiring. This function itself never modifies `candles` or its
    `symbol` column; it only computes the mapping.
    """
    df = candles.sort_values(["symbol", "date"])
    dates = pd.to_datetime(df["date"])
    gap_days = dates.groupby(df["symbol"]).diff().dt.days
    is_new_segment = (gap_days > max_gap_calendar_days).fillna(False)
    segment_idx = is_new_segment.groupby(df["symbol"]).cumsum()
    segment_symbol = df["symbol"].astype(str) + "__seg" + segment_idx.astype(str)
    return segment_symbol.reindex(candles.index)
