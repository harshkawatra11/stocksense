"""Trading-gap segmentation tests (Phase G1 bug fix). The load-bearing
property: a symbol that halts and reopens must be split into distinct
segments so a bar-sequence pct_change/shift computed downstream cannot
span the halt -- proven directly by checking that the FIRST row of a
new segment is never adjacent (in segment terms) to the row before the
gap, while an unaffected symbol's segmentation is a pure rename with no
behavioral change."""

from __future__ import annotations

import pandas as pd
import pytest

from stocksense.data.liquidity import flag_stale_reopening_rows, segment_symbols_by_trading_gap


def _row(symbol, d, close=100.0):
    return {"symbol": symbol, "date": pd.Timestamp(d), "close": close}


def test_continuously_traded_symbol_gets_a_single_segment() -> None:
    rows = [_row("STEADY", f"2024-01-{d:02d}") for d in range(1, 11)]
    candles = pd.DataFrame(rows)
    seg = segment_symbols_by_trading_gap(candles)
    assert (seg == "STEADY__seg0").all()


def test_halt_and_reopen_splits_into_two_segments() -> None:
    rows = [
        _row("HALTED", "2024-01-01"), _row("HALTED", "2024-01-02"), _row("HALTED", "2024-01-03"),
        # 300-day halt
        _row("HALTED", "2024-11-01"), _row("HALTED", "2024-11-02"), _row("HALTED", "2024-11-03"),
    ]
    candles = pd.DataFrame(rows)
    seg = segment_symbols_by_trading_gap(candles)

    pre_halt = seg[candles["date"] < "2024-11-01"]
    post_halt = seg[candles["date"] >= "2024-11-01"]
    assert set(pre_halt) == {"HALTED__seg0"}
    assert set(post_halt) == {"HALTED__seg1"}


def test_multiple_halts_increment_segment_further() -> None:
    rows = [
        _row("MULTI", "2024-01-01"),
        _row("MULTI", "2024-03-01"),   # 60-day gap -> new segment
        _row("MULTI", "2024-03-02"),
        _row("MULTI", "2024-08-01"),   # ~150-day gap -> another new segment
    ]
    candles = pd.DataFrame(rows)
    seg = segment_symbols_by_trading_gap(candles)
    assert list(seg) == ["MULTI__seg0", "MULTI__seg1", "MULTI__seg1", "MULTI__seg2"]


def test_ordinary_weekend_gap_does_not_split() -> None:
    # Friday -> Monday is a 3-calendar-day gap, well under the 10-day default
    rows = [_row("NORMAL", "2024-01-05"), _row("NORMAL", "2024-01-08")]
    candles = pd.DataFrame(rows)
    seg = segment_symbols_by_trading_gap(candles)
    assert (seg == "NORMAL__seg0").all()


def test_segment_symbol_aligned_to_original_index_after_reordering_input() -> None:
    """The function sorts internally but must return a Series indexed
    like the CALLER's frame, not the sorted one -- otherwise a caller
    doing candles.assign(symbol=segment_symbols_by_trading_gap(candles))
    would silently misalign rows."""
    rows = [_row("B", "2024-01-02"), _row("A", "2024-01-01"), _row("B", "2024-01-01")]
    candles = pd.DataFrame(rows)  # deliberately out of (symbol, date) order
    seg = segment_symbols_by_trading_gap(candles)
    assert list(seg.index) == list(candles.index)
    # row 0 is B on 2024-01-02; row 2 is B on 2024-01-01 -- same segment either way
    assert seg.loc[0] == "B__seg0"
    assert seg.loc[1] == "A__seg0"
    assert seg.loc[2] == "B__seg0"


def test_multiple_symbols_do_not_interfere() -> None:
    rows = [
        _row("X", "2024-01-01"), _row("X", "2024-09-01"),  # halt
        _row("Y", "2024-01-01"), _row("Y", "2024-01-02"),  # no halt
    ]
    candles = pd.DataFrame(rows)
    seg = segment_symbols_by_trading_gap(candles)
    assert seg.iloc[0] == "X__seg0"
    assert seg.iloc[1] == "X__seg1"
    assert seg.iloc[2] == "Y__seg0"
    assert seg.iloc[3] == "Y__seg0"


def test_flag_stale_reopening_rows_identifies_the_gap_row() -> None:
    rows = [_row("HALTED", "2024-01-01"), _row("HALTED", "2024-09-01")]
    candles = pd.DataFrame(rows)
    flagged = flag_stale_reopening_rows(candles)
    assert len(flagged) == 1
    assert flagged.iloc[0]["symbol"] == "HALTED"
    assert flagged.iloc[0]["gap_calendar_days"] > 200


def test_flag_stale_reopening_rows_empty_for_continuous_symbol() -> None:
    rows = [_row("STEADY", f"2024-01-{d:02d}") for d in range(1, 6)]
    candles = pd.DataFrame(rows)
    flagged = flag_stale_reopening_rows(candles)
    assert flagged.empty
