"""Intraday label tests (Phase E2). The whole point of first_touch_label
is that intraday P&L is PATH-DEPENDENT -- these tests pin down: target
hit first, stop hit first, both touched in the same bar (conservative
stop-wins convention), a time-based exit when neither is touched, never
reaching into the next session, and the session-bounded forward-return
label going NaN rather than reaching past a session's last bar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocksense.labels.intraday_labels import add_session_forward_return, first_touch_label, precompute_sessions


def _bar(symbol, ts, o, h, l, c, v=1000.0):
    return {"symbol": symbol, "ts": pd.Timestamp(ts), "open": o, "high": h, "low": l, "close": c, "volume": v}


# ---- add_session_forward_return ----

def test_session_forward_return_computed_within_session() -> None:
    bars = pd.DataFrame([
        _bar("X", "2026-01-05 09:15", 100, 100, 100, 100),
        _bar("X", "2026-01-05 09:20", 101, 101, 101, 101),
        _bar("X", "2026-01-05 09:25", 105, 105, 105, 105),
    ])
    out = add_session_forward_return(bars, horizon_bars=2)
    assert out.iloc[0]["fwd_ret_2b"] == pytest.approx((105 / 100) - 1.0)


def test_session_forward_return_nan_past_session_end() -> None:
    day1 = pd.DataFrame([_bar("X", f"2026-01-05 09:{15+5*i}", 100, 100, 100, 100) for i in range(3)])
    day2 = pd.DataFrame([_bar("X", f"2026-01-06 09:{15+5*i}", 200, 200, 200, 200) for i in range(3)])
    bars = pd.concat([day1, day2], ignore_index=True)

    out = add_session_forward_return(bars, horizon_bars=2)
    # last bar of day 1 has no future bars left in ITS OWN session
    last_day1_row = out[(out["ts"].dt.normalize() == pd.Timestamp("2026-01-05"))].iloc[-1]
    assert pd.isna(last_day1_row["fwd_ret_2b"])
    # and critically, it must not have reached into day2's 200-price bars
    # to compute a fabricated overnight return


# ---- first_touch_label ----

def test_target_hit_before_stop() -> None:
    bars = pd.DataFrame([
        _bar("X", "2026-01-05 09:15", 100, 100.5, 99.5, 100),
        _bar("X", "2026-01-05 09:16", 100, 103, 99.8, 102),   # target (102) touched, stop (98.5) not
        _bar("X", "2026-01-05 09:17", 102, 104, 101, 103),
    ])
    entries = pd.DataFrame([{"symbol": "X", "entry_ts": "2026-01-05 09:15", "entry_price": 100.0}])
    out = first_touch_label(bars, entries, stop_pct=0.015, target_pct=0.02, max_holding_minutes=60)
    row = out.iloc[0]
    assert row["outcome"] == "target"
    assert row["exit_price"] == pytest.approx(102.0)
    assert row["ret"] == pytest.approx(0.02)


def test_stop_hit_before_target() -> None:
    bars = pd.DataFrame([
        _bar("X", "2026-01-05 09:15", 100, 100.5, 99.5, 100),
        _bar("X", "2026-01-05 09:16", 100, 100.2, 98.0, 98.5),  # stop (98.5) touched, target not
        _bar("X", "2026-01-05 09:17", 98.5, 99, 98, 98.7),
    ])
    entries = pd.DataFrame([{"symbol": "X", "entry_ts": "2026-01-05 09:15", "entry_price": 100.0}])
    out = first_touch_label(bars, entries, stop_pct=0.015, target_pct=0.02, max_holding_minutes=60)
    row = out.iloc[0]
    assert row["outcome"] == "stop"
    assert row["exit_price"] == pytest.approx(98.5)
    assert row["ret"] == pytest.approx(-0.015)


def test_both_touched_in_same_bar_conservatively_records_stop() -> None:
    bars = pd.DataFrame([
        _bar("X", "2026-01-05 09:15", 100, 100.5, 99.5, 100),
        # one wild bar that spans both the target (102) and stop (98.5)
        _bar("X", "2026-01-05 09:16", 100, 105, 95, 101),
    ])
    entries = pd.DataFrame([{"symbol": "X", "entry_ts": "2026-01-05 09:15", "entry_price": 100.0}])
    out = first_touch_label(bars, entries, stop_pct=0.015, target_pct=0.02, max_holding_minutes=60)
    assert out.iloc[0]["outcome"] == "stop"  # conservative convention, never overstates the label


def test_time_exit_when_neither_touched() -> None:
    bars = pd.DataFrame([
        _bar("X", "2026-01-05 09:15", 100, 100.5, 99.5, 100),
        _bar("X", "2026-01-05 09:16", 100, 100.3, 99.7, 100.1),
        _bar("X", "2026-01-05 09:17", 100.1, 100.4, 99.9, 100.2),
    ])
    entries = pd.DataFrame([{"symbol": "X", "entry_ts": "2026-01-05 09:15", "entry_price": 100.0}])
    out = first_touch_label(bars, entries, stop_pct=0.015, target_pct=0.02, max_holding_minutes=3)
    row = out.iloc[0]
    assert row["outcome"] == "time_exit"
    assert row["exit_price"] == pytest.approx(100.2)  # last bar's close within the holding window


def test_never_crosses_into_the_next_session() -> None:
    """An MIS position does not exist overnight -- a stop/target only
    reachable in tomorrow's bars must not be recognized as this entry's
    exit."""
    day1 = pd.DataFrame([_bar("X", "2026-01-05 15:25", 100, 100.2, 99.8, 100)])
    day2 = pd.DataFrame([_bar("X", "2026-01-06 09:15", 100, 200, 50, 150)])  # huge move, but next day
    bars = pd.concat([day1, day2], ignore_index=True)
    entries = pd.DataFrame([{"symbol": "X", "entry_ts": "2026-01-05 15:25", "entry_price": 100.0}])
    out = first_touch_label(bars, entries, stop_pct=0.015, target_pct=0.02, max_holding_minutes=600)
    row = out.iloc[0]
    assert row["outcome"] == "time_exit"  # NOT stop or target from day2's bar
    assert row["exit_price"] == pytest.approx(100.0)  # entry's own session has no more bars


def test_no_data_for_a_session_with_no_bars_at_all() -> None:
    bars = pd.DataFrame([_bar("X", "2026-01-05 09:15", 100, 100, 100, 100)])
    entries = pd.DataFrame([{"symbol": "X", "entry_ts": "2026-02-10 09:15", "entry_price": 100.0}])
    out = first_touch_label(bars, entries, stop_pct=0.015, target_pct=0.02)
    assert out.iloc[0]["outcome"] == "no_data"
    assert pd.isna(out.iloc[0]["ret"])


# ---- PERFORMANCE FIX: precompute_sessions lets a caller with many
# entries against the same bars_1min build the grouping once (see
# module docstring on precompute_sessions for the real incident this
# closes -- the E4 sweep's first fold ran 6.5+ hours before this fix). ----

def test_precomputed_sessions_gives_identical_result_to_the_default_path() -> None:
    bars = pd.DataFrame([
        _bar("X", "2026-01-05 09:15", 100, 100.5, 99.5, 100),
        _bar("X", "2026-01-05 09:16", 100, 103, 99.8, 102),
        _bar("X", "2026-01-05 09:17", 102, 104, 101, 103),
    ])
    entries = pd.DataFrame([{"symbol": "X", "entry_ts": "2026-01-05 09:15", "entry_price": 100.0}])

    default_result = first_touch_label(bars, entries, stop_pct=0.015, target_pct=0.02, max_holding_minutes=60)
    sessions = precompute_sessions(bars)
    precomputed_result = first_touch_label(
        bars, entries, stop_pct=0.015, target_pct=0.02, max_holding_minutes=60, sessions=sessions,
    )
    pd.testing.assert_frame_equal(default_result, precomputed_result)


def test_precomputed_sessions_reused_across_many_calls_stays_correct() -> None:
    """The actual access pattern this fix targets: MANY separate
    first_touch_label calls (one per trade, as
    simulate_intraday_trades_for_fold does) against the SAME
    precomputed sessions dict -- each call must still get the right
    answer for its own entry, not bleed state from a previous call."""
    bars = pd.DataFrame([
        _bar("A", "2026-01-05 09:15", 100, 100.2, 99.8, 100),
        _bar("A", "2026-01-05 09:16", 100, 103, 99.8, 102),  # A: target hit
        _bar("B", "2026-01-05 09:15", 50, 50.1, 49.9, 50),
        _bar("B", "2026-01-05 09:16", 50, 50.2, 48.0, 48.5),  # B: stop hit
    ])
    sessions = precompute_sessions(bars)

    entry_a = pd.DataFrame([{"symbol": "A", "entry_ts": "2026-01-05 09:15", "entry_price": 100.0}])
    entry_b = pd.DataFrame([{"symbol": "B", "entry_ts": "2026-01-05 09:15", "entry_price": 50.0}])

    result_a = first_touch_label(bars, entry_a, stop_pct=0.02, target_pct=0.015, sessions=sessions)
    result_b = first_touch_label(bars, entry_b, stop_pct=0.02, target_pct=0.015, sessions=sessions)

    assert result_a.iloc[0]["outcome"] == "target"
    assert result_b.iloc[0]["outcome"] == "stop"
