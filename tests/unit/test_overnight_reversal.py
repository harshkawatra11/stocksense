"""Strategy family 1: the overnight/intraday tug of war (Lou, Polk & Skouras).

Signal = adj_open / prev_adj_close - 1, cross-sectionally demeaned, restricted
to prev_gap_sessions == 1 -- a genuine one-session overnight return, computed
on ADJUSTED prices so a split ex-date is never read as a fabricated gap.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stocksense.strategies.overnight_reversal import (
    OvernightReversalConfig,
    compute_overnight_signal,
    daily_pnl,
    select_positions,
)

D = date(2026, 1, 5)


def _panel(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ------------------------------------------------------------- compute_overnight_signal
def test_signal_is_adj_open_over_prev_adj_close_minus_one():
    panel = _panel([
        {"date": D, "symbol": "A", "adj_open": 98.0, "prev_adj_close": 100.0, "prev_gap_sessions": 1},
    ])
    out = compute_overnight_signal(panel, demean=False, winsorise_pct=0.0)
    assert out.loc[0, "signal"] == pytest.approx(-0.02)


def test_signal_drops_rows_where_gap_is_not_one_session():
    panel = _panel([
        {"date": D, "symbol": "A", "adj_open": 98.0, "prev_adj_close": 100.0, "prev_gap_sessions": 1},
        {"date": D, "symbol": "B", "adj_open": 98.0, "prev_adj_close": 100.0, "prev_gap_sessions": 3},
        {"date": D, "symbol": "C", "adj_open": 98.0, "prev_adj_close": np.nan, "prev_gap_sessions": np.nan},
    ])
    out = compute_overnight_signal(panel, demean=False, winsorise_pct=0.0)
    assert list(out["symbol"]) == ["A"]


def test_signal_demean_subtracts_the_per_date_cross_sectional_mean():
    panel = _panel([
        {"date": D, "symbol": "A", "adj_open": 98.0, "prev_adj_close": 100.0, "prev_gap_sessions": 1},  # -0.02
        {"date": D, "symbol": "B", "adj_open": 103.0, "prev_adj_close": 100.0, "prev_gap_sessions": 1},  # +0.03
    ])
    out = compute_overnight_signal(panel, demean=True, winsorise_pct=0.0)
    # mean is 0.005 -> demeaned: -0.025, +0.025
    assert out.set_index("symbol").loc["A", "signal"] == pytest.approx(-0.025)
    assert out.set_index("symbol").loc["B", "signal"] == pytest.approx(0.025)


def test_signal_winsorise_clips_extreme_tails():
    rows = [
        {"date": D, "symbol": f"S{i}", "adj_open": 100.0 * (1 + i * 0.001), "prev_adj_close": 100.0, "prev_gap_sessions": 1}
        for i in range(100)
    ]
    # inject one extreme outlier
    rows.append({"date": D, "symbol": "OUTLIER", "adj_open": 200.0, "prev_adj_close": 100.0, "prev_gap_sessions": 1})
    panel = _panel(rows)
    out = compute_overnight_signal(panel, demean=False, winsorise_pct=0.01)
    raw_max = (panel["adj_open"] / panel["prev_adj_close"] - 1).max()
    clipped_max = out["signal"].max()
    assert clipped_max < raw_max


# ------------------------------------------------------------------- select_positions
def test_select_positions_long_picks_lowest_signal_names():
    # side="long": buy the overnight LOSERS (most negative signal).
    panel = _panel([
        {"date": D, "symbol": "A", "signal": -0.05},
        {"date": D, "symbol": "B", "signal": -0.01},
        {"date": D, "symbol": "C", "signal": 0.02},
        {"date": D, "symbol": "D", "signal": 0.06},
    ])
    cfg = OvernightReversalConfig(side="long", n_positions=2, min_overnight_move=0.0)
    picks = select_positions(panel, cfg)
    assert set(picks["symbol"]) == {"A", "B"}
    assert (picks["side"] == 1).all()


def test_select_positions_short_picks_highest_signal_names():
    panel = _panel([
        {"date": D, "symbol": "A", "signal": -0.05},
        {"date": D, "symbol": "B", "signal": -0.01},
        {"date": D, "symbol": "C", "signal": 0.02},
        {"date": D, "symbol": "D", "signal": 0.06},
    ])
    cfg = OvernightReversalConfig(side="short", n_positions=2, min_overnight_move=0.0)
    picks = select_positions(panel, cfg)
    assert set(picks["symbol"]) == {"C", "D"}
    assert (picks["side"] == -1).all()


def test_select_positions_long_short_takes_both_tails():
    panel = _panel([
        {"date": D, "symbol": "A", "signal": -0.05},
        {"date": D, "symbol": "B", "signal": -0.01},
        {"date": D, "symbol": "C", "signal": 0.02},
        {"date": D, "symbol": "D", "signal": 0.06},
    ])
    cfg = OvernightReversalConfig(side="long_short", n_positions=1, min_overnight_move=0.0)
    picks = select_positions(panel, cfg)
    long_side = picks[picks["side"] == 1]["symbol"].tolist()
    short_side = picks[picks["side"] == -1]["symbol"].tolist()
    assert long_side == ["A"]
    assert short_side == ["D"]


def test_select_positions_respects_min_overnight_move():
    panel = _panel([
        {"date": D, "symbol": "A", "signal": -0.002},  # below threshold, excluded
        {"date": D, "symbol": "B", "signal": -0.05},
    ])
    cfg = OvernightReversalConfig(side="long", n_positions=5, min_overnight_move=0.01)
    picks = select_positions(panel, cfg)
    assert list(picks["symbol"]) == ["B"]


# ----------------------------------------------------------------------- daily_pnl
def test_daily_pnl_long_wins_when_price_rises_from_open_to_close():
    positions = _panel([{"date": D, "symbol": "A", "side": 1}])
    prices = _panel([{"date": D, "symbol": "A", "adj_open": 100.0, "adj_close": 102.0}])
    ret = daily_pnl(positions, prices, charges_bps=0.0)
    assert ret.loc[D] == pytest.approx(0.02)


def test_daily_pnl_short_wins_when_price_falls_from_open_to_close():
    positions = _panel([{"date": D, "symbol": "A", "side": -1}])
    prices = _panel([{"date": D, "symbol": "A", "adj_open": 100.0, "adj_close": 98.0}])
    ret = daily_pnl(positions, prices, charges_bps=0.0)
    assert ret.loc[D] == pytest.approx(0.02)


def test_daily_pnl_averages_equally_across_positions():
    positions = _panel([
        {"date": D, "symbol": "A", "side": 1},
        {"date": D, "symbol": "B", "side": 1},
    ])
    prices = _panel([
        {"date": D, "symbol": "A", "adj_open": 100.0, "adj_close": 110.0},  # +10%
        {"date": D, "symbol": "B", "adj_open": 100.0, "adj_close": 100.0},  # 0%
    ])
    ret = daily_pnl(positions, prices, charges_bps=0.0)
    assert ret.loc[D] == pytest.approx(0.05)


def test_daily_pnl_subtracts_round_trip_charges():
    positions = _panel([{"date": D, "symbol": "A", "side": 1}])
    prices = _panel([{"date": D, "symbol": "A", "adj_open": 100.0, "adj_close": 102.0}])
    ret = daily_pnl(positions, prices, charges_bps=10.62)
    # +2% gross, minus 10.62 bps = 0.1062% round-trip charge
    assert ret.loc[D] == pytest.approx(0.02 - 0.001062)


def test_daily_pnl_is_zero_on_dates_with_no_positions():
    positions = _panel([]).assign(date=pd.Series(dtype="object"), symbol=pd.Series(dtype="object"), side=pd.Series(dtype="int"))
    prices = _panel([{"date": D, "symbol": "A", "adj_open": 100.0, "adj_close": 102.0}])
    ret = daily_pnl(positions, prices, charges_bps=0.0)
    assert ret.empty or D not in ret.index
