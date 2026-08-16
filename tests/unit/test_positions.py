"""FIFO round-trip reconstruction — the edge cases named in the plan:
short sells, partial fills, same-day reversals, multi-day carries."""

from __future__ import annotations

import pandas as pd

from stocksense.statements.positions import reconstruct_positions


def _trade(symbol, segment, date, time, side, qty, price):
    return {
        "symbol": symbol, "segment": segment, "trade_date": date, "trade_time": time,
        "side": side, "quantity": qty, "price": price,
    }


def test_simple_long_round_trip() -> None:
    trades = pd.DataFrame(
        [
            _trade("RELIANCE", "equity_delivery", "2024-01-15", "09:20:00", "buy", 10, 2500.0),
            _trade("RELIANCE", "equity_delivery", "2024-01-16", "10:00:00", "sell", 10, 2550.0),
        ]
    )
    pos = reconstruct_positions(trades)
    assert len(pos) == 1
    row = pos.iloc[0]
    assert row["gross_pnl"] == 500.0
    assert row["net_pnl"] < row["gross_pnl"]  # charges deducted
    assert row["is_intraday"] == False


def test_partial_fill_splits_into_multiple_positions() -> None:
    trades = pd.DataFrame(
        [
            _trade("TCS", "equity_intraday", "2024-01-17", "09:16:00", "buy", 20, 3800.0),
            _trade("TCS", "equity_intraday", "2024-01-17", "10:30:00", "sell", 5, 3820.0),
            _trade("TCS", "equity_intraday", "2024-01-17", "15:00:00", "sell", 15, 3810.0),
        ]
    )
    pos = reconstruct_positions(trades)
    assert len(pos) == 2
    assert pos["quantity"].sum() == 20.0
    assert set(pos["exit_price"]) == {3820.0, 3810.0}


def test_short_sell_then_cover() -> None:
    trades = pd.DataFrame(
        [
            _trade("INFY", "equity_intraday", "2024-01-18", "09:20:00", "sell", 10, 1500.0),
            _trade("INFY", "equity_intraday", "2024-01-18", "11:00:00", "buy", 10, 1490.0),
        ]
    )
    pos = reconstruct_positions(trades)
    assert len(pos) == 1
    row = pos.iloc[0]
    assert row["gross_pnl"] == (1500.0 - 1490.0) * 10  # profited from the price drop
    assert row["is_intraday"] == True


def test_unmatched_trailing_quantity_produces_no_position() -> None:
    trades = pd.DataFrame([_trade("WIPRO", "equity_delivery", "2024-01-20", "10:00:00", "buy", 10, 500.0)])
    pos = reconstruct_positions(trades)
    assert len(pos) == 0  # position still open, nothing to report yet


def test_multi_day_carry_forward() -> None:
    trades = pd.DataFrame(
        [
            _trade("HDFCBANK", "equity_delivery", "2024-01-01", "09:20:00", "buy", 5, 1600.0),
            _trade("HDFCBANK", "equity_delivery", "2024-03-15", "14:00:00", "sell", 5, 1700.0),
        ]
    )
    pos = reconstruct_positions(trades)
    assert len(pos) == 1
    assert pos.iloc[0]["gross_pnl"] == 500.0
    assert pos.iloc[0]["holding_seconds"] > 60 * 60 * 24 * 60  # well over 60 days


def test_empty_trades_returns_empty_frame() -> None:
    pos = reconstruct_positions(pd.DataFrame(columns=["symbol", "segment", "trade_date", "trade_time", "side", "quantity", "price"]))
    assert len(pos) == 0
    assert "position_id" in pos.columns
