"""
FIFO round-trip reconstruction (docs/12-statement-forensics.md).

Raw trades are buy/sell events; a "position" is the FIFO-matched
buy-to-sell (or sell-to-buy, for short positions) round trip that
actually determines P&L. This is where behavioral diagnostics attach —
you can't measure "holding losers too long" on individual trade rows,
only on reconstructed positions with a duration.

FIFO, not LIFO or average-cost, because it is what Indian tax law
(Income Tax Act) uses for capital gains matching, so the reconstructed
positions are consistent with what a Tax P&L statement would report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from stocksense.execution.cost_model import compute_charges


@dataclass
class _Lot:
    quantity: float
    price: float
    trade_date: str
    trade_time: str | None


@dataclass
class _OpenBook:
    long_lots: list[_Lot] = field(default_factory=list)
    short_lots: list[_Lot] = field(default_factory=list)


def _is_intraday(segment: str) -> bool:
    return segment in ("equity_intraday", "fno_futures", "fno_options")


def reconstruct_positions(trades: pd.DataFrame) -> pd.DataFrame:
    """FIFO-match buys and sells per (symbol, segment), producing one row
    per completed round trip. Handles: short sells (sell-before-buy opens
    a short lot), partial fills (a single sell can close multiple buy
    lots and vice versa), and multi-day holds (lots simply carry forward
    across statement boundaries since matching is symbol/segment-keyed,
    not date-keyed).

    Unmatched trailing quantity (a position still open at the end of the
    available trade history) is left in the book and does not produce a
    position row — it isn't closed yet, so there's no P&L to report.
    """
    if trades.empty:
        return _empty_positions_frame()

    trades = trades.sort_values(["trade_date", "trade_time"], na_position="first").reset_index(drop=True)

    books: dict[tuple[str, str], _OpenBook] = {}
    # AUDIT FIX: position_id used to be built from
    # symbol|segment|open_date|open_price|close_date|close_price|quantity
    # -- no TIME component. Confirmed live against the user's real
    # tradebook (444 positions reconstructed, only 434 persisted): an
    # active intraday trader (40+ trades/day) genuinely opens and closes
    # the same symbol at the same round price more than once in a single
    # day, producing identical position_id strings for real, DISTINCT
    # round trips. write_positions' ON CONFLICT DO NOTHING then silently
    # dropped the second one as if it were a duplicate re-ingest, not a
    # real position -- corrupting the persisted aggregate by exactly the
    # collision count. Fixed structurally with a per-(symbol, segment)
    # monotonic sequence counter, which is unique by construction
    # regardless of price/timestamp granularity, rather than trusting
    # open_time/close_time strings to always disambiguate.
    seq_counters: dict[tuple[str, str], int] = {}
    out_rows: list[dict] = []

    for _, t in trades.iterrows():
        key = (t["symbol"], t["segment"])
        book = books.setdefault(key, _OpenBook())
        qty = float(t["quantity"])
        price = float(t["price"])
        side = t["side"]
        trade_date = t["trade_date"]
        trade_time = t.get("trade_time")

        if side == "buy":
            remaining = qty
            # first close any open short lots (FIFO)
            while remaining > 1e-9 and book.short_lots:
                lot = book.short_lots[0]
                matched = min(remaining, lot.quantity)
                seq = seq_counters[key] = seq_counters.get(key, 0) + 1
                out_rows.append(
                    _close_position(t["symbol"], t["segment"], lot, price, trade_date, trade_time, matched, is_short=True, seq=seq)
                )
                lot.quantity -= matched
                remaining -= matched
                if lot.quantity <= 1e-9:
                    book.short_lots.pop(0)
            if remaining > 1e-9:
                book.long_lots.append(_Lot(remaining, price, trade_date, trade_time))
        else:  # sell
            remaining = qty
            while remaining > 1e-9 and book.long_lots:
                lot = book.long_lots[0]
                matched = min(remaining, lot.quantity)
                seq = seq_counters[key] = seq_counters.get(key, 0) + 1
                out_rows.append(
                    _close_position(t["symbol"], t["segment"], lot, price, trade_date, trade_time, matched, is_short=False, seq=seq)
                )
                lot.quantity -= matched
                remaining -= matched
                if lot.quantity <= 1e-9:
                    book.long_lots.pop(0)
            if remaining > 1e-9:
                book.short_lots.append(_Lot(remaining, price, trade_date, trade_time))

    if not out_rows:
        return _empty_positions_frame()
    return pd.DataFrame(out_rows)


def _close_position(symbol, segment, open_lot: _Lot, close_price: float, close_date: str,
                     close_time: str | None, quantity: float, is_short: bool, seq: int) -> dict:
    entry_price = open_lot.price if not is_short else open_lot.price
    exit_price = close_price
    if is_short:
        gross_pnl = (open_lot.price - close_price) * quantity  # sold high, buying back to close
    else:
        gross_pnl = (close_price - open_lot.price) * quantity

    is_intraday = _is_intraday(segment) and open_lot.trade_date == close_date
    charge_segment = segment if segment != "equity_delivery" or not is_intraday else "equity_intraday"

    open_side = "sell" if is_short else "buy"
    close_side = "buy" if is_short else "sell"
    open_charges = compute_charges(charge_segment, open_side, quantity, open_lot.price)
    close_charges = compute_charges(charge_segment, close_side, quantity, close_price)
    total_charges = open_charges.total_charges + close_charges.total_charges
    net_pnl = gross_pnl - total_charges

    holding_seconds = _holding_seconds(open_lot.trade_date, open_lot.trade_time, close_date, close_time)

    return {
        "position_id": f"{symbol}|{segment}|{open_lot.trade_date}|{open_lot.price}|{close_date}|{close_price}|{quantity}|{seq}",
        "symbol": symbol,
        "segment": segment,
        "open_date": open_lot.trade_date,
        "open_time": open_lot.trade_time,
        "close_date": close_date,
        "close_time": close_time,
        "quantity": quantity,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_pnl": gross_pnl,
        "charges": total_charges,
        "net_pnl": net_pnl,
        "holding_seconds": holding_seconds,
        "is_intraday": is_intraday,
        "mae": None,  # requires intraday candle data, filled in separately when available
        "mfe": None,
    }


def _holding_seconds(open_date: str, open_time: str | None, close_date: str, close_time: str | None) -> int | None:
    try:
        o = pd.Timestamp(f"{open_date} {open_time or '00:00:00'}")
        c = pd.Timestamp(f"{close_date} {close_time or '00:00:00'}")
        return max(0, int((c - o).total_seconds()))
    except Exception:
        return None


def _empty_positions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "position_id", "symbol", "segment", "open_date", "open_time", "close_date", "close_time",
            "quantity", "entry_price", "exit_price", "gross_pnl", "charges", "net_pnl",
            "holding_seconds", "is_intraday", "mae", "mfe",
        ]
    )
