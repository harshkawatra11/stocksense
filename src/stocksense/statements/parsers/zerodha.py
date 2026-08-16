"""
Zerodha Console tradebook parser.

Observed column layout (Zerodha Console tradebook CSV export):
symbol/tradingsymbol, isin, trade_date, exchange, segment, series,
trade_type (buy/sell), auction, quantity, price, trade_id, order_id,
order_execution_time. Column names are matched fuzzily (see
parsers/base.py) rather than hardcoded exactly, since this list is
compiled from public documentation, not a guaranteed-stable spec.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stocksense.statements.parsers.base import (
    canonical_row_id,
    find_column,
    infer_segment,
    read_tabular,
    to_source_row_json,
)

_REQUIRED_HINTS = ["tradingsymbol", "trade_type", "order_execution_time"]

_ALIASES = {
    "symbol": ["symbol", "tradingsymbol", "trading_symbol"],
    "isin": ["isin"],
    "trade_date": ["trade_date", "date"],
    "trade_time": ["order_execution_time", "trade_time", "time"],
    "exchange": ["exchange"],
    "segment": ["segment"],
    "series": ["series"],
    "side": ["trade_type", "side", "transaction_type"],
    "quantity": ["quantity", "qty"],
    "price": ["price", "trade_price"],
    "order_id": ["order_id"],
    "trade_id": ["trade_id"],
}


class ZerodhaParser:
    broker = "zerodha"

    def detect(self, path: Path) -> bool:
        try:
            df = read_tabular(path)
        except Exception:
            return False
        cols_norm = {c.strip().lower().replace(" ", "_") for c in df.columns}
        return all(find_column(list(df.columns), _ALIASES[k]) for k in ("symbol", "side", "quantity", "price")) and (
            "tradingsymbol" in cols_norm or "order_execution_time" in cols_norm
        )

    def parse(self, path: Path) -> pd.DataFrame:
        df = read_tabular(path)
        cols = list(df.columns)
        col = {k: find_column(cols, v) for k, v in _ALIASES.items()}

        missing = [k for k in ("symbol", "trade_date", "side", "quantity", "price") if col[k] is None]
        if missing:
            raise ValueError(f"Zerodha parser: missing required columns {missing} in {path.name}")

        out_rows = []
        for _, row in df.iterrows():
            symbol = str(row[col["symbol"]]).strip()
            side = str(row[col["side"]]).strip().lower()
            side = "buy" if side.startswith("b") else "sell"
            quantity = float(row[col["quantity"]])
            price = float(row[col["price"]])
            trade_date = str(row[col["trade_date"]])[:10]
            segment_raw = str(row[col["segment"]]) if col["segment"] else ""
            series = str(row[col["series"]]) if col["series"] else ""
            exchange = str(row[col["exchange"]]) if col["exchange"] else "NSE"

            out_rows.append(
                {
                    "trade_id": canonical_row_id("zerodha", str(row.get(col["order_id"], "")), symbol, trade_date, str(quantity), str(price)),
                    "broker": "zerodha",
                    "symbol": symbol,
                    "isin": str(row[col["isin"]]).strip() if col["isin"] else None,
                    "segment": infer_segment(segment_raw or exchange, series),
                    "trade_date": trade_date,
                    "trade_time": str(row[col["trade_time"]]) if col["trade_time"] else None,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "value": quantity * price,
                    "order_id": str(row[col["order_id"]]) if col["order_id"] else None,
                    "exchange": exchange or "NSE",
                    "product_type": None,  # Zerodha tradebook doesn't carry MIS/CNC; joined from Tax P&L if available
                    "source_row_json": to_source_row_json(row),
                }
            )
        return pd.DataFrame(out_rows)
