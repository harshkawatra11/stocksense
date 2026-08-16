"""
Upstox trade report parser.

Upstox's export column names are less publicly documented than
Zerodha's (verified during planning: public sources describe the
download flow but not a stable header spec), so this parser leans
harder on fuzzy alias matching and treats `product` (MIS/CNC/NRML) as
optional-but-preferred — when present it lets us classify
equity_intraday vs equity_delivery directly instead of guessing from
segment/series alone, which is what Zerodha's parser has to do.
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

_ALIASES = {
    "symbol": ["symbol", "instrument", "trading_symbol", "scrip_name", "trading symbol"],
    "isin": ["isin"],
    "trade_date": ["trade_date", "date", "order_date", "exchange_time"],
    "trade_time": ["trade_time", "time", "order_time", "exchange_time"],
    "exchange": ["exchange", "exch"],
    "segment": ["segment", "instrument_type"],
    "side": ["side", "transaction_type", "trade_type", "buy_sell", "order_type"],
    "quantity": ["quantity", "qty", "filled_qty", "traded_qty"],
    "price": ["price", "trade_price", "average_price", "avg_price"],
    "order_id": ["order_id", "order_no"],
    "trade_id": ["trade_id", "trade_no"],
    "product_type": ["product", "product_type"],
}


class UpstoxParser:
    broker = "upstox"

    def detect(self, path: Path) -> bool:
        try:
            df = read_tabular(path)
        except Exception:
            return False
        cols = list(df.columns)
        has_core = all(find_column(cols, _ALIASES[k]) for k in ("symbol", "side", "quantity", "price"))
        cols_norm = {c.strip().lower().replace(" ", "_") for c in cols}
        upstox_hint = bool({"product", "instrument_type", "scrip_name"} & cols_norm)
        return has_core and upstox_hint

    def parse(self, path: Path) -> pd.DataFrame:
        df = read_tabular(path)
        cols = list(df.columns)
        col = {k: find_column(cols, v) for k, v in _ALIASES.items()}

        missing = [k for k in ("symbol", "trade_date", "side", "quantity", "price") if col[k] is None]
        if missing:
            raise ValueError(f"Upstox parser: missing required columns {missing} in {path.name}")

        out_rows = []
        for _, row in df.iterrows():
            symbol = str(row[col["symbol"]]).strip()
            side_raw = str(row[col["side"]]).strip().lower()
            side = "buy" if side_raw.startswith("b") else "sell"
            quantity = float(row[col["quantity"]])
            price = float(row[col["price"]])
            trade_date = str(row[col["trade_date"]])[:10]
            segment_raw = str(row[col["segment"]]) if col["segment"] else ""
            exchange = str(row[col["exchange"]]) if col["exchange"] else "NSE"
            product = str(row[col["product_type"]]).strip().upper() if col["product_type"] else None

            segment = infer_segment(segment_raw or exchange)
            if segment == "equity_delivery" and product == "MIS":
                segment = "equity_intraday"

            out_rows.append(
                {
                    "trade_id": canonical_row_id("upstox", str(row.get(col["order_id"], "")), symbol, trade_date, str(quantity), str(price)),
                    "broker": "upstox",
                    "symbol": symbol,
                    "isin": str(row[col["isin"]]).strip() if col["isin"] else None,
                    "segment": segment,
                    "trade_date": trade_date,
                    "trade_time": str(row[col["trade_time"]]) if col["trade_time"] else None,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "value": quantity * price,
                    "order_id": str(row[col["order_id"]]) if col["order_id"] else None,
                    "exchange": exchange or "NSE",
                    "product_type": product,
                    "source_row_json": to_source_row_json(row),
                }
            )
        return pd.DataFrame(out_rows)
