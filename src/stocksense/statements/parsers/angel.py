"""
Angel One tradebook parser.

Angel One's XLSX exports ("Trades_History_*.xlsx", "TradesAndCharges"
sheet) are NOT a flat table -- confirmed by inspecting a real export
during planning: several dozen rows of report metadata (client code,
date range, a charges summary broken into multiple sub-sections) precede
the real per-fill header row ("Scrip/Contract", "Buy/Sell", ...), and the
file has multiple sheets, only one of which carries trade-level rows at
all (Broking Ledger / Charges / P&L sheets in the sibling export files
have no per-fill data). detect()/parse() locate that header row and
sheet dynamically by content (scanning for the literal "Scrip/Contract"
first cell), never by a fixed skiprows count or sheet name, since the
preamble length is not guaranteed stable across export versions -- the
same reasoning parsers/base.py already states for fuzzy column matching,
applied one level up (row location, not just column names).

Two more structural differences from Zerodha/Upstox that this parser
handles rather than assumes away:
- Price is split into "Buy Price" / "Sell Price" columns, only one
  populated per row depending on "Buy/Sell" -- coalesced here into the
  single canonical `price` field.
- "Order Type" carries Intraday/Delivery directly (Angel's own product
  classification) -- a stronger signal than Zerodha's segment/series
  guess or Upstox's MIS/CNC inference, used directly for segment
  classification rather than routed through infer_segment's fallback.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stocksense.statements.parsers.base import canonical_row_id, find_column, to_source_row_json

_HEADER_MARKER = "scrip/contract"  # normalized form of the real header row's first cell
_MAX_PREAMBLE_SCAN_ROWS = 100

_ALIASES = {
    "symbol": ["scrip/contract", "scrip", "contract"],
    "side": ["buy/sell"],
    "buy_price": ["buy_price"],
    "sell_price": ["sell_price"],
    "quantity": ["quantity"],
    "order_type": ["order_type"],
    "exchange": ["exchange"],
    "order_id": ["order_id"],
    "trade_id": ["trade_id"],
    "trade_date": ["date"],
}


def _find_tradebook_sheet_and_header(path: Path) -> tuple[str, int] | None:
    """Scans every sheet for the row whose first cell is the real
    tradebook header, returning (sheet_name, header_row_index) for the
    first match, or None if no sheet in this file has one."""
    try:
        xl = pd.ExcelFile(path)
    except Exception:
        return None
    for sheet in xl.sheet_names:
        raw = xl.parse(sheet, header=None, nrows=_MAX_PREAMBLE_SCAN_ROWS)
        if raw.empty:
            continue
        first_col = raw.iloc[:, 0].astype(str).str.strip().str.lower()
        matches = first_col[first_col == _HEADER_MARKER]
        if not matches.empty:
            return (sheet, int(matches.index[0]))
    return None


class AngelOneParser:
    broker = "angelone"

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() not in (".xlsx", ".xls"):
            return False
        return _find_tradebook_sheet_and_header(path) is not None

    def parse(self, path: Path) -> pd.DataFrame:
        located = _find_tradebook_sheet_and_header(path)
        if located is None:
            raise ValueError(f"Angel One parser: no tradebook header row found in {path.name}")
        sheet, header_row = located
        df = pd.read_excel(path, sheet_name=sheet, skiprows=header_row)

        cols = list(df.columns)
        col = {k: find_column(cols, v) for k, v in _ALIASES.items()}
        missing = [k for k in ("symbol", "side", "buy_price", "sell_price", "quantity", "trade_date") if col[k] is None]
        if missing:
            raise ValueError(f"Angel One parser: missing required columns {missing} in {path.name}")

        out_rows = []
        for _, row in df.iterrows():
            if pd.isna(row[col["symbol"]]):
                continue  # trailing blank/summary row after the last real fill
            symbol = str(row[col["symbol"]]).strip()
            side_raw = str(row[col["side"]]).strip().lower()
            side = "buy" if side_raw.startswith("b") else "sell"
            price = row[col["buy_price"]] if side == "buy" else row[col["sell_price"]]
            price = float(price)
            quantity = float(row[col["quantity"]])
            trade_date = str(row[col["trade_date"]])[:10]
            order_type = str(row[col["order_type"]]).strip().upper() if col["order_type"] else ""
            exchange = str(row[col["exchange"]]).strip() if col["exchange"] else "NSE"
            order_id = str(row[col["order_id"]]) if col["order_id"] else ""
            trade_id_raw = str(row[col["trade_id"]]) if col["trade_id"] else ""

            product = "MIS" if order_type == "INTRADAY" else ("CNC" if order_type == "DELIVERY" else None)
            segment = "equity_intraday" if order_type == "INTRADAY" else "equity_delivery"

            out_rows.append({
                # trade_id hash includes Angel's own per-fill Trade ID, not just
                # Order ID -- a single order can produce multiple partial fills
                # (verified in a real export: one Order ID across several Trade
                # IDs with different quantities), so Order ID alone under-
                # distinguishes rows the way Zerodha/Upstox's single-fill-per-
                # order assumption doesn't have to worry about.
                "trade_id": canonical_row_id("angelone", f"{order_id}:{trade_id_raw}", symbol, trade_date, str(quantity), str(price)),
                "broker": "angelone",
                "symbol": symbol,
                "isin": None,  # not present in this export's tradebook sheet
                "segment": segment,
                "trade_date": trade_date,
                "trade_time": None,  # this export carries date only, not time-of-fill
                "side": side,
                "quantity": quantity,
                "price": price,
                "value": quantity * price,
                "order_id": order_id or None,
                "exchange": exchange or "NSE",
                "product_type": product,
                "source_row_json": to_source_row_json(row),
            })
        return pd.DataFrame(out_rows)
