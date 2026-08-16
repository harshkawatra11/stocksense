"""
Statement parser contract (docs/12-statement-forensics.md).

Every broker parser normalizes to the same canonical `trades` schema
(stocksense.data.store's `trades` table) regardless of broker export
quirks. detect() sniffs file *headers*, never the filename — a renamed
file must still be identified correctly, and a wrongly-named file must
not be silently misparsed.

Column matching is fuzzy (case/whitespace-insensitive, alias-based)
rather than hardcoded to one exact header set, because broker CSV/XLSX
column names have been observed to drift across export format versions
— the same lesson this project already learned once from yfinance's
adjustment-factor discontinuities: don't trust an external format to
stay fixed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import pandas as pd

CANONICAL_TRADE_COLUMNS = [
    "trade_id", "statement_id", "broker", "symbol", "isin", "segment",
    "trade_date", "trade_time", "side", "quantity", "price", "value",
    "order_id", "exchange", "product_type", "source_row_json",
]


class StatementParser(Protocol):
    broker: str

    def detect(self, path: Path) -> bool: ...
    def parse(self, path: Path) -> pd.DataFrame: ...


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tabular(path: Path) -> pd.DataFrame:
    """Read CSV or XLSX transparently; broker exports come in both."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, dtype=str)


def find_column(columns: list[str], aliases: list[str]) -> str | None:
    """Case/whitespace/underscore-insensitive match against a list of
    acceptable header names for one logical field."""
    normalized = {c: c.strip().lower().replace(" ", "_").replace("-", "_") for c in columns}
    alias_set = {a.strip().lower().replace(" ", "_").replace("-", "_") for a in aliases}
    for orig, norm in normalized.items():
        if norm in alias_set:
            return orig
    return None


def infer_segment(exchange_or_segment: str, series: str | None = None) -> str:
    """Map a broker's raw exchange/segment/series field to our canonical
    segment vocabulary. Conservative: unrecognized values fall through to
    equity_delivery rather than raising, since statement forensics must
    degrade gracefully on an unexpected row rather than abort the whole
    ingestion (a single malformed row must not sink the report)."""
    s = (exchange_or_segment or "").strip().upper()
    series = (series or "").strip().upper()
    if s in ("NFO", "BFO", "FO") or series in ("FUTSTK", "FUTIDX"):
        return "fno_futures"
    if series in ("OPTSTK", "OPTIDX"):
        return "fno_options"
    if s in ("CDS", "MCX", "BCD"):
        return "commodity"
    return "equity_delivery"  # refined to equity_intraday by product_type downstream


def canonical_row_id(broker: str, order_id: str, symbol: str, trade_date: str, quantity: str, price: str) -> str:
    raw = f"{broker}|{order_id}|{symbol}|{trade_date}|{quantity}|{price}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def to_source_row_json(row: pd.Series) -> str:
    return json.dumps({k: (None if pd.isna(v) else str(v)) for k, v in row.items()})
