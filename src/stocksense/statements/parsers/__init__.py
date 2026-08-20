from __future__ import annotations

from stocksense.statements.parsers.angel import AngelOneParser
from stocksense.statements.parsers.base import CANONICAL_TRADE_COLUMNS, StatementParser
from stocksense.statements.parsers.upstox import UpstoxParser
from stocksense.statements.parsers.zerodha import ZerodhaParser

PARSERS: list[StatementParser] = [ZerodhaParser(), UpstoxParser(), AngelOneParser()]


def detect_parser(path) -> StatementParser | None:
    """Try each registered parser's detect() until one claims the file.
    Order matters only in the pathological case where two parsers would
    both claim a file; detect() implementations are written to be
    specific enough that this should not happen in practice."""
    for parser in PARSERS:
        if parser.detect(path):
            return parser
    return None


__all__ = ["CANONICAL_TRADE_COLUMNS", "StatementParser", "PARSERS", "detect_parser"]
