"""Golden-file parser tests on hand-authored synthetic statements — no
real user data in the repo. Per docs/12-statement-forensics.md: a parsing
bug here silently corrupts every downstream diagnostic, so exact parsed
output is asserted, not just "parses without error"."""

from __future__ import annotations

from pathlib import Path

import pytest

from stocksense.statements.parsers import detect_parser
from stocksense.statements.parsers.upstox import UpstoxParser
from stocksense.statements.parsers.zerodha import ZerodhaParser

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "statements"


def test_zerodha_detect() -> None:
    parser = ZerodhaParser()
    assert parser.detect(FIXTURES / "zerodha_sample.csv")
    assert not UpstoxParser().detect(FIXTURES / "zerodha_sample.csv")


def test_upstox_detect() -> None:
    parser = UpstoxParser()
    assert parser.detect(FIXTURES / "upstox_sample.csv")
    assert not ZerodhaParser().detect(FIXTURES / "upstox_sample.csv")


def test_auto_detect_routes_correctly() -> None:
    assert detect_parser(FIXTURES / "zerodha_sample.csv").broker == "zerodha"
    assert detect_parser(FIXTURES / "upstox_sample.csv").broker == "upstox"


def test_zerodha_parse_exact_values() -> None:
    df = ZerodhaParser().parse(FIXTURES / "zerodha_sample.csv")
    assert len(df) == 4
    row = df.iloc[0]
    assert row["symbol"] == "RELIANCE"
    assert row["side"] == "buy"
    assert row["quantity"] == 10.0
    assert row["price"] == 2500.50
    assert row["value"] == 25005.0
    assert row["isin"] == "INE002A01018"
    assert row["segment"] == "equity_delivery"
    assert row["broker"] == "zerodha"

    sell_row = df.iloc[1]
    assert sell_row["side"] == "sell"
    assert sell_row["quantity"] == 10.0
    assert sell_row["price"] == 2510.00


def test_upstox_parse_exact_values_and_intraday_classification() -> None:
    df = UpstoxParser().parse(FIXTURES / "upstox_sample.csv")
    assert len(df) == 3

    mis_row = df.iloc[0]
    assert mis_row["symbol"] == "INFY"
    assert mis_row["side"] == "buy"
    assert mis_row["product_type"] == "MIS"
    assert mis_row["segment"] == "equity_intraday"  # MIS -> intraday, not delivery

    cnc_row = df.iloc[2]
    assert cnc_row["symbol"] == "HDFCBANK"
    assert cnc_row["product_type"] == "CNC"
    assert cnc_row["segment"] == "equity_delivery"


def test_trade_ids_are_unique_and_deterministic() -> None:
    df1 = ZerodhaParser().parse(FIXTURES / "zerodha_sample.csv")
    df2 = ZerodhaParser().parse(FIXTURES / "zerodha_sample.csv")
    assert df1["trade_id"].is_unique
    assert list(df1["trade_id"]) == list(df2["trade_id"])  # deterministic, not random


def test_source_row_json_preserves_raw_data() -> None:
    import json

    df = ZerodhaParser().parse(FIXTURES / "zerodha_sample.csv")
    raw = json.loads(df.iloc[0]["source_row_json"])
    assert raw["symbol"] == "RELIANCE"
    assert raw["trade_id"] == "T001"


def test_missing_required_columns_raises() -> None:
    import pandas as pd
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        bad_path = Path(d) / "bad.csv"
        pd.DataFrame({"foo": [1, 2]}).to_csv(bad_path, index=False)
        with pytest.raises(ValueError):
            ZerodhaParser().parse(bad_path)
