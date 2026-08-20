"""Golden-file tests for the Angel One parser, on a hand-authored
synthetic workbook shaped like a real export (verified during planning
against an actual Angel One "Trades_History" download) -- no real user
data in the repo, matching test_statement_parsers.py's convention.

Built programmatically (not committed as a binary .xlsx) so the fixture
stays diffable and its exact shape is visible in this file: a preamble
of report-metadata rows, a "TradesAndCharges"-style header row appearing
partway down the sheet, then real fill rows -- exactly the structure
that broke pandas' default header=0 assumption when first inspected."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stocksense.statements.parsers import detect_parser
from stocksense.statements.parsers.angel import AngelOneParser
from stocksense.statements.parsers.upstox import UpstoxParser
from stocksense.statements.parsers.zerodha import ZerodhaParser

_TRADEBOOK_HEADER = [
    "Scrip/Contract", "Buy/Sell", "Buy Price", "Sell Price", "Quantity",
    "Brokerage", "GST", "STT", "Sebi Tax", "Exchange Turnover Charges",
    "Stamp Duty", "Other Charges", "IPFT Charges", "Order Type", "Segment",
    "Exchange", "Order ID", "Trade ID", "Date",
]


def _write_angel_workbook(path: Path, rows: list[list], extra_sheets: dict[str, list[list]] | None = None) -> None:
    """Writes a workbook shaped like a real Angel One export: preamble
    rows (client info, charges summary) BEFORE the real header, on a
    sheet named "TradesAndCharges" -- plus optional other sheets (like
    the real export's "Broking Ledger"/"Charges") that carry no
    trade-level data at all, to prove detect() doesn't false-positive
    on them."""
    preamble = [
        ["ClientCode", "AACI000000"],
        ["DateOfDownload", "2026-08-18"],
        [],
        ["Date Range"],
        ["StartDate", "EndDate"],
        ["2026-07-19", "2026-08-17"],
        [],
        ["Charges Summary"],
        ["Total Trades", str(len(rows))],
        [],
        ["Trade Charges"],
        ["Brokerage", "100.0"],
        [],
        ["TradeBook And Charges"],
        _TRADEBOOK_HEADER,
    ] + rows

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(preamble).to_excel(writer, sheet_name="TradesAndCharges", index=False, header=False)
        if extra_sheets:
            for sheet_name, sheet_rows in extra_sheets.items():
                pd.DataFrame(sheet_rows).to_excel(writer, sheet_name=sheet_name, index=False, header=False)


@pytest.fixture()
def angel_workbook(tmp_path) -> Path:
    rows = [
        # Buy, Intraday
        ["RELIANCE", "Buy", 2500.50, None, 10, 5.0, 0.9, 0.02, 0.4, 0.5, 0, 0, 0, "Intraday", "CAPITAL", "NSE", "ORD001", "TRD001", "2026-08-01"],
        # Sell, Intraday, same order (a partial-fill split -- real Angel exports do this)
        ["RELIANCE", "Sell", None, 2510.00, 10, 5.0, 0.9, 0.02, 0.4, 0, 0, 0, 0, "Intraday", "CAPITAL", "NSE", "ORD001", "TRD002", "2026-08-01"],
        # Buy, Delivery
        ["HDFCBANK", "Buy", 1650.00, None, 15, 8.0, 1.4, 0.03, 0.6, 0.8, 5.0, 0, 0, "Delivery", "CAPITAL", "NSE", "ORD002", "TRD003", "2026-08-02"],
    ]
    path = tmp_path / "Trades_History_AACI000000.xlsx"
    _write_angel_workbook(
        path, rows,
        extra_sheets={"Broking Ledger": [["Client Basic Information"], ["ClientCode", "AACI000000"]]},
    )
    return path


def test_angel_detect_finds_the_tradebook_sheet(angel_workbook) -> None:
    parser = AngelOneParser()
    assert parser.detect(angel_workbook)
    assert not ZerodhaParser().detect(angel_workbook)
    assert not UpstoxParser().detect(angel_workbook)


def test_angel_detect_false_for_non_angel_files(tmp_path) -> None:
    path = tmp_path / "not_angel.xlsx"
    pd.DataFrame({"foo": [1, 2], "bar": [3, 4]}).to_excel(path, index=False)
    assert not AngelOneParser().detect(path)


def test_auto_detect_routes_angel_correctly(angel_workbook) -> None:
    parser = detect_parser(angel_workbook)
    assert parser is not None
    assert parser.broker == "angelone"


def test_angel_parse_exact_values_and_price_coalescing(angel_workbook) -> None:
    df = AngelOneParser().parse(angel_workbook)
    assert len(df) == 3

    buy_row = df.iloc[0]
    assert buy_row["symbol"] == "RELIANCE"
    assert buy_row["side"] == "buy"
    assert buy_row["price"] == 2500.50  # coalesced from Buy Price, not Sell Price (which was NaN)
    assert buy_row["quantity"] == 10.0
    assert buy_row["value"] == 25005.0
    assert buy_row["segment"] == "equity_intraday"
    assert buy_row["product_type"] == "MIS"
    assert buy_row["broker"] == "angelone"

    sell_row = df.iloc[1]
    assert sell_row["side"] == "sell"
    assert sell_row["price"] == 2510.00  # coalesced from Sell Price, not Buy Price (which was NaN)

    delivery_row = df.iloc[2]
    assert delivery_row["symbol"] == "HDFCBANK"
    assert delivery_row["segment"] == "equity_delivery"
    assert delivery_row["product_type"] == "CNC"


def test_angel_trade_ids_distinguish_partial_fills_under_the_same_order(angel_workbook) -> None:
    """Regression: two fills sharing an Order ID but different Trade IDs
    (a real pattern in Angel exports -- one order, multiple partial
    fills) must not collide into the same trade_id."""
    df = AngelOneParser().parse(angel_workbook)
    buy_row, sell_row = df.iloc[0], df.iloc[1]
    assert buy_row["order_id"] == sell_row["order_id"] == "ORD001"
    assert buy_row["trade_id"] != sell_row["trade_id"]


def test_angel_trade_ids_are_deterministic(angel_workbook) -> None:
    df1 = AngelOneParser().parse(angel_workbook)
    df2 = AngelOneParser().parse(angel_workbook)
    assert list(df1["trade_id"]) == list(df2["trade_id"])
    assert df1["trade_id"].is_unique


def test_angel_source_row_json_preserves_raw_data(angel_workbook) -> None:
    import json

    df = AngelOneParser().parse(angel_workbook)
    raw = json.loads(df.iloc[0]["source_row_json"])
    assert raw["Scrip/Contract"] == "RELIANCE"


def test_angel_ignores_trailing_blank_rows(tmp_path) -> None:
    rows = [
        ["INFY", "Buy", 1500.0, None, 5, 2.0, 0.3, 0.01, 0.1, 0.1, 0, 0, 0, "Intraday", "CAPITAL", "NSE", "ORD010", "TRD010", "2026-08-05"],
        [None] * 19,  # a genuinely blank trailing row, as real exports sometimes have
    ]
    path = tmp_path / "Trades_History_AACI000001.xlsx"
    _write_angel_workbook(path, rows)
    df = AngelOneParser().parse(path)
    assert len(df) == 1


def test_angel_parse_raises_on_missing_required_columns(tmp_path) -> None:
    path = tmp_path / "bad_angel.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([["Scrip/Contract", "Buy/Sell"], ["RELIANCE", "Buy"]]).to_excel(
            writer, sheet_name="TradesAndCharges", index=False, header=False
        )
    with pytest.raises(ValueError):
        AngelOneParser().parse(path)


def test_angel_does_not_false_positive_on_non_tradebook_sheets_only(tmp_path) -> None:
    """A file with e.g. only a 'Broking Ledger'/'Charges'-shaped sheet
    (no TradesAndCharges header anywhere) must not be claimed by
    AngelOneParser -- matches the real YourStatement_*.xlsx export,
    which carries no per-fill trade data at all."""
    path = tmp_path / "YourStatement_AACI000000.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([["Client Basic Information"], ["ClientCode", "AACI000000"], ["Opening Balance", "0"]]).to_excel(
            writer, sheet_name="Broking Ledger", index=False, header=False
        )
    assert not AngelOneParser().detect(path)
