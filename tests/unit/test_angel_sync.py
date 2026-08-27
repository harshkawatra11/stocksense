"""Phase J1.3: Angel One sync tests. Network fully mocked -- the field
mappings in normalize_holdings/normalize_positions are pinned against
REAL response shapes captured live against the user's own account
before this module was written (see angel_sync.py's module docstring),
not guessed from documentation."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from stocksense.brokers.angel_session import BrokerAuthError, TransientBrokerError
from stocksense.brokers.angel_sync import normalize_holdings, normalize_positions, sync_angel
from stocksense.data.store import Store


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


_REAL_POSITION_ROW = {
    "tradingsymbol": "MOTILALOFS-EQ", "exchange": "NSE", "producttype": "INTRADAY",
    "netqty": "0", "buyqty": "65", "sellqty": "65", "buyavgprice": "1041.18",
    "sellavgprice": "1037.52", "ltp": "1038.1", "realised": "-237.90", "unrealised": "-0.00",
}

_REAL_HOLDING_ROW = {
    "tradingsymbol": "RELIANCE-EQ", "exchange": "NSE", "isin": "INE002A01018",
    "quantity": "10", "t1quantity": "0", "averageprice": "1300.5", "ltp": "1350.0",
    "close": "1345.0", "profitandloss": "495.0",
}


def test_normalize_positions_uses_real_field_names() -> None:
    df = normalize_positions([_REAL_POSITION_ROW], date(2026, 8, 28))
    row = df.iloc[0]
    assert row["symbol"] == "MOTILALOFS"
    assert row["net_qty"] == 0.0
    assert row["buy_avg"] == pytest.approx(1041.18)
    assert row["realised"] == pytest.approx(-237.90)


def test_normalize_holdings_uses_real_field_names() -> None:
    df = normalize_holdings([_REAL_HOLDING_ROW], date(2026, 8, 28))
    row = df.iloc[0]
    assert row["symbol"] == "RELIANCE"
    assert row["quantity"] == 10.0
    assert row["avg_price"] == pytest.approx(1300.5)
    assert row["isin"] == "INE002A01018"


def test_normalize_positions_empty_list() -> None:
    df = normalize_positions([], date(2026, 8, 28))
    assert df.empty
    assert list(df.columns)  # still has the right schema, just no rows


def test_normalize_tolerates_missing_or_blank_numeric_fields() -> None:
    row = dict(_REAL_POSITION_ROW)
    row["netqty"] = ""
    df = normalize_positions([row], date(2026, 8, 28))
    assert df.iloc[0]["net_qty"] == 0.0


def test_sync_angel_records_auth_failure_without_raising(tmp_store) -> None:
    with patch("stocksense.brokers.angel_sync.login", side_effect=BrokerAuthError("bad totp")):
        result = sync_angel(tmp_store, settings=MagicMock())
    assert result.status == "auth_failure"
    runs = tmp_store.read_broker_sync_runs("angelone")
    assert len(runs) == 1
    assert runs.iloc[0]["status"] == "auth_failure"


def test_sync_angel_records_transient_failure_without_raising(tmp_store) -> None:
    with patch("stocksense.brokers.angel_sync.login", side_effect=TransientBrokerError("timeout")):
        result = sync_angel(tmp_store, settings=MagicMock())
    assert result.status == "transient_failure"


def test_sync_angel_writes_holdings_and_positions_on_success(tmp_store) -> None:
    fake_client = MagicMock()
    fake_client.holding.return_value = {"status": True, "data": [_REAL_HOLDING_ROW]}
    fake_client.position.return_value = {"status": True, "data": [_REAL_POSITION_ROW]}

    with patch("stocksense.brokers.angel_sync.login", return_value=(MagicMock(), fake_client)):
        result = sync_angel(tmp_store, settings=MagicMock())

    assert result.status == "ok"
    assert result.n_holdings == 1
    assert result.n_positions == 1
    holdings = tmp_store.read_broker_holdings("angelone")
    positions = tmp_store.read_broker_positions_snapshot("angelone")
    assert len(holdings) == 1
    assert len(positions) == 1


def test_sync_angel_resyncing_same_day_overwrites_not_duplicates(tmp_store) -> None:
    fake_client = MagicMock()
    fake_client.holding.return_value = {"status": True, "data": [_REAL_HOLDING_ROW]}
    fake_client.position.return_value = {"status": True, "data": [_REAL_POSITION_ROW]}

    with patch("stocksense.brokers.angel_sync.login", return_value=(MagicMock(), fake_client)):
        sync_angel(tmp_store, settings=MagicMock())
        sync_angel(tmp_store, settings=MagicMock())

    assert len(tmp_store.read_broker_holdings("angelone")) == 1
    assert len(tmp_store.read_broker_positions_snapshot("angelone")) == 1
    # but two sync RUNS are recorded -- the audit trail is append-only
    assert len(tmp_store.read_broker_sync_runs("angelone")) == 2


def test_broker_sync_cli_reports_status(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STOCKSENSE_DUCKDB_PATH", str(tmp_path / "test.duckdb"))
    from typer.testing import CliRunner

    from stocksense.brokers.angel_sync import SyncResult
    from stocksense.cli.main import app

    def fake_sync_angel(store, settings, scopes):
        return SyncResult(sync_id="abc123", status="ok", n_holdings=2, n_positions=1, error=None)

    monkeypatch.setattr("stocksense.cli.main.sync_angel", fake_sync_angel)
    result = CliRunner().invoke(app, ["broker-sync"])

    assert result.exit_code == 0
    assert "status=ok" in result.output
    assert "holdings synced: 2" in result.output


def test_broker_sync_cli_exits_nonzero_on_auth_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STOCKSENSE_DUCKDB_PATH", str(tmp_path / "test.duckdb"))
    from typer.testing import CliRunner

    from stocksense.brokers.angel_sync import SyncResult
    from stocksense.cli.main import app

    def fake_sync_angel(store, settings, scopes):
        return SyncResult(sync_id="abc123", status="auth_failure", n_holdings=0, n_positions=0, error="bad totp")

    monkeypatch.setattr("stocksense.cli.main.sync_angel", fake_sync_angel)
    result = CliRunner().invoke(app, ["broker-sync"])

    assert result.exit_code == 1


def test_broker_sync_cli_rejects_unsupported_broker(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STOCKSENSE_DUCKDB_PATH", str(tmp_path / "test.duckdb"))
    from typer.testing import CliRunner

    from stocksense.cli.main import app

    result = CliRunner().invoke(app, ["broker-sync", "--broker", "zerodha"])
    assert result.exit_code == 1


def test_sync_angel_one_scope_failing_does_not_abort_the_other(tmp_store) -> None:
    fake_client = MagicMock()
    fake_client.holding.side_effect = RuntimeError("boom")
    fake_client.position.return_value = {"status": True, "data": [_REAL_POSITION_ROW]}

    with patch("stocksense.brokers.angel_sync.login", return_value=(MagicMock(), fake_client)):
        result = sync_angel(tmp_store, settings=MagicMock())

    assert result.status == "partial"
    assert result.n_positions == 1
    assert "holdings" in result.error
