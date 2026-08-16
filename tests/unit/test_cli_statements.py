"""End-to-end CLI test for statement-ingest -> kundli. Caught a real bug
once: write_trades used 'INSERT ... SELECT *' which binds by column
POSITION, not name, silently misaligning trade_time into trade_date's
slot once statement_id was appended out of DDL order. Store.write_*
methods now bind by explicit column name; this test guards the full
path end-to-end so a regression fails loudly here, not silently in
production."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stocksense.agent.claude_cli import AgentResult

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "statements"


@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKSENSE_DUCKDB_PATH", str(tmp_path / "test.duckdb"))

    import stocksense.statements.report as report_mod

    def fake_invoke(req, store=None, job_run_id=None):
        return AgentResult(
            agent_run_id="test", output_text="FAKE NARRATIVE", status="ok", error=None,
            started_at=datetime.now(timezone.utc), finished_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr(report_mod, "invoke", fake_invoke)
    return CliRunner()


def test_ingest_then_kundli_end_to_end(cli_env) -> None:
    from stocksense.cli.main import app

    fixture = str(FIXTURES / "zerodha_sample.csv")
    result = cli_env.invoke(app, ["statement-ingest", fixture])
    assert result.exit_code == 0, result.output
    assert "Ingested 4 trades" in result.output

    kundli_result = cli_env.invoke(app, ["kundli"])
    assert kundli_result.exit_code == 0, kundli_result.output
    assert "Positions analyzed: 2" in kundli_result.output
    assert "FAKE NARRATIVE" in kundli_result.output


def test_reingest_same_file_is_idempotent(cli_env) -> None:
    from stocksense.cli.main import app

    fixture = str(FIXTURES / "zerodha_sample.csv")
    cli_env.invoke(app, ["statement-ingest", fixture])
    result = cli_env.invoke(app, ["statement-ingest", fixture])
    assert result.exit_code == 0
    assert "Already ingested" in result.output


def test_trade_dates_and_times_land_in_correct_columns(cli_env) -> None:
    """The regression test for the exact bug found: trade_time values
    (HH:MM:SS strings) must not end up cast into the trade_date column."""
    from stocksense.cli.main import app
    from stocksense.core.config import get_settings
    from stocksense.data.store import Store

    fixture = str(FIXTURES / "zerodha_sample.csv")
    cli_env.invoke(app, ["statement-ingest", fixture])

    settings = get_settings()
    store = Store(settings.duckdb_path)
    trades = store.read_trades()
    store.close()

    assert len(trades) == 4
    first = trades.iloc[0]
    assert str(first["trade_date"])[:10] == "2024-01-15"
    assert str(first["trade_time"]).startswith("09:20:15")
