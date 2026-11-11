from __future__ import annotations

from datetime import datetime

import polars as pl
from typer.testing import CliRunner

from stocksense.cli import app
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock
from stocksense.data import io
from stocksense.data.status import status_table
from tests.fixtures.synth import make_market, write_market

runner = CliRunner()
NOW = FixedClock(datetime(2026, 11, 11, 10, tzinfo=IST))


def _row(t: pl.DataFrame, name: str) -> dict:
    return t.filter(pl.col("dataset") == name).row(0, named=True)


def test_status_table_columns_and_values(tmp_data_root, tmp_data_dir):
    m = make_market(n=40, n_sessions=60, seed=3)
    write_market(m, tmp_data_dir)
    cal = TradingCalendar.from_store(tmp_data_root, NOW)
    t = status_table(tmp_data_root, cal, NOW)
    assert t.columns == ["dataset", "first_date", "last_date", "rows", "sessions_missing", "fresh", "issues"]
    row = _row(t, "bhavcopy_eq")
    sessions = sorted(m.eq["date"].unique().to_list())
    assert row["first_date"] == sessions[0]
    assert row["last_date"] == sessions[-1]
    assert row["rows"] == m.eq.height
    assert row["sessions_missing"] == 0
    assert row["fresh"] is False  # synthetic data ends in 2020; clock is 2026-11-11
    assert row["issues"] == len(sessions)  # every session is below the 500-row floor (40 stocks)
    assert _row(t, "bhavcopy_delivery")["rows"] == m.delivery.height


def test_sessions_missing_counts_calendar_sessions_without_rows(tmp_data_root, tmp_data_dir):
    m = make_market(n=40, n_sessions=30, seed=3)
    write_market(m, tmp_data_dir)
    sessions = sorted(m.eq["date"].unique().to_list())
    drop = [sessions[10], sessions[11]]
    # rewrite delivery without two middle sessions
    for key in sorted({d.strftime("%Y-%m") for d in sessions}):
        part = m.delivery.filter(pl.col("date").dt.strftime("%Y-%m") == key, ~pl.col("date").is_in(drop))
        io.write_partition(tmp_data_root, "bhavcopy_delivery", key, part)
    cal = TradingCalendar.from_store(tmp_data_root, NOW)
    dl = _row(status_table(tmp_data_root, cal, NOW), "bhavcopy_delivery")
    # the eq-derived calendar has every session; delivery lacks exactly the two dropped ones
    assert dl["sessions_missing"] == 2


def test_fresh_true_when_last_completed_session_is_present(tmp_data_root, tmp_data_dir):
    m = make_market(n=40, n_sessions=30, seed=3)
    write_market(m, tmp_data_dir)
    last = max(m.eq["date"].to_list())
    clock = FixedClock(datetime(last.year, last.month, last.day, 20, tzinfo=IST))
    cal = TradingCalendar.from_store(tmp_data_root, clock)
    t = status_table(tmp_data_root, cal, clock)
    assert _row(t, "bhavcopy_eq")["fresh"] is True


def test_cli_data_status_prints_table(tmp_data_root, tmp_data_dir):
    write_market(make_market(n=40, n_sessions=30, seed=3), tmp_data_dir)
    res = runner.invoke(app, ["data-status"], obj=tmp_data_root)
    assert res.exit_code == 0, res.output
    assert "bhavcopy_eq" in res.output
    assert "sessions_missing" in res.output
    assert "fresh" in res.output


def test_cli_data_status_empty_store(tmp_data_root):
    res = runner.invoke(app, ["data-status"], obj=tmp_data_root)
    assert res.exit_code == 1
    assert "no data" in res.output.lower()
