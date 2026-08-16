"""CLI backfill resumability test: proves that an interruption partway
through a backfill leaves already-fetched days genuinely queryable in
the database, not just cached to disk. This is the exact property that
was missing before fetch_range became a generator -- a kill mid-run used
to lose all database progress even though the raw files survived."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest


def _bhavcopy_df(d: date, symbol="AAA"):
    return pd.DataFrame([{
        "symbol": symbol, "series": "EQ", "date": d, "open": 100.0, "high": 101.0,
        "low": 99.0, "close": 100.5, "prev_close": 100.0, "volume": 1000.0,
        "turnover_inr": 100500.0, "era": "legacy",
    }])


@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKSENSE_DUCKDB_PATH", str(tmp_path / "test.duckdb"))
    from typer.testing import CliRunner
    return CliRunner()


def test_interrupted_backfill_keeps_already_fetched_days_in_db(cli_env, monkeypatch) -> None:
    """Simulates a kill on the 3rd of 5 days: fetch_range yields days 1
    and 2 normally, then raises on day 3 (as a forced kill would
    interrupt the generator mid-iteration). Days 1 and 2 must already be
    queryable in the database despite the process never reaching the
    end of the range."""
    from stocksense.cli.main import app

    dates = [date(2024, 1, 15), date(2024, 1, 16), date(2024, 1, 17)]

    def fake_fetch_range(start, end, kind):
        yield dates[0], _bhavcopy_df(dates[0])
        yield dates[1], _bhavcopy_df(dates[1])
        raise KeyboardInterrupt("simulated kill signal")

    monkeypatch.setattr("stocksense.cli.main.fetch_range", fake_fetch_range)

    # Click converts an unhandled KeyboardInterrupt into SystemExit(130)
    # (the standard SIGINT exit code) rather than re-raising it -- the
    # property under test is that the DB already has the pre-interruption
    # days, not the exact exception type surfaced by the test runner.
    result = cli_env.invoke(app, ["backfill-nse-archive", "--start", "2024-01-15", "--end", "2024-01-19", "--kind", "cm"])
    assert result.exit_code == 130

    from stocksense.core.config import get_settings
    from stocksense.data.store import Store

    store = Store(get_settings().duckdb_path)
    rows = store.read_bhavcopy_eq()
    store.close()

    assert len(rows) == 2  # both pre-interruption days survived in the DB
    assert set(rows["date"].astype(str)) == {"2024-01-15", "2024-01-16"}


def test_resumed_backfill_after_interruption_completes_the_range(cli_env, monkeypatch) -> None:
    """After the simulated kill above, a second invocation covering the
    same range should be able to add the remaining days -- proving the
    two-phase story (interrupt keeps partial progress, re-run completes
    it) actually works end to end."""
    from stocksense.cli.main import app

    dates = [date(2024, 1, 15), date(2024, 1, 16), date(2024, 1, 17)]

    def first_attempt(start, end, kind):
        yield dates[0], _bhavcopy_df(dates[0])
        yield dates[1], _bhavcopy_df(dates[1])
        raise KeyboardInterrupt("simulated kill")

    monkeypatch.setattr("stocksense.cli.main.fetch_range", first_attempt)
    first_result = cli_env.invoke(app, ["backfill-nse-archive", "--start", "2024-01-15", "--end", "2024-01-17", "--kind", "cm"])
    assert first_result.exit_code == 130

    def second_attempt(start, end, kind):
        # a real resume would skip re-fetching cached days via
        # _cached_or_fetch; this fake simulates that by only yielding
        # the day that was never reached the first time
        yield dates[2], _bhavcopy_df(dates[2])

    monkeypatch.setattr("stocksense.cli.main.fetch_range", second_attempt)
    result = cli_env.invoke(app, ["backfill-nse-archive", "--start", "2024-01-15", "--end", "2024-01-17", "--kind", "cm"])
    assert result.exit_code == 0

    from stocksense.core.config import get_settings
    from stocksense.data.store import Store

    store = Store(get_settings().duckdb_path)
    rows = store.read_bhavcopy_eq()
    store.close()

    assert len(rows) == 3  # all three days now present
    assert set(rows["date"].astype(str)) == {"2024-01-15", "2024-01-16", "2024-01-17"}
