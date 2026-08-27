"""Phase J0.1: connect_with_retry is the actual fix for the nightly
scheduled jobs dying outright on a single momentary DuckDB lock
collision (reconcile.log/daily_backfill.log both terminated their whole
run on one IOException, no retry). Verified live before writing this
that a plain read_only flag does NOT avoid the collision -- this
DuckDB build's locking is fully exclusive between any two connections
except reader-vs-reader -- so retry-with-backoff is the load-bearing
fix, not a nicety."""

from __future__ import annotations

from unittest.mock import patch

import duckdb
import pytest

from stocksense.data.store import Store, connect_with_retry


def test_connect_with_retry_succeeds_immediately_when_unlocked(tmp_path) -> None:
    db_path = tmp_path / "test.duckdb"
    Store(db_path).close()

    with patch("stocksense.data.store.time.sleep") as sleep_mock:
        store = connect_with_retry(db_path)
        try:
            assert store.con.execute("SELECT 1").fetchone() == (1,)
        finally:
            store.close()
    sleep_mock.assert_not_called()


def test_connect_with_retry_recovers_from_transient_lock(tmp_path) -> None:
    db_path = tmp_path / "test.duckdb"
    Store(db_path).close()

    real_store_init = Store.__init__
    calls = {"n": 0}

    def _flaky_init(self, path, read_only=False):
        calls["n"] += 1
        if calls["n"] < 3:
            raise duckdb.IOException("simulated lock collision")
        real_store_init(self, path, read_only=read_only)

    with patch.object(Store, "__init__", _flaky_init), patch("stocksense.data.store.time.sleep") as sleep_mock:
        store = connect_with_retry(db_path, attempts=5, delay_s=1.0)
        try:
            assert calls["n"] == 3
            assert sleep_mock.call_count == 2  # slept between attempt 1->2 and 2->3, not after success
        finally:
            store.close()


def test_connect_with_retry_gives_up_after_max_attempts(tmp_path) -> None:
    db_path = tmp_path / "test.duckdb"

    def _always_locked(self, path, read_only=False):
        raise duckdb.IOException("simulated permanent lock")

    with patch.object(Store, "__init__", _always_locked), patch("stocksense.data.store.time.sleep") as sleep_mock:
        with pytest.raises(duckdb.IOException):
            connect_with_retry(db_path, attempts=3, delay_s=1.0)
    assert sleep_mock.call_count == 2  # slept between each of the 3 attempts, not after the last failure


def test_connect_with_retry_does_not_retry_non_io_errors(tmp_path) -> None:
    db_path = tmp_path / "test.duckdb"

    def _raises_value_error(self, path, read_only=False):
        raise ValueError("not a lock problem")

    with patch.object(Store, "__init__", _raises_value_error), patch("stocksense.data.store.time.sleep") as sleep_mock:
        with pytest.raises(ValueError):
            connect_with_retry(db_path)
    sleep_mock.assert_not_called()
