"""Phase J0.1: read-only Store connections must themselves reject writes
and skip schema/migration DDL, and must be able to coexist with EACH
OTHER (several desktop-API requests in flight at once).

What this test file deliberately does NOT claim: that a read-only
connection can open against an already-open writer, or vice versa.
Verified live with a real two-process test before writing any of this
fix: this DuckDB build's file locking is fully exclusive between any
two connections except reader-vs-reader -- a read-only connection
blocks (and is blocked by) a writer exactly like read-write-vs-
read-write does. The actual fix for the nightly jobs dying on a lock
collision is retry-with-backoff (test_connect_with_retry.py), not
read-only mode; see Store.__init__'s docstring for the full story."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from stocksense.data.store import Store


def test_read_only_store_rejects_writes(tmp_path) -> None:
    db_path = tmp_path / "test.duckdb"
    Store(db_path).close()  # create the file with the real schema first

    reader = Store(db_path, read_only=True)
    try:
        with pytest.raises(duckdb.Error):
            reader.con.execute("INSERT INTO app_settings (key, value) VALUES ('x', 'y')")
    finally:
        reader.close()


def test_read_only_store_skips_schema_and_migration_ddl(tmp_path, monkeypatch) -> None:
    """A read-only Store must never attempt CREATE TABLE / ALTER TABLE --
    both are writes, and would themselves fail (or, worse, silently
    require the DB to already be perfectly up to date) against a
    read-only connection. Guards against a future refactor accidentally
    re-adding the SCHEMA/migration calls to the read-only branch."""
    db_path = tmp_path / "test.duckdb"
    Store(db_path).close()

    calls = []
    original_execute = duckdb.DuckDBPyConnection.execute

    def _tracking_execute(self, query, *args, **kwargs):
        calls.append(query)
        return original_execute(self, query, *args, **kwargs)

    monkeypatch.setattr(duckdb.DuckDBPyConnection, "execute", _tracking_execute)
    reader = Store(db_path, read_only=True)
    reader.close()

    assert not any("CREATE TABLE" in c for c in calls)
    assert not any("ALTER TABLE" in c for c in calls)


def test_read_only_and_read_write_store_collide_in_the_same_process(tmp_path) -> None:
    """Regression, found live: this DuckDB Python client refuses to open
    a second connection to a path with a DIFFERENT read_only config
    while a connection to that path is already open in the SAME
    process -- not merely the same-process-same-config restriction the
    other tests here exercise. This is exactly what broke
    `test_get_log_of_finished_job_reads_from_disk` when two /api/jobs/*
    endpoints briefly used `_store(read_only=True)` alongside
    JobRegistry's own read-write `Store` calls in server/jobs.py --
    fixed by keeping every server/app.py endpoint on the read-write
    default rather than mixing configs in one process. Pinned here so a
    future change doesn't reintroduce read_only=True into that module
    without rediscovering this the hard way."""
    db_path = tmp_path / "test.duckdb"
    writer = Store(db_path)
    try:
        with pytest.raises(duckdb.Error):
            Store(db_path, read_only=True)
    finally:
        writer.close()


def test_two_read_only_stores_can_open_concurrently(tmp_path) -> None:
    db_path = tmp_path / "test.duckdb"
    Store(db_path).close()

    a = Store(db_path, read_only=True)
    b = Store(db_path, read_only=True)
    try:
        assert a.con.execute("SELECT 1").fetchone() == (1,)
        assert b.con.execute("SELECT 1").fetchone() == (1,)
    finally:
        a.close()
        b.close()
