from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from stocksense.core.clock import FixedClock
from stocksense.core.paths import Paths
from stocksense.core.state import State


def test_paths_layout(tmp_data_root):
    p = Paths(tmp_data_root)
    root = tmp_data_root.paths.data_root
    assert p.root == root
    assert p.parquet == root / "parquet" and p.state == root / "state" and p.logs == root / "logs"
    assert p.dataset_dir("bhavcopy_eq") == root / "parquet" / "bhavcopy_eq"
    assert p.sqlite == root / "state" / "stocksense.sqlite"


def test_migrate_creates_tables_and_is_idempotent(tmp_data_root, fixed_clock):
    st = State(tmp_data_root, fixed_clock)
    st.migrate()
    st.migrate()
    con = st.connect()
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"job_runs", "kv", "schema_version"} <= names
    assert [r[0] for r in con.execute("SELECT version FROM schema_version")] == [1]
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_two_migrations_applied_twice(tmp_data_root, fixed_clock):
    migs = [(1, "CREATE TABLE a(x INTEGER);"), (2, "CREATE TABLE b(y INTEGER);")]
    st = State(tmp_data_root, fixed_clock, migrations=migs)
    st.migrate()
    st.migrate()
    con = st.connect()
    assert [r[0] for r in con.execute("SELECT version FROM schema_version ORDER BY version")] == [1, 2]
    # a later task appends a migration: only the new one is applied
    st2 = State(tmp_data_root, fixed_clock, migrations=[*migs, (3, "CREATE TABLE c(z INTEGER);")])
    st2.migrate()
    assert [r[0] for r in con.execute("SELECT version FROM schema_version ORDER BY version")] == [1, 2, 3]


def test_failed_migration_rolls_back(tmp_data_root, fixed_clock):
    st = State(tmp_data_root, fixed_clock, migrations=[(1, "CREATE TABLE a(x); CREATE TABLE a(x);")])
    with pytest.raises(sqlite3.Error):
        st.migrate()
    con = st.connect()
    assert con.execute("SELECT count(*) FROM sqlite_master WHERE name='a'").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM schema_version").fetchone()[0] == 0


def test_reader_selects_while_writer_in_transaction(tmp_data_root, fixed_clock):
    st = State(tmp_data_root, fixed_clock)
    st.migrate()
    st.kv_set("k", "old")
    writer = st.connect()
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("UPDATE kv SET value='new' WHERE key='k'")
    reader = st.connect()
    # WAL: reader sees the last committed value, does not block
    assert reader.execute("SELECT value FROM kv WHERE key='k'").fetchone()[0] == "old"
    writer.execute("COMMIT")
    assert reader.execute("SELECT value FROM kv WHERE key='k'").fetchone()[0] == "new"


def test_job_lifecycle(tmp_data_root, fixed_clock):
    st = State(tmp_data_root, fixed_clock)
    st.migrate()
    d = date(2026, 11, 12)
    assert st.job_status("eod", d) is None
    assert st.job_start("eod", d) == 1
    assert st.job_status("eod", d) == "running"
    st.job_finish("eod", d, 1, "done", "ok")
    assert st.job_status("eod", d) == "done"
    # next attempt is max+1 and the latest attempt wins
    assert st.job_start("eod", d) == 2
    st.job_finish("eod", d, 2, "failed", "boom")
    assert st.job_status("eod", d) == "failed"
    row = (
        st.connect()
        .execute("SELECT status, started_at, finished_at, detail FROM job_runs WHERE attempt=2")
        .fetchone()
    )
    assert row[0] == "failed" and row[1].endswith("+05:30") and row[2].endswith("+05:30") and row[3] == "boom"
    # trade_date may be given as a string too
    assert st.job_status("eod", "2026-11-12") == "failed"


def test_stale_running_is_failed(tmp_data_root, fixed_clock):
    st = State(tmp_data_root, fixed_clock)
    st.migrate()
    d = date(2026, 11, 12)
    st.job_start("eod", d)  # started 10:00
    later = FixedClock(fixed_clock.now() + timedelta(hours=5, minutes=59))
    assert State(tmp_data_root, later).job_status("eod", d) == "running"
    later = FixedClock(fixed_clock.now() + timedelta(hours=6, minutes=1))
    assert State(tmp_data_root, later).job_status("eod", d) == "failed"


def test_job_finish_rejects_bad_status(tmp_data_root, fixed_clock):
    st = State(tmp_data_root, fixed_clock)
    st.migrate()
    st.job_start("eod", date(2026, 11, 12))
    with pytest.raises(ValueError):
        st.job_finish("eod", date(2026, 11, 12), 1, "running", "")


def test_kv(tmp_data_root, fixed_clock):
    st = State(tmp_data_root, fixed_clock)
    st.migrate()
    assert st.kv_get("a") is None
    assert st.kv_get("a", "dflt") == "dflt"
    st.kv_set("a", "1")
    st.kv_set("a", "2")
    assert st.kv_get("a") == "2"
