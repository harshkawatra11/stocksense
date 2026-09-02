"""Storage tests.

The load-bearing one is `test_readers_work_while_a_writer_holds_the_lock`. It
pins down the property the whole storage design exists to provide, and it is a
regression test for a real production failure: the previous build opened readers
with DuckDB's `read_only=True` believing that let them coexist with a writer. It
does not -- cross-process it still raises IOException -- and every nightly job
died on it while the desktop app was open.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from stocksense.data.store import Reader, Store


def _bhav_row(symbol: str, d: date, **over) -> dict:
    row = dict(
        symbol=symbol, series="EQ", date=d, open=100.0, high=105.0, low=99.0,
        close=104.0, prev_close=100.0, last_price=104.0, volume=10_000.0,
        turnover_inr=1_040_000.0, n_trades=500.0, era="udiff",
    )
    row.update(over)
    return row


@pytest.fixture()
def paths(tmp_path):
    return tmp_path / "hot.duckdb", tmp_path / "parquet"


# ------------------------------------------------------------------ bulk I/O
def test_write_then_read_roundtrip(paths):
    db, pq = paths
    with Store(db, pq) as s:
        s.write_bhavcopy_eq(pd.DataFrame([_bhav_row("RELIANCE", date(2026, 8, 28))]))

    with Reader(pq) as r:
        got = r.bhavcopy_eq()
        assert len(got) == 1
        assert got.symbol.iloc[0] == "RELIANCE"
        assert r.bhavcopy_bounds() == (date(2026, 8, 28), date(2026, 8, 28))


def test_rewriting_the_same_key_is_idempotent_not_duplicated(paths):
    """A resumable backfill replays windows that may have been half-written.

    Replaying must be a no-op that leaves ONE row, with the newest values --
    not a second copy. Without this the row count silently inflates on every
    interrupted-then-resumed run.
    """
    db, pq = paths
    d = date(2026, 8, 28)
    with Store(db, pq) as s:
        s.write_bhavcopy_eq(pd.DataFrame([_bhav_row("RELIANCE", d, close=100.0)]))
        s.write_bhavcopy_eq(pd.DataFrame([_bhav_row("RELIANCE", d, close=999.0)]))

    with Reader(pq) as r:
        got = r.bhavcopy_eq()
        assert len(got) == 1, "replaying a window duplicated rows"
        assert got.close.iloc[0] == 999.0, "replay did not win over the stale copy"


def test_rows_are_partitioned_by_month(paths):
    """Partitioning bounds the cost of an upsert: one month is rewritten, not
    the whole dataset."""
    db, pq = paths
    with Store(db, pq) as s:
        s.write_bhavcopy_eq(
            pd.DataFrame(
                [
                    _bhav_row("A", date(2026, 7, 15)),
                    _bhav_row("A", date(2026, 8, 15)),
                    _bhav_row("A", date(2026, 9, 15)),
                ]
            )
        )
    parts = sorted(p.name for p in (pq / "bhavcopy_eq").glob("*.parquet"))
    assert parts == ["part-2026-07.parquet", "part-2026-08.parquet", "part-2026-09.parquet"]


def test_missing_columns_raise_rather_than_writing_partial_rows(paths):
    db, pq = paths
    with Store(db, pq) as s:
        with pytest.raises(ValueError, match="missing columns"):
            s.write_bhavcopy_eq(pd.DataFrame([{"symbol": "A", "date": date(2026, 8, 1)}]))


def test_no_temp_files_survive_a_write(paths):
    """Partition writes go through a temp file + atomic rename, so a reader can
    never see a half-written partition. Nothing may be left behind."""
    db, pq = paths
    with Store(db, pq) as s:
        s.write_bhavcopy_eq(pd.DataFrame([_bhav_row("A", date(2026, 8, 3))]))
    assert list((pq / "bhavcopy_eq").glob(".*tmp")) == []


# --------------------------------------------------------------- small tables
def test_ingest_runs_drive_resumability(paths):
    """`completed_units` is what makes a backfill resumable. A FAILED unit must
    not be treated as done, or the gap is never repaired. An EMPTY unit must be
    treated as done -- a market holiday genuinely has no rows, and re-fetching
    it every run is how a backfill never finishes."""
    db, pq = paths
    now = datetime.now()
    with Store(db, pq) as s:
        for unit, status in [("d1", "ok"), ("d2", "empty"), ("d3", "failed")]:
            s.record_ingest_run(
                dict(
                    run_id=f"r-{unit}", source="nse_bhavcopy", unit=unit, started_at=now,
                    finished_at=now, status=status, rows_written=0, attempt=1, error=None,
                )
            )
        assert s.completed_units("nse_bhavcopy") == {"d1", "d2"}


def test_publish_makes_small_tables_readable_without_the_duckdb_file(paths):
    db, pq = paths
    with Store(db, pq) as s:
        s.write_corporate_actions(
            pd.DataFrame(
                [
                    dict(
                        symbol="RELIANCE", ex_date=date(2024, 10, 28), action_type="bonus",
                        ratio_num=1.0, ratio_den=1.0, factor_price=0.5, dividend_amount=None,
                        face_before=10.0, face_after=10.0, subject_raw="Bonus 1:1",
                        parse_status="ok",
                    )
                ]
            )
        )
    # __exit__ published. The reader never opens the DuckDB file.
    with Reader(pq) as r:
        ca = r.corporate_actions()
        assert len(ca) == 1
        assert ca.factor_price.iloc[0] == 0.5


def test_reader_on_an_empty_store_returns_empty_frames_not_errors(paths):
    """The app must open and be usable with no data ingested yet."""
    _, pq = paths
    with Reader(pq) as r:
        assert r.bhavcopy_eq().empty
        assert r.corporate_actions().empty
        assert r.ingest_runs().empty
        assert r.bhavcopy_bounds() == (None, None)


# ------------------------------------------------------------ THE key property
_WRITER_SRC = textwrap.dedent(
    """
    import sys, time
    import pandas as pd
    from datetime import date
    from stocksense.data.store import Store
    db, pq, flag = sys.argv[1], sys.argv[2], sys.argv[3]
    s = Store(db, pq)
    s.write_bhavcopy_eq(pd.DataFrame([dict(
        symbol="LOCKTEST", series="EQ", date=date(2026, 8, 28), open=1.0, high=2.0,
        low=0.5, close=1.5, prev_close=1.4, last_price=1.5, volume=1.0,
        turnover_inr=1.0, n_trades=1.0, era="udiff")]))
    s.publish()
    open(flag, "w").write("up")     # signal: parquet is on disk, lock still held
    time.sleep(25)                  # keep holding the DuckDB write lock
    s.close()
    """
)


def test_readers_work_while_a_writer_holds_the_lock(tmp_path):
    """THE property the storage design exists to provide.

    A separate process holds the DuckDB write lock. Readers must still work.
    This is a regression test for the failure that killed the previous build's
    nightly jobs, and it also proves the negative: opening the DuckDB FILE
    read-only does not work, which is why bulk data lives in Parquet.
    """
    db, pq = tmp_path / "hot.duckdb", tmp_path / "parquet"
    flag = tmp_path / "writer_up.flag"
    script = tmp_path / "writer.py"
    script.write_text(_WRITER_SRC, encoding="utf-8")

    writer = subprocess.Popen([sys.executable, str(script), str(db), str(pq), str(flag)])
    try:
        deadline = datetime.now() + timedelta(seconds=45)
        while not flag.exists() and datetime.now() < deadline:
            if writer.poll() is not None:
                pytest.fail("writer process exited before signalling readiness")
        assert flag.exists(), "writer never came up"

        # 1. Parquet readers: must all succeed, concurrently, lock or no lock.
        for _ in range(3):
            with Reader(pq) as r:
                got = r.bhavcopy_eq()
                assert len(got) == 1
                assert got.symbol.iloc[0] == "LOCKTEST"

        # 2. The negative result that justifies the whole design: opening the
        #    DuckDB file, even read-only, is refused while the writer holds it.
        import duckdb

        with pytest.raises(duckdb.IOException):
            duckdb.connect(str(db), read_only=True)
    finally:
        writer.kill()
        writer.wait(timeout=30)


def test_time_column_dtype_is_stable_across_ingest_and_reload(paths):
    """Regression: a python `date` written to Parquet comes back as a pandas
    Timestamp. If the dtype depends on whether a frame came from ingest or from
    disk, equality joins against the trading calendar silently drop rows. The
    store normalises on write and `bhavcopy_bounds` hands back plain dates."""
    db, pq = paths
    with Store(db, pq) as s:
        s.write_bhavcopy_eq(pd.DataFrame([_bhav_row("A", date(2026, 8, 28))]))

    with Reader(pq) as r:
        got = r.bhavcopy_eq()
        assert pd.api.types.is_datetime64_any_dtype(got["date"]), "date must be datetime64 on disk"
        lo, hi = r.bhavcopy_bounds()
        assert isinstance(lo, date) and not isinstance(lo, pd.Timestamp)
        assert (lo, hi) == (date(2026, 8, 28), date(2026, 8, 28))
