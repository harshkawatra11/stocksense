"""Storage: a single-writer DuckDB catalogue plus a lock-free Parquet read surface.

WHY THIS SHAPE (measured on this machine, 2026-09-02, not assumed):

DuckDB allows exactly one process to hold a database file, and `read_only=True`
does NOT get you around it. With a writer holding `hot.duckdb`, a second process
opening the same file read-only fails:

    _duckdb.IOException: IO Error: Cannot open file "...": File is already open

That is precisely the failure that killed the previous build's nightly jobs --
the desktop app held a connection, and every scheduled ingest died on the lock
with the ledger silently accumulating nothing. Opening readers `read_only=True`
was the fix that plan proposed, and the probe above shows it does not work.

What DOES work, verified in the same test: three concurrent processes reading
Parquet through an in-memory DuckDB connection, while a fourth process held the
DuckDB file lock. Parquet has no lock.

So:

    Store   -- the single WRITER. Owns stocksense.duckdb, which holds only small
               mutable state (ingest telemetry, corporate actions, and later the
               attempt registry / orders / arming). Bulk rows never live here.
    Reader  -- LOCK-FREE, any number of concurrent processes. Reads only Parquet.
               This is what the research workers, the API server and the UI use.

`Store.publish()` snapshots the small DuckDB tables out to Parquet so that
readers have exactly one access path for everything. The rule to keep in your
head: **nothing outside this module opens the DuckDB file.**
"""

from __future__ import annotations

import os
import shutil
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

# Small, mutable, upsert-heavy tables. These live in DuckDB because they need
# transactional ON CONFLICT semantics, and they are small enough (tens of
# thousands of rows) that snapshotting them to Parquet on publish is cheap.
SCHEMA = """
-- Parsed from NSE free-text corporate-action subject lines. `parse_status` is
-- ok or unparsed, and unparsed rows are KEPT: silent skips are how this rots,
-- and the known-unparseable classes (rights issues, buybacks, demergers,
-- schemes of arrangement) must stay visible and countable.
CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol          VARCHAR NOT NULL,
    ex_date         DATE    NOT NULL,
    action_type     VARCHAR NOT NULL,
    ratio_num       DOUBLE,
    ratio_den       DOUBLE,
    factor_price    DOUBLE,
    dividend_amount DOUBLE,
    face_before     DOUBLE,
    face_after      DOUBLE,
    subject_raw     VARCHAR NOT NULL,
    parse_status    VARCHAR NOT NULL,
    PRIMARY KEY (symbol, ex_date, subject_raw)
);

-- Every ingest attempt, including failures and retries. This is what the
-- desktop pipeline monitor renders, and it is the reason there are no scheduled
-- tasks: a job that fails must be VISIBLE, with its error, not silently absent.
CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id       VARCHAR   NOT NULL PRIMARY KEY,
    source       VARCHAR   NOT NULL,
    unit         VARCHAR   NOT NULL,
    started_at   TIMESTAMP NOT NULL,
    finished_at  TIMESTAMP,
    status       VARCHAR   NOT NULL,
    rows_written BIGINT    NOT NULL DEFAULT 0,
    attempt      INTEGER   NOT NULL DEFAULT 1,
    error        VARCHAR
);
CREATE INDEX IF NOT EXISTS ix_ingest_runs_source_unit ON ingest_runs (source, unit);
"""

# ---------------------------------------------------------------- bulk datasets
# Partitioned Parquet. The partition key is chosen so that an upsert rewrites a
# BOUNDED amount of data: one month of bhavcopy is ~22 days x ~2,500 symbols,
# which is trivial to rewrite, whereas a single flat file would not be.

BULK_SCHEMAS: dict[str, dict[str, Any]] = {
    "bhavcopy_eq": {
        "columns": [
            "symbol", "series", "date", "open", "high", "low", "close",
            "prev_close", "last_price", "volume", "turnover_inr", "n_trades", "era",
        ],
        "keys": ["symbol", "series", "date"],
        "partition_on": "date",  # year-month
    },
    "bhavcopy_delivery": {
        "columns": ["symbol", "series", "date", "deliv_qty", "deliv_pct"],
        "keys": ["symbol", "series", "date"],
        "partition_on": "date",
    },
    "intraday_bars": {
        "columns": ["symbol", "ts", "interval", "open", "high", "low", "close", "volume"],
        "keys": ["symbol", "ts", "interval"],
        "partition_on": "ts",
    },
}

SMALL_TABLES = ("corporate_actions", "ingest_runs")


def _partition_key(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.strftime("%Y-%m")


class StoreLocked(RuntimeError):
    """Another process already holds the single write lock.

    Raised instead of DuckDB's raw IOException, which says "The process cannot
    access the file because it is being used by another process" and leaves the
    reader to work out that this is by design, not corruption. The previous build
    wasted real time on exactly that message.
    """


def _raise_if_locked(exc: Exception, path: Path) -> None:
    text = str(exc)
    if "already open" not in text and "being used by another process" not in text:
        return
    pid = ""
    for token in text.replace("(", " ").replace(")", " ").split():
        if token.isdigit() and len(token) >= 3:
            pid = f" (PID {token})"
    raise StoreLocked(
        f"another StockSense writer already holds {path.name}{pid}.\n"
        "This is BY DESIGN: exactly one process may write at a time.\n"
        "  - Wait for the running job (check with: stocksense data-status), or\n"
        "  - stop it, then re-run -- every backfill is resumable and loses nothing.\n"
        "Reads are never blocked: data-status and the API use the lock-free "
        "Parquet Reader and work fine right now."
    ) from exc


class Store:
    """The single writer. Nothing else may open the DuckDB file.

    Bulk frames are appended to partitioned Parquet; small mutable tables go to
    DuckDB and are snapshotted to Parquet by `publish()`.
    """

    def __init__(self, duckdb_path: str | Path, parquet_root: str | Path) -> None:
        self.duckdb_path = Path(duckdb_path)
        self.parquet_root = Path(parquet_root)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        try:
            self.con = duckdb.connect(str(self.duckdb_path))
        except Exception as exc:
            _raise_if_locked(exc, self.duckdb_path)
            raise
        self.con.execute(SCHEMA)

    # -------------------------------------------------------------- bulk I/O
    def _dataset_dir(self, dataset: str) -> Path:
        return self.parquet_root / dataset

    def write_bulk(self, dataset: str, df: pd.DataFrame) -> int:
        """Upsert `df` into a partitioned Parquet dataset. Idempotent.

        Idempotence is not a nicety: the whole resumable-backfill design depends
        on being able to replay a window that may have been half-written when a
        process died. Re-running a window must be a no-op, not a duplicate.

        The write is atomic per partition -- a temp file is renamed into place --
        so a reader can never observe a half-written partition, and a crash
        mid-write leaves the previous partition intact.
        """
        spec = BULK_SCHEMAS[dataset]
        if df.empty:
            return 0
        missing = [c for c in spec["columns"] if c not in df.columns]
        if missing:
            raise ValueError(f"{dataset}: missing columns {missing}")

        frame = df[spec["columns"]].copy()
        # Normalise the time column to datetime64 ON WRITE. Parquet round-trips a
        # python `date` back as a pandas Timestamp, so without this the dtype
        # depends on whether a frame came from ingest or from disk -- and a
        # date/Timestamp mismatch silently breaks equality joins against the
        # trading calendar, which is the single most common source of quiet
        # wrongness in this codebase's history.
        frame[spec["partition_on"]] = pd.to_datetime(frame[spec["partition_on"]])
        frame["_part"] = _partition_key(frame[spec["partition_on"]])
        out_dir = self._dataset_dir(dataset)
        out_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for part, chunk in frame.groupby("_part", sort=True):
            chunk = chunk.drop(columns=["_part"])
            target = out_dir / f"part-{part}.parquet"
            if target.exists():
                chunk = pd.concat([pd.read_parquet(target), chunk], ignore_index=True)
            # keep="last" => the newly-written rows win over any existing copy
            chunk = chunk.drop_duplicates(subset=spec["keys"], keep="last")
            chunk = chunk.sort_values(spec["keys"]).reset_index(drop=True)

            tmp = out_dir / f".{target.name}.{uuid.uuid4().hex}.tmp"
            chunk.to_parquet(tmp, index=False)
            os.replace(tmp, target)  # atomic on Windows and POSIX
            written += len(chunk)
        return written

    def write_bhavcopy_eq(self, df: pd.DataFrame) -> int:
        return self.write_bulk("bhavcopy_eq", df)

    def write_delivery(self, df: pd.DataFrame) -> int:
        return self.write_bulk("bhavcopy_delivery", df)

    def write_intraday_bars(self, df: pd.DataFrame) -> int:
        return self.write_bulk("intraday_bars", df)

    # ------------------------------------------------------------ small tables
    def _upsert(self, table: str, df: pd.DataFrame, cols: list[str], keys: list[str]) -> int:
        if df.empty:
            return 0
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{table}: missing columns {missing}")
        payload = df[cols]
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c not in keys)
        collist = ", ".join(cols)
        self.con.register("_upsert_src", payload)
        try:
            self.con.execute(
                f"INSERT INTO {table} ({collist}) SELECT {collist} FROM _upsert_src "
                f"ON CONFLICT ({', '.join(keys)}) DO UPDATE SET {updates}"
            )
        finally:
            self.con.unregister("_upsert_src")
        return len(payload)

    CA_COLS = [
        "symbol", "ex_date", "action_type", "ratio_num", "ratio_den", "factor_price",
        "dividend_amount", "face_before", "face_after", "subject_raw", "parse_status",
    ]

    def write_corporate_actions(self, df: pd.DataFrame) -> int:
        return self._upsert(
            "corporate_actions", df, self.CA_COLS, ["symbol", "ex_date", "subject_raw"]
        )

    INGEST_COLS = [
        "run_id", "source", "unit", "started_at", "finished_at",
        "status", "rows_written", "attempt", "error",
    ]

    def record_ingest_run(self, row: dict[str, Any]) -> None:
        self.con.execute(
            f"INSERT OR REPLACE INTO ingest_runs ({', '.join(self.INGEST_COLS)}) "
            f"VALUES ({', '.join(['?'] * len(self.INGEST_COLS))})",
            [row.get(c) for c in self.INGEST_COLS],
        )

    def completed_units(self, source: str) -> set[str]:
        """Units already ingested successfully -- the basis of resumability.

        Status `empty` counts as done: a market holiday genuinely has no rows,
        and re-fetching it on every run is how a backfill never finishes.
        """
        rows = self.con.execute(
            "SELECT DISTINCT unit FROM ingest_runs WHERE source = ? AND status IN ('ok','empty')",
            [source],
        ).fetchall()
        return {r[0] for r in rows}

    # ----------------------------------------------------------------- publish
    def publish(self) -> list[str]:
        """Snapshot the small DuckDB tables to Parquet.

        After this, readers have exactly ONE access path for every dataset --
        Parquet -- and never need the DuckDB file. Call it at the end of any
        command that mutated a small table.
        """
        published = []
        for table in SMALL_TABLES:
            out_dir = self._dataset_dir(table)
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / "part-all.parquet"
            tmp = out_dir / f".{target.name}.{uuid.uuid4().hex}.tmp"
            self.con.execute(f"COPY (SELECT * FROM {table}) TO '{tmp.as_posix()}' (FORMAT PARQUET)")
            os.replace(tmp, target)
            published.append(table)
        return published

    # --------------------------------------------------------------- lifecycle
    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self.publish()
        finally:
            self.close()


class Reader:
    """Lock-free reads over the Parquet surface.

    Safe to open in any number of concurrent processes, including while an
    ingest is running and holding the DuckDB write lock -- verified with three
    concurrent readers against an active writer. Research workers, the API
    server and the UI all use this and never touch the DuckDB file.
    """

    def __init__(self, parquet_root: str | Path) -> None:
        self.parquet_root = Path(parquet_root)
        self.con = duckdb.connect()  # in-memory: no file, no lock

    def _glob(self, dataset: str) -> str:
        return (self.parquet_root / dataset / "*.parquet").as_posix()

    def exists(self, dataset: str) -> bool:
        d = self.parquet_root / dataset
        return d.is_dir() and any(d.glob("*.parquet"))

    def sql(self, query: str, params: list[Any] | None = None) -> pd.DataFrame:
        """Run arbitrary SQL. Use `{dataset}` placeholders for Parquet globs."""
        for name in list(BULK_SCHEMAS) + list(SMALL_TABLES):
            query = query.replace(f"{{{name}}}", f"read_parquet('{self._glob(name)}')")
        return self.con.execute(query, params or []).fetchdf()

    def bhavcopy_eq(
        self,
        symbols: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        series: str | None = "EQ",
    ) -> pd.DataFrame:
        if not self.exists("bhavcopy_eq"):
            return pd.DataFrame(columns=BULK_SCHEMAS["bhavcopy_eq"]["columns"])
        sql = "SELECT * FROM {bhavcopy_eq} WHERE 1=1"
        params: list[Any] = []
        if series:
            sql += " AND series = ?"
            params.append(series)
        if symbols:
            sql += f" AND symbol IN ({', '.join(['?'] * len(symbols))})"
            params += symbols
        if start:
            sql += " AND date >= ?"
            params.append(start)
        if end:
            sql += " AND date <= ?"
            params.append(end)
        return self.sql(sql + " ORDER BY symbol, date", params)

    def bhavcopy_bounds(self) -> tuple[date | None, date | None]:
        """First and last ingested trading date, as plain `date` objects.

        Deliberately converted from the stored datetime64: callers use these to
        build date ranges and compare against the trading calendar, and a
        Timestamp leaking out here is how a `date == Timestamp` comparison
        quietly evaluates False.
        """
        if not self.exists("bhavcopy_eq"):
            return (None, None)
        row = self.sql("SELECT min(date) AS lo, max(date) AS hi FROM {bhavcopy_eq}")
        if row.empty or pd.isna(row.lo.iloc[0]):
            return (None, None)
        return (pd.Timestamp(row.lo.iloc[0]).date(), pd.Timestamp(row.hi.iloc[0]).date())

    def corporate_actions(self, symbols: list[str] | None = None) -> pd.DataFrame:
        if not self.exists("corporate_actions"):
            return pd.DataFrame(columns=Store.CA_COLS)
        sql = "SELECT * FROM {corporate_actions}"
        params: list[Any] = []
        if symbols:
            sql += f" WHERE symbol IN ({', '.join(['?'] * len(symbols))})"
            params += symbols
        return self.sql(sql + " ORDER BY symbol, ex_date", params)

    def ingest_runs(self, limit: int = 200) -> pd.DataFrame:
        if not self.exists("ingest_runs"):
            return pd.DataFrame(columns=Store.INGEST_COLS)
        return self.sql(f"SELECT * FROM {{ingest_runs}} ORDER BY started_at DESC LIMIT {int(limit)}")

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Reader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def reset_store(duckdb_path: str | Path, parquet_root: str | Path) -> None:
    """Delete everything. Used by tests; never call this from application code."""
    p = Path(duckdb_path)
    if p.exists():
        p.unlink()
    r = Path(parquet_root)
    if r.exists():
        shutil.rmtree(r)
