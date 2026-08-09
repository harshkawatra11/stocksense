"""
DuckDB store. Embedded, single-writer, per docs/02-data-layer.md's choice
rationale — no server to install or supervise.

All writes are idempotent upserts keyed on (symbol, date, source): running
ingestion twice for the same date must not duplicate rows
(docs/05-nightly-pipeline.md's idempotency requirement).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol      VARCHAR NOT NULL,
    date        DATE    NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    adj_close   DOUBLE,
    volume      DOUBLE,
    source      VARCHAR NOT NULL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS universe (
    symbol       VARCHAR NOT NULL PRIMARY KEY,
    name         VARCHAR,
    first_seen   DATE,
    last_seen    DATE,
    is_active    BOOLEAN,
    delisted     BOOLEAN
);

CREATE TABLE IF NOT EXISTS predictions (
    run_id        VARCHAR NOT NULL,
    symbol        VARCHAR NOT NULL,
    as_of_date    DATE NOT NULL,
    horizon_bars  INTEGER NOT NULL,
    score         DOUBLE,
    rank          INTEGER,
    model_version VARCHAR,
    PRIMARY KEY (run_id, symbol, as_of_date, horizon_bars)
);

-- docs/02-data-layer.md "model_registry": one row per trained model
-- version. The Gate (stocksense.models.gate) is the only writer.
CREATE TABLE IF NOT EXISTS model_registry (
    model_id            VARCHAR NOT NULL PRIMARY KEY,
    model_type          VARCHAR NOT NULL,   -- e.g. 'cross_sectional_ranker'
    horizon_bars        INTEGER NOT NULL,
    top_n                INTEGER,
    feature_schema_version VARCHAR NOT NULL,
    training_start       DATE,
    training_end          DATE,
    hyperparameters_json  VARCHAR,
    random_seed           INTEGER,
    created_at             TIMESTAMP NOT NULL,
    metrics_json           VARCHAR,          -- aggregate + per-fold + per-regime, per docs/06
    gate_decision           VARCHAR,          -- 'promote' | 'reject'
    gate_reason              VARCHAR,
    lifecycle_state           VARCHAR NOT NULL, -- candidate | shadow | live | archived | rolled_back
    artifact_path              VARCHAR NOT NULL, -- path to the serialized model file
    promoted_at                 TIMESTAMP,
    rolled_back_at                TIMESTAMP
);

-- docs/02-data-layer.md "job_runs": the heartbeat. Always written, even
-- on hard failure — this is how "did last night actually run?" gets
-- answered (docs/08-operations.md).
CREATE TABLE IF NOT EXISTS job_runs (
    run_id        VARCHAR NOT NULL PRIMARY KEY,
    job_name       VARCHAR NOT NULL,
    started_at      TIMESTAMP NOT NULL,
    finished_at      TIMESTAMP,
    status            VARCHAR NOT NULL,  -- completed | failed | aborted | interrupted
    detail_json        VARCHAR
);
"""


class Store:
    """Thin wrapper around a DuckDB file. One instance = one connection.

    DuckDB permits a single writer (docs/02-data-layer.md's accepted
    trade-off); this class does not attempt to hide that, callers should
    not open concurrent writable connections to the same file.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.con = duckdb.connect(str(path))
        self.con.execute(SCHEMA)

    def upsert_candles(self, df: pd.DataFrame) -> int:
        """Upsert OHLCV rows keyed on (symbol, date, source).

        Expects columns: symbol, date, open, high, low, close, adj_close,
        volume, source. Returns the number of rows upserted.
        """
        required = {"symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"upsert_candles missing columns: {missing}")

        self.con.register("_incoming", df[list(required)])
        self.con.execute(
            """
            INSERT INTO candles
            SELECT symbol, date, open, high, low, close, adj_close, volume, source
            FROM _incoming
            ON CONFLICT (symbol, date) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                adj_close = excluded.adj_close,
                volume = excluded.volume,
                source = excluded.source
            """
        )
        self.con.unregister("_incoming")
        return len(df)

    def read_candles(self, symbols: list[str] | None = None) -> pd.DataFrame:
        if symbols:
            placeholders = ",".join(["?"] * len(symbols))
            query = f"SELECT * FROM candles WHERE symbol IN ({placeholders}) ORDER BY symbol, date"
            return self.con.execute(query, symbols).fetchdf()
        return self.con.execute("SELECT * FROM candles ORDER BY symbol, date").fetchdf()

    def upsert_universe(self, df: pd.DataFrame) -> int:
        required = {"symbol", "name", "first_seen", "last_seen", "is_active", "delisted"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"upsert_universe missing columns: {missing}")
        self.con.register("_uni", df[list(required)])
        self.con.execute(
            """
            INSERT INTO universe
            SELECT symbol, name, first_seen, last_seen, is_active, delisted FROM _uni
            ON CONFLICT (symbol) DO UPDATE SET
                name = excluded.name,
                first_seen = excluded.first_seen,
                last_seen = excluded.last_seen,
                is_active = excluded.is_active,
                delisted = excluded.delisted
            """
        )
        self.con.unregister("_uni")
        return len(df)

    # ---- model registry: the Gate is the only writer (docs/01-architecture.md) ----

    def insert_model_registry_row(self, row: dict) -> None:
        cols = list(row.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_list = ", ".join(cols)
        self.con.execute(
            f"INSERT INTO model_registry ({col_list}) VALUES ({placeholders})",
            [row[c] for c in cols],
        )

    def update_model_lifecycle(
        self, model_id: str, lifecycle_state: str, promoted_at=None, rolled_back_at=None
    ) -> None:
        self.con.execute(
            """
            UPDATE model_registry
            SET lifecycle_state = ?,
                promoted_at = COALESCE(?, promoted_at),
                rolled_back_at = COALESCE(?, rolled_back_at)
            WHERE model_id = ?
            """,
            [lifecycle_state, promoted_at, rolled_back_at, model_id],
        )

    def get_live_model(self, model_type: str, horizon_bars: int) -> pd.DataFrame:
        return self.con.execute(
            """
            SELECT * FROM model_registry
            WHERE model_type = ? AND horizon_bars = ? AND lifecycle_state = 'live'
            ORDER BY promoted_at DESC LIMIT 1
            """,
            [model_type, horizon_bars],
        ).fetchdf()

    def read_model_registry(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM model_registry ORDER BY created_at").fetchdf()

    # ---- job heartbeat (docs/05-nightly-pipeline.md step 15, always runs) ----

    def start_job_run(self, run_id: str, job_name: str, started_at) -> None:
        self.con.execute(
            "INSERT INTO job_runs (run_id, job_name, started_at, status) VALUES (?, ?, ?, 'running')",
            [run_id, job_name, started_at],
        )

    def finish_job_run(self, run_id: str, status: str, finished_at, detail_json: str | None = None) -> None:
        self.con.execute(
            "UPDATE job_runs SET status = ?, finished_at = ?, detail_json = ? WHERE run_id = ?",
            [status, finished_at, detail_json, run_id],
        )

    def read_job_runs(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM job_runs ORDER BY started_at DESC").fetchdf()

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
