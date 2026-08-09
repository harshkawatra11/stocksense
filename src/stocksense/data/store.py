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

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
