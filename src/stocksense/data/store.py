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

-- docs/12-statement-forensics.md: raw uploaded broker statement files,
-- content-hashed so re-uploading the same file is a no-op, not a
-- duplicate ingestion.
CREATE TABLE IF NOT EXISTS statements (
    statement_id   VARCHAR NOT NULL PRIMARY KEY,
    broker         VARCHAR NOT NULL,
    statement_type VARCHAR NOT NULL,  -- tradebook | tax_pnl
    file_path      VARCHAR NOT NULL,
    file_hash      VARCHAR NOT NULL,
    period_start   DATE,
    period_end     DATE,
    ingested_at    TIMESTAMP NOT NULL,
    row_count      INTEGER,
    parse_status   VARCHAR NOT NULL  -- ok | partial | failed
);

-- Canonical cross-broker trade schema. source_row_json preserves the raw
-- parsed row so a parsing bug is always forensically recoverable, the
-- same discipline that found the ADANIENT adjustment-factor bug.
CREATE TABLE IF NOT EXISTS trades (
    trade_id       VARCHAR NOT NULL PRIMARY KEY,
    statement_id   VARCHAR NOT NULL,
    broker         VARCHAR NOT NULL,
    symbol         VARCHAR NOT NULL,
    isin           VARCHAR,
    segment        VARCHAR NOT NULL,  -- equity_delivery | equity_intraday | fno | currency | commodity
    trade_date     DATE NOT NULL,
    trade_time     VARCHAR,
    side           VARCHAR NOT NULL,  -- buy | sell
    quantity       DOUBLE NOT NULL,
    price          DOUBLE NOT NULL,
    value          DOUBLE NOT NULL,
    order_id       VARCHAR,
    exchange       VARCHAR,
    product_type   VARCHAR,           -- MIS | CNC | NRML
    source_row_json VARCHAR
);

-- Exact Indian charge breakdown per trade (stocksense.execution.cost_model.compute_charges).
CREATE TABLE IF NOT EXISTS trade_charges (
    trade_id       VARCHAR NOT NULL PRIMARY KEY,
    brokerage      DOUBLE NOT NULL,
    stt            DOUBLE NOT NULL,
    exchange_txn   DOUBLE NOT NULL,
    sebi_fee       DOUBLE NOT NULL,
    stamp_duty     DOUBLE NOT NULL,
    gst            DOUBLE NOT NULL,
    total_charges  DOUBLE NOT NULL,
    cost_bps       DOUBLE NOT NULL
);

-- FIFO-matched buy->sell round trips, the unit of behavioral analysis.
CREATE TABLE IF NOT EXISTS positions (
    position_id    VARCHAR NOT NULL PRIMARY KEY,
    symbol         VARCHAR NOT NULL,
    segment        VARCHAR NOT NULL,
    open_date      DATE NOT NULL,
    open_time      VARCHAR,
    close_date     DATE NOT NULL,
    close_time     VARCHAR,
    quantity       DOUBLE NOT NULL,
    entry_price    DOUBLE NOT NULL,
    exit_price     DOUBLE NOT NULL,
    gross_pnl      DOUBLE NOT NULL,
    charges        DOUBLE NOT NULL,
    net_pnl        DOUBLE NOT NULL,
    holding_seconds INTEGER,
    is_intraday    BOOLEAN NOT NULL,
    mae            DOUBLE,  -- max adverse excursion, requires intraday candles
    mfe            DOUBLE   -- max favorable excursion, requires intraday candles
);

-- One row per (run, metric, cohort). Severity thresholds are
-- pre-registered in docs/12-statement-forensics.md before being run
-- against real data, the same discipline that fixed the gate.
CREATE TABLE IF NOT EXISTS diagnostics (
    run_id         VARCHAR NOT NULL,
    as_of          DATE NOT NULL,
    metric_name    VARCHAR NOT NULL,
    metric_value   DOUBLE,
    metric_unit    VARCHAR,
    severity       VARCHAR,  -- ok | notable | high | critical
    cohort         VARCHAR NOT NULL DEFAULT 'all',
    detail_json    VARCHAR,
    PRIMARY KEY (run_id, metric_name, cohort)
);

-- "What if" replays of actual fills under modified rules. Arithmetic on
-- history, explicitly not predictions.
CREATE TABLE IF NOT EXISTS counterfactuals (
    run_id            VARCHAR NOT NULL,
    scenario_name     VARCHAR NOT NULL,
    actual_pnl        DOUBLE NOT NULL,
    scenario_pnl      DOUBLE NOT NULL,
    delta_pnl         DOUBLE NOT NULL,
    n_trades_affected INTEGER NOT NULL,
    detail_json       VARCHAR,
    PRIMARY KEY (run_id, scenario_name)
);

-- RAG corpus (docs/14-rag.md). embedding is nullable so FTS-only degraded
-- mode (no Ollama) still works — see rag/index.py.
CREATE TABLE IF NOT EXISTS rag_documents (
    doc_id         VARCHAR NOT NULL PRIMARY KEY,
    source_type    VARCHAR NOT NULL,
    source_ref     VARCHAR,
    title          VARCHAR,
    content        VARCHAR NOT NULL,
    content_hash   VARCHAR NOT NULL,
    indexed_at     TIMESTAMP NOT NULL,
    metadata_json  VARCHAR
);

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id       VARCHAR NOT NULL PRIMARY KEY,
    doc_id         VARCHAR NOT NULL,
    chunk_index    INTEGER NOT NULL,
    content        VARCHAR NOT NULL,
    token_count    INTEGER,
    embedding      FLOAT[768]
);

-- Every Claude CLI invocation, in and out. The audit trail that makes
-- "the agent never invents a number" a checkable claim, not a promise.
CREATE TABLE IF NOT EXISTS agent_runs (
    agent_run_id   VARCHAR NOT NULL PRIMARY KEY,
    job_run_id     VARCHAR,
    skill_name     VARCHAR,
    prompt_hash    VARCHAR NOT NULL,
    input_json     VARCHAR NOT NULL,
    output_text    VARCHAR,
    model          VARCHAR,
    started_at     TIMESTAMP NOT NULL,
    finished_at    TIMESTAMP,
    status         VARCHAR NOT NULL,  -- ok | unverified_numbers | error | timeout
    error          VARCHAR,
    cost_estimate  DOUBLE
);

-- docs/19-foreman.md: the self-building harness's goal queue. source
-- distinguishes goals the user asked for from ones the Foreman proposed
-- to itself, so the ledger can answer "did I ask for this."
CREATE TABLE IF NOT EXISTS goals (
    goal_id          VARCHAR NOT NULL PRIMARY KEY,
    source           VARCHAR NOT NULL,  -- user | self_assess | adversary | maintenance
    prompt           VARCHAR NOT NULL,
    status           VARCHAR NOT NULL,  -- queued|planning|executing|verifying|blocked|done|abandoned
    priority         INTEGER NOT NULL DEFAULT 5,
    created_at       TIMESTAMP NOT NULL,
    completed_at     TIMESTAMP,
    parent_goal_id   VARCHAR,
    result_summary   VARCHAR
);

-- One row per tool invocation within a goal. This is what makes "what
-- did the Foreman actually do" answerable from the database rather than
-- reconstructed from scrollback -- the same principle as job_runs.
CREATE TABLE IF NOT EXISTS build_ledger (
    entry_id         VARCHAR NOT NULL PRIMARY KEY,
    goal_id          VARCHAR NOT NULL,
    task_name        VARCHAR NOT NULL,
    tool             VARCHAR NOT NULL,
    action           VARCHAR NOT NULL,   -- read | write | exec | network
    diff_summary     VARCHAR,
    files_touched    VARCHAR,            -- JSON list
    verdict          VARCHAR,            -- ok | failed | blocked_protected | adversary_rejected
    ci_run_url       VARCHAR,
    attempts         INTEGER NOT NULL DEFAULT 1,
    tokens_estimate  DOUBLE,
    created_at       TIMESTAMP NOT NULL
);

-- Every attempt to write a protected path, whether blocked (the normal
-- case) or -- if this table is ever inconsistent with reality -- not.
-- The point of this table is that a pattern of the agent repeatedly
-- reaching for evaluation/gate.py is VISIBLE, not silent.
CREATE TABLE IF NOT EXISTS protected_violations (
    violation_id     VARCHAR NOT NULL PRIMARY KEY,
    goal_id          VARCHAR NOT NULL,
    path             VARCHAR NOT NULL,
    attempted_at     TIMESTAMP NOT NULL,
    action_taken     VARCHAR NOT NULL  -- blocked_routed_to_pr
);

-- docs/17-data-spine.md: point-in-time NSE archive data. Distinct from
-- `candles` (yfinance, adjusted, current-universe-only) -- these are
-- RAW bhavcopy rows, the source that makes a genuine point-in-time
-- universe possible (see data/universe_pit.py).
CREATE TABLE IF NOT EXISTS bhavcopy_eq (
    symbol        VARCHAR NOT NULL,
    series        VARCHAR NOT NULL,
    date          DATE NOT NULL,
    open          DOUBLE,
    high          DOUBLE,
    low           DOUBLE,
    close         DOUBLE,
    prev_close    DOUBLE,
    volume        DOUBLE,
    turnover_inr  DOUBLE,
    era           VARCHAR NOT NULL,
    PRIMARY KEY (symbol, series, date)
);

CREATE TABLE IF NOT EXISTS bhavcopy_delivery (
    symbol        VARCHAR NOT NULL,
    series        VARCHAR NOT NULL,
    date          DATE NOT NULL,
    delivery_qty  DOUBLE,
    delivery_pct  DOUBLE,
    PRIMARY KEY (symbol, series, date)
);

CREATE TABLE IF NOT EXISTS bhavcopy_fo (
    symbol         VARCHAR NOT NULL,
    instrument     VARCHAR NOT NULL,
    expiry_date    VARCHAR NOT NULL,
    strike         DOUBLE NOT NULL,
    option_type    VARCHAR NOT NULL,
    date           DATE NOT NULL,
    open           DOUBLE,
    high           DOUBLE,
    low            DOUBLE,
    close          DOUBLE,
    open_interest  DOUBLE,
    chg_in_oi      DOUBLE,
    era            VARCHAR NOT NULL,
    PRIMARY KEY (symbol, instrument, expiry_date, strike, option_type, date)
);

-- Daily spend/activity cap enforcement.
CREATE TABLE IF NOT EXISTS budget (
    period_date      DATE NOT NULL PRIMARY KEY,
    invocations      INTEGER NOT NULL DEFAULT 0,
    tokens_estimate  DOUBLE NOT NULL DEFAULT 0,
    goals_completed  INTEGER NOT NULL DEFAULT 0
);
"""

# AUDIT: predictions was created but never written to (CRITICAL-1). These
# columns are added defensively via _MIGRATIONS below rather than baked
# into predictions' CREATE TABLE, so a database created before this
# change migrates in place instead of needing to be rebuilt.
_PREDICTIONS_MIGRATION_COLUMNS = {
    "horizon_type": "VARCHAR",           # 'monthly' | 'intraday'
    "predicted_return": "DOUBLE",
    "confidence": "DOUBLE",
    "feature_snapshot_hash": "VARCHAR",
    "graded_at": "TIMESTAMP",
    "actual_return": "DOUBLE",
    "grade_json": "VARCHAR",
}


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
        self._migrate_predictions_columns()

    def _migrate_predictions_columns(self) -> None:
        """Defensive ALTER for columns added after predictions' original
        CREATE TABLE, so a database created before this change gains them
        in place instead of needing to be rebuilt (see
        _PREDICTIONS_MIGRATION_COLUMNS above predictions' schema)."""
        existing = {
            row[1] for row in self.con.execute("PRAGMA table_info('predictions')").fetchall()
        }
        for col, col_type in _PREDICTIONS_MIGRATION_COLUMNS.items():
            if col not in existing:
                self.con.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")

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

    # ---- predictions: the reconcile loop (CRITICAL-1) writes here ----

    def write_predictions(self, df: pd.DataFrame) -> int:
        cols = [
            "run_id", "symbol", "as_of_date", "horizon_bars", "score", "rank", "model_version",
            "horizon_type", "predicted_return", "confidence", "feature_snapshot_hash",
        ]
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"write_predictions missing columns: {missing}")
        self.con.register("_pred", df[cols])
        self.con.execute(
            f"""
            INSERT INTO predictions ({", ".join(cols)})
            SELECT {", ".join(cols)} FROM _pred
            ON CONFLICT (run_id, symbol, as_of_date, horizon_bars) DO UPDATE SET
                score = excluded.score, rank = excluded.rank, model_version = excluded.model_version,
                horizon_type = excluded.horizon_type, predicted_return = excluded.predicted_return,
                confidence = excluded.confidence, feature_snapshot_hash = excluded.feature_snapshot_hash
            """
        )
        self.con.unregister("_pred")
        return len(df)

    def read_ungraded_predictions(self, as_of_before) -> pd.DataFrame:
        """Predictions whose horizon has matured (as_of_date + horizon_bars
        trading days <= as_of_before) but which have not been graded yet."""
        return self.con.execute(
            "SELECT * FROM predictions WHERE graded_at IS NULL AND as_of_date <= ? ORDER BY as_of_date",
            [as_of_before],
        ).fetchdf()

    def grade_prediction(self, run_id: str, symbol: str, as_of_date, horizon_bars: int,
                          actual_return: float, grade_json: str, graded_at) -> None:
        self.con.execute(
            """
            UPDATE predictions SET actual_return = ?, grade_json = ?, graded_at = ?
            WHERE run_id = ? AND symbol = ? AND as_of_date = ? AND horizon_bars = ?
            """,
            [actual_return, grade_json, graded_at, run_id, symbol, as_of_date, horizon_bars],
        )

    def read_predictions(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM predictions ORDER BY as_of_date, symbol").fetchdf()

    # ---- statement forensics (docs/12) ----

    def insert_statement(self, row: dict) -> None:
        cols = list(row.keys())
        self.con.execute(
            f"INSERT INTO statements ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
            [row[c] for c in cols],
        )

    def find_statement_by_hash(self, file_hash: str) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM statements WHERE file_hash = ?", [file_hash]).fetchdf()

    def write_trades(self, df: pd.DataFrame) -> int:
        # AUDIT FIX: "SELECT *" binds by column POSITION, not name — a
        # DataFrame whose column order doesn't exactly match the trades
        # DDL (e.g. statement_id appended after parsing rather than
        # inserted in schema order) silently misaligns columns instead of
        # erroring, which is how trade_time once landed in trade_date's
        # slot. Select explicitly by name instead.
        cols = [
            "trade_id", "statement_id", "broker", "symbol", "isin", "segment",
            "trade_date", "trade_time", "side", "quantity", "price", "value",
            "order_id", "exchange", "product_type", "source_row_json",
        ]
        self.con.register("_trades", df[cols])
        self.con.execute(f"INSERT INTO trades ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _trades")
        self.con.unregister("_trades")
        return len(df)

    def read_trades(self, broker: str | None = None) -> pd.DataFrame:
        if broker:
            return self.con.execute("SELECT * FROM trades WHERE broker = ? ORDER BY trade_date, trade_time", [broker]).fetchdf()
        return self.con.execute("SELECT * FROM trades ORDER BY trade_date, trade_time").fetchdf()

    def write_trade_charges(self, df: pd.DataFrame) -> int:
        # AUDIT FIX: bind by explicit column list, not "SELECT *" — see
        # write_trades' comment for the exact failure mode this avoids.
        cols = ["trade_id", "brokerage", "stt", "exchange_txn", "sebi_fee", "stamp_duty", "gst", "total_charges", "cost_bps"]
        self.con.register("_charges", df[cols])
        self.con.execute(f"INSERT INTO trade_charges ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _charges")
        self.con.unregister("_charges")
        return len(df)

    def write_positions(self, df: pd.DataFrame) -> int:
        cols = [
            "position_id", "symbol", "segment", "open_date", "open_time", "close_date", "close_time",
            "quantity", "entry_price", "exit_price", "gross_pnl", "charges", "net_pnl",
            "holding_seconds", "is_intraday", "mae", "mfe",
        ]
        self.con.register("_positions", df[cols])
        self.con.execute(f"INSERT INTO positions ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _positions")
        self.con.unregister("_positions")
        return len(df)

    def read_positions(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM positions ORDER BY open_date, open_time").fetchdf()

    def write_diagnostics(self, df: pd.DataFrame) -> int:
        cols = ["run_id", "as_of", "metric_name", "metric_value", "metric_unit", "severity", "cohort", "detail_json"]
        self.con.register("_diag", df[cols])
        self.con.execute(
            f"""
            INSERT INTO diagnostics ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _diag
            ON CONFLICT (run_id, metric_name, cohort) DO UPDATE SET
                metric_value = excluded.metric_value, severity = excluded.severity, detail_json = excluded.detail_json
            """
        )
        self.con.unregister("_diag")
        return len(df)

    def read_diagnostics(self, run_id: str) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM diagnostics WHERE run_id = ?", [run_id]).fetchdf()

    def read_latest_diagnostics(self) -> pd.DataFrame:
        """Diagnostics from the most recent Kundli run (by as_of), for
        callers (the desktop dashboard) that want 'the current picture'
        without already knowing a specific run_id."""
        latest = self.con.execute("SELECT run_id FROM diagnostics ORDER BY as_of DESC LIMIT 1").fetchone()
        if latest is None:
            return pd.DataFrame()
        return self.read_diagnostics(latest[0])

    def write_counterfactuals(self, df: pd.DataFrame) -> int:
        cols = ["run_id", "scenario_name", "actual_pnl", "scenario_pnl", "delta_pnl", "n_trades_affected", "detail_json"]
        self.con.register("_cf", df[cols])
        self.con.execute(
            f"""
            INSERT INTO counterfactuals ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _cf
            ON CONFLICT (run_id, scenario_name) DO UPDATE SET
                actual_pnl = excluded.actual_pnl, scenario_pnl = excluded.scenario_pnl,
                delta_pnl = excluded.delta_pnl, n_trades_affected = excluded.n_trades_affected,
                detail_json = excluded.detail_json
            """
        )
        self.con.unregister("_cf")
        return len(df)

    def read_counterfactuals(self, run_id: str) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM counterfactuals WHERE run_id = ?", [run_id]).fetchdf()

    # ---- agent audit trail (docs/11) ----

    def insert_agent_run(self, row: dict) -> None:
        cols = list(row.keys())
        self.con.execute(
            f"INSERT INTO agent_runs ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
            [row[c] for c in cols],
        )

    def read_agent_runs(self, limit: int = 50) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT ?", [limit]).fetchdf()

    # ---- Foreman: goals, ledger, protected-path violations, budget ----

    def insert_goal(self, row: dict) -> None:
        cols = list(row.keys())
        self.con.execute(
            f"INSERT INTO goals ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
            [row[c] for c in cols],
        )

    def update_goal_status(self, goal_id: str, status: str, completed_at=None, result_summary: str | None = None) -> None:
        self.con.execute(
            "UPDATE goals SET status = ?, completed_at = COALESCE(?, completed_at), "
            "result_summary = COALESCE(?, result_summary) WHERE goal_id = ?",
            [status, completed_at, result_summary, goal_id],
        )

    def read_goals(self, status: str | None = None) -> pd.DataFrame:
        if status:
            return self.con.execute("SELECT * FROM goals WHERE status = ? ORDER BY priority, created_at", [status]).fetchdf()
        return self.con.execute("SELECT * FROM goals ORDER BY created_at DESC").fetchdf()

    def insert_ledger_entry(self, row: dict) -> None:
        cols = list(row.keys())
        self.con.execute(
            f"INSERT INTO build_ledger ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
            [row[c] for c in cols],
        )

    def read_ledger(self, goal_id: str | None = None) -> pd.DataFrame:
        if goal_id:
            return self.con.execute("SELECT * FROM build_ledger WHERE goal_id = ? ORDER BY created_at", [goal_id]).fetchdf()
        return self.con.execute("SELECT * FROM build_ledger ORDER BY created_at DESC").fetchdf()

    def insert_protected_violation(self, row: dict) -> None:
        cols = list(row.keys())
        self.con.execute(
            f"INSERT INTO protected_violations ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
            [row[c] for c in cols],
        )

    def read_protected_violations(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM protected_violations ORDER BY attempted_at DESC").fetchdf()

    def get_or_create_budget(self, period_date) -> dict:
        row = self.con.execute("SELECT * FROM budget WHERE period_date = ?", [period_date]).fetchdf()
        if row.empty:
            self.con.execute(
                "INSERT INTO budget (period_date, invocations, tokens_estimate, goals_completed) VALUES (?, 0, 0, 0)",
                [period_date],
            )
            return {"period_date": period_date, "invocations": 0, "tokens_estimate": 0.0, "goals_completed": 0}
        return row.iloc[0].to_dict()

    def increment_budget(self, period_date, invocations: int = 0, tokens_estimate: float = 0.0, goals_completed: int = 0) -> None:
        self.get_or_create_budget(period_date)
        self.con.execute(
            "UPDATE budget SET invocations = invocations + ?, tokens_estimate = tokens_estimate + ?, "
            "goals_completed = goals_completed + ? WHERE period_date = ?",
            [invocations, tokens_estimate, goals_completed, period_date],
        )

    # ---- point-in-time NSE archive data (docs/17-data-spine.md) ----

    def write_bhavcopy_eq(self, df: pd.DataFrame) -> int:
        cols = ["symbol", "series", "date", "open", "high", "low", "close", "prev_close", "volume", "turnover_inr", "era"]
        self.con.register("_bc_eq", df[cols])
        self.con.execute(
            f"""
            INSERT INTO bhavcopy_eq ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _bc_eq
            ON CONFLICT (symbol, series, date) DO UPDATE SET
                open = excluded.open, high = excluded.high, low = excluded.low, close = excluded.close,
                prev_close = excluded.prev_close, volume = excluded.volume,
                turnover_inr = excluded.turnover_inr, era = excluded.era
            """
        )
        self.con.unregister("_bc_eq")
        return len(df)

    def read_bhavcopy_eq(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM bhavcopy_eq"
        params = []
        if start and end:
            query += " WHERE date BETWEEN ? AND ?"
            params = [start, end]
        return self.con.execute(query + " ORDER BY date, symbol", params).fetchdf()

    def write_bhavcopy_delivery(self, df: pd.DataFrame) -> int:
        cols = ["symbol", "series", "date", "delivery_qty", "delivery_pct"]
        self.con.register("_bc_deliv", df[cols])
        self.con.execute(
            f"""
            INSERT INTO bhavcopy_delivery ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _bc_deliv
            ON CONFLICT (symbol, series, date) DO UPDATE SET
                delivery_qty = excluded.delivery_qty, delivery_pct = excluded.delivery_pct
            """
        )
        self.con.unregister("_bc_deliv")
        return len(df)

    def read_bhavcopy_delivery(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM bhavcopy_delivery"
        params = []
        if start and end:
            query += " WHERE date BETWEEN ? AND ?"
            params = [start, end]
        return self.con.execute(query + " ORDER BY date, symbol", params).fetchdf()

    def write_bhavcopy_fo(self, df: pd.DataFrame) -> int:
        cols = ["symbol", "instrument", "expiry_date", "strike", "option_type", "date",
                "open", "high", "low", "close", "open_interest", "chg_in_oi", "era"]
        self.con.register("_bc_fo", df[cols])
        self.con.execute(
            f"""
            INSERT INTO bhavcopy_fo ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _bc_fo
            ON CONFLICT (symbol, instrument, expiry_date, strike, option_type, date) DO UPDATE SET
                open = excluded.open, high = excluded.high, low = excluded.low, close = excluded.close,
                open_interest = excluded.open_interest, chg_in_oi = excluded.chg_in_oi, era = excluded.era
            """
        )
        self.con.unregister("_bc_fo")
        return len(df)

    def read_bhavcopy_fo(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM bhavcopy_fo"
        params = []
        if start and end:
            query += " WHERE date BETWEEN ? AND ?"
            params = [start, end]
        return self.con.execute(query + " ORDER BY date, symbol", params).fetchdf()

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
