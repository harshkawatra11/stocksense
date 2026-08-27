"""
DuckDB store. Embedded, single-writer, per docs/02-data-layer.md's choice
rationale — no server to install or supervise.

All writes are idempotent upserts keyed on (symbol, date, source): running
ingestion twice for the same date must not duplicate rows
(docs/05-nightly-pipeline.md's idempotency requirement).
"""

from __future__ import annotations

import time
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

-- Corporate actions (splits/bonuses/dividends), parsed from NSE's
-- corporateActions API (data/corporate_actions.py). bhavcopy_eq carries
-- NO adjustment for these -- verified directly: prev_close equals the
-- prior raw close across confirmed splits (LICI, ECLERX, PASHUPATI).
-- This table is what data/adjust.py backs out an adjusted price series
-- from; without it, raw bhavcopy prices contain fake -50% to -90%
-- one-day "returns" on every split/bonus date.
CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol         VARCHAR NOT NULL,
    ex_date        DATE NOT NULL,
    action_type    VARCHAR NOT NULL,  -- split | bonus | dividend | ignore | unparsed
    ratio_num      DOUBLE,
    ratio_den      DOUBLE,
    factor_price   DOUBLE NOT NULL,   -- price-return adjustment factor (splits/bonuses only)
    dividend_amount DOUBLE,           -- Rs per share; total-return factor computed at apply time (needs ex-date price)
    face_before    DOUBLE,
    face_after     DOUBLE,
    subject_raw    VARCHAR NOT NULL,
    parse_status   VARCHAR NOT NULL,  -- ok | unparsed
    PRIMARY KEY (symbol, ex_date, subject_raw)
);

-- Phase E1: 1-minute intraday bars from Upstox. `interval` is stored
-- explicitly (not assumed '1minute') so a future coarser-grain fetch can
-- share this table without a migration. ts is IST wall-clock, matching
-- what Upstox returns and what a trader reasons about directly -- no UTC
-- conversion, since this project only ever trades one exchange/timezone.
CREATE TABLE IF NOT EXISTS intraday_bars (
    symbol      VARCHAR NOT NULL,
    ts          TIMESTAMP NOT NULL,
    interval    VARCHAR NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    volume      DOUBLE,
    PRIMARY KEY (symbol, ts, interval)
);

-- Phase E1: symbol -> Upstox instrument_key resolution, cached so the
-- ~2,600-row instrument master doesn't need re-fetching every run and so
-- unmapped symbols are a queryable, auditable fact rather than a log line.
CREATE TABLE IF NOT EXISTS upstox_instrument_map (
    symbol          VARCHAR NOT NULL,
    isin            VARCHAR,
    instrument_key  VARCHAR,
    resolved        BOOLEAN NOT NULL,  -- false if no Upstox instrument matched this symbol
    PRIMARY KEY (symbol)
);

-- Phase F1: desktop control-center job tracking. Distinct from job_runs
-- (the harness graph runner's node-level table) -- this tracks a whole
-- CLI subprocess triggered from the UI (a backfill, foreman run, etc.),
-- durable across a server restart so a still-running multi-hour job
-- isn't silently forgotten. Written only at job START and FINISH (never
-- polled/updated mid-run) -- live progress while running comes from the
-- in-process JobRegistry's in-memory buffer, not from this table, since
-- the job's own subprocess holds DuckDB's single-writer lock for its
-- entire duration and a mid-run UPDATE from the server process would
-- itself block on that lock.
CREATE TABLE IF NOT EXISTS ui_jobs (
    job_id       VARCHAR NOT NULL PRIMARY KEY,
    command      VARCHAR NOT NULL,
    args_json    VARCHAR NOT NULL,
    pid          INTEGER,
    status       VARCHAR NOT NULL,  -- running | completed | failed | stopped
    started_at   TIMESTAMP NOT NULL,
    finished_at  TIMESTAMP,
    log_path     VARCHAR
);

-- Phase F2/F4: local app settings + the Claude-CLI access authorize/
-- decline flag. Key-value on purpose (not a Settings-mirroring typed
-- table) -- new keys (risk thresholds in E5, per-role model/effort in
-- F4) get added without a migration.
CREATE TABLE IF NOT EXISTS app_settings (
    key          VARCHAR NOT NULL PRIMARY KEY,
    value        VARCHAR,
    updated_at   TIMESTAMP NOT NULL
);

-- Phase F3: measured (not official) Claude usage, aggregated from the
-- CLI's own local session transcripts (~/.claude/projects/**/*.jsonl).
-- claude_usage_offsets is what makes re-scanning incremental -- a byte
-- offset per file, never re-parsing the 600MB+ of history that already
-- accumulated before this feature existed.
CREATE TABLE IF NOT EXISTS claude_usage_offsets (
    file_path    VARCHAR NOT NULL PRIMARY KEY,
    byte_offset  BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS claude_usage_events (
    event_id               VARCHAR NOT NULL PRIMARY KEY,
    ts                     TIMESTAMP NOT NULL,
    model                  VARCHAR,
    input_tokens           BIGINT NOT NULL DEFAULT 0,
    output_tokens          BIGINT NOT NULL DEFAULT 0,
    cache_creation_tokens  BIGINT NOT NULL DEFAULT 0,
    cache_read_tokens      BIGINT NOT NULL DEFAULT 0
);

-- Phase J2: paper trading. Deliberately a UNIT book, not a rupee one --
-- no capital-denominated column exists anywhere below. This preserves
-- the invariant Phase G established (docs/STATUS.md: "no capital, real
-- or paper, has been committed... no account size lives in config, a
-- model, the gate, or the prediction ledger") and matches
-- research/verdict_intraday.md's own precedent: EXPOSURE_INR lives only
-- inside a retired, non-live research script, never in a table a live
-- system reads. Whole-share divisibility is answered on demand via
-- optimizer.sizing.min_capital_for_full_positions, computed the same
-- way /api/brief already does -- never stored.
CREATE TABLE IF NOT EXISTS paper_accounts (
    account_id         VARCHAR NOT NULL PRIMARY KEY,
    name               VARCHAR NOT NULL,
    model_id           VARCHAR NOT NULL,
    model_type         VARCHAR NOT NULL,
    horizon_bars       INTEGER NOT NULL,
    top_n              INTEGER NOT NULL,
    cap_band           VARCHAR,
    fill_rule          VARCHAR NOT NULL,  -- 'rebalance_date_close' today; a future validated
                                           -- entry-timing rule (Phase J4a) is a NEW value here,
                                           -- never a silent redefinition of an existing one.
    no_trade_band      DOUBLE NOT NULL DEFAULT 0.02,
    created_at         TIMESTAMP NOT NULL,
    closed_at          TIMESTAMP,
    status             VARCHAR NOT NULL,  -- 'active' | 'closed'
    notes              VARCHAR
);

CREATE TABLE IF NOT EXISTS paper_orders (
    order_id           VARCHAR NOT NULL PRIMARY KEY,
    account_id         VARCHAR NOT NULL,
    rebalance_date     DATE NOT NULL,
    symbol             VARCHAR NOT NULL,
    action             VARCHAR NOT NULL,  -- optimizer.rebalance.RebalanceAction.action
    current_weight     DOUBLE NOT NULL,
    target_weight      DOUBLE NOT NULL,
    weight_delta       DOUBLE NOT NULL,
    fill_rule          VARCHAR NOT NULL,
    fill_price         DOUBLE,
    fill_status        VARCHAR NOT NULL,  -- 'filled' | 'rejected'
    rejection_reason   VARCHAR,
    charges_fraction   DOUBLE NOT NULL,   -- execution.cost_model.compute_charges at notional 1.0,
                                           -- same convention optimizer.rebalance already uses
    model_id           VARCHAR NOT NULL,
    created_at         TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_positions (
    account_id         VARCHAR NOT NULL,
    symbol             VARCHAR NOT NULL,
    open_date          DATE NOT NULL,
    close_date         DATE,
    weight             DOUBLE NOT NULL,
    entry_price        DOUBLE NOT NULL,
    exit_price         DOUBLE,
    gross_return       DOUBLE,
    charges_fraction   DOUBLE,
    net_return         DOUBLE,
    status             VARCHAR NOT NULL,  -- 'open' | 'closed'
    open_order_id      VARCHAR NOT NULL,
    close_order_id     VARCHAR,
    PRIMARY KEY (account_id, symbol, open_date)
);

-- Phase J4c (docs/09-open-questions.md's OQ-11, finally built): the
-- evaluation-attempt counter. This gate has been run against ~40+
-- distinct configurations across this project's history with no
-- multiplicity correction anywhere -- research/gate_criteria_
-- preregistration.md names this exact gap as the thing it does NOT
-- close ("if this exact gate is run repeatedly against re-tunings of
-- the model... until one clears it, the gate becomes an overfitting
-- instrument again"). evaluation/attempts.py is the enforcement layer;
-- this table is just the append-only record it writes to. Deliberately
-- does NOT touch evaluation/gate.py -- evaluate_gate already accepts an
-- injected `criteria` parameter, so a stricter GateCriteria is
-- CONSTRUCTED and passed in, never hand-edited into the protected file.
CREATE TABLE IF NOT EXISTS evaluation_attempts (
    attempt_id              VARCHAR NOT NULL PRIMARY KEY,
    hypothesis_id            VARCHAR NOT NULL,
    preregistration_path      VARCHAR NOT NULL,
    preregistration_hash       VARCHAR NOT NULL,
    holdout_id                  VARCHAR NOT NULL,
    holdout_spec_json            VARCHAR NOT NULL,
    attempt_index                 INTEGER NOT NULL,  -- assigned by the DB, never caller-supplied
    registered_at                  TIMESTAMP NOT NULL,
    registered_by                   VARCHAR NOT NULL,  -- 'user' | 'foreman'
    status                           VARCHAR NOT NULL,  -- 'registered' | 'run' | 'abandoned'
    base_alpha                       DOUBLE NOT NULL,
    gate_alpha_used                   DOUBLE,
    result_verdict                     VARCHAR,          -- 'pass' | 'fail' | 'inconclusive'
    result_metrics_json                 VARCHAR,
    notes                                 VARCHAR,
    UNIQUE (hypothesis_id, holdout_id, preregistration_hash)
);

-- Phase J1: Angel One SmartAPI read-only sync. `broker_sync_runs` is the
-- durability/audit record (was this transient or an auth failure? did
-- reconciliation agree with the FIFO reconstruction?); `broker_holdings`
-- and `broker_positions_snapshot` are point-in-time snapshots (one row
-- per (broker, as_of_date, symbol) -- re-syncing the same day overwrites,
-- it does not accumulate duplicates). Trade/order-level sync (which
-- feeds `trades`/`positions` directly) is a deliberate follow-up, not
-- built in this pass -- see angel_sync.py's module docstring.
CREATE TABLE IF NOT EXISTS broker_sync_runs (
    sync_id          VARCHAR NOT NULL PRIMARY KEY,
    broker           VARCHAR NOT NULL,
    started_at       TIMESTAMP NOT NULL,
    finished_at      TIMESTAMP,
    status           VARCHAR NOT NULL,  -- 'ok' | 'partial' | 'transient_failure' | 'auth_failure'
    scopes_json      VARCHAR NOT NULL,
    n_holdings       INTEGER,
    n_positions      INTEGER,
    session_source   VARCHAR,           -- 'cached' | 'fresh_login'
    error            VARCHAR
);

CREATE TABLE IF NOT EXISTS broker_holdings (
    broker              VARCHAR NOT NULL,
    as_of_date          DATE NOT NULL,
    symbol              VARCHAR NOT NULL,
    exchange            VARCHAR,
    isin                VARCHAR,
    quantity            DOUBLE,
    t1_quantity         DOUBLE,
    avg_price           DOUBLE,
    ltp                 DOUBLE,
    close_price         DOUBLE,
    pnl                 DOUBLE,
    synced_at           TIMESTAMP NOT NULL,
    PRIMARY KEY (broker, as_of_date, symbol)
);

CREATE TABLE IF NOT EXISTS broker_positions_snapshot (
    broker              VARCHAR NOT NULL,
    as_of_date          DATE NOT NULL,
    symbol              VARCHAR NOT NULL,
    exchange            VARCHAR,
    product             VARCHAR NOT NULL,
    net_qty             DOUBLE,
    buy_qty             DOUBLE,
    buy_avg             DOUBLE,
    sell_qty            DOUBLE,
    sell_avg            DOUBLE,
    ltp                 DOUBLE,
    realised            DOUBLE,
    unrealised           DOUBLE,
    synced_at            TIMESTAMP NOT NULL,
    PRIMARY KEY (broker, as_of_date, symbol, product)
);

CREATE TABLE IF NOT EXISTS paper_daily_nav (
    account_id             VARCHAR NOT NULL,
    date                   DATE NOT NULL,
    nav_units              DOUBLE NOT NULL,
    daily_return           DOUBLE,
    cum_return             DOUBLE,
    benchmark_nav_units    DOUBLE,
    benchmark_daily_return DOUBLE,
    benchmark_cum_return   DOUBLE,
    n_positions            INTEGER NOT NULL,
    drawdown               DOUBLE,
    PRIMARY KEY (account_id, date)
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


def connect_with_retry(
    path: Path, *, read_only: bool = False, attempts: int = 5, delay_s: float = 60.0,
) -> "Store":
    """Phase J0.1: the real fix for the nightly scheduled jobs dying
    outright on a single momentary lock collision. Found live:
    `reconcile.log`/`daily_backfill.log` both terminated their ENTIRE
    run on one `duckdb.IOException` the first time the desktop API (or
    another job) happened to hold the file open at the exact instant the
    scheduled task started -- no retry existed anywhere. Verified
    two-process before writing this: this DuckDB build's locking is
    fully exclusive between any two connections except reader-vs-reader
    (see `Store.__init__`'s docstring), so contention is expected and
    normal, not a sign anything is broken -- a request-scoped `Store` in
    the desktop API holds its lock for milliseconds, so a short retry
    loop resolves the overwhelming majority of collisions without the
    scheduled job needing to abandon its run.

    Retries on `duckdb.IOException` specifically (the exception this
    project has twice observed for a locked file) — any other exception
    propagates immediately, since retrying a real error (bad SQL, a
    corrupt file) would just waste `attempts * delay_s` before failing
    anyway."""
    last_error: duckdb.IOException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return Store(path, read_only=read_only)
        except duckdb.IOException as e:
            last_error = e
            if attempt < attempts:
                time.sleep(delay_s)
    assert last_error is not None
    raise last_error


class Store:
    """Thin wrapper around a DuckDB file. One instance = one connection.

    DuckDB permits a single writer (docs/02-data-layer.md's accepted
    trade-off); this class does not attempt to hide that, callers should
    not open concurrent writable connections to the same file.
    """

    def __init__(self, path: Path, read_only: bool = False):
        """`read_only=True` opens a read-only DuckDB connection instead of
        the default read-write one.

        MEASURED, not assumed (Phase J0, verified live on this Windows +
        DuckDB 1.5.5 setup with a two-process test before trusting it):
        this file's locking is NOT "single writer, many concurrent
        readers." It is fully mutually exclusive between ANY two
        connections except reader-vs-reader -- a read-only connection
        blocks a writer trying to open the same file, and a writer blocks
        a read-only connection, exactly as read-write-vs-read-write does.
        Only two simultaneously-open read-only connections coexist. So
        `read_only=True` does NOT, by itself, stop a GET request from
        blocking a scheduled write job (or vice versa) -- the actual fix
        for that is holding every connection as briefly as possible
        (already the pattern: open, query, close, per request) plus
        retry-with-backoff on the writer side (see `connect_with_retry`
        below), which is what closes the real bug found live:
        `reconcile.log`/`daily_backfill.log` both died outright on a
        single `IOException: ... already open in ... python.exe`
        because neither the CLI nor the scheduled-task wrapper ever
        retried a momentary lock collision.

        What `read_only=True` DOES still buy: multiple simultaneous
        read-only connections (several desktop-app requests in flight,
        or a read-only connection alongside another read-only one) no
        longer need to serialize behind DuckDB's exclusive-writer lock,
        and a caller that only ever reads can no longer accidentally
        run the schema-creation/migration DDL below. Schema creation and
        the predictions migration are both write operations, so both are
        skipped for a read-only connection -- opening read-only against a
        database that doesn't exist yet correctly fails, rather than
        silently creating one."""
        self.path = path
        self.read_only = read_only
        if read_only:
            self.con = duckdb.connect(str(path), read_only=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
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
        # AUDIT FIX: position_id is DETERMINISTIC (positions.py derives it
        # from symbol/segment/dates/prices/qty, not a random uuid), so a
        # plain INSERT crashed on the primary-key constraint the second
        # time kundli ran against the same trades -- the exact scenario
        # a user re-running a report hits immediately. ON CONFLICT DO
        # NOTHING is correct here (not DO UPDATE): a reconstruction from
        # the same trades always recomputes the same numbers, so there's
        # nothing to update, only a no-op to allow.
        self.con.register("_positions", df[cols])
        self.con.execute(
            f"""
            INSERT INTO positions ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _positions
            ON CONFLICT (position_id) DO NOTHING
            """
        )
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

    def write_corporate_actions(self, df: pd.DataFrame) -> int:
        cols = ["symbol", "ex_date", "action_type", "ratio_num", "ratio_den", "factor_price",
                "dividend_amount", "face_before", "face_after", "subject_raw", "parse_status"]
        self.con.register("_ca", df[cols])
        self.con.execute(
            f"""
            INSERT INTO corporate_actions ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _ca
            ON CONFLICT (symbol, ex_date, subject_raw) DO UPDATE SET
                action_type = excluded.action_type, ratio_num = excluded.ratio_num,
                ratio_den = excluded.ratio_den, factor_price = excluded.factor_price,
                dividend_amount = excluded.dividend_amount, face_before = excluded.face_before,
                face_after = excluded.face_after, parse_status = excluded.parse_status
            """
        )
        self.con.unregister("_ca")
        return len(df)

    def read_corporate_actions(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM corporate_actions"
        params = []
        if start and end:
            query += " WHERE ex_date BETWEEN ? AND ?"
            params = [start, end]
        return self.con.execute(query + " ORDER BY symbol, ex_date", params).fetchdf()

    def write_intraday_bars(self, df: pd.DataFrame) -> int:
        cols = ["symbol", "ts", "interval", "open", "high", "low", "close", "volume"]
        self.con.register("_ib", df[cols])
        self.con.execute(
            f"""
            INSERT INTO intraday_bars ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _ib
            ON CONFLICT (symbol, ts, interval) DO UPDATE SET
                open = excluded.open, high = excluded.high, low = excluded.low,
                close = excluded.close, volume = excluded.volume
            """
        )
        self.con.unregister("_ib")
        return len(df)

    def read_intraday_bars(
        self, symbols: list[str] | None = None, start: str | None = None, end: str | None = None,
        interval: str = "1minute",
    ) -> pd.DataFrame:
        query = "SELECT * FROM intraday_bars WHERE interval = ?"
        params: list = [interval]
        if symbols:
            query += f" AND symbol IN ({', '.join(['?'] * len(symbols))})"
            params += symbols
        if start and end:
            query += " AND ts BETWEEN ? AND ?"
            params += [start, end]
        return self.con.execute(query + " ORDER BY symbol, ts", params).fetchdf()

    def write_upstox_instrument_map(self, df: pd.DataFrame) -> int:
        cols = ["symbol", "isin", "instrument_key", "resolved"]
        self.con.register("_uim", df[cols])
        self.con.execute(
            f"""
            INSERT INTO upstox_instrument_map ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _uim
            ON CONFLICT (symbol) DO UPDATE SET
                isin = excluded.isin, instrument_key = excluded.instrument_key, resolved = excluded.resolved
            """
        )
        self.con.unregister("_uim")
        return len(df)

    def read_upstox_instrument_map(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM upstox_instrument_map ORDER BY symbol").fetchdf()

    # ---- Phase F1: UI job tracking ----

    def insert_ui_job(self, row: dict) -> None:
        cols = list(row.keys())
        self.con.execute(
            f"INSERT INTO ui_jobs ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
            [row[c] for c in cols],
        )

    def finish_ui_job(self, job_id: str, status: str, finished_at) -> None:
        self.con.execute(
            "UPDATE ui_jobs SET status = ?, finished_at = ? WHERE job_id = ?",
            [status, finished_at, job_id],
        )

    def read_ui_jobs(self, limit: int = 50) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM ui_jobs ORDER BY started_at DESC LIMIT ?", [limit]).fetchdf()

    def read_ui_job(self, job_id: str) -> dict | None:
        df = self.con.execute("SELECT * FROM ui_jobs WHERE job_id = ?", [job_id]).fetchdf()
        return None if df.empty else df.iloc[0].to_dict()

    # ---- Phase F2/F4: key-value app settings ----

    def set_app_setting(self, key: str, value: str | None) -> None:
        self.con.execute(
            """
            INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            [key, value],
        )

    def get_app_setting(self, key: str) -> str | None:
        row = self.con.execute("SELECT value FROM app_settings WHERE key = ?", [key]).fetchone()
        return row[0] if row else None

    def read_app_settings(self) -> dict:
        rows = self.con.execute("SELECT key, value FROM app_settings").fetchall()
        return dict(rows)

    # ---- Phase F3: Claude usage tracking ----

    def get_usage_offset(self, file_path: str) -> int:
        row = self.con.execute("SELECT byte_offset FROM claude_usage_offsets WHERE file_path = ?", [file_path]).fetchone()
        return int(row[0]) if row else 0

    def set_usage_offset(self, file_path: str, byte_offset: int) -> None:
        self.con.execute(
            """
            INSERT INTO claude_usage_offsets (file_path, byte_offset) VALUES (?, ?)
            ON CONFLICT (file_path) DO UPDATE SET byte_offset = excluded.byte_offset
            """,
            [file_path, byte_offset],
        )

    def insert_usage_events(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        cols = ["event_id", "ts", "model", "input_tokens", "output_tokens", "cache_creation_tokens", "cache_read_tokens"]
        self.con.register("_usage_ev", df[cols])
        self.con.execute(
            f"""
            INSERT INTO claude_usage_events ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _usage_ev
            ON CONFLICT (event_id) DO NOTHING
            """
        )
        self.con.unregister("_usage_ev")
        return len(df)

    def read_usage_events(self, since=None) -> pd.DataFrame:
        if since is not None:
            return self.con.execute("SELECT * FROM claude_usage_events WHERE ts >= ? ORDER BY ts", [since]).fetchdf()
        return self.con.execute("SELECT * FROM claude_usage_events ORDER BY ts").fetchdf()

    # ---- Phase J2: paper trading (unit book, no capital column anywhere) ----

    def insert_paper_account(self, row: dict) -> None:
        cols = [
            "account_id", "name", "model_id", "model_type", "horizon_bars", "top_n",
            "cap_band", "fill_rule", "no_trade_band", "created_at", "closed_at", "status", "notes",
        ]
        self.con.execute(
            f"INSERT INTO paper_accounts ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
            [row.get(c) for c in cols],
        )

    def read_paper_accounts(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM paper_accounts ORDER BY created_at").fetchdf()

    def get_paper_account(self, account_id: str) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM paper_accounts WHERE account_id = ?", [account_id]).fetchdf()

    def close_paper_account(self, account_id: str, closed_at) -> None:
        self.con.execute(
            "UPDATE paper_accounts SET status = 'closed', closed_at = ? WHERE account_id = ?",
            [closed_at, account_id],
        )

    def write_paper_orders(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        cols = [
            "order_id", "account_id", "rebalance_date", "symbol", "action", "current_weight",
            "target_weight", "weight_delta", "fill_rule", "fill_price", "fill_status",
            "rejection_reason", "charges_fraction", "model_id", "created_at",
        ]
        self.con.register("_porders", df[cols])
        self.con.execute(f"INSERT INTO paper_orders ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _porders")
        self.con.unregister("_porders")
        return len(df)

    def read_paper_orders(self, account_id: str) -> pd.DataFrame:
        return self.con.execute(
            "SELECT * FROM paper_orders WHERE account_id = ? ORDER BY rebalance_date, symbol", [account_id]
        ).fetchdf()

    def rebalance_dates_recorded(self, account_id: str) -> list:
        return self.con.execute(
            "SELECT DISTINCT rebalance_date FROM paper_orders WHERE account_id = ? ORDER BY rebalance_date",
            [account_id],
        ).fetchdf()["rebalance_date"].tolist()

    def upsert_paper_positions(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        cols = [
            "account_id", "symbol", "open_date", "close_date", "weight", "entry_price", "exit_price",
            "gross_return", "charges_fraction", "net_return", "status", "open_order_id", "close_order_id",
        ]
        self.con.register("_ppos", df[cols])
        self.con.execute(
            f"""
            INSERT INTO paper_positions ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _ppos
            ON CONFLICT (account_id, symbol, open_date) DO UPDATE SET
                close_date = excluded.close_date, exit_price = excluded.exit_price,
                gross_return = excluded.gross_return, charges_fraction = excluded.charges_fraction,
                net_return = excluded.net_return, status = excluded.status,
                close_order_id = excluded.close_order_id
            """
        )
        self.con.unregister("_ppos")
        return len(df)

    def read_paper_positions(self, account_id: str, status: str | None = None) -> pd.DataFrame:
        if status is not None:
            return self.con.execute(
                "SELECT * FROM paper_positions WHERE account_id = ? AND status = ? ORDER BY open_date, symbol",
                [account_id, status],
            ).fetchdf()
        return self.con.execute(
            "SELECT * FROM paper_positions WHERE account_id = ? ORDER BY open_date, symbol", [account_id]
        ).fetchdf()

    def upsert_paper_daily_nav(self, row: dict) -> None:
        cols = [
            "account_id", "date", "nav_units", "daily_return", "cum_return", "benchmark_nav_units",
            "benchmark_daily_return", "benchmark_cum_return", "n_positions", "drawdown",
        ]
        self.con.execute(
            f"""
            INSERT INTO paper_daily_nav ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})
            ON CONFLICT (account_id, date) DO UPDATE SET
                nav_units = excluded.nav_units, daily_return = excluded.daily_return,
                cum_return = excluded.cum_return, benchmark_nav_units = excluded.benchmark_nav_units,
                benchmark_daily_return = excluded.benchmark_daily_return,
                benchmark_cum_return = excluded.benchmark_cum_return,
                n_positions = excluded.n_positions, drawdown = excluded.drawdown
            """,
            [row.get(c) for c in cols],
        )

    def read_paper_daily_nav(self, account_id: str) -> pd.DataFrame:
        return self.con.execute(
            "SELECT * FROM paper_daily_nav WHERE account_id = ? ORDER BY date", [account_id]
        ).fetchdf()

    # ---- Phase J4c: the evaluation-attempt registry ----

    def register_evaluation_attempt(self, row: dict) -> int:
        """Assigns attempt_index inside the SAME transaction as the
        insert -- computed here, never accepted as a caller-supplied
        value, so it cannot be spoofed or raced. Returns the assigned
        index."""
        existing = self.con.execute(
            "SELECT COALESCE(MAX(attempt_index), 0) FROM evaluation_attempts WHERE holdout_id = ?",
            [row["holdout_id"]],
        ).fetchone()[0]
        attempt_index = int(existing) + 1
        cols = [
            "attempt_id", "hypothesis_id", "preregistration_path", "preregistration_hash",
            "holdout_id", "holdout_spec_json", "attempt_index", "registered_at", "registered_by",
            "status", "base_alpha", "gate_alpha_used", "result_verdict", "result_metrics_json", "notes",
        ]
        payload = dict(row)
        payload["attempt_index"] = attempt_index
        self.con.execute(
            f"INSERT INTO evaluation_attempts ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
            [payload.get(c) for c in cols],
        )
        return attempt_index

    def count_evaluation_attempts(self, holdout_id: str) -> int:
        return int(self.con.execute(
            "SELECT COUNT(*) FROM evaluation_attempts WHERE holdout_id = ?", [holdout_id]
        ).fetchone()[0])

    def read_evaluation_attempts(self, holdout_id: str | None = None) -> pd.DataFrame:
        if holdout_id is not None:
            return self.con.execute(
                "SELECT * FROM evaluation_attempts WHERE holdout_id = ? ORDER BY attempt_index", [holdout_id]
            ).fetchdf()
        return self.con.execute("SELECT * FROM evaluation_attempts ORDER BY registered_at").fetchdf()

    def update_evaluation_attempt_result(
        self, attempt_id: str, status: str, gate_alpha_used: float | None,
        result_verdict: str | None, result_metrics_json: str | None,
    ) -> None:
        self.con.execute(
            """
            UPDATE evaluation_attempts SET status = ?, gate_alpha_used = ?,
                result_verdict = ?, result_metrics_json = ? WHERE attempt_id = ?
            """,
            [status, gate_alpha_used, result_verdict, result_metrics_json, attempt_id],
        )

    # ---- Phase J1: Angel One broker sync ----

    def insert_broker_sync_run(self, row: dict) -> None:
        cols = [
            "sync_id", "broker", "started_at", "finished_at", "status", "scopes_json",
            "n_holdings", "n_positions", "session_source", "error",
        ]
        self.con.execute(
            f"INSERT INTO broker_sync_runs ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})",
            [row.get(c) for c in cols],
        )

    def read_broker_sync_runs(self, broker: str | None = None, limit: int = 50) -> pd.DataFrame:
        if broker is not None:
            return self.con.execute(
                "SELECT * FROM broker_sync_runs WHERE broker = ? ORDER BY started_at DESC LIMIT ?", [broker, limit]
            ).fetchdf()
        return self.con.execute("SELECT * FROM broker_sync_runs ORDER BY started_at DESC LIMIT ?", [limit]).fetchdf()

    def upsert_broker_holdings(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        cols = [
            "broker", "as_of_date", "symbol", "exchange", "isin", "quantity", "t1_quantity",
            "avg_price", "ltp", "close_price", "pnl", "synced_at",
        ]
        self.con.register("_bhold", df[cols])
        self.con.execute(
            f"""
            INSERT INTO broker_holdings ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _bhold
            ON CONFLICT (broker, as_of_date, symbol) DO UPDATE SET
                exchange = excluded.exchange, isin = excluded.isin, quantity = excluded.quantity,
                t1_quantity = excluded.t1_quantity, avg_price = excluded.avg_price, ltp = excluded.ltp,
                close_price = excluded.close_price, pnl = excluded.pnl, synced_at = excluded.synced_at
            """
        )
        self.con.unregister("_bhold")
        return len(df)

    def read_broker_holdings(self, broker: str, as_of_date=None) -> pd.DataFrame:
        if as_of_date is not None:
            return self.con.execute(
                "SELECT * FROM broker_holdings WHERE broker = ? AND as_of_date = ? ORDER BY symbol",
                [broker, as_of_date],
            ).fetchdf()
        return self.con.execute(
            "SELECT * FROM broker_holdings WHERE broker = ? ORDER BY as_of_date DESC, symbol", [broker]
        ).fetchdf()

    def upsert_broker_positions_snapshot(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        cols = [
            "broker", "as_of_date", "symbol", "exchange", "product", "net_qty", "buy_qty",
            "buy_avg", "sell_qty", "sell_avg", "ltp", "realised", "unrealised", "synced_at",
        ]
        self.con.register("_bpos", df[cols])
        self.con.execute(
            f"""
            INSERT INTO broker_positions_snapshot ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _bpos
            ON CONFLICT (broker, as_of_date, symbol, product) DO UPDATE SET
                exchange = excluded.exchange, net_qty = excluded.net_qty, buy_qty = excluded.buy_qty,
                buy_avg = excluded.buy_avg, sell_qty = excluded.sell_qty, sell_avg = excluded.sell_avg,
                ltp = excluded.ltp, realised = excluded.realised, unrealised = excluded.unrealised,
                synced_at = excluded.synced_at
            """
        )
        self.con.unregister("_bpos")
        return len(df)

    def read_broker_positions_snapshot(self, broker: str, as_of_date=None) -> pd.DataFrame:
        if as_of_date is not None:
            return self.con.execute(
                "SELECT * FROM broker_positions_snapshot WHERE broker = ? AND as_of_date = ? ORDER BY symbol",
                [broker, as_of_date],
            ).fetchdf()
        return self.con.execute(
            "SELECT * FROM broker_positions_snapshot WHERE broker = ? ORDER BY as_of_date DESC, symbol", [broker]
        ).fetchdf()

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
