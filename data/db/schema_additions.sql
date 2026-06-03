-- StockSense schema additions (Phase 1)
-- Apply after the base schema.sql has been initialized.
-- Safe to re-run: all statements use IF NOT EXISTS.

-- ------------------------------------------------------------------ --
-- F&O daily aggregates                                                 --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS fo_daily (
    time        TIMESTAMPTZ NOT NULL,
    ticker      VARCHAR(20) NOT NULL REFERENCES stocks(ticker),
    total_oi    BIGINT,
    oi_change   BIGINT,
    put_oi      BIGINT,
    call_oi     BIGINT,
    pcr         NUMERIC(10, 4),
    PRIMARY KEY (time, ticker)
);

SELECT create_hypertable('fo_daily', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_fo_daily_ticker ON fo_daily(ticker, time DESC);

-- ------------------------------------------------------------------ --
-- Model accuracy (already in base schema; guard with IF NOT EXISTS)    --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS model_accuracy (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(50) NOT NULL,
    signal_type     VARCHAR(10) NOT NULL,
    timeframe       VARCHAR(20) NOT NULL,
    period_start    TIMESTAMPTZ NOT NULL,
    period_end      TIMESTAMPTZ NOT NULL,
    total_signals   INTEGER DEFAULT 0,
    correct_signals INTEGER DEFAULT 0,
    accuracy        NUMERIC(5, 4),
    avg_confidence  NUMERIC(5, 4),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------------ --
-- SLM training pairs (Claude response + situation context)             --
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS slm_training_pairs (
    id                  SERIAL PRIMARY KEY,
    ticker              VARCHAR(20) REFERENCES stocks(ticker),
    situation_date      DATE NOT NULL,
    situation_context   TEXT NOT NULL,
    claude_response     TEXT NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_slm_training_ticker ON slm_training_pairs(ticker, situation_date DESC);
