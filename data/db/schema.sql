CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE stocks (
    ticker VARCHAR(20) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    sector VARCHAR(100),
    industry VARCHAR(100),
    exchange VARCHAR(10) DEFAULT 'NSE',
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE ohlcv_daily (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(20) NOT NULL REFERENCES stocks(ticker),
    open NUMERIC(12,2),
    high NUMERIC(12,2),
    low NUMERIC(12,2),
    close NUMERIC(12,2),
    volume BIGINT,
    adj_close NUMERIC(12,2),
    PRIMARY KEY (time, ticker)
);
SELECT create_hypertable('ohlcv_daily', 'time', if_not_exists => TRUE);

CREATE TABLE ohlcv_1min (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(20) NOT NULL REFERENCES stocks(ticker),
    open NUMERIC(12,2),
    high NUMERIC(12,2),
    low NUMERIC(12,2),
    close NUMERIC(12,2),
    volume BIGINT,
    PRIMARY KEY (time, ticker)
);
SELECT create_hypertable('ohlcv_1min', 'time', if_not_exists => TRUE);

CREATE TABLE portfolio (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL REFERENCES stocks(ticker),
    quantity INTEGER NOT NULL,
    avg_price NUMERIC(12,2) NOT NULL,
    buy_date TIMESTAMPTZ NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL REFERENCES stocks(ticker),
    signal_type VARCHAR(10) NOT NULL CHECK (signal_type IN ('BUY','SELL','HOLD')),
    timeframe VARCHAR(20) NOT NULL CHECK (timeframe IN ('intraday','swing','longterm')),
    price_at_signal NUMERIC(12,2) NOT NULL,
    target_price NUMERIC(12,2),
    stop_loss NUMERIC(12,2),
    ml_confidence NUMERIC(5,4),
    slm_confidence NUMERIC(5,4),
    claude_confidence NUMERIC(5,4),
    final_confidence NUMERIC(5,4),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active','hit_target','hit_stop','expired','cancelled')),
    fired_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    actual_close NUMERIC(12,2)
);

CREATE TABLE signal_reasoning (
    id SERIAL PRIMARY KEY,
    signal_id INTEGER NOT NULL REFERENCES signals(id),
    model_name VARCHAR(50) NOT NULL,
    reasoning TEXT NOT NULL,
    raw_output JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE model_accuracy (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    signal_type VARCHAR(10) NOT NULL,
    timeframe VARCHAR(20) NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    total_signals INTEGER DEFAULT 0,
    correct_signals INTEGER DEFAULT 0,
    accuracy NUMERIC(5,4),
    avg_confidence NUMERIC(5,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE learnings (
    id SERIAL PRIMARY KEY,
    learning_date DATE NOT NULL,
    learning_type VARCHAR(50) NOT NULL CHECK (learning_type IN ('eod_review','weekly_review','pattern_failure','pattern_success')),
    ticker VARCHAR(20) REFERENCES stocks(ticker),
    signal_id INTEGER REFERENCES signals(id),
    title VARCHAR(300) NOT NULL,
    body TEXT NOT NULL,
    tags TEXT[],
    raw_claude_output TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE fo_daily (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(20) NOT NULL REFERENCES stocks(ticker),
    total_oi BIGINT,
    oi_change BIGINT,
    put_oi BIGINT,
    call_oi BIGINT,
    pcr NUMERIC(10,4),
    PRIMARY KEY (time, ticker)
);
SELECT create_hypertable('fo_daily', 'time', if_not_exists => TRUE);

CREATE TABLE slm_training_pairs (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) REFERENCES stocks(ticker),
    trade_date DATE NOT NULL,
    situation_context TEXT NOT NULL,
    claude_response TEXT NOT NULL,
    decision VARCHAR(10),
    confidence NUMERIC(5,4),
    reasoning TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE feature_importance (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    importance_score NUMERIC(10,6) NOT NULL,
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_signals_ticker ON signals(ticker);
CREATE INDEX idx_signals_fired_at ON signals(fired_at DESC);
CREATE INDEX idx_learnings_date ON learnings(learning_date DESC);
CREATE INDEX idx_portfolio_active ON portfolio(active, ticker);
