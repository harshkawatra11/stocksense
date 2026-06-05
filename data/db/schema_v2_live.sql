-- ============================================================
-- StockSense schema v2 — live multi-timeframe + capital tracking
-- Idempotent: safe to run repeatedly.
-- ============================================================

-- 1. Allow concurrent multi-horizon timeframe labels on signals.
--    Old constraint only permitted intraday/swing/longterm.
ALTER TABLE signals DROP CONSTRAINT IF EXISTS signals_timeframe_check;
ALTER TABLE signals ADD CONSTRAINT signals_timeframe_check
    CHECK (timeframe IN ('30m','2h','1D','3D','5D','intraday','swing','longterm'));

-- 2. Per-signal annotations for the live phase.
ALTER TABLE signals ADD COLUMN IF NOT EXISTS horizon_days       INTEGER;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS affordable         BOOLEAN;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS shares_affordable  INTEGER;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS macro_sector_score NUMERIC(5,4);

-- 3. Account / capital state. Latest row = current capital.
CREATE TABLE IF NOT EXISTS account (
    id             SERIAL PRIMARY KEY,
    cash_available NUMERIC(12,2) NOT NULL DEFAULT 0,
    cash_reserve   NUMERIC(12,2) NOT NULL DEFAULT 0,
    updated_at     TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Decision ledger — every action taken on a signal, including passes,
--    so the learning loop can learn from what was NOT acted on too.
CREATE TABLE IF NOT EXISTS decisions (
    id          SERIAL PRIMARY KEY,
    signal_id   INTEGER REFERENCES signals(id),
    ticker      VARCHAR(20) REFERENCES stocks(ticker),
    action      VARCHAR(10) NOT NULL CHECK (action IN ('BUY','SELL','PASS','SKIP')),
    quantity    INTEGER DEFAULT 0,
    price       NUMERIC(12,2),
    cash_after  NUMERIC(12,2),
    rationale   TEXT,
    outcome     VARCHAR(20),          -- filled later: win/loss/flat/open
    pnl         NUMERIC(12,2),        -- realized P&L when closed
    decided_at  TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_decisions_ticker ON decisions(ticker);
CREATE INDEX IF NOT EXISTS idx_decisions_decided_at ON decisions(decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_timeframe ON signals(timeframe);
