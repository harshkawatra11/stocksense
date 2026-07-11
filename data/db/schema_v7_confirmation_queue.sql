-- ============================================================
-- StockSense schema v7 — human-confirmed trade approval queue
-- Execution wiring intentionally deferred — see WHAT_TO_DO_NEXT.txt Section 5.
-- This table only records human approve/reject decisions on proposed trades;
-- nothing places real orders yet. Trades must always be human approve/reject
-- only, never auto-executed, per explicit user requirement.
-- Idempotent: safe to run repeatedly.
-- ============================================================

CREATE TABLE IF NOT EXISTS pending_trade_confirmations (
    id            SERIAL PRIMARY KEY,
    signal_id     INTEGER,              -- loose reference to signals.id; no FK constraint (signal may be transient)
    ticker        VARCHAR(20) NOT NULL,
    action        VARCHAR(10) NOT NULL, -- BUY | SELL
    quantity      INTEGER NOT NULL,
    price         NUMERIC(12, 2) NOT NULL,
    reasoning     TEXT,
    status        VARCHAR(20) NOT NULL DEFAULT 'PENDING', -- PENDING | APPROVED | REJECTED | EXPIRED
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_pending_trade_confirmations_status
    ON pending_trade_confirmations (status);
