-- ============================================================
-- StockSense schema v7 — Upstox data-layer foundation
-- Adds ISIN to stocks (needed to build Upstox instrument_key =
-- "NSE_EQ|<ISIN>") and a tiny single-row-per-day token cache for
-- the daily-expiring Upstox OAuth access token.
-- Idempotent: safe to run repeatedly.
-- ============================================================

ALTER TABLE stocks ADD COLUMN IF NOT EXISTS isin VARCHAR(20);
CREATE INDEX IF NOT EXISTS idx_stocks_isin ON stocks (isin);

-- One row per acquired token; get_valid_token() reads the most recent
-- row and checks acquired_at is "today" (IST). Old rows are left in
-- place for audit/debugging rather than deleted.
CREATE TABLE IF NOT EXISTS upstox_token (
    id SERIAL PRIMARY KEY,
    token TEXT NOT NULL,
    acquired_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_upstox_token_acquired_at ON upstox_token (acquired_at DESC);
