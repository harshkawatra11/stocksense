-- ============================================================
-- StockSense schema v8 — sandbox order execution on the confirmation queue
--
-- Wires backend/routers/confirmations.py's approve() endpoint to actually
-- place an order via Upstox's SANDBOX API (sandbox.upstox.com/v2 — risk-free
-- rehearsal, no real money, separate token from the live Analytics Token).
-- Every row this app has ever placed an order for is sandbox; `is_sandbox`
-- exists so a future live-order stage can't be confused with this one by
-- accident — it defaults TRUE and nothing in this codebase sets it FALSE.
-- Idempotent: safe to run repeatedly.
-- ============================================================

ALTER TABLE pending_trade_confirmations ADD COLUMN IF NOT EXISTS order_id VARCHAR(64);
ALTER TABLE pending_trade_confirmations ADD COLUMN IF NOT EXISTS execution_status VARCHAR(20);
-- NULL (not yet attempted) | PLACED | FAILED
ALTER TABLE pending_trade_confirmations ADD COLUMN IF NOT EXISTS execution_detail TEXT;
ALTER TABLE pending_trade_confirmations ADD COLUMN IF NOT EXISTS is_sandbox BOOLEAN NOT NULL DEFAULT TRUE;
