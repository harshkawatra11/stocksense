-- ============================================================
-- StockSense schema v3 — intelligent signals + position re-analysis + activity log
-- Idempotent.
-- ============================================================

-- 1. Signal: target comes from the forecast, plus a fractional ETA to reach it.
ALTER TABLE signals ADD COLUMN IF NOT EXISTS target_eta_days  NUMERIC(5,2);  -- e.g. 2.5
ALTER TABLE signals ADD COLUMN IF NOT EXISTS predicted_path   JSONB;          -- forecast close path
ALTER TABLE signals ADD COLUMN IF NOT EXISTS expected_move_pct NUMERIC(6,2);

-- 2. Position re-analysis: each held position gets re-checked vs its original
--    prediction as its horizon approaches. One row per re-analysis event.
CREATE TABLE IF NOT EXISTS position_reviews (
    id            SERIAL PRIMARY KEY,
    portfolio_id  INTEGER REFERENCES portfolio(id),
    signal_id     INTEGER REFERENCES signals(id),
    ticker        VARCHAR(20) REFERENCES stocks(ticker),
    reviewed_at   TIMESTAMPTZ DEFAULT NOW(),
    days_elapsed      NUMERIC(5,2),   -- since entry
    eta_days          NUMERIC(5,2),   -- original predicted ETA
    entry_price       NUMERIC(12,2),
    target_price      NUMERIC(12,2),
    current_price     NUMERIC(12,2),
    progress_pct      NUMERIC(6,2),   -- % of the predicted move achieved
    status            VARCHAR(20),    -- on_track | ahead | behind | target_hit | stopped | expired
    verdict           VARCHAR(10),    -- HOLD | EXIT | ADD
    reasoning         TEXT
);
CREATE INDEX IF NOT EXISTS idx_position_reviews_ticker ON position_reviews(ticker);
CREATE INDEX IF NOT EXISTS idx_position_reviews_reviewed_at ON position_reviews(reviewed_at DESC);

-- 3. Activity log — the full lifecycle the user sees in the app:
--    SUGGESTED, RATED (liked/disliked + why), BOUGHT, HELD, REANALYZED, EXITED.
CREATE TABLE IF NOT EXISTS activity_log (
    id          SERIAL PRIMARY KEY,
    event_type  VARCHAR(20) NOT NULL,   -- SUGGESTED | RATED | BOUGHT | SOLD | REANALYZED | NOTE
    ticker      VARCHAR(20) REFERENCES stocks(ticker),
    signal_id   INTEGER REFERENCES signals(id),
    rating      VARCHAR(10),            -- LIKE | DISLIKE (when event_type = RATED)
    note        TEXT,                   -- free text, e.g. "confidence too low"
    payload     JSONB,                  -- snapshot of relevant data
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_activity_log_created_at ON activity_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_activity_log_ticker ON activity_log(ticker);
