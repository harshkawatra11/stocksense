-- ============================================================
-- StockSense schema v8 — learnings lifecycle
-- Kills unfalsifiable, ever-accumulating learnings prose: every learning now
-- expires, and gets tracked for whether it was actually worth applying.
-- Idempotent: safe to run repeatedly.
--
-- Simplification (see intelligence/eod_review.py for the real logic):
--   applies_count — incremented once per row every time get_active_learnings()
--                   returns it (i.e. every time it's included in a Claude
--                   synthesis/calibration prompt).
--   hit_count     — a rough daily proxy, NOT precise per-learning causal
--                   attribution (that would require tagging which learning
--                   drove which trade, which nothing in this codebase does
--                   today). Once per day, after EOD review resolves the day's
--                   decisions, every learning that was applied that day gets
--                   hit_count += 1 if that day's SELL win-rate was >= 50%.
--                   So hit_count/applies_count approximates "how often was
--                   this learning active on a day that went well" — a signal
--                   of harmlessness/usefulness, not proof of causation.
-- ============================================================

ALTER TABLE learnings ADD COLUMN IF NOT EXISTS applies_count INTEGER DEFAULT 0;
ALTER TABLE learnings ADD COLUMN IF NOT EXISTS hit_count INTEGER DEFAULT 0;
ALTER TABLE learnings ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;

-- Backfill expires_at for any pre-existing rows (created_at + 60 days).
UPDATE learnings SET expires_at = created_at + INTERVAL '60 days' WHERE expires_at IS NULL;

-- Plain (non-partial) index — a partial index predicate on NOW() isn't
-- allowed (NOW() isn't IMMUTABLE), so filter at query time instead.
CREATE INDEX IF NOT EXISTS idx_learnings_expires_at ON learnings (expires_at);
