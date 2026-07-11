-- ============================================================
-- StockSense schema v6 — Stage 0 truth layer (component health)
-- Stores per-signal pipeline component provenance so the frontend can
-- show "Kronos: pretrained (not NSE)" / "Claude: skipped" honestly
-- instead of silently degrading.
-- Idempotent: safe to run repeatedly.
-- ============================================================

-- Shape (per the shared status contract):
-- {
--   "kronos":     {"status": "ok"|"degraded"|"unavailable", "detail": "...", "source": "finetuned"|"pretrained"|"mock"|"unavailable"},
--   "lightgbm":   {"status": "ok"|"degraded"|"unavailable", "detail": "...", "stale": false},
--   "llm_synthesis": {"status": "ok"|"degraded"|"unavailable", "detail": "...", "source": "claude"|"codex"|"gemini"|"skipped"},
--   "macro":      {"status": "ok"|"degraded"|"unavailable", "detail": "...", "source": "ollama_local"|"ollama_cloud"|"neutral_fallback"}
-- }
ALTER TABLE signals ADD COLUMN IF NOT EXISTS components_json JSONB;

CREATE INDEX IF NOT EXISTS idx_signals_components_json ON signals USING GIN (components_json);
