-- ============================================================
-- StockSense schema v5 — pluggable LLM providers
-- Stores the user's chosen macro + synthesis engines (no secrets).
-- Idempotent: safe to run repeatedly.
-- ============================================================

-- Generic key/value app configuration, JSON-valued.
-- Keys used by the provider system:
--   macro_provider     -> {"mode": "local|cloud", "model": "qwen2.5:7b", "has_key": false}
--   synthesis_provider -> {"backend": "claude|codex|gemini", "fast_model": "...", "deep_model": "..."}
-- Secrets (e.g. the Ollama Cloud key) live in .env only; we persist has_key, never the key.
CREATE TABLE IF NOT EXISTS app_config (
    key        TEXT PRIMARY KEY,
    value      JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
