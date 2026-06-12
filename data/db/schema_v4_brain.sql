-- ============================================================
-- StockSense schema v4 — autonomous brain
-- Adaptive parameters, audit history, job heartbeat, idempotency.
-- Idempotent: safe to run repeatedly.
-- ============================================================

-- 1. Adaptive parameters the brain tunes within hard bounds.
CREATE TABLE IF NOT EXISTS brain_params (
    param_name  TEXT PRIMARY KEY,
    value       DOUBLE PRECISION NOT NULL,
    min_value   DOUBLE PRECISION NOT NULL,
    max_value   DOUBLE PRECISION NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by  TEXT NOT NULL DEFAULT 'seed',   -- seed | claude_calibration | manual
    reason      TEXT DEFAULT ''
);

INSERT INTO brain_params (param_name, value, min_value, max_value) VALUES
  ('confidence_threshold', 0.55, 0.45, 0.80),
  ('max_position_pct',     0.25, 0.05, 0.50),
  ('max_open_positions',   5,    1,    10),
  ('macro_nudge_cap',      0.10, 0.00, 0.20),
  ('agreement_boost',      0.05, 0.00, 0.15)
ON CONFLICT (param_name) DO NOTHING;

-- 2. Audit trail of every parameter change (who, why, when).
CREATE TABLE IF NOT EXISTS brain_param_history (
    id          BIGSERIAL PRIMARY KEY,
    param_name  TEXT NOT NULL,
    old_value   DOUBLE PRECISION,
    new_value   DOUBLE PRECISION NOT NULL,
    changed_by  TEXT NOT NULL,
    reason      TEXT DEFAULT '',
    changed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_brain_param_history_time
    ON brain_param_history (changed_at DESC);

-- 3. Scheduler heartbeat — one row per job execution.
CREATE TABLE IF NOT EXISTS job_runs (
    id          BIGSERIAL PRIMARY KEY,
    job_id      TEXT NOT NULL,
    started_at  TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    status      TEXT NOT NULL DEFAULT 'running',  -- running | ok | error
    summary     TEXT DEFAULT '',
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_job_runs_job_time ON job_runs (job_id, started_at DESC);

-- 4. Idempotency backstop: at most one [AUTO] decision per signal.
CREATE UNIQUE INDEX IF NOT EXISTS uq_decisions_auto_signal
    ON decisions (signal_id)
    WHERE rationale LIKE '[AUTO]%' AND signal_id IS NOT NULL;
