-- ============================================================
-- StockSense schema v9 — hard removal of Kronos
--
-- Kronos was archived behind KRONOS_ENABLED (default false) months ago in
-- favor of the 3-seed LightGBM ensemble + quantile regressors. The flag,
-- its config properties, and every code reference have been removed
-- (2026-07-13) — this migration removes the last DB residue: the
-- signals.kronos_confidence column (always NULL going forward, since no
-- code path writes it anymore) and any historical signal_reasoning rows
-- attributed to model_name='kronos'.
-- Idempotent: safe to run repeatedly.
-- ============================================================

ALTER TABLE signals DROP COLUMN IF EXISTS kronos_confidence;
DELETE FROM signal_reasoning WHERE model_name = 'kronos';
