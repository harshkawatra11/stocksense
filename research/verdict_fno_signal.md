# F&O Positioning Signal: Verdict

**Date:** 2026-08-25
**Pre-registration:** research/preregistration_fno_signal.md (parameters fixed before this run)

## Overall: NO PASS -- same-day signal not found, with or without F&O positioning data

Baseline and treatment run on the IDENTICAL F&O-eligible point-in-time universe, horizon=1, using evaluation/gate.py unmodified with research/gate_criteria_preregistration.md's defaults.

## Per-combination verdicts

| feature_set | top_n | passed | reason | n_folds | mean_alpha_net | hit_rate_pvalue |
|---|---|---|---|---|---|---|
| baseline_price_only | 10 | FAIL | mean net alpha -0.0098% <= threshold +0.0000% | 96 | -0.00010 | 0.8693 |
| baseline_price_only | 20 | FAIL | mean net alpha -0.0351% <= threshold +0.0000% | 96 | -0.00035 | 0.9997 |
| treatment_price_plus_fno | 10 | FAIL | hit rate 48% (46/96) not significant vs chance: one-sided binomial p=0.695 > alpha=0.1 | 96 | +0.00010 | 0.6950 |
| treatment_price_plus_fno | 20 | FAIL | mean net alpha -0.0244% <= threshold +0.0000% | 96 | -0.00024 | 0.9079 |
