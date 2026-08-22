# Bhavcopy Point-in-Time Gate Re-Run: Verdict

**Date:** 2026-08-22
**Pre-registration:** research/preregistration_bhavcopy_rerun.md (parameters fixed before this run)

## Overall: AT LEAST ONE COMBINATION PASSES

Evaluated independently per (cap_band, horizon, top_n) at the pre-registered 25bps cost point, using evaluation/gate.py unmodified with research/gate_criteria_preregistration.md's defaults.

## Per-combination verdicts

| cap_band | horizon | top_n | passed | reason | n_folds | mean_alpha_net | hit_rate_pvalue |
|---|---|---|---|---|---|---|---|
| full_pit | 10 | 10 | PASS | all criteria passed | 25 | +0.01672 | 0.0001 |
| full_pit | 10 | 20 | PASS | all criteria passed | 25 | +0.01425 | 0.0000 |
| full_pit | 20 | 10 | PASS | all criteria passed | 13 | +0.01990 | 0.0001 |
| full_pit | 20 | 20 | PASS | all criteria passed | 13 | +0.02172 | 0.0001 |
| large | 10 | 10 | FAIL | hit rate 48% (12/25) not significant vs chance: one-sided binomial p=0.655 > alpha=0.1 | 25 | +0.00060 | 0.6550 |
| large | 10 | 20 | FAIL | hit rate 52% (13/25) not significant vs chance: one-sided binomial p=0.500 > alpha=0.1 | 25 | +0.00122 | 0.5000 |
| large | 20 | 10 | FAIL | hit rate 54% (7/13) not significant vs chance: one-sided binomial p=0.500 > alpha=0.1 | 13 | +0.00212 | 0.5000 |
| large | 20 | 20 | FAIL | hit rate 69% (9/13) not significant vs chance: one-sided binomial p=0.133 > alpha=0.1 | 13 | +0.00225 | 0.1334 |
| mid | 10 | 10 | PASS | all criteria passed | 25 | +0.00560 | 0.0001 |
| mid | 10 | 20 | PASS | all criteria passed | 25 | +0.00441 | 0.0005 |
| mid | 20 | 10 | PASS | all criteria passed | 13 | +0.00722 | 0.0112 |
| mid | 20 | 20 | PASS | all criteria passed | 13 | +0.00401 | 0.0461 |
| small | 10 | 10 | PASS | all criteria passed | 25 | +0.00694 | 0.0001 |
| small | 10 | 20 | PASS | all criteria passed | 25 | +0.00681 | 0.0000 |
| small | 20 | 10 | PASS | all criteria passed | 13 | +0.01003 | 0.0017 |
| small | 20 | 20 | PASS | all criteria passed | 13 | +0.00889 | 0.0017 |
