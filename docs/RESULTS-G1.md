# G1 results

## 4. Kronos zero-shot features (T7.5d)

Pre-registered reduced plan (docs/DECISIONS.md, Phase 7): Kronos-small zero-shot, float32, lookback 128, 5-step horizon, 10 paths, 150 most liquid names, every 10th session 2016-01-04 to 2025-09-30 (242 dates, 36,300 rows, research period only). Incremental test: ridge (alpha 100) on all base ranked features without vs with the seven `kr_*` features, 40 % train-only then 4 expanding blocks with purge; admission needs mean incremental IC >= 0.002 and paired t >= 2.

| target | test dates | IC without | IC with | mean diff | paired t | admitted |
|---|---|---|---|---|---|---|
| y_oc | 146 | +0.0509 | +0.0502 | -0.0007 | -0.48 | no |
| y_c2o_5 | 145 | +0.0088 | +0.0106 | +0.0018 | +0.79 | no |

Standalone mean daily Spearman IC (naive t): `kr_ret5_mean` +0.030 (3.90) on y_oc, +0.012 (1.46) on y_c2o_5, -0.001 on y_c2o_20; `kr_up_prob` +0.028 (4.07) on y_oc; `kr_oc1_mean` +0.024 (3.22) on y_oc; `kr_range5` -0.036 (-2.75) on y_oc; `kr_co1_mean` +0.017 (2.47) on y_c2o_5. `kr_ret5_mean` correlates -0.39 with ranked `mom_5`, -0.25 with `ret_1`, -0.05 with `vol_21`.

Verdict: Kronos zero-shot: no incremental value established (low power: 242 dates x about 150 stocks). The standalone y_oc signal is largely short-term reversal already in the base features. Not disproven; the features are not added to the feature version.
