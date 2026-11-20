# Program G2 results

## 1. S1 cost and turnover grid (card C5, family `g2-s1cost`)

Run 2026-11-19/20 on the fixed GBM scores (`s1-gbm-y_c2o_10-e0f7276025`, `s1-gbm-y_c2o_20-747bb3abf2`), research window 2014-01-01 to 2025-09-30, k_max 10, no overlay (the values of the pre-C5 best trial `g1-s1-y_c2o_10-cce12fb757`), auto-resume 20, u500 hysteresis universe flag, capital grid plus 2x slippage, reference capital Rs 5 lakh. Meta-label stage not applied (DECISIONS, C5). All 16 trials were registered before any ran. Every number is at Rs 5 lakh. Cost drag is the yearly cost plus slippage paid as a fraction of capital.

| variant | target | net Sharpe | 2x-slip Sharpe | excess t (v1, raw) | excess t (v2, exposure-matched) | positive years | max DD | mean turnover/day | avg hold (sessions) | cost drag (annual, of capital) |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 (exit rank 50, min hold h/2) | y_c2o_10 | 1.311 | 0.921 | -1.23 | +1.21 | 0.83 | 0.171 | 0.115 | 9.1 | 0.190 |
| V2 (V1 + free extension, rebuy block 5) | y_c2o_10 | 1.266 | 0.796 | -1.38 | +1.06 | 0.75 | 0.177 | 0.108 | 9.0 | 0.157 |
| V3 (V2 + score EMA 3) | y_c2o_10 | 1.465 | 0.776 | -1.03 | +1.90 | 0.75 | 0.175 | 0.096 | 10.3 | 0.177 |
| V4 (V2 + score EMA 5) | y_c2o_10 | 1.308 | 0.752 | -1.13 | +1.26 | 0.83 | 0.209 | 0.098 | 11.0 | 0.187 |
| V5 (V4 + switch hurdle) | y_c2o_10 | 1.308 | 0.752 | -1.13 | +1.26 | 0.83 | 0.209 | 0.098 | 11.0 | 0.187 |
| V6 (V5 + decile ER map) | y_c2o_10 | 1.636 | 1.461 | -0.04 | +2.24 | 0.83 | 0.188 | 0.108 | 11.8 | 0.320 |
| V7 (V6, exit rank 75) | y_c2o_10 | 1.521 | 1.341 | -0.43 | +1.97 | 0.83 | 0.159 | 0.101 | 12.4 | 0.237 |
| V8 (V6 + max fixed cost 10 bps) | y_c2o_10 | 1.687 | 1.292 | -0.01 | +2.26 | 0.83 | 0.199 | 0.104 | 11.6 | 0.293 |
| V1 (exit rank 50, min hold h/2) | y_c2o_20 | 1.459 | 1.164 | -0.47 | +1.84 | 1.00 | 0.193 | 0.067 | 18.0 | 0.135 |
| V2 (V1 + free extension, rebuy block 5) | y_c2o_20 | 1.366 | 1.112 | -0.79 | +1.45 | 0.92 | 0.192 | 0.065 | 18.0 | 0.111 |
| V3 (V2 + score EMA 3) | y_c2o_20 | 1.398 | 1.273 | -0.58 | +1.79 | 0.92 | 0.196 | 0.061 | 19.4 | 0.122 |
| V4 (V2 + score EMA 5) | y_c2o_20 | 1.052 | 1.004 | -1.46 | +0.38 | 0.75 | 0.217 | 0.053 | 21.1 | 0.075 |
| V5 (V4 + switch hurdle) | y_c2o_20 | 1.052 | 1.004 | -1.46 | +0.38 | 0.75 | 0.217 | 0.053 | 21.1 | 0.075 |
| V6 (V5 + decile ER map) | y_c2o_20 | 1.586 | 1.212 | -0.13 | +1.41 | 0.83 | 0.194 | 0.056 | 21.3 | 0.133 |
| V7 (V6, exit rank 75) | y_c2o_20 | 1.632 | 1.328 | +0.00 | +2.01 | 0.92 | 0.189 | 0.053 | 22.6 | 0.126 |
| V8 (V6 + max fixed cost 10 bps) | y_c2o_20 | 1.484 | 1.447 | -0.53 | +1.05 | 0.83 | 0.195 | 0.053 | 21.4 | 0.108 |
| pre-C5 best (meta-label) (comparison, g1-s1-y_c2o_10-cce12fb757) | y_c2o_10 | 1.166 | 0.222 | -1.70 | +0.08 | 0.75 | 0.172 | 0.186 | 5.5 | 0.252 |
| pre-C5 plain base (comparison, g1-s1-y_c2o_10-87b9794e81) | y_c2o_10 | 1.109 | 0.347 | -1.80 | -0.04 | 0.67 | 0.182 | 0.188 | 5.5 | 0.258 |

Trial ids:

- V1 y_c2o_10: `g2-s1cost-y_c2o_10-9d13d7d89c`
- V2 y_c2o_10: `g2-s1cost-y_c2o_10-15ae2b9f17`
- V3 y_c2o_10: `g2-s1cost-y_c2o_10-8319a5e2b2`
- V4 y_c2o_10: `g2-s1cost-y_c2o_10-9c6e5c4c53`
- V5 y_c2o_10: `g2-s1cost-y_c2o_10-c3f5dbd8de`
- V6 y_c2o_10: `g2-s1cost-y_c2o_10-97b995f190`
- V7 y_c2o_10: `g2-s1cost-y_c2o_10-6aa25c2ce5`
- V8 y_c2o_10: `g2-s1cost-y_c2o_10-40f8615b42`
- V1 y_c2o_20: `g2-s1cost-y_c2o_20-117c59f688`
- V2 y_c2o_20: `g2-s1cost-y_c2o_20-c14caf66e3`
- V3 y_c2o_20: `g2-s1cost-y_c2o_20-d411fbf233`
- V4 y_c2o_20: `g2-s1cost-y_c2o_20-45913f0d32`
- V5 y_c2o_20: `g2-s1cost-y_c2o_20-5e85958c54`
- V6 y_c2o_20: `g2-s1cost-y_c2o_20-99af68b13a`
- V7 y_c2o_20: `g2-s1cost-y_c2o_20-f18ffbc8b1`
- V8 y_c2o_20: `g2-s1cost-y_c2o_20-a4a88b729a`

Observations (read from the table, not tuned):

- Every one of the 16 variants beats the pre-C5 best on net Sharpe (1.05 to 1.69 against 1.17 for the meta-label trial and 1.11 for its plain base) and on 2x-slippage Sharpe (0.75 to 1.46 against 0.22). Mean turnover falls from 0.19 to 0.05-0.12 per day and the average hold rises from 5.5 to 9-23 sessions. The rank-decay exit at rank 50 with a minimum hold (V1) is where most of that comes from.
- The free extension and rebuy block (V2) did not help by themselves (net 1.27 against 1.31 at 10 days, 1.37 against 1.46 at 20 days).
- V4 and V5 are identical to the last digit in both targets: the switch hurdle never produced a swap in these backtests. It is a no-op here, not evidence for or against switching.
- The convex decile expected-return map (V6) is the largest single step at 10 days (net 1.31 to 1.64, 2x 0.75 to 1.46) and raises excess t against the raw benchmark from -1.1 to -0.04.
- V8 (max fixed cost 10 bps, fewer and larger positions) has the best net Sharpe (1.687) but a lower 2x-slippage Sharpe than V6 (1.292 against 1.461) and a higher cost drag than the variants with longer holds. V7 at 20 days (net 1.632, 2x 1.328, excess t v2 2.01) is the best 20-day variant.
- Eight variants per target were compared on the same research window, so the winner's Sharpe is an upward-biased selection among 16; the gate's PBO line is the check on that.

### Candidate and gate

Candidate (best net Sharpe at Rs 5 lakh, the rule written in the card): `g2-s1cost-y_c2o_10-40f8615b42` = V8 on `y_c2o_10`: exit rank 50, min hold 5, free extension, rebuy block 5, switch hurdle, score EMA 5, decile expected-return map, max fixed cost 10 bps.

All trial ids the candidate depends on (the GBM model trial) predate `effective_from` 2026-11-20, so the official verdict is gate v1 (`config/gate.toml`). Gate v2 is run for information only: the candidate's own portfolio trials were registered on 2026-11-19, before `effective_from`, so v2 would not select itself here.

Gate v1 (official), `stocksense validate g2-s1cost-y_c2o_10-40f8615b42 --gate-version 1`:

| # | line | value | threshold | result |
|---|---|---|---|---|
| 1 | mean_rank_ic | 0.0768 | >= 0.02 | PASS |
| 2 | ic_tstat | 26.5029 | >= 3 | PASS |
| 3 | net_sharpe | 1.6868 | >= 1 | PASS |
| 4 | net_sharpe_2x_slip | 1.2917 | >= 0.5 | PASS |
| 5 | excess_tstat | -0.0062 | >= 2 | FAIL |
| 6 | frac_positive_years | 0.8333 | >= 0.7 | PASS |
| 7 | dsr | 0.0000 (N = 119 trials) | >= 0.95 | FAIL |
| 8 | pbo | 0.5228 (16 trials, 2908 days) | <= 0.3 | FAIL |
| 9 | max_drawdown | 0.1985 | <= 0.25 | PASS |

Gate v2 (informational, same command with `--gate-version 2`):

| # | line | value | threshold | result |
|---|---|---|---|---|
| 1 | mean_rank_ic | 0.0768 | >= 0.02 | PASS |
| 2 | ic_tstat | 26.5029 | >= 3 | PASS |
| 3 | net_sharpe | 1.6868 | >= 1 | PASS |
| 4 | net_sharpe_2x_slip | 1.2917 | >= 0.5 | PASS |
| 5 | excess_tstat | 2.2617 | >= 2 | PASS |
| 6 | frac_positive_years | 0.8333 | >= 0.7 | PASS |
| 7 | dsr | 1.0000 (N = 119 trials) | >= 0.95 | PASS |
| 8 | pbo | 0.5228 (16 trials, 2908 days) | <= 0.3 | FAIL |
| 9 | max_drawdown | 0.1985 | <= 0.25 | PASS |

### Verdict

The cost work did what it was meant to do: against `g1-s1-y_c2o_10-cce12fb757` net Sharpe rises from 1.17 to 1.69 and the 2x-slippage Sharpe from 0.22 to 1.29, with turnover down from 0.19 to 0.10 per day and holds more than twice as long. The candidate still FAILS the official research gate (v1) on three lines: excess t (-0.01: against the raw equal-weight benchmark the book has no excess return), DSR (0.00 with the all-trials Sharpe variance and N = 119) and PBO (0.52: with the 16 variants the in-sample winner is a coin flip out of sample). Under v2 (informational) it fails only the PBO line, the excess t (2.26) and DSR (1.00) lines pass. The PBO result means the choice among these 16 variants cannot be shown to carry over out of sample, so this candidate does not go to the lockbox. It is the best available S1 policy and is kept as the paper-trading policy, labelled unvalidated; the 60-session paper gate decides.

`config/gate_v2.toml` `effective_from` needs no change. The official verdict for these trials is v1.

## 2. Model families (cards D1-D11)

Evaluation protocol of PLAN-G2 Part 4.2: test folds 2016-2025 (10 folds), base = `s1-gbm-y_c2o_10-e0f7276025` re-scored on the same window (mean IC 0.0747), S1 net Sharpe from the pre-C5 policy of D0.5 (k_max 10, no overlay, auto-resume 20, Rs 5 lakh).

| family | trial | mean IC | t | corr with base | orth IC (t NW) | blend gain (t NW) | S1 net Sharpe | 2x slip | turnover/day | admission |
|---|---|---|---|---|---|---|---|---|---|---|
| dens (DoubleEnsemble, K=6, decay 0.9) | `g2-dens-y_c2o_10-553d191061` | 0.0752 | 25.91 | 0.88 | 0.0218 (5.17) | 0.0020 (1.89) | 0.85 | 0.05 | 0.144 | ADMITTED (orthogonal IC) |

dens verdict:

- Selection on the inner validation of fold 2016 (6 configs, mean IC 0.0981 to 0.1017) chose K=6, decay 0.9; differences among configs were within 0.004, so the choice is weak. Walk-forward IC 0.0752 (t 25.9, NW t 10.5) is about equal to the GBM (0.0747 on the same window); by year it ranges 0.105 (2016) to 0.021 (2025).
- Its score is 0.88 correlated with the GBM and adds orthogonal IC 0.022 (t 5.2), so it passes the pre-registered admission rule on the orth line; the 50/50 blend gain is only +0.0020 (t 1.9), below the 0.003 line, so the benefit is small.
- Alone in S1 it gives net Sharpe 0.85 and 2x-slip 0.05 (trial `g2-s1-y_c2o_10-736a9989de`), weaker than the GBM book under the same pre-C5 policy; turnover is low (0.144). The 2025 fold (IC 0.021) is its weak spot. It enters the D12 ensemble as the stronger-baseline member.

## 3. Probability and stochastic-process options on the C5 winner (cards E1-E3, family `g2-s1cost`)

Run 2026-11-20 with `stocksense research s1-cost --grid e` (log `data_store/logs/g2_E_grid.log`). Same fixed scores, window, k_max 10, no overlay, auto-resume 20, u500 hysteresis flag and Rs 5 lakh reference as section 1. Base E0 is the C5 winner V8 (the identical spec, so the same trial `g2-s1cost-y_c2o_10-40f8615b42` was loaded, not re-run); each other variant changes exactly one thing in V8. HAR/GARCH forecasts are causal (HAR refit each January on data to the previous 31 December, GARCH refit at each month end on the last 1,000 returns, 1,352 names, cached in `data_store/cache/volforecast/`). Feature version 2 has NOT been built (card E9 does that once); nothing here touches `config/strategies.toml` or live code.

| variant | trial | net Sharpe | 2x-slip Sharpe | excess t (v1, raw) | excess t (v2, exposure-matched) | positive years | max DD | mean turnover/day | avg hold (sessions) | cost drag (annual, of capital) |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 = V8 (base) | `g2-s1cost-y_c2o_10-40f8615b42` | 1.687 | 1.292 | -0.01 | +2.26 | 0.83 | 0.199 | 0.104 | 11.6 | 0.293 |
| E1 V8 + `vol_model=har` | `g2-s1cost-y_c2o_10-ae4b593140` | 1.497 | 1.400 | -0.53 | +1.32 | 0.67 | 0.187 | 0.100 | 11.7 | 0.230 |
| E2 V8 + `vol_model=garch` | `g2-s1cost-y_c2o_10-3ecc1c301e` | 1.813 | 1.331 | +0.36 | +3.21 | 0.83 | 0.196 | 0.106 | 11.7 | 0.357 |
| E3 V8 + `er_map=decile_js` | `g2-s1cost-y_c2o_10-39cae29708` | 1.877 | 1.326 | +0.47 | +3.22 | 0.83 | 0.174 | 0.107 | 11.8 | 0.383 |
| E4 V8 + Kelly c = 0.25 | `g2-s1cost-y_c2o_10-0190db08db` | 1.641 | 1.246 | -0.11 | +1.95 | 0.92 | 0.192 | 0.108 | 11.6 | 0.294 |
| E5 V8 + Kelly c = 0.5 | `g2-s1cost-y_c2o_10-505b5a494f` | 1.753 | 1.161 | +0.26 | +2.18 | 0.83 | 0.195 | 0.110 | 11.7 | 0.347 |

Paired daily net-return difference against E0 (Newey-West t, lags 5): E1 -2.7% a year (t -1.54), E2 +1.8% (t +0.91), E3 +2.4% (t +1.18), E4 -0.5% (t -0.28), E5 +1.3% (t +0.71). Daily-return correlation with E0 is 0.83 to 0.86 for all five.

### Best variant and gate

Best by the card's rule (net Sharpe at Rs 5 lakh): E3 `g2-s1cost-y_c2o_10-39cae29708` (the same policy options as V8, expected-return map `decile_js`). `stocksense validate g2-s1cost-y_c2o_10-39cae29708 --gate-version 1` and `--gate-version 2` (the family now has 21 finished trials and the registry N = 127):

| # | line | value | threshold | v1 (official: its GBM trial predates `effective_from`) | v2 (informational) |
|---|---|---|---|---|---|
| 1 | mean_rank_ic | 0.0768 | >= 0.02 | PASS | PASS |
| 2 | ic_tstat | 26.5029 | >= 3 | PASS | PASS |
| 3 | net_sharpe | 1.8773 | >= 1 | PASS | PASS |
| 4 | net_sharpe_2x_slip | 1.3258 | >= 0.5 | PASS | PASS |
| 5 | excess_tstat | v1 0.4711, v2 3.2209 | >= 2 | FAIL | PASS |
| 6 | frac_positive_years | 0.8333 | >= 0.7 | PASS | PASS |
| 7 | dsr | 0.0000 (v1), 1.0000 (v2), N = 127 | >= 0.95 | FAIL | PASS |
| 8 | pbo | 0.2891 (21 trials, 2908 days) | <= 0.3 | PASS | PASS |
| 9 | max_drawdown | 0.1743 | <= 0.25 | PASS | PASS |

Gate v1: FAIL on lines 5 and 7 (the two lines B1 found mis-specified). Gate v2 (informational): PASS on all nine lines; the C5 winner V8 failed v2 line 8 (PBO 0.52 over 16 trials) and now passes it (0.289 over 21 trials, margin 0.011 from the 0.30 limit).

### Verdict

- E3 (James-Stein decile map) and E2 (GARCH vol) each beat the V8 base on BOTH net Sharpe and 2x-slip Sharpe (E3 1.877 / 1.326, E2 1.813 / 1.331, base 1.687 / 1.292). By the pre-written rule E3, the best on net, is recorded as a **candidate for the live policy only**; `config/strategies.toml` is unchanged (it still reads `er_map = "decile"` with V8's policy options) and no live code reads the new options.
- The evidence is weak. The gains over the base are 0.19 (E3) and 0.13 (E2) Sharpe, +2.4% and +1.8% a year, with paired NW t of 1.18 and 0.91: not distinguishable from zero. Five variants were compared against one base on the same window, so the best of them is an upward-biased pick. The PBO of 0.289 improved only because the five new trials are highly correlated with the existing family (daily-return correlation 0.83 to 0.86); it should not be read as stronger out-of-sample evidence. Treat E3 as "no harm, possibly a small gain", not as a validated improvement.
- E3 raises cost drag (0.383 a year against 0.293) and gross exposure (0.629 against 0.610) while lowering max drawdown (0.174 against 0.199); James-Stein shrinkage is mild here (about 25,000 rows per decile make the sampling variance small, and overlapping labels make it optimistic), so `decile_js` is close to `decile` and the difference is mostly in how the isotonic map is shaped.
- E1 (HAR vol) is the one variant that lowers net Sharpe (1.497) but has the highest 2x-slip Sharpe (1.400) and the lowest cost drag (0.230); its positive-years share fell to 0.67. E2 (GARCH) raises net Sharpe but not the 2x line by more than 0.04. HAR and GARCH are on a different scale than `vol_21` (calibrated by a close-to-close ratio for HAR; GARCH uses close-to-close returns directly) and the 10-day GARCH forecast is held for a month, so the stop and weights are sized somewhat differently from the base; no attempt was made to tune this.
- Kelly sizing (E4, E5) did not help: c = 0.25 is slightly worse than the base on both lines, c = 0.5 is better on net (1.753) but worst on 2x slip (1.161). The two c values do give different books (Kelly weights below the volatility-targeted room are the likely reason; not investigated), but there is no consistent gain over the score-over-vol sizing.
- Nothing here was chosen by looking at the lockbox; the lockbox stays closed.

### 3.2 Portfolio options of cards E4 and E6 on the E3 winner (family `g2-s1cost`)

Run 2026-11-20 with `stocksense research s1-cost --grid e2` (log `data_store/logs/g2_E2_grid.log`). Same fixed scores, window, k_max 10, no breadth overlay, auto-resume 20, u500 hysteresis flag and Rs 5 lakh reference as sections 1 and 3. The base R0 is the E3 winner (V8 options with `er_map=decile_js`; identical spec, so the same trial `g2-s1cost-y_c2o_10-39cae29708` was loaded, not re-run). Each other variant changes only the portfolio option named: R1 `regime_overlay` (gross x 0.5 when the filtered 3-state HMM bear probability is above 0.6; the probability is 0.6 or more on about 19% of the sessions of 2015-2025, parameters refit each January on data to the previous 31 December), R2 `stop_model=evt` (stop at `max(8%, ES99 x vol x sqrt(h))` instead of `max(8%, 3.0 x vol x sqrt(h))`; the pooled-by-tercile GPD ES99 multiplier averages 3.83 vol units, range 2.99 to 4.58, so the extreme-value stop is up to about 50% wider than the fixed one), R3 both. All three new trials were registered before any was run. Nothing here touches `config/strategies.toml` or live code.

| variant | trial | net Sharpe | 2x-slip Sharpe | excess t (v1, raw) | excess t (v2, exposure-matched) | positive years | max DD | mean turnover/day | avg hold (sessions) | cost drag (annual, of capital) | mean gross | paired net difference vs R0 (NW t, lags 5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 = E3 winner (base) | `g2-s1cost-y_c2o_10-39cae29708` | 1.877 | 1.326 | +0.47 | +3.22 | 0.83 | 0.174 | 0.107 | 11.8 | 0.383 | 0.629 | - |
| R1 R0 + regime overlay | `g2-s1cost-y_c2o_10-ea16321659` | 1.673 | 1.287 | -0.27 | +1.76 | 0.83 | 0.190 | 0.103 | 11.7 | 0.272 | 0.605 | -3.7% a year (t -1.94) |
| R2 R0 + EVT stop | `g2-s1cost-y_c2o_10-cd28c9d82b` | 1.927 | 1.263 | +0.62 | +3.20 | 0.83 | 0.174 | 0.108 | 11.8 | 0.421 | 0.640 | +0.7% a year (t +0.48) |
| R3 R0 + both | `g2-s1cost-y_c2o_10-52def95f1a` | 1.611 | 1.191 | -0.52 | +1.94 | 0.75 | 0.170 | 0.097 | 11.6 | 0.238 | 0.565 | -5.1% a year (t -2.66) |

Daily-return correlation with R0: R1 0.85, R2 0.91, R3 0.85. Exits by reason (all folds, trades): stop 32 (R0), 30 (R1), 16 (R2), 13 (R3); rank_decay 1,118 / 982 / 1,163 / 921; horizon 841 / 805 / 867 / 830.

### Best variant and gate (3.2)

Best by the card's rule (net Sharpe at Rs 5 lakh): R2 `g2-s1cost-y_c2o_10-cd28c9d82b` (stored as the run's candidate). `stocksense validate g2-s1cost-y_c2o_10-cd28c9d82b --gate-version 1` and `--gate-version 2` (registry N = 130, family has 24 finished trials):

| # | line | value | threshold | v1 (official: its GBM trial predates `effective_from`) | v2 (informational) |
|---|---|---|---|---|---|
| 1 | mean_rank_ic | 0.0768 | >= 0.02 | PASS | PASS |
| 2 | ic_tstat | 26.5029 | >= 3 | PASS | PASS |
| 3 | net_sharpe | 1.9266 | >= 1 | PASS | PASS |
| 4 | net_sharpe_2x_slip | 1.2634 | >= 0.5 | PASS | PASS |
| 5 | excess_tstat | v1 0.6153, v2 3.2026 | >= 2 | FAIL | PASS |
| 6 | frac_positive_years | 0.8333 | >= 0.7 | PASS | PASS |
| 7 | dsr | 0.0000 (v1), 1.0000 (v2), N = 130 | >= 0.95 | FAIL | PASS |
| 8 | pbo | 0.2162 (24 trials, 2908 days) | <= 0.3 | PASS | PASS |
| 9 | max_drawdown | 0.1741 | <= 0.25 | PASS | PASS |

Gate v1: FAIL on lines 5 and 7 (the two lines B1 found mis-specified), as for every earlier variant; gate v2 (informational): PASS on all nine lines.

### Verdict (3.2)

- **E4 regime overlay: no.** Halving gross when the filtered bear probability exceeds 0.6 lowers net Sharpe from 1.877 to 1.673 and the 2x-slip Sharpe from 1.326 to 1.287, raises max drawdown (0.190 against 0.174) and costs 3.7% a year (paired t -1.94, the only paired difference near significance in this grid); its v2 excess t falls to 1.76, below the gate's 2. Combined with the EVT stop (R3) it is the worst variant (net 1.611, paired t -2.66, positive years 0.75). The overlay is not adopted; the S1 breadth overlay and this one both reduce exposure in weak markets and both cost return on this window, which is consistent with the book's own volatility targeting already shrinking gross when risk rises.
- **E6 EVT stop: no clear effect.** The extreme-value stop is wider than the fixed one (ES99 multiplier 3.0 to 4.6 against the fixed 3.0), so stops fire half as often (16 against 32 across 2,120 trades). Net Sharpe rises by 0.05 (1.927 against 1.877; +0.7% a year, paired t +0.48), the 2x-slip Sharpe falls by 0.06 (1.263 against 1.326), cost drag rises (0.421 against 0.383) and max drawdown is the same (0.174). That is not distinguishable from zero. By the pre-written rule R2 is recorded as a candidate for the live policy only; `config/strategies.toml` is unchanged and no live code reads `stop_model` or `regime_overlay`. Three variants were compared against one base on the same window, so the best-of pick is upward biased.
- The feature-based items are NOT evaluated here: the E4 regime and change-point features (`regime_p_bear`, `regime_p_bull`, `cp_prob`, `run_length_mean`), the E5 OU and pair features (`ou_sscore`, `ou_halflife`, `resid_rev_5`, `pair_z`), the E7 event features (`ann_intensity_2/10`, eight category counts, `result_drift_proxy`) and E1's HAR/GARCH features need the single feature-version-2 build and a `g2-feat` GBM trial, which is card E9. Their modules (`features/regime.py`, `features/ou.py`, `features/events_v2.py`) are not in `FEATURE_MODULES` yet; their unit and truncation tests pass.
- Nothing here was chosen by looking at the lockbox; the lockbox stays closed.
