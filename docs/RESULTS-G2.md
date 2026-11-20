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
