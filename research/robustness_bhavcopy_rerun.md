# Robustness Check: `full_pit` h=10/n=10 (+1.672%/rebalance)

**Date:** 2026-08-27
**Status:** disclosed robustness check, NOT a re-tune. No threshold in
`evaluation/gate.py` or `research/gate_criteria_preregistration.md` is
touched or reconsidered here. `research/verdict_bhavcopy_rerun.md` remains
the pre-registered result of record regardless of what this document finds.

## Why this exists

`verdict_bhavcopy_rerun.md`'s `full_pit`/h=10/n=10 result (+1.672% mean net
alpha/rebalance, 25 folds, one-sided binomial p=0.0001) is 3.5–5.5x the
`large`-cap-band magnitude on the same pipeline, and this codebase has
twice already produced a "too good" number that turned out to be a data
artifact (Run 2's ADANIENT adjustment-factor bug; the first run of this
exact sweep, which hit 10–25x Run 4's alpha before the halted-symbol
reopening bug was found and fixed by
`data/liquidity.py:segment_symbols_by_trading_gap`). Per the project's own
precedent, an unusually large number gets checked before it gets trusted.
This is that check for the surviving result.

## What was tested

**1. Is the alpha driven by a small number of extreme single-name folds?**

No. Read `research/bhavcopy_rerun_fold_results.csv` directly for
`cap_band='full_pit', horizon_bars=10, top_n=10, cost_bps=25.0` (25 folds):
`alpha_net` ranges from −0.23% to +4.55%, mean +1.672%, std 1.17%. The
distribution is smooth and unimodal — no fold is an order of magnitude away
from its neighbors, unlike the ADANIENT-class bug's signature (a single
fold with an ~8x price discontinuity). 22 of 25 folds are positive. This is
not the failure pattern found before.

**2. Where does `full_pit`'s alpha actually come from, relative to the
named cap bands?**

This is the real finding. `universe_pit.CAP_BANDS` partitions the
turnover-ranked, point-in-time-tradeable universe as:

| band | turnover-rank percentile | symbols on 2025-06-30 |
|---|---|---|
| `large` | 0.80 – 1.00 | 352 |
| `mid` | 0.50 – 0.80 | 527 |
| `small` | 0.15 – 0.50 | 615 |
| **(uncovered)** | **0.00 – 0.15** | **263** |
| `full_pit` | everything ≥ ₹50L/day turnover & ≥ ₹5 price | 1,757 = sum of all four |

**`full_pit` includes an entire bottom-15%-by-turnover segment that none of
`large`/`mid`/`small` individually cover, and that segment was never its
own row in `verdict_bhavcopy_rerun.md`'s table.** Measured on the same date:
this segment's median 60-day average turnover is **₹85 lakh/day**, roughly
**5x less liquid** than `small`'s median (₹4.24cr/day), with a lower median
price too (₹117 vs ₹258).

The full comparison table (25bps cost, all horizons/top_n) shows a
monotonic pattern — alpha rises as the universe gets less liquid, and
`full_pit` consistently *exceeds even `small`*, which it should not do if
it were simply a turnover-weighted blend of the three named bands:

| cap_band | h | n | mean net alpha | mean gross alpha | mean IC | mean hit rate |
|---|---|---|---|---|---|---|
| large | 10 | 10 | +0.060% | +0.214% | 0.016 | 0.55 |
| mid | 10 | 10 | +0.560% | +0.734% | 0.059 | 0.61 |
| small | 10 | 10 | +0.694% | +0.869% | 0.094 | 0.58 |
| **full_pit** | 10 | 10 | **+1.672%** | **+1.851%** | 0.087 | 0.65 |
| large | 20 | 20 | +0.225% | +0.371% | 0.021 | 0.69 |
| mid | 20 | 20 | +0.401% | +0.574% | 0.079 | 0.63 |
| small | 20 | 20 | +0.889% | +1.072% | 0.112 | 0.64 |
| **full_pit** | 20 | 20 | **+2.172%** | **+2.359%** | 0.096 | 0.72 |

The pattern "less liquid → more measured alpha" is consistent with two very
different explanations, and this check cannot distinguish them without a
full re-run (deliberately not done here, to stay a disclosed check rather
than a re-tune):

- **(a) Genuine size/illiquidity premium.** Small, thinly-covered names are
  a classical source of real (if fragile) equity anomalies. Plausible, and
  consistent with `mid`/`small` also passing the gate on their own.
- **(b) An execution-realism gap the backtest doesn't model.** The gate's
  cost assumption is a flat bps figure (`compute_charges`), the same for a
  ₹4cr/day name and an ₹85L/day name. Real slippage, market impact, and
  achievable fill size scale *worse* than linearly as liquidity drops — the
  exact concern `execution/fill_model.py`'s participation cap exists to
  catch for the (unrelated, already-retired) intraday track, but the
  `full_pit`/`mid`/`small` daily/monthly backtest has no equivalent check.
  Notably, `full_pit`'s mean IC (0.087) is *lower* than `small`'s (0.094)
  despite `full_pit`'s much higher mean alpha — the outperformance is not
  coming from better rank correlation, it's coming from larger realized
  return magnitudes in the tail, which is exactly the shape an
  under-costed illiquidity effect would produce.

**3. Halt-segment / label-boundary check.** `full_pit`'s universe is the
widest and therefore the most exposed to `data/liquidity.py`'s
halt-segmentation fix (the bug that inflated the sweep's *first* run).
Confirmed the fix is still active and unmodified in the current code path
(`_load_and_prepare` in `research/bhavcopy_rerun_sweep.py` calls
`segment_symbols_by_trading_gap` unconditionally) and that this document's
own fold-smoothness check (point 1) shows none of the extreme-single-fold
signature that bug originally produced.

## What this does NOT establish

This check does not re-run the backtest with a tighter liquidity floor —
that would be a new experiment requiring its own pre-registration before
running, not a robustness check on an existing, already-judged result. It
also does not measure real slippage on the excluded bottom-15% segment,
which only live paper trading (Phase J2) can actually do.

## Conclusion

The `full_pit`/h=10/n=10 result is **not** exhibiting the ADANIENT-class
data-corruption signature the project has twice found before — the fold
distribution is smooth, and the halt-segmentation fix is active. But it
**is** disproportionately sourced from a genuinely more illiquid segment
than any of `mid`/`small` individually cover, and the gate's flat-bps cost
model has no mechanism to detect whether that segment's realistic execution
cost is higher than assumed. This is a real, disclosed limitation, not a
disqualification: `mid` and `small` also independently clear the gate at
smaller but still-real magnitudes (+0.56% and +0.69%/rebalance at h=10),
and neither depends on the uncovered bottom-15% segment at all.

**Recommendation carried into J2 (paper trading):** the paper account
should be run primarily on `mid`/`small` (or a `full_pit` variant with an
explicit, disclosed liquidity floor at the `small` band's lower edge), not
on unrestricted `full_pit`, until real fills demonstrate the bottom-15%
segment's alpha survives contact with actual achievable execution — exactly
the question paper trading exists to answer. The currently-live model
(`cross_sectional_ranker_h10_n10_20260823T164657`) is trained on
unrestricted `full_pit`; whether to keep it live, or promote a
`mid`/`small`-trained candidate alongside it for comparison, is a decision
for J2's shadow period, not for this document to make.
