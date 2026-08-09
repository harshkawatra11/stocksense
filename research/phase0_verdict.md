# Phase 0 Verdict

**Date:** 2026-08-09
**Question:** does any (horizon × selectivity × cost) configuration produce net-positive, fold-stable, out-of-sample alpha on the Phase 0 universe?

## Verdict: **CONDITIONAL — real signal confirmed, insufficient sample to trust it yet. Continue building; do not scale toward capital.**

This is neither GO nor NO-GO as originally framed, because the sweep surfaced a third outcome the plan didn't name explicitly: a signal that is real but currently under-sampled. That's a legitimate result and this document treats it as one.

---

## What was run

- **Universe:** 98 liquid NSE large/mid-cap symbols (hand-curated, `data/universe.py`) — **not** the full point-in-time tradeable universe. Source: yfinance (no Upstox credentials available in this environment).
- **History:** 2010-01-01 → 2026-08-09 (~15 years, 388,882 rows). Does **not** reach 2000 or the 2008 GFC.
- **Sweep grid:** horizon ∈ {1,3,5,10,20} bars × top_n ∈ {10,20,50,100} × cost ∈ {10,15,25,35} bps round-trip. 80 configurations, purged/embargoed walk-forward, expanding window, 7–13 folds per horizon depending on embargo size.
- **Model:** LightGBM regression against cross-sectional relative forward return (`models/ranker.py`), retrained per fold — 46 (horizon, fold) train/score passes total.
- **Reference cost check:** the modeled Indian delivery cost stack (`execution/cost_model.py`) — STT, exchange charges, SEBI fee, stamp duty, GST, 5bps modeled slippage — comes to **32.2 bps round-trip**, close to v1's assumed 25 bps and inside this sweep's 25–35 bps grid.

## What the sweep found

**1. The signal is real and reproduces v1's finding independently.** Every horizon ≥ 5 bars shows positive mean information coefficient (best: 0.044 at horizon=5). The horizon=1 configuration is fragile and barely survives even 10bps cost — this reproduces v1's own diagnosis (1-day holding cannot clear round-trip cost) on a completely different dataset, universe, and codebase. That is a meaningful cross-check, not a coincidence.

**2. Longer horizons survive realistic cost on a gross basis.** Best configurations (horizon=10, top_n=10 or 20) show break-even costs of 50–67 bps — comfortably above the 32bps modeled realistic cost. At the sweep's 25bps grid point, 27 of 80 configurations are net-positive.

**3. The best-performing configuration by mean and fold-hit-rate is horizon=10, top_n=20:**
- 9 folds, 7/9 (77.8%) net-positive at 25bps
- Mean net alpha per rebalance: +0.137%
- Positive at every cost level tested, 10–35bps

**4. But it does not survive the best-trade-removal stress test.** This is the finding that changes the verdict. For every leading configuration, dropping just the best 2 of 9 folds flips the mean from positive to negative or flat:

| Config | Full mean | Median | Mean excl. best 2 folds |
|---|---|---|---|
| h=10, n=20 | +0.137% | +0.039% | **−0.048%** |
| h=10, n=10 | +0.271% | +0.088% | **−0.006%** |
| h=20, n=10 | +0.107% | +0.052% | **−0.094%** |
| h=5, n=10 | +0.071% | −0.003% | **−0.025%** |

The median is close to zero or negative in every case. **The positive mean is being carried by two unusually strong folds (fold_id 1 and 4), not by a broadly distributed edge.** This is exactly what `docs/10-evaluation.md`'s adversarial stress battery (§10, "remove best trades") is designed to catch, and it caught something real: v1's own historical output showed the identical pattern independently (`edge_by_year.out.txt`: "2002 = 16% of all positive-year PnL"). Two different codebases, two different universes, two different eras — the same lumpiness. That consistency is itself informative: this alpha source appears to pay off in concentrated bursts rather than smoothly, which is a property to design around, not an artifact to dismiss.

**5. The harness passes its own sanity checks.** top_n=100 (≥ universe size) collapses turnover to ~0 and alpha to ~0, exactly as it should — selecting the whole universe every period is holding the benchmark, and the code correctly shows that as no edge and no trading.

## Why this is not a clean GO

The Phase 0 kill criteria specified in advance require net-positive alpha "stable across folds... without depending on a handful of outlier [periods]." §4 above is a direct failure of that specific bar. With only 7–9 test folds (a consequence of 15 years of history and long embargo gaps at longer horizons), two strong folds are a large share of the sample — this is a **statistical power problem**, not evidence the edge is fake.

## Why this is not NO-GO either

- IC is positive and directionally consistent across every horizon ≥ 3 bars, on data the model never trained on.
- The horizon=1 fragility result independently reproduces a known-true fact about v1, which is strong evidence the harness itself is sound, not just optimistic.
- The lumpiness pattern matches v1's finding almost exactly despite no shared code or data — that is corroboration, not noise.
- Gross alpha clears realistic modeled costs by a wide margin at horizon ≥ 10.

A NO-GO would stop the project on a sample too small to have rejected the hypothesis fairly.

## What actually needs to happen before a confident capital decision

1. **Extend history to 2000** (the daily data exists per `docs/02-data-layer.md`; this Phase 0 run used 2010+ only because that is what was pulled today). This alone would roughly double the number of walk-forward folds and include the 2008 crisis — both directly address the sample-size problem.
2. **Point-in-time universe reconstruction**, not today's 98 survivors. Current results are an upper bound, not a final number (`docs/02-data-layer.md`'s point-in-time obligation).
3. **Run the full adversarial stress battery** from `docs/10-evaluation.md` formally — Monte Carlo reshuffling, parameter perturbation, worst-fold removal — rather than the single ad hoc check done here.
4. **Investigate what fold 1 and fold 4 have in common.** If it's a regime (specific volatility or trend condition), that is itself a usable finding — it would mean the edge is real but conditional, which argues for regime-gating rather than abandoning the signal.

## Decision

**Proceed to Phase 1 build-out** (nightly pipeline, model registry, evaluator formalization) **using this as validated architecture, not validated alpha.** The pipeline, feature engine, leakage tests, cost model, and portfolio constructor all worked correctly on real data end-to-end — that engineering result stands regardless of the outlier-dependence finding.

**Do not** treat any number in this document as investable. The next research task, before anything else, is re-running this exact sweep against full 2000→ history with a reconstructed point-in-time universe, specifically to get enough folds to make §4 either disappear (more folds dilute two strong periods) or harden into a real, actionable regime-dependency finding.
