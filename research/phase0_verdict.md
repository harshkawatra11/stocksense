# Phase 0 Verdict

**Date:** 2026-08-09 (revised same day — extended history run)
**Question:** does any (horizon × selectivity × cost) configuration produce net-positive, fold-stable, out-of-sample alpha on the Phase 0 universe?

## Verdict: **GO. Proceed to Phase 1.** The monthly-horizon signal is real, cost-robust with a wide margin, and — critically — now survives the best-trade-removal stress test that the initial (2010–2026) run failed.

This document was written twice in one session. The first pass (preserved below as "Run 1") returned a conditional result: real signal, insufficient sample to trust the mean over the median. The fix identified at the time — extend history back to 2000 to get more walk-forward folds — was executed immediately after, and it resolved the open question. This is Run 2's verdict; Run 1 is kept for the record because the reasoning that got here matters as much as the destination.

---

## Run 2: full 2000–2026 history

- **Universe:** same 98-symbol liquid NSE set as Run 1 — still **not** the full point-in-time tradeable universe. This limitation is unchanged and still caveats every number below (see "What is still not proven").
- **History:** 2000-01-03 → 2026-08-07, confirmed via yfinance (which does reach back to 2000 for these symbols — verified directly, not assumed). 558,438 rows, up from 388,882 in Run 1.
- **Effect on fold count:** roughly 1.6–1.8× more folds at every horizon, exactly as predicted:

| Horizon (bars) | Run 1 folds (2010–2026) | Run 2 folds (2000–2026) |
|---|---|---|
| 1 | 13 | 22 |
| 3 | 12 | 21 |
| 5 | 11 | 19 |
| 10 | 9 | 16 |
| 20 | 7 | **11** |

## What changed

**The best configuration shifted from horizon=10 to horizon=20** (roughly one month of trading bars), and it is now a substantially stronger result than anything in Run 1:

| Metric (h=20, top_n=20, 25bps cost) | Run 1 (n/a — too few folds) | Run 2 |
|---|---|---|
| Folds | — | **11** |
| Mean net alpha / rebalance | — | **+0.765%** |
| Median | — | **+0.687%** |
| % folds net-positive | — | **9/11 (82%)** |
| Mean IC | — | **0.052** |
| Break-even cost | — | **171 bps** |

**The best-trade-removal stress test now passes.** This is the decisive change. Dropping the 2 best of 11 folds:

| Config | Full mean | Median | Mean excl. best 2 folds |
|---|---|---|---|
| h=20, n=20 | +0.765% | +0.687% | **+0.484%** — still solidly positive |
| h=20, n=10 | +1.056% | +0.594% | **+0.304%** — still positive |
| h=5, n=10 | +0.144% | +0.155% | **+0.083%** — still positive |
| h=10, n=20 | +0.196% | +0.019% | −0.020% — still fragile, consistent with Run 1 |

Compare this to Run 1, where every leading config went negative under the identical test. The h=20 signal does not have that problem: excluding its two best months, it still clears zero by a wide margin.

**Per-fold detail for the winning configuration (h=20, n=20, 25bps) makes the case directly** — the edge is now distributed across most of the sample, not concentrated in one or two windows:

```
fold  alpha_net   ic       hit_rate
 0    +2.32%      0.139    0.83
 1    +0.33%      0.042    0.75
 2    +1.73%      0.060    0.83
 3    +1.22%      0.062    0.92
 4    +0.48%      0.058    0.58
 5    +0.24%      0.099    0.58
 6    +0.69%      0.058    0.58
 7    +0.99%      0.064    0.83
 8    +1.06%      0.048    0.58
 9    −0.59%     −0.038    0.50
10    −0.06%     −0.024    0.83
```

Nine of eleven folds are individually positive, spanning a 26-year sample that now includes the 2008 crisis and the COVID crash inside the training history of later folds. Only fold 9 is a clear loser; fold 10 is roughly flat. That is a materially different shape from Run 1's "two outlier months carry the whole result."

**Break-even cost is now large enough to matter.** 171 bps for h=20/n=20, 201 bps for h=20/n=10 — five to six times the 32bps realistic modeled cost. Run 1's best margin was roughly 50–67bps. This gap is now big enough to absorb a materially wrong slippage assumption and still clear.

## What stayed the same (and still matters)

- **The horizon=1 fragility finding reproduces again** on the larger sample, still barely clearing 10bps cost — the same conclusion as Run 1 and as v1's original diagnosis. Three independent confirmations now (v1, Run 1, Run 2) of the same fact: this alpha source cannot survive same-day round-trip costs, regardless of how much history is used to measure it.
- **top_n=100 sanity check still behaves correctly** — turnover and alpha both collapse toward zero when top_n exceeds the universe size, confirming the harness has not changed behavior in a way that would fabricate an edge.
- **The universe is still 98 hand-picked, currently-liquid, currently-listed symbols.** This is unchanged from Run 1 and is not fixed by adding more history — it is fixed by point-in-time universe reconstruction, which is separate work.

## What is still not proven

Being direct about the remaining gap between this result and something investable:

1. **Survivorship bias.** All 98 symbols are today's liquid large/mid-caps. The sample says nothing about names that delisted, went illiquid, or fell out of the index over 26 years — and those are disproportionately likely to have been the *bad* outcomes, meaning realized results on the true point-in-time universe are plausibly weaker than this. This is the single largest remaining source of overstatement and the next thing to fix, per `docs/02-data-layer.md`'s point-in-time obligation.
2. **Slippage is modeled, not replayed.** No order-book fidelity exists in this environment. 171bps of headroom is large enough to absorb a materially wrong slippage assumption, but "large enough to absorb being wrong" is not the same as "verified."
3. **11 folds is a large improvement over 7, not a large number in absolute terms.** The per-fold table above is the actual evidence, and readers should look at it rather than trust the summary statistic alone.
4. **Only one ablation has been run** (best-trade removal). The full battery from `docs/10-evaluation.md` §10 — Monte Carlo reshuffling, parameter perturbation ±20%, latency injection, worst-trade removal, universe perturbation — has not yet been executed formally.
5. **Fold 9's loss is unexplained.** Worth understanding before this goes further — if it corresponds to a specific regime (2018 IL&FS stress, or similar), that is a usable finding about when the strategy fails, in the same spirit as the F&O contradiction checks planned for the Investigator layer.

## Decision

**GO.** Proceed to Phase 1: nightly pipeline, model registry, gate, evaluator formalization — built around a **monthly rebalance horizon (h≈20 trading bars)**, not the daily cadence originally implied by v1's design. This is now the headline architectural consequence of Phase 0: StockSense is a monthly-rebalance quant system, not a daily one, because that is where the evidence says the edge actually survives costs.

Before any capital, real or paper, touches this signal: point-in-time universe reconstruction (item 1) and the full stress battery (item 4) are the two specific, named prerequisites — not vague future work, but the literal next research tasks.
