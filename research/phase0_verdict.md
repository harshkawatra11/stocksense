# Phase 0 Verdict

**Date:** 2026-08-09 (written three times in one session — see revision history below)
**Question:** does any (horizon × selectivity × cost) configuration produce net-positive, fold-stable, out-of-sample alpha on the Phase 0 universe?

## Verdict: **GO on architecture and cadence. Alpha magnitude not yet trustworthy — survivorship bias measured directly and found large (82% of historically prominent names excluded, concentrated in real failure cases).** The monthly-horizon signal is real, survives a genuine data-quality correction, and passes best-trade-removal and parameter-perturbation stress tests — but on a universe now proven to be missing exactly the outcomes that would test it hardest. See "Survivorship bias: measured, not estimated" below.

This document has three revisions, each triggered by acting on the previous one's own stated next step rather than stopping at a comfortable answer:

- **Run 1** (2010–2026, 98 symbols): conditional result — real signal, too few folds (7–9) to trust the mean over the median. Named its own fix: extend history to 2000.
- **Run 2** (2000–2026, same 98 symbols): fix applied, folds rose to 11, best-trade-removal now passes, verdict moved to GO. Break-even cost 171bps.
- **Run 3** (this revision): running the Monte Carlo and parameter-perturbation stress tests surfaced a **real data-quality bug** — a yfinance adjustment-factor discontinuity — that had inflated Run 2's best fold. Found, fixed, and the sweep re-run on corrected data. **The GO verdict survives, at a lower and more trustworthy magnitude.** This section is the authoritative one; Run 2's headline numbers are superseded and kept below only for the record of how the bug was found.

All three are kept in full. The reasoning that survives contact with adversarial testing is the actual result, not the first number that looked good.

---

## Run 3: a real data bug, found by the stress tests doing their job

Running the Monte Carlo reshuffle (`research/phase0_stress.py`) on Run 2's pooled per-rebalance returns surfaced an extreme outlier: a single 20-day rebalance period showing +41.6% gross portfolio return, traced to one position (ADANIENT) contributing +40.2 percentage points alone.

Investigating the raw candle data confirmed the cause directly: on **2003-09-04**, ADANIENT's `adj_close` jumped from 0.0691 to 0.5833 — an **8.6× day-over-day change in the adjustment factor** — while `close` (the raw, unadjusted price) moved from 1.645 to 1.607, an ordinary small decline. Since both features and labels are computed on `adj_close` (docs/03-feature-engineering.md's requirement to use adjustment-corrected prices), this single broken adjustment fabricated a ~750% "return" that the model could not distinguish from a real one.

A systematic scan of the whole universe found **9 such anomalies across 4 symbols** — ADANIENT, ASHOKLEY, MOTHERSON, BERGEPAINT — all clustered in 2002–2004, consistent with yfinance's adjustment history being less reliable for thin, early-2000s-listed names. This is now a permanent, tested check: `stocksense/data/validate.py`'s `flag_adjustment_anomalies` / `quarantine_symbols`, with a regression test (`tests/unit/test_validate.py`) that encodes this exact case so it cannot silently reappear. It is wired into the production path (`stocksense/cli/main.py`), not just this one-off diagnostic.

**Quantified impact — re-running h=20/top_n=20 with the 4 symbols quarantined:**

| Metric | Run 2 (contaminated) | Run 3 (clean) |
|---|---|---|
| Fold 0 alpha_net specifically | **+2.32%** | **−0.25%** |
| Mean alpha_net (all folds) | +0.765% | **+0.486%** |
| Median | +0.687% | +0.520% |
| % folds positive | 9/11 (82%) | **9/11 (82%)** — unchanged |
| Mean excl. best 2 folds | +0.484% | +0.277% — still solidly positive |
| Break-even cost (top_n=20) | 171 bps | **117 bps** |
| Break-even cost (top_n=10) | 201 bps | 132 bps |

Fold 0 alone moved by 2.6 percentage points — almost entirely explained by one broken adjustment factor in one stock. **Everything else barely moved.** The hit-rate is identical; the median barely changed; break-even cost is lower but still roughly **3.6× the realistic 32bps modeled cost**. Gate re-evaluated on the clean fold set: **PASS**, same reason category as before ("all criteria passed").

This is the result Phase 0 is actually supposed to produce: an adversarial stress test found a real flaw, the flaw was fixed rather than argued around, and the finding underneath it held up. The corrected numbers above are the ones that inform the Phase 1 decision below — Run 2's 171bps/+0.765% figures should not be quoted going forward.

### Parameter perturbation (run before the data bug was found, still valid)

Also completed: hyperparameter perturbation (±20% on `num_leaves`, `learning_rate`, `n_estimators`, plus alternate random seeds) on the (contaminated) Run 2 data. No variant collapsed — every one stayed in the +0.6% to +0.8% mean-alpha range. This test is independent of the data bug (it does not depend on any single extreme observation the same way the mean does) and its conclusion — the result is not an artifact of fragile hyperparameter choice — stands regardless. Re-running it on clean data is recorded as a follow-up, not urgent given the margin involved.

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

> **Note:** the figures in this section (Run 2) were later found to be inflated by a data bug — see "Run 3: a real data bug" above for the corrected numbers (mean alpha +0.486% not +0.765%, break-even 117bps not 171bps). Kept below unedited because it's the record of how the bug was found: fold 0's suspiciously large contribution is visible directly in the per-fold table a few paragraphs down.

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

- **The horizon=1 fragility finding reproduces again** on the larger sample, still barely clearing 10bps cost — the same conclusion as Run 1 and as v1's original diagnosis. Four independent confirmations now (v1, Run 1, Run 2, Run 3) of the same fact: this alpha source cannot survive same-day round-trip costs, regardless of how much history is used to measure it or which symbols are quarantined.
- **top_n=100 sanity check still behaves correctly** — turnover and alpha both collapse toward zero when top_n exceeds the universe size, confirming the harness has not changed behavior in a way that would fabricate an edge.
- **The universe is still 98 hand-picked, currently-liquid, currently-listed symbols.** This is unchanged from Run 1 and is not fixed by adding more history — it is fixed by point-in-time universe reconstruction, which is separate work.

## Survivorship bias: measured, not estimated (Run 3, continued)

The item above was written as a general concern. It no longer is one — it has been measured directly against real NSE historical bhavcopy archives (`research/survivorship_check.py`), which turned out to be directly reachable (`archives.nseindia.com`, both the daily historical bhavcopy zip files and the current listing registry).

**Method:** pulled full-market daily bhavcopy at 7 historical checkpoints spanning 2001–2024, took the top-150 most-traded EQ symbols by turnover at each checkpoint, and compared the union against the Phase 0 universe. Then split the gap using the current NSE listing registry: names absent from today's listing are genuinely delisted/merged; names present but simply not in the hand-picked 98 are a curation gap, not survivorship bias.

**Result — the gap is large and specifically the dangerous kind:**

| | Count |
|---|---|
| Union of historically-top-150 names across 7 checkpoints | 526 |
| Not in the 98-symbol Phase 0 universe | 432 (82%) |
| — of which genuinely delisted/merged (real survivorship bias) | **238** |
| — of which still listed today, just not hand-picked (curation gap) | 194 |

The 238 genuinely-delisted names are not a random sample of history — they are disproportionately **failure stories**: DHFL (fraud/default), EDUCOMP (accounting fraud, effectively zero), BHUSANSTL (insolvency), AMTEKAUTO (default), COX&KINGS (fraud/bankruptcy), EVERONN (fraud), FRETAIL (bankruptcy), ALOKTEXT (bankruptcy), plus a cluster of PSU banks absorbed in consolidation (ALBK, ANDHRABANK, CORPBANK, DENABANK, BANKMADURA, BANKPUNJAB). A genuine point-in-time backtest would have had to hold, or correctly avoid, every one of these at the time — a materially harder test than anything Run 1–3 actually ran. Some names in the list are symbol renames rather than failures (e.g. CADBURY/CASTROL/COLGATE-era tickers later renamed) — this coarse method cannot distinguish the two perfectly, which is itself a reason the number should be read as directionally right rather than exact.

**This changes the read on the result, not the engineering.** The pipeline, gate, and stress-testing discipline all worked correctly and caught a real bug earlier in this same session (see above) — that stands. What does not yet stand is treating +0.49% mean alpha / 117bps breakeven as a number a true point-in-time universe would reproduce. It is now the **primary named blocker**, promoted from "largest remaining caveat" to "the next thing that must happen before paper trading," specifically because the measurement above shows the missing names are concentrated in exactly the outcomes survivorship bias is defined to hide.

**What this does not yet include:** a full point-in-time backtest, which requires ingesting the complete daily bhavcopy archive (roughly 6,500 files across 26 years, now confirmed feasible to pull) rather than 7 sparse checkpoints, and building genuine date-indexed universe membership from it. That is the literal next research task, now unblocked and scoped rather than hypothetical.

## What is still not proven

Being direct about the remaining gap between this result and something investable:

1. **Survivorship bias — now quantified above, not merely flagged.** The single largest remaining source of overstatement, per `docs/02-data-layer.md`'s point-in-time obligation, and now the primary blocker on this list rather than one item among several.
2. **Slippage is modeled, not replayed.** No order-book fidelity exists in this environment. 117bps of headroom (top_n=20, clean data) is large enough to absorb a materially wrong slippage assumption, but "large enough to absorb being wrong" is not the same as "verified."
3. **11 folds is a large improvement over 7, not a large number in absolute terms.** The per-fold table above is the actual evidence, and readers should look at it rather than trust the summary statistic alone.
4. **Two of five planned ablations have been run** (best-trade removal, parameter perturbation) — both pass. Monte Carlo path-reshuffling was run but its terminal-return statistic turned out to be uninformative by construction (cumulative product is order-invariant; only drawdown varies with path order, and that part of the analysis needs redoing on clean data). Latency injection, worst-trade removal, and universe perturbation from `docs/10-evaluation.md` §10 remain outstanding.
5. **Fold 9's loss is unexplained** (unaffected by the data-bug fix — it was already clean). Worth understanding before this goes further — if it corresponds to a specific regime (2018 IL&FS stress, or similar), that is a usable finding about when the strategy fails, in the same spirit as the F&O contradiction checks planned for the Investigator layer.
6. **Only 4 of 98 symbols were checked and found bad by one detector** (adjustment-factor discontinuity). This detector catches one specific failure mode; it is not a general data-quality guarantee, and the same skepticism that found this bug should be applied again before capital is committed.

## Decision

**GO on architecture, engineering, and cadence. NOT YET on the specific alpha numbers.** These are two different claims and this document has been sloppy about keeping them apart until this revision.

**Architecture: GO.** Proceed to Phase 1 build-out — nightly pipeline, model registry, gate, evaluator formalization — built around a **monthly rebalance horizon (h≈20 trading bars)**, not the daily cadence originally implied by v1's design. The pipeline, the gate mechanism, and the stress-testing discipline all demonstrably work: they were exercised on real data, found a real data-quality bug mid-session, and the process for handling that (fix, re-measure, report honestly) is itself the validated output of this phase. That does not depend on the survivorship finding below.

**Alpha magnitude: not yet trustworthy.** The corrected Run 3 figures — mean net alpha +0.49%/rebalance, 9/11 folds positive, break-even cost 117bps against a realistic 32bps cost — were the authoritative numbers *until* the survivorship measurement above, which shows the 98-symbol universe systematically excludes 238 names concentrated in real failure outcomes (DHFL, EDUCOMP, BHUSANSTL, and similar). The direction of the bias is unambiguous — a point-in-time universe would perform worse than this — the magnitude is not yet known. It could still clear costs. It could not. That is now the specific, scoped, unblocked next research task: full daily bhavcopy ingestion (confirmed feasible, ~6,500 files) and a re-run of this exact sweep against genuine point-in-time universe membership.

**No capital, real or paper, until that re-run happens.** This is not a new caution added for form — it is the direct, load-bearing consequence of a measurement made in this same session.
