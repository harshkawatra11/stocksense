# Walk-Forward Extension: Does Macro Nudge / Claude Synthesis Add Value?

Generated: 2026-07-12

**Question asked:** does the FULL pipeline (ensemble → macro sector nudge →
Claude synthesis confirm/reject) meaningfully outperform the raw-ML-only
walk-forward backtest (`backtest/reports/walkforward_20260712_133701.md`,
net expectancy +10.4 bps/trade, 45.3% win rate, underperforms NIFTYBEES
buy-and-hold: +1.0% vs +14.4% over 2024-2026)?

**Short answer: this cannot be genuinely tested with the data that exists
today.** Below is exactly what was checked, why it falls short, and the one
narrow thing the existing data *can* say (which is not reassuring).

---

## 1. What the code actually does (read, not re-derived)

`intelligence/signal_pipeline.py`:

- `apply_macro_nudge()` (line 49): adds `cap * sector_score` to confidence for
  BUY signals, where `sector_score` comes from `intelligence/macro_context.py`
  (an LLM/RSS-headline-driven sector sentiment score, `-1.0..+1.0`), cap
  default 0.10 (adaptive via `brain_params`). This happens in
  `run_single_ticker_multi` (line 620) — it modifies `confidence` before the
  `gate` threshold check (line 626), so it can flip a signal from
  below-threshold to fired or vice versa, but it never changes `signal`
  itself (BUY stays BUY).
- `_claude_synthesis()` (line 809): takes the top `TOP_SIGNALS_FOR_CLAUDE`
  signals by confidence (one per ticker, best timeframe), calls
  `intelligence.claude_cli.intraday_signal_check()`, and on `REJECT` demotes
  `signal_type` from BUY to **HOLD** in the DB (line 848) — i.e. synthesis is
  a hard filter on the already-saved signal, not a backtest-time overlay.
  This only runs on `TOP_SIGNALS_FOR_CLAUDE` signals per cycle (not all BUYs),
  and only when `settings.CLAUDE_SYNTHESIS_ENABLED`.

So, per the production code, an apples-to-apples "full pipeline" backtest
would need, for every OOS trade in the walk-forward: (a) a sector score at
that historical date, and (b) a Claude CONFIRM/REJECT verdict at that
historical date. Neither exists for the 2019–2026 walk-forward window.

## 2. What's actually stored in the DB — checked directly

Queried the live `signals` table (2702 rows total):

| Check | Result |
|---|---|
| Total signals rows | 2,702 |
| `fired_at` range | **2026-06-04 to 2026-06-25** (10 distinct trading days, all in the last ~5 weeks) |
| Rows with `macro_sector_score` populated | 2,687 (but 2,675 of those are exactly `0.0000` — only **12 rows** have a nonzero macro score) |
| Rows with `claude_confidence` populated (i.e. reached synthesis) | 906 |
| Rows with `actual_close` populated (resolved) | 611 |
| Resolved BUY rows with `claude_confidence` present | 131 |
| Resolved BUY rows *without* synthesis (didn't make top-N that cycle) | 373 |

Two disqualifying problems, found by inspecting actual rows:

1. **No historical overlap with the walk-forward window.** The walk-forward
   backtest scores OOS trades from 2019 through mid-2026. Production signal
   history — the only place macro scores and Claude verdicts are ever
   recorded — starts 2026-06-04. There is no ticker+date join key that
   exists in both datasets. Macro/synthesis data for 2019–2025 simply was
   never generated (the live pipeline didn't exist yet), so it cannot be
   backtested against those trades even in principle without re-running the
   macro/Claude layers historically — which is exactly the expensive
   re-run the task said to avoid.

2. **The `actual_close` "resolution" column is not a real forward-price
   outcome.** Sampled 10 resolved BUY rows directly: in every case
   `actual_close == price_at_signal` to the cent, and `resolved_at` is
   ~55-60 minutes after `fired_at` (e.g. id 1378: fired 09:17:06, resolved
   10:16:32, `price_at_signal` = `actual_close` = 1466.70). This is
   consistent with an expiry/stub job that closes out stale "active" rows
   without ever fetching a real subsequent price — not a genuine trade
   outcome. Confirmed in aggregate: **all 504 resolved BUY signals with a
   non-null `actual_close` show exactly 0.0 return, 0% "win rate"** — which
   is itself informative (the resolution pipeline is broken/not wired to
   real forward prices), but it means the "611 resolved signals" cannot be
   used to compute a real win rate or expectancy for any subset (with or
   without synthesis, with or without a macro nudge).

3. **The macro nudge has barely fired historically anyway.** Of 2,687 rows
   with a `macro_sector_score` value, only 12 are nonzero. So even setting
   aside the outcome-resolution problem, there isn't a meaningfully-sized
   nonzero-vs-zero macro comparison available in production data — the
   macro layer was returning neutral (0.0) almost every cycle during the
   window it has run in, which on its own means the ±0.10 nudge has had
   almost no practical effect on which signals fired during this period. (Not
   evaluated here: whether that's because news pipeline was cached/thin
   in this specific 3-week stretch, or a broader default-to-neutral pattern —
   would need `intelligence/macro_context.py`'s own logs to say, and that's
   outside this task's read-only scope.)

## 3. What was and wasn't attempted

- **Macro nudge replay against walk-forward trades**: not feasible. Checked
  for a `macro_sector_score` (or similar) column already populated on
  historical `ohlcv_daily`-era data — it does not exist; the column only
  lives on `signals`, and `signals` only goes back to 2026-06-04.
- **Claude synthesis replay against walk-forward trades**: not feasible for
  the same reason — `claude_confidence`/synthesis verdicts only exist for
  906 of the 2,702 production-era signals, all within the same 3-week window,
  with no realized-outcome data to score them against (see #2 above).
- **A same-window (2026-06-04 to 2026-06-25) comparison of "reached
  synthesis" vs "didn't reach synthesis" BUY signals**: attempted, but
  blocked by the broken outcome resolution (#2) — both subsets show 0.0
  mean return because `actual_close` isn't real. This is not a null result
  ("synthesis doesn't matter") — it's a "the comparison can't be computed"
  result, and reporting it as a null finding would be exactly the kind of
  fabricated positivity the project's ethos rules out.

No new backtest script was built beyond the queries above, because there is
no dataset to run one against — an extended `walkforward.py` mode would have
nothing to differ from the existing raw-ML run once you take away
non-existent historical macro/synthesis data and a broken outcome column.
Building a script whose output would necessarily be `n=0` for two of the
three requested comparison arms did not seem like a useful deliverable; the
numbers above are the actual finding.

## 4. Bottom line

| Comparison requested | Status |
|---|---|
| (a) Raw ensemble only | Already have this: **+10.4 bps/trade, 45.3% win rate, +1.0% cumulative vs NIFTYBEES +14.4%** (existing report) |
| (b) Ensemble + macro nudge | **Cannot be tested.** No historical macro scores exist before 2026-06-04; production macro scores in the only window that does exist are ≈100% neutral (12/2687 nonzero), so even a same-window comparison has no real sample. |
| (c) Ensemble + macro + synthesis filter | **Cannot be tested.** Synthesis verdicts only exist for the same 3-week production window, and the only outcome-resolution column (`actual_close`) in that window is not a real forward price — it's an expiry stub that always equals the entry price. There is no way to compute win rate/expectancy for signals that were CONFIRMed vs REJECTed vs never-reached-synthesis with current data. |

**Recommendation, not asked for but relevant:** two prerequisites would need
to exist before this question is answerable at all: (1) a working outcome
resolver that actually fetches a real N-days-later price for expired
signals (the current one appears to be a no-op stub — worth flagging to
whoever owns `intelligence/eod_review.py` or wherever the expiry job lives,
since it silently defeats any live win-rate tracking, not just this
analysis), and (2) enough elapsed production time (months, not weeks) with
that resolver working before a same-window synthesis-vs-no-synthesis
comparison would have a real sample size. Re-running macro/Claude
historically against the 2019-2026 walk-forward trades, as the task
correctly anticipated, would be the only way to get an answer for the full
history, and that is the expensive path this task asked to avoid.
