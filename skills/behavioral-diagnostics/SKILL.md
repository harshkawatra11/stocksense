---
name: behavioral-diagnostics
description: The 13-metric dosha catalogue - exact formulas and what each measures. Use when explaining why a specific diagnostic fired or what it means.
allowed-tools: []
---

# Behavioral Diagnostics — the Dosha Catalogue

Thirteen deterministic metrics computed by `statements.diagnostics` from
reconstructed positions. Each is arithmetic — this skill exists to
narrate *what a fired diagnostic means*, not to recompute it.

| Dosha | What it measures | Formula (informal) |
|---|---|---|
| **Cost drag** | Charges as a fraction of gross P&L. >100% means charges alone erased more than was made. | `total_charges / abs(gross_pnl)` |
| **Disposition effect** | The textbook bias: holding losers longer than winners, hoping they recover, while cutting winners early. | `median_hold(losers) / median_hold(winners)` |
| **Revenge trading** | Elevated trade frequency in the 30 minutes right after a loss, vs. baseline frequency. | post-loss trades / expected-at-baseline-rate |
| **Overtrading** | How much of gross P&L is consumed by the sheer number of trades taken. | charges / gross, plus trades/day |
| **Position sizing chaos** | How inconsistent position sizes are — erratic sizing often reflects emotional rather than systematic decisions. | coefficient of variation of position value |
| **Martingale escalation** | Whether position size grows after a losing streak — the classic "double down to get even" pattern. | correlation(size, consecutive prior losses) |
| **Averaging down** | Adding to a position that's already losing. | fraction of positions that closed as losers |
| **Opening-bell bleed** | Losses concentrated in the first 15 minutes of the session, when volatility and spreads are typically worst. | first-15-min P&L / total loss |
| **Expectancy** | The core edge/no-edge number: expected P&L per trade given the actual win rate and average win/loss size. | `win_rate*avg_win - loss_rate*avg_loss` |
| **Concentration** | How much total exposure sits in a single symbol. | max single-symbol exposure / total exposure |
| **Drawdown profile** | Peak-to-trough equity decline, and how it compares to typical monthly P&L. | max(cumulative P&L − running peak) |
| **Tail dependence** | Whether overall profitability depends on a small number of outlier trades. | P&L excluding the best 5% of trades |
| **Time-of-day edge** | Whether a specific time-of-day bucket is a consistent net drag. | worst 15-min-bucket P&L / total P&L |

## Reading severity correctly

Severity (`ok`/`notable`/`high`/`critical`) is assigned from
**pre-registered thresholds**, not from how a given user's number
compares to other users — there is no cross-user benchmarking in this
system (see `claude-report-writing`'s anti-pattern list: never invent a
"typical trader" comparison). A `critical` cost-drag finding means the
ratio crossed a fixed, principle-based bar (>100%, i.e. charges exceeded
gross profit), not that it's worse than some percentile of other traders.

## When a diagnostic returns `None`/`ok` for lack of data

Several diagnostics (`revenge_trading`, `martingale_escalation`,
`tail_dependence`) require a minimum sample and return a neutral/`ok`
result with `detail: {"reason": "insufficient_data"}` rather than a false
signal on thin data. Narrate this as "not enough trades to assess" — not
as "no problem found," which overstates the finding.

## Composite reading: which doshas tend to travel together

- **Disposition effect + averaging down** often co-occur — both are
  expressions of reluctance to realize a loss.
- **Revenge trading + martingale escalation** often co-occur — both are
  expressions of loss-chasing, one in frequency, one in size.
- **High cost drag + overtrading** are close to the same underlying cause
  (too many trades) viewed through two lenses (total charges vs. per-trade
  frequency) — don't present them as two independent findings without
  noting the overlap.
