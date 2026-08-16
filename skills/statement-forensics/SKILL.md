---
name: statement-forensics
description: Broker statement parsing conventions, FIFO position reconstruction, and the Kundli methodology. Use whenever explaining how a diagnostic or counterfactual was derived from raw trades.
allowed-tools: []
---

# Statement Forensics — the Kundli

The pipeline from a raw broker export to a behavioral diagnosis:
`statements.parsers` (broker-specific, fuzzy header matching) → canonical
`trades` rows → `statements.positions.reconstruct_positions` (FIFO
matching) → `statements.diagnostics` (13 "doshas") →
`statements.counterfactuals` (7 "what if" replays) →
`statements.report.generate_kundli` (the narrated output, which uses
`claude-report-writing`'s rule).

## Why FIFO, specifically

Indian capital-gains matching (Income Tax Act) uses First-In-First-Out —
the oldest open lot is closed first. `reconstruct_positions` matches
buys to sells this way, not LIFO or average-cost, so reconstructed
positions are consistent with what a real Tax P&L statement would report,
and holding-duration-dependent logic (STCG/LTCG classification, the
`india-cost-model` intraday/delivery distinction) is correct.

## What a "position" is, and isn't

A position is a completed round trip — a matched buy-to-sell (or, for a
short, sell-to-buy) pair, possibly a *partial* fill of a larger order. A
single buy of 100 shares followed by two sells of 40 and 60 produces
**two** position rows, not one. An unmatched trailing quantity (bought
but not yet sold, at the end of available trade history) produces **no**
position row — it's still open, so there's no P&L to report yet. When
narrating "how many trades did you make," be precise about whether you
mean raw trade rows or reconstructed positions — they are not the same
count.

## The dosha catalogue (behavioral-diagnostics has the full formulas)

Each dosha is a **pre-registered, deterministic** metric — severity
thresholds are fixed before being run against any specific user's data,
the same discipline that fixed `evaluation/gate.py`'s overfitting problem
in this project's research history
(`research/gate_criteria_preregistration.md`). Do not suggest adjusting a
threshold to make a particular result read better; if a threshold seems
wrong, that's a finding to raise, not something to quietly work around in
a single report.

## Counterfactuals are historical arithmetic, not predictions

Every counterfactual ("what if you'd never traded the first 15 minutes")
replays *actual, already-happened* fills under a modified rule. State
this plainly whenever narrating one: it says what would have happened to
*that exact sequence of trades*, not what would happen if the same
behavior change were applied going forward — changed behavior changes
market impact and the trader's own subsequent decisions in ways this
arithmetic cannot model.

## Sample-size discipline

`MIN_POSITIONS_FOR_SIGNIFICANCE = 30` (the reference guidance backing
this: ≥30 minimum, 100+ preferred, before treating a behavioral pattern
as more than noise). A fact sheet with `insufficient_sample: true` should
be narrated with an explicit low-confidence caveat, not silently treated
the same as a well-populated one.
