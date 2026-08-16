---
name: data-quality-forensics
description: Adjustment-factor anomalies, survivorship bias, and point-in-time discipline - what this project learned the hard way. Use when reasoning about whether a data source or historical figure can be trusted.
allowed-tools: []
---

# Data Quality Forensics

Encodes a real, expensive lesson from this project's history: a Monte
Carlo stress test once surfaced a single 20-day rebalance period showing
+41.6% gross portfolio return. Investigation traced it to one position
(ADANIENT) whose `adj_close` jumped 8.6x day-over-day on 2003-09-04 while
the raw `close` barely moved — a broken adjustment factor fabricating a
~750% "return" that had inflated the single strongest fold of an entire
research run everyone already believed. This skill is that discipline,
generalized.

## Adjusted vs. unadjusted close — the specific failure mode

`adj_close` is corporate-action-adjusted (used for features/labels);
`close` is what actually printed (used for display and cost
calculation). Conflating the two is the classic source of silent
training-data corruption. **The detector**: flag any day where
`adj_close / close` changes by more than ~1.5x from the prior day while
`close` itself moves ordinarily — that ratio discontinuity is the
signature of a broken adjustment, not a real corporate action (real
splits/bonuses produce a *consistent* ratio shift, not a one-day spike
followed by reversion).

`stocksense.data.validate.flag_adjustment_anomalies` /
`quarantine_symbols` implement this and are wired into the production
path — any new data source integration should route through this before
being trusted, not bypass it because "this source is supposed to be
clean."

## Survivorship bias — measured, not assumed

This project didn't just flag survivorship bias as a caveat — it
measured it directly against real NSE bhavcopy archives
(`research/survivorship_check.py`): 82% of historically-prominent names
(union of top-150-by-turnover across 9 checkpoints spanning 2001–2024)
were absent from the hand-picked 98-symbol research universe, and the
238 genuinely-delisted ones among them were disproportionately failure
stories (DHFL, EDUCOMP, BHUSANSTL — fraud, insolvency, effectively
worthless). A universe silently missing exactly the outcomes that would
test a strategy hardest overstates performance in a specific, dangerous
direction.

The follow-up discipline, also from this project's history: **bound the
damage before assuming it's fatal.** `research/survivorship_bound.py`
injects synthetic delisting shocks (weighted toward weaker-scoring held
names, since failures don't happen uniformly) and finds the break-even
shock rate — compared against the *measured* real delisting rate, not a
guess.

## Point-in-time discipline, generally

A feature or a universe membership rule must never use information that
wouldn't have been available on the date it's applied to. This applies
beyond adjustment factors: a "liquid enough to trade" universe filter
computed with hindsight (using a stock's *eventual* full history to
decide it was liquid on some past date) is a subtler version of the same
mistake survivorship bias is. `test_leakage.py`'s truncation-invariance
tests (recompute a feature on truncated history, assert past values are
bit-identical) are the mechanical check for this — not a naming
convention, a real proof.

## When reasoning about a data anomaly

Don't assume a big fold or big single-trade outcome is either "real
signal" or "must be a bug" without checking. The actual discipline that
worked here: run the stress tests, let an *outlier* trigger investigation
(a Monte Carlo reshuffle surfacing an extreme single period is exactly
the kind of signal to chase), and confirm against the raw (unadjusted)
data before concluding either way.
