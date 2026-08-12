# StockSense

A quantitative research project for the NSE: does a cross-sectional, monthly-rebalance signal survive realistic transaction costs, out of sample, on real data?

**Current answer: probably, on architecture and cadence — not yet confirmed on magnitude.** See [research/phase0_verdict.md](research/phase0_verdict.md) for the full evidence trail, including a real data bug found and fixed mid-research, and a measured (not estimated) survivorship-bias gap that is the current blocking issue.

## Start here

1. **[research/phase0_verdict.md](research/phase0_verdict.md)** — what the evidence actually shows, revised three times as it was tested.
2. **[docs/STATUS.md](docs/STATUS.md)** — what the `docs/` architecture set describes (aspirational, written before code existed) versus what's actually built. Read this before trusting any individual doc.

## What's actually built

A Python research pipeline, not the product `docs/` describes:

```
src/stocksense/
  core/        config, trading calendar, typed domain objects
  data/        yfinance ingestion, DuckDB store, adjustment-anomaly validation
  features/    leak-tested feature engine (price/volume/candlestick/context)
  labels/      cross-sectional relative forward return, horizon-parameterized
  models/      LightGBM cross-sectional ranker + versioned model registry
  portfolio/   target-weight construction, no-trade band, turnover budget
  execution/   Indian equity cost model
  evaluation/  purged walk-forward validation + promote/reject gate
  cli/         train-candidate / predict / registry commands
```

Run via `stocksense.cli.main` (train, evaluate, gate, register, and score a real 20-name monthly portfolio against live NSE data). No scheduler, no UI, no LLM layer, no Telegram — those are documented in `docs/` as the intended product but do not exist yet, on purpose: building them before knowing whether the underlying signal survives costs would be building on an unproven foundation.

## What's proven so far

- A real, cost-surviving cross-sectional edge at a **monthly** rebalance horizon (not daily — daily was tried and demonstrably fails on costs, independently confirmed four times across two codebases).
- A genuine data-quality bug (a yfinance adjustment-factor discontinuity) found by adversarial stress testing, fixed, and the result re-verified on clean data.
- A measured survivorship-bias gap (82% of historically prominent NSE names absent from the current 98-symbol universe, concentrated in real failure cases) that is the specific, named blocker before any further validation is trustworthy.

## What isn't built yet

The reconcile/learning loop that is this project's actual thesis — an immutable prediction ledger, grading, calibration tracking — has **zero implementation**. What exists today is a rigorous backtest, not a system that learns from its own mistakes in production. See `docs/STATUS.md` for the full built-vs-aspirational breakdown.
