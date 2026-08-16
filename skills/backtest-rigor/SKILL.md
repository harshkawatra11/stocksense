---
name: backtest-rigor
description: Purged/embargoed walk-forward validation, pre-registration, and the stress battery. Use whenever reasoning about whether a research result should be trusted, or explaining a gate verdict.
allowed-tools: []
---

# Backtest Rigor

This project's research history is the argument for this skill: four
verdict revisions, each triggered by taking the previous version's own
stated next step seriously rather than stopping at a comfortable answer.
The discipline below is what survived contact with adversarial testing —
treat it as load-bearing, not aspirational.

## Purge vs. embargo — a distinction this project got wrong once

**Purge** = remove training data whose label window overlaps the test
window. Must cover exactly the label horizon — if predicting 20-bar
forward return, purge 20 bars around the test window boundary.

**Embargo** = a small additional buffer for residual serial correlation,
separate from purging.

This project's original embargo was `horizon + 252 bars` (the full
feature lookback) — over-conservative by roughly 13x, because a test-day
feature legitimately depending on pre-test history **is not leakage**;
that's how the model runs live. Leakage is *future into past*, not *past
into present*. Fixing this (embargo = horizon + a small buffer, not
horizon + full lookback) roughly doubled the usable fold count. If asked
to reason about embargo size, use the corrected principle, not the
original mistake — both are documented in `research/phase0_verdict.md`
precisely so the mistake isn't silently repeated.

## Pre-registration — commit criteria before the run they judge

`research/gate_criteria_preregistration.md` exists because the original
gate thresholds (`min_pct_folds_positive=0.6`, `n_best_folds_to_drop=2`)
were chosen *after* seeing the results they were meant to judge — the
exact evaluator-overfitting failure `docs/10-evaluation.md` names as the
deepest risk in the whole design, committed in the same session that
documented the risk.

The rule going forward: any new gate criteria, any new research
threshold, gets written down and committed **before** the sweep that will
be judged against it. If asked to help design or adjust a gate criterion,
check whether results already exist — if they do, the criterion is
compromised regardless of how principled it looks, and that should be
said plainly rather than worked around.

## The current pre-registered gate (evaluation/gate.py)

Four criteria, all must pass: ≥10 folds, mean net alpha > 0, one-sided
exact binomial hit-rate test p ≤ 0.10, and mean alpha after dropping the
best 15% of folds still > 0. A candidate must also strictly beat the
incumbent live model. **None of these are adjustable from the CLI on
purpose** — `train_candidate`'s docstring states this explicitly, because
allowing ad-hoc overrides would recreate the exact failure pre-
registration exists to prevent.

## The stress battery, and what "passing" actually proves

- **Best-trade-removal**: does the result survive dropping its best few
  folds? If not, the edge is concentrated in one or two lucky windows,
  not a real distributed effect.
- **Parameter perturbation**: does ±20% on hyperparameters collapse the
  result? If yes, the original configuration was fit to noise.
- **Survivorship-bound Monte Carlo**: inject synthetic delisting shocks
  and find the break-even rate — compare against a *measured*, not
  assumed, real delisting rate.
- **Data-quality stress** (`data-quality-forensics` skill): a single
  broken adjustment factor in one symbol once inflated an entire fold —
  found by exactly this kind of adversarial testing, not by inspection.

None of these tests, individually or together, prove a signal is real —
they narrow the space of ways it could be fake. Narrate results
accordingly: "survives the stress battery" is evidence, not proof, and
should never be reported as "proven."
