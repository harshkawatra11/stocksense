# Overnight–Intraday Reversal: Pre-Registration

**Date:** 2026-08-29
**Committed before:** any sweep script for this hypothesis has been written or run. Only the
underlying data has been inspected (see "What has already been seen" below) — no
overnight-return, intraday-return, or IC number of any kind has been computed. This ordering
is the entire point of this document, per this project's own established pattern
(`gate_criteria_preregistration.md`, `preregistration_bhavcopy_rerun.md`, `preregistration_intraday.md`).

## Why this hypothesis, and why now

Six independent attempts at intraday cross-sectional ranking on this pipeline have failed —
most recently `verdict_intraday.md`'s 0/15 folds, `gross_expectancy` negative in every fold
before a single rupee of cost. That failure has one specific shape: rank an ML-feature
cross-section, hold intraday, and hope. This hypothesis is structurally different: it is a
single, economically-motivated overnight→open reversal signal, documented at ~5× the
magnitude of conventional short-term reversal (Della Corte & Kosowski, "Overnight–Intraday
Reversal Everywhere"), and it is genuinely a same-session MIS position (enter at the open,
exit at the close), which is what the user actually asked for.

This is **attempt #7** against same-day/intraday prediction on this codebase, and it is
registered as such — see "Multiplicity accounting" below. It is not exempt from the
possibility of an 7th honest FAIL.

## What has already been seen (full disclosure)

Before writing this document: `bhavcopy_eq`'s `open`/`prev_close`/`close` columns were
checked for basic data quality only — **6,536,761 EQ rows, 2010-01-04 → 2026-08-24, zero
nulls in `open` or `prev_close`, zero non-positive opens**. No overnight return, no intraday
return, no correlation with anything, and no IC of any kind has been computed. This is
disclosed in full per this project's own standing rule.

## The signal, defined precisely

For each symbol on each trading date `d` (raw `close`/`open`/`prev_close` from `bhavcopy_eq`,
**not** adjusted — see "Adjustment" below):

```
overnight_ret(d) = open(d) / prev_close(d) - 1      # prev session's close -> today's open
intraday_ret(d)  = close(d) / open(d) - 1           # today's open -> today's close
```

Signal = the cross-sectional z-score of `overnight_ret(d)` across the tradeable universe on
date `d` (via `features/registry.py`'s `@factor` decorator, computed cross-sectionally — this
is a cross-sectional factor, not a per-symbol time-series one, so it plugs into the registry
as a factor whose `fn` receives the full day's cross-section, not one symbol's history; this
is the one place this hypothesis's implementation deviates from `registry.py`'s per-symbol
contract, and is called out explicitly in the sweep script's own module docstring when
written).

**Position:** enter at date `d`'s open, priced with `intraday_ret(d)` as the realized return,
exit at date `d`'s close. One session, never held overnight — the deliberate exclusion of
`verdict_intraday.md`'s already-falsified path (holding an ML-ranked cross-section across a
session) in favor of a signal known to the literature *specifically for* the overnight→open
transition.

### Adjustment

Raw (unadjusted) `open`/`close`/`prev_close` are used, not `adj_close`. A corporate action
(split/bonus) creates a large jump in raw `prev_close` vs `open` that would otherwise be
misread as a real overnight return. **Mitigation, fixed here before any result exists:** any
symbol-date with a `corporate_actions` row where `ex_date == d` is EXCLUDED from that date's
cross-section entirely (not adjusted, excluded) — the simplest, least-assumption-laden fix,
consistent with `data/adjust.py`'s existing "quarantine over correct-and-hope" philosophy for
detected anomalies elsewhere in this codebase.

## The three variants — all three reported, whatever the outcome

| Variant | Construction |
|---|---|
| **Long-only** | Long the bottom-`top_n` by overnight return (the overnight losers) at the open, exit at the close. No short leg. |
| **Short-only** | Short the top-`top_n` by overnight return (the overnight winners) at the open, exit at the close. |
| **Long/short** | Both legs simultaneously, equal-weighted long and short books, matching Della Corte & Kosowski's own published construction. |

This project's own research (`research/robustness_bhavcopy_rerun.md` and the wider record)
has repeatedly found India-specific asymmetries the US literature does not have — most
directly, post-2011 NSE shows **persistently positive overnight and negative intraday
drift**, per the desk research behind this pre-registration. That drift cuts *for* the short
leg and *against* the long leg. All three variants are measured and reported independently,
specifically so this asymmetry — if present — is visible rather than averaged away inside a
single long/short number.

## Parameters, and where each comes from

| Parameter | Value | Source |
|---|---|---|
| `price_source` | `"bhavcopy"` | Only source with raw (unadjusted) `open`/`prev_close`/`close` — `candles` (yfinance) does not carry a reliable raw-open field for this universe. |
| `use_point_in_time_universe` | `True` | Same survivorship-bias discipline as every post-Phase-D2 run. |
| `cap_bands` | `full_pit`, `large`, `mid`, `small` (via `universe_pit.CAP_BANDS`, unmodified) | `research/robustness_bhavcopy_rerun.md` already found this pipeline's edge is cap-band-dependent (`large` fails on significance where `mid`/`small`/`full_pit` pass) — reporting all four here rather than assuming the pattern repeats. |
| `top_n_grid` | `(5, 10, 20)` | `5`/`10` bracket the user's own stated 5–10 name intraday portfolio; `20` is the wider cross-check already used throughout this project's other sweeps. |
| Holding period | One session (open→close), never overnight | The literature's own construction; also the only genuinely *intraday* (same-day MIS) position this hypothesis can claim, as distinct from the already-falsified overnight-to-overnight or multi-day holds. |
| Cost model | `execution/cost_model.compute_charges(segment="equity_intraday", side=..., quantity=..., price=...)` on **both legs**, unmodified | The exact, verified Indian MIS charge stack — 8.3bps round-trip on ₹100k, per `docs/STATUS.md`'s own verification against Zerodha's published sheet. |
| Fill realism | `execution/fill_model.py`'s participation cap (`max_participation_pct` default) and circuit-lock filter, unmodified | The two things that actually decide viability at ₹17,500 — a signal that only "works" by assuming fills a real order book could not have produced is not a result. |
| Fold construction | `evaluation/walkforward.make_folds`, `horizon_bars=1` (the label matures same-session), `test_window_bars=63`, `min_train_bars=500`, embargo default, unmodified | Reused completely unchanged. `horizon_bars=1` is correct here specifically because the label (`intraday_ret`) is realized same-day — this is NOT a re-opening of the already-falsified "predict `fwd_ret_1b` from an ML cross-section" path; the *signal* is the economically-motivated overnight return, not a model output. |
| Gate | `evaluation/gate.py`, **unmodified**; `gate_criteria_preregistration.md`'s defaults | Same gate as every prior run. `evaluate_gate` is called with the SAME `GateCriteria()` for every variant/cap-band cell — no per-cell tuning. |
| Data window | **2010-01-04 → 2024-12-31 only** | Phase K1's sealed vault. 2025-01-01 onward is withheld by `data/loader.load_candles`'s default ceiling and is not looked at for this research phase. |

## Multiplicity accounting

This hypothesis is registered via `evaluation/attempts.register_attempt` **before** the sweep
runs, against a `holdout_id` computed from the full parameter spec above (`cap_bands` ×
`top_n_grid` × 3 variants = 36 cells; each cell registers as its own attempt via
`evaluation.attempts`, so the count feeds the Deflated Sharpe Ratio's `N` honestly rather than
treating "the hypothesis" as a single trial when it is actually 36 measured cells).

**If this is not the first attempt registered against this exact `holdout_id`** (e.g. a
future session re-runs this exact spec), `evaluation.attempts.criteria_for_attempt` tightens
`hit_rate_significance_alpha` via Bonferroni before any gate evaluation — this document does
not hand-pick a stricter threshold itself; the registry computes it from the actual attempt
count.

## What counts as a pass or fail — written before any result exists

Evaluated independently per (cap_band, variant, top_n) — **36 cells, all 36 reported**:

- **Gate PASS** (`evaluation/gate.py`, unmodified criteria): that cell is a candidate for the
  final sealed-vault gate (Deflated Sharpe Ratio ≥ 0.95, Probability of Backtest Overfitting
  ≤ 0.5, per `evaluation/robustness.py`) — the vault is unsealed **once**, for this
  `hypothesis_id`, only if at least one cell reaches this stage.
- **Gate FAIL** across all 36 cells: this is the **seventh** honest FAIL of same-day/intraday
  prediction on this codebase and is committed as such — `research/verdict_overnight_intraday.md`
  reports it in full, per the standing precedent of Phases E, G, and I. The fallback hypothesis
  named in the plan (intraday momentum, first-half-hour-predicts-last-half-hour, using the
  91.2M-bar minute spine) becomes the next pre-registered attempt, not a silent pivot inside
  this same document.
- **If the long-only and short-only variants disagree sharply** (one passes, one fails) —
  that is direct evidence of the hypothesized India-specific overnight/intraday drift
  asymmetry, reported as a finding in its own right, not resolved by picking whichever variant
  looks better.

No parameter in the tables above — cap bands, `top_n_grid`, cost model, fold construction, or
gate criteria — may be changed after this commit based on the sweep's result.
