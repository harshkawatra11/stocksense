---
name: india-cost-model
description: Exact Indian equity/F&O transaction cost arithmetic - STT, exchange charges, SEBI fee, stamp duty, GST, brokerage. Use whenever explaining, estimating, or reasoning about the cost of a trade or portfolio rebalance.
allowed-tools: []
---

# India Cost Model

The maths that killed v1 of this product: a strategy showing ~23bps gross
alpha against ~25bps of real transaction cost is not a strategy, it's
noise with a very convincing backtest. Every cost claim in this codebase
traces to `stocksense.execution.cost_model.compute_charges` — this skill
explains that model so it can be reasoned about and narrated correctly,
not recomputed by hand.

## The components, and which leg they apply to

| Component | Delivery | Intraday (MIS) | F&O Futures | F&O Options |
|---|---|---|---|---|
| STT | 0.1%, **both legs** | 0.025%, **sell leg only** | 0.0125%, sell leg only | 0.0625% of premium, sell leg only |
| Exchange txn (NSE) | 0.00307%, both legs | same | same | same |
| SEBI fee | ₹10/crore, both legs | same | same | same |
| Stamp duty | 0.015%, **buy leg only** | 0.003%, buy leg only | same | same |
| Brokerage | Usually ₹0 (discount brokers) | min(₹20/order, 0.03%) | same | same |
| GST | 18% on (brokerage + exchange txn + SEBI fee) — **not** on STT or stamp duty | same | same | same |

## The number that mattered most in this project's history

**Intraday is cheaper than delivery, not more expensive.** A round trip
on ₹100,000: intraday ≈ ₹82.68 (8.3bps), delivery ≈ ₹222.48 (22.2bps).
This project's own research once argued the opposite by (wrongly) using
delivery's STT rate to reason about intraday economics — a retraction is
recorded in `research/phase0_verdict.md`. If you ever find yourself about
to say "intraday costs more because more trades," check whether you mean
"more round trips accumulate more cost" (true) versus "each round trip
costs more" (false) — these get conflated easily and shouldn't be.

## STT is the load-bearing asymmetry

Delivery's 0.1% applies on *both* legs; intraday's 0.025% applies on the
sell leg *only*. This single fact is why intraday's total STT (0.025%) is
4x cheaper than delivery's (0.2% combined) per round trip, even before
counting the other components. Getting the leg wrong when estimating cost
is the single most common way to be off by a large factor.

## GST scope, a common mistake

GST is 18% on (brokerage + exchange transaction charge + SEBI fee) only.
It is **not** charged on STT or stamp duty — those are already government
levies, and taxing a tax would be double-counting. A cost estimate that
applies 18% to the whole trade value rather than just these three
components will overstate cost meaningfully, especially for
delivery trades where brokerage is often zero and STT dominates.

## What this skill does not cover

Live slippage/market impact — the cost model's `slippage_bps` parameter
is a modeled assumption (default conservative, not measured from real
fills), separate from the statutory charges above. Do not conflate "the
statutory cost is X bps" with "the total realistic cost including
slippage is X bps" — say which one you mean.
