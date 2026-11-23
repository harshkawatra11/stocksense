# The tier list: a hypothesis, not a fact

## What it is

A trader's public ranking sorts strategies into tiers: F (avoid), C (mediocre), B (decent). We have not verified who made the list or on what evidence, so StockSense treats it as a set of hypotheses and measures each one. The F-tier strategies serve as negative controls: if the machinery works, they should fail the gate.

## How it is computed

Each strategy is coded as exact rules in `strategies/tiers.py`, pre-registered in `config/strategy_library.toml` before any result exists, run through the walk-forward with costs, and judged by the nine-line gate. Then RESULTS-M section 4 compares claimed tier order (F = 0, C = 1, B = 2) with the measured median net Sharpe and the Spearman rank correlation between them.

| Strategy key | Rule in one line |
|---|---|
| tier_f_trendline | long when price touches a rising line through the last two confirmed swing lows and closes up |
| tier_f_rsi_oversold | long when RSI(14) crosses up through 30 |
| tier_f_ict_unicorn | breaker order block overlapped by a fair-value gap, entry on return into the overlap |
| tier_c_vwap_mr | long below VWAP minus 2 sigma with RSI(2) under 10, exit at VWAP |
| tier_c_btst | buy at close on a strong, high-volume close in an up market, sell next open |
| tier_c_orb_15 / _30 | opening range breakout with volume confirmation |
| tier_c_macro_trend / _fii | index timing from USDINR, Brent, US10Y and FII positioning |
| tier_b_absorption, tier_b_cvd_divergence | adverse order flow rules (notes 02, 04) |
| tier_b_str_1d / _5d, tier_b_smt | short-term reversal (Jegadeesh 1990; Lehmann 1990) and NIFTY/BANKNIFTY divergence |
| tier_b_amt_80, _ib_ext, _open_drive, _poor_extreme | auction market theory rules (note 08) |
| tier_b_turtle | 20-day breakout, exit under the 10-day low |

Interpretations to correct if wrong: "BTS" read as BTST, "8-tier" as B-tier, "STR" as short-term reversal (SMT divergence added in case), "adverse order flow" as absorption and CVD-price divergence.

## What it says about flow and positioning

Only the B-tier items use flow information; the F and C items are price patterns or index timing.

## What NSE data allows and how StockSense computes it

Daily strategies use stored adjusted prices from 2010; intraday ones use 5-minute bars from 2024-09, about 12 months, so intraday results have low power. The index 5-minute bars needed by `tier_b_smt` are probed first; if absent the strategy is marked `await_data`.

## Evidence

Short-term reversal is documented in US equities (Jegadeesh 1990; Lehmann 1990). Surveys of technical analysis find mixed profitability and strong sensitivity to data snooping and costs (Park and Irwin 2007). Support and resistance has some evidence in foreign exchange (Osler 2000). Whether a public tier list predicts NSE results is what RESULTS-M section 4 measures.

## How traders misuse it

Trusting a ranking without trade counts, costs or out-of-sample tests; testing many variants and reporting the best; confusing a good-looking equity curve with an edge.

## Where it appears in the dashboard

Strategies page: library table with a tier column and the "tier list versus evidence" bar chart.

## Sources

- https://doi.org/10.1111/j.1540-6261.1990.tb05110.x
- https://doi.org/10.2307/2937816
- https://doi.org/10.1111/j.1467-6419.2007.00519.x
- https://doi.org/10.3905/jpm.2014.40.5.094
- https://scholar.google.com/scholar?q=Osler+support+for+resistance+technical+analysis+and+intraday+exchange+rates
