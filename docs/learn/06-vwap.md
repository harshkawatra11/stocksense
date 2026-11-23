# VWAP and its bands

## What it is

The volume-weighted average price is the average price of the session so far, weighting each price by the volume traded at it. It is the benchmark many institutions are measured against when they execute large orders, so it acts as a reference level during the day.

## How it is computed

Typical price tp = (high + low + close) / 3 per bar. VWAP = cumulative sum of (tp x volume) divided by cumulative volume, reset each session. Bands are VWAP plus and minus k standard deviations, where the volume-weighted standard deviation is the square root of the cumulative sum of volume x (tp minus running VWAP) squared, divided by cumulative volume; k = 1, 2, 3. An anchored VWAP starts the same sum at a chosen bar (a swing low, an earnings bar) instead of the open. When a bar has zero volume (index symbols), weight 1 is used, as Market Lab does.

## What it says about flow and positioning

Price above VWAP means that, on average, buyers paid up through the day; below means sellers accepted lower prices. A buy order executed at a price below VWAP did better than the average participant. Because many algorithms target VWAP, the line can attract trade.

## What NSE data allows and how StockSense computes it

Bars from Yahoo 1-minute (live), Angel candles, stored 5-minute bars from 2024-09 and 1-minute bars from 2026-03. `orderflow/vwap.py` provides `session_vwap`, `bands` and `anchored_vwap`. The mean-reversion strategy `tier_c_vwap_mr` (long below minus 2 sigma with RSI(2) under 10, exit at VWAP) is in the pre-registered library as a test of the common claim.

## Evidence

Berkowitz, Logue and Noser (1988) introduced VWAP as an execution benchmark. There is little published evidence that VWAP bands generate trading profits after costs for retail traders; the claim is tested here rather than assumed.

## How traders misuse it

Treating VWAP as support or resistance with no test; applying it to the first minutes of the session when it carries almost no information; forgetting that VWAP resets daily and lags in a trend.

## Where it appears in the dashboard

Order Flow workbench candle chart (VWAP, plus or minus 1 and 2 sigma, click-to-anchor), and the tape chart modal's smart-money layers.

## Sources

- https://doi.org/10.1111/j.1540-6261.1988.tb02590.x
- https://scholar.google.com/scholar?q=VWAP+volume+weighted+average+price+execution+benchmark
