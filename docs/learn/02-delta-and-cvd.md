# Delta and cumulative volume delta (CVD)

## What it is

Delta is buying volume minus selling volume over an interval. Cumulative volume delta (CVD) is the running total of delta through the session. A rising CVD with a rising price says buyers are driving the move; a rising price with a falling CVD is called a divergence.

## How it is computed

- Tick rule: a trade at a higher price than the previous trade is a buy, a lower price is a sell, an unchanged price copies the previous classification.
- Lee-Ready (1991): a trade above the quote midpoint is a buy, below it is a sell, at the midpoint fall back to the tick rule.
- Bulk volume classification (BVC, Easley, Lopez de Prado and O'Hara 2012): for a bar, buy volume = V x Phi(dC / sigma), where dC is the close-to-close price change, sigma its standard deviation over the previous bars, and Phi the normal distribution function; delta = 2 x buy - V.
- Close location value (CLV): delta = V x (2C - H - L) / (H - L), the volume split by where the bar closed within its range.
- CVD = cumulative sum of delta, reset each session.

## What it says about flow and positioning

Delta shows which side was more aggressive within a bar. CVD shows whether that aggression accumulates. Divergence between price and CVD suggests that the move is being made on thin or opposing flow, which some traders read as exhaustion.

## What NSE data allows and how StockSense computes it

We do not have aggressor-tagged trades (note 00), so `orderflow/delta.py` provides `delta_bvc` (the default, with the standard deviation estimated from the previous 50 bars only), `delta_clv`, and `delta_tick` for forward-captured snapshots. History: 5-minute bars from 2024-09, 1-minute from 2026-03. `divergence` flags a new 20-bar price low with CVD above its value at the prior low (bullish) and the mirror case (bearish).

## Evidence

Easley, Lopez de Prado and O'Hara (2012) show BVC reproduces tick-classified volume well enough for their flow-toxicity measure on futures. Lee and Ready (1991) document that the midpoint test is more accurate than the tick rule alone but still misclassifies trades near the quote. We have not yet measured how well BVC on 1-minute NSE bars tracks true aggressor volume; any strategy using CVD is therefore tested in the gated runs (`tier_b_cvd_divergence`, `flow_cvd_trend`) rather than assumed to work.

## How traders misuse it

Reading estimated delta as exchange-reported; drawing conclusions from CVD on a single stock bar without regard to its volume; treating divergences as reversal signals without testing their hit rate.

## Where it appears in the dashboard

Order Flow workbench: the CVD pane and divergence markers; Money Flow stock rows: `cvd_slope_30m`.

## Sources

- https://doi.org/10.1093/rfs/hhs053
- https://doi.org/10.1111/j.1540-6261.1991.tb02683.x
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1695596
