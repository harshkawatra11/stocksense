# Volume profile

## What it is

A volume profile is a histogram of traded volume by price instead of by time. The price with the most volume is the point of control (POC). The value area is the price range containing a set share (conventionally 70%) of the volume. High-volume nodes are prices where much business was done; low-volume nodes are prices price crossed quickly.

## How it is computed

Choose a bin size (0.05 rupees or 0.05% of price, whichever is larger). Spread each bar's volume uniformly over the bins between its low and high. Sum over the bars of the session or range. The POC is the bin with the largest volume. The value area starts at the POC and adds, one step at a time, the larger of the next bin above and next bin below (ties go above) until 70% of total volume is covered; its edges are VAH (value area high) and VAL (value area low).

## What it says about flow and positioning

Acceptance: heavy trade at a price means both sides found it fair. Rejection: thin volume with fast movement means one side was in control. Price tends to be attracted back toward high-volume prices and to move quickly through low-volume ones, which is the idea behind value-area strategies (note 08).

## What NSE data allows and how StockSense computes it

Real volume at each price needs tick data. We distribute each bar's volume over its range, an approximation that smooths the true distribution, using stored 5-minute and 1-minute bars (`orderflow/profile.py`, `volume_profile`). The previous session's VAH, VAL and POC feed the strategies `tier_b_amt_80` and `tier_b_poor_extreme`.

## Evidence

Value-area rules come from practitioner texts (Steidlmayer and Koy 1986; Dalton, Jones and Dalton) rather than academic studies. Whether they have an edge on NSE after costs is exactly what the gated runs of the AMT strategies measure.

## How traders misuse it

Reading a profile built from 5-minute bars as if it showed exact prices of trades; using a profile from too few sessions; ignoring that value moves, so yesterday's value area may be stale.

## Where it appears in the dashboard

Order Flow workbench: the profile histogram docked at the right of the chart with POC, VAH and VAL lines.

## Sources

- https://www.worldcat.org/search?q=Mind+Over+Markets+Dalton+Jones+Dalton
- https://www.worldcat.org/search?q=Markets+and+Market+Logic+Steidlmayer+Koy
