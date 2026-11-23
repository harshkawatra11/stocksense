# Market profile and auction market theory

## What it is

Auction market theory (AMT) treats a market as a continuous auction whose job is to find prices where trade is fair for both sides. Market profile records the day as time-price opportunities (TPOs): the day is cut into 30-minute periods labelled A, B, C and so on; each letter is stamped at every price traded in that period. The result is a bell-shaped (or not) picture of where the market spent time. AMT calls the market "in balance" when it rotates inside a range and "out of balance" when it trends to a new area of value; the "law of the market" is that price leaves balance only with participation and returns to value when it is rejected.

## How it is computed

- Periods: A 09:15 to 09:44 through M 15:15 to 15:29 (13 periods).
- Initial balance (IB): the range of the first two periods (09:15 to 10:14).
- Single prints: prices touched in only one period inside the range, between multi-period prices. They show fast one-sided moves.
- Poor high or low: two or more periods end at the extreme price, a sign of unfinished business, often revisited.
- Day types (Dalton): trend, normal, normal variation, neutral, double distribution, non-trend, tested in this order (see `orderflow/profile.py:day_type`).
- Open types: open drive, open test drive, open rejection reverse, open auction in or out of range of the previous value area.
- The 80% rule: if price opens outside the previous day's value area and then trades back inside it and holds for two consecutive periods, price tends to traverse the value area to its other side about 80% of the time (practitioner claim).

## What it says about flow and positioning

Where the market opens relative to yesterday's value (inside, outside, drive) says whether participants are accepting or rejecting yesterday's price. An initial-balance extension with volume says larger timeframe participants joined the day's direction.

## What NSE data allows and how StockSense computes it

5-minute bars from 2024-09 give TPOs at 30-minute resolution; 5-minute bars are fine because TPOs only need the period high and low. `orderflow/profile.py` provides `tpo_profile`, `day_type`, `open_type`. The strategy keys `tier_b_amt_80`, `tier_b_ib_ext`, `tier_b_open_drive` and `tier_b_poor_extreme` test the practitioner claims with pre-registered costs and walk-forward, so the 80% rule is measured on NSE instead of assumed.

## Evidence

The rules come from practitioner books (Steidlmayer and Koy 1986; Dalton, Jones and Dalton 2013). We know of no peer-reviewed validation on Indian stocks, and our history is about 12 months, so even a positive result will have low power; RESULTS-M states this.

## How traders misuse it

Quoting the 80% rule as if it were a law; classifying day types after the fact, when the day is over, instead of in real time; using a single day's profile with no volume context.

## Where it appears in the dashboard

Order Flow workbench: TPO letters grid, IB bracket, single-print shading, day-type and open-type badges.

## Sources

- https://www.worldcat.org/search?q=Mind+Over+Markets+Dalton+Jones+Dalton
- https://www.worldcat.org/search?q=Markets+and+Market+Logic+Steidlmayer+Koy
- https://scholar.google.com/scholar?q=auction+market+theory+market+profile+empirical+evidence
