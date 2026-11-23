# Delivery, FII/DII and participant positioning

## What it is

Delivery percentage is the share of a day's traded quantity that was actually delivered (settled in shares) rather than squared off intraday. FII and DII are foreign and domestic institutional investors; NSE publishes their daily cash-market buy and sell values. Participant-wise open interest shows long and short positions in index and stock futures and options for four client types: FII, DII, PRO (proprietary) and CLIENT (retail).

## How it is computed

- Delivery z-score: today's delivery percentage minus the mean of the previous 20 sessions, divided by their standard deviation (ddof 1; at least 15 observations); today is not in its own baseline. A delivery spike is z above 2 on a day the close is above the open.
- Delivery value = delivered quantity x close.
- Index futures long ratio per client type = long / (long + short); the 1- and 5-session changes are the reading.
- FII/DII net = buy value minus sell value.

## What it says about flow and positioning

High delivery on a rise suggests holders took shares to keep, not day traders. Rising FII index-futures long ratio says foreign institutions are adding index exposure through derivatives; FII cash selling with long ratio falling says broad de-risking.

## What NSE data allows and how StockSense computes it

Delivery per stock since 2019-08 (`bhavcopy_delivery`), participant OI since 2019-12 (`participant_oi`), FII/DII cash only about 17 days (`fii_dii`, building forward). `flow/delivery.py` and `flow/participants.py` compute the above. Cash-market FII/DII history before our collection started is not available from an archive file we have found; the card records this in DECISIONS.

## Evidence

Published studies of foreign institutional flows in India (see the search link below) mostly report a positive contemporaneous relation between FII flows and market returns, with continuing debate about the direction of causality; we have not verified a specific result and make no claim from them. Delivery as a return predictor has weaker published evidence. `flow_deliv_spike` and `flow_fii_index` are pre-registered to test both on our data.

## How traders misuse it

Reading one day's FII figure as a forecast; mistaking provisional end-of-day figures for final; treating delivery spikes on corporate-action days (index rebalancing, buybacks) as informed buying.

## Where it appears in the dashboard

Money Flow market strip (FII and DII cash, participant long ratios with 60-day sparklines) and the delivery z column.

## Sources

- https://www.nseindia.com/reports/fii-dii
- https://www.nseindia.com/all-reports-derivatives
- https://scholar.google.com/scholar?q=foreign+institutional+investors+flows+India+stock+returns+causality
