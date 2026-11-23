# The 09:00 pre-open session

## What it is

NSE's regular session starts at 09:15, but the exchange runs a pre-open call auction from 09:00. Orders entered then are not matched one by one; they are collected and matched together at a single opening price. That price and the buy and sell quantities waiting at it are the first flow signal of the day. Block deals have their own window before it (08:45 to 09:00).

## How it is computed

- Indicative equilibrium price (IEP): the price at which the largest quantity can be matched, ties broken by smaller order imbalance, then by closeness to the previous close.
- IEP gap = (IEP divided by previous close minus 1) times 100.
- Imbalance = (buy quantity minus sell quantity) divided by (buy plus sell quantity), from -1 to +1.
- Follow-through: whether the sign of the IEP gap matches the sign of price at 09:30 versus the 09:15 open.

## What it says about flow and positioning

A positive gap with large buy-side imbalance means more demand than supply at the indicative price: participants who want to trade at the open lean to buy. The effect is most informative for names with news overnight or large global moves. A random-close element at the end of the order entry phase is meant to stop last-second manipulation.

## What NSE data allows and how StockSense computes it

NSE's pre-open JSON has IEP, previous close, final price and buy and sell quantities per symbol; `ingest_misc.parse_preopen` parses it. Our stored `preopen` dataset runs 2026-09-23 to 2026-10-21 and then has a gap; `flow/preopen.py` fills it from now on while the server runs. **Timing caveat:** the F&O segment pre-open began on 2025-12-08 and secondary sources (broker and news articles, not NSE's own pages, which we could not read) report that the internal phases changed from 2026-09-07: orders of both kinds until about 09:05, then limit orders only until a random close between 09:08 and 09:10, matching to 09:12, a buffer to 09:15. Because we have not confirmed those clock times from an NSE circular, the capture loop polls through 09:00 to 09:15 and takes the official snapshot after the first sign that matching has begun (the first payload with a final price) instead of at a hard-coded 09:08. Card M3.5 records the live observation.

## Evidence

Biais, Hillion and Spatt (1999) studied price discovery in the pre-opening period of the Paris Bourse and found that indicative prices become more informative as the opening nears. We have not measured how much the NSE indicative gap predicts the first hours; `flow_preopen_imb` is registered with status `await_data` until at least 120 sessions are stored.

## How traders misuse it

Trading on indicative prices early in the order-entry phase, when orders can still be pulled; assuming an extreme IEP gap will persist; ignoring that thinly traded names have wide gaps with tiny quantities.

## Where it appears in the dashboard

Money Flow pre-open board: live from 09:00, then a "pre-open versus first 15 minutes" view with follow-through ticks.

## Sources

- https://www.nseindia.com/static/products-services/equity-market-pre-open
- https://nsearchives.nseindia.com//web/mediaattachment/2025-12/FAQs_for_PreOpen_Session_in_Equity_Derivatives_FO_Segment_20251202123640.pdf
- https://nsearchives.nseindia.com/web/sites/default/files/inline-files/FAQs_Block_Deal_Window.pdf
- https://scholar.google.com/scholar?q=Biais+Hillion+Spatt+price+discovery+and+learning+during+the+preopening+period+in+the+Paris+Bourse
