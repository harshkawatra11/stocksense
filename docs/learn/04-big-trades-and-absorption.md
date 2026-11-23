# Big trades, bulk and block deals, and absorption

## What it is

Large orders leave marks. Exchanges publish some of them after the fact (bulk and block deals). Inside the day, a bar with unusually large volume is a proxy for a large trade. Absorption is when heavy volume trades at a level without moving price, as if a large passive order is soaking up the other side.

## How it is computed

- Bulk deal: a trade (or total of trades by one client) above 0.5% of a company's listed shares in a session, disclosed by the exchange after the close.
- Block deal: a large single trade executed in a special window with a minimum size; the morning window is 08:45 to 09:00 using the previous close as reference, the afternoon window 14:05 to 14:20 using the 13:45 to 14:00 volume-weighted price. Minimum size and price band have been revised by SEBI in 2025 (reported figures differ between sources: 25 crore rupees and a 3% band in recent reports against older 10 crore and 1%); check the exchange's current FAQ before relying on a number.
- Volume spike bar: bar volume greater than five times the median volume of the same minute of the session over the previous 20 sessions; reactions measured at +5, +15 and +30 minutes.
- Absorption flag: volume at least three times the median of the previous 20 bars with a range at most half their median range, located within 0.2% of the day's high or low, a VWAP band, or the previous day's value-area edge.

## What it says about flow and positioning

A big print says someone with size acted, not whether they are buying or selling; the direction of the bar and what happens next gives the clue. Absorption at lows with price holding suggests a buyer is accepting all sellers' supply; absorption at highs suggests the mirror.

## What NSE data allows and how StockSense computes it

`flow/deals.py` stores bulk and block deals from the NSE deals file (forward from 2026-11-11) and summarises per symbol and day (value bought, sold, net, top client). `orderflow/bigtrades.py` and `absorption.py` work on 5-minute bars (from 2024-09), using only past sessions for the time-of-day median. A configurable list of client names to highlight lives in `config/flow_clients.toml` (empty by default).

## Evidence

Studies of block trades show price effects and information content, but magnitudes depend on market and period; we do not claim an edge for Indian block deals. The tier-list strategy `tier_b_absorption` is registered in the pre-registered library so its value is measured, not assumed.

## How traders misuse it

Assuming every large print is "smart money"; reading absorption without checking that volume and range are unusual for that stock at that time of day; acting on a deal disclosed after the close as if it were live.

## Where it appears in the dashboard

Money Flow deals list (last five days); Order Flow big trades and absorption list with reaction columns.

## Sources

- https://nsearchives.nseindia.com/web/sites/default/files/inline-files/FAQs_Block_Deal_Window.pdf
- https://www.nseindia.com/report-detail/display-bulk-and-block-deals
- https://scholar.google.com/scholar?q=block+trades+price+impact+information+content+Holthausen+Leftwich+Mayers
