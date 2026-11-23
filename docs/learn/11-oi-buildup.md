# Open-interest buildup

## What it is

Open interest is the number of futures or options contracts not yet closed. Reading OI together with price gives four classic states: long buildup (price up, OI up), short buildup (price down, OI up), short covering (price up, OI down), long unwinding (price down, OI down).

## How it is computed

Per stock and day: near-month futures price change versus the previous session's price of the same contract, and total futures OI change versus the previous session. The near month rolls to the next expiry on the session before expiry. Rollover percentage near expiry = next-month OI divided by total OI in the last five sessions. Basis = (futures settle divided by underlying minus 1) times 100. `derivs.oi_price_quad` gives the label.

## What it says about flow and positioning

Rising OI with rising price means new long positions are being opened; rising OI with falling price means new shorts. Falling OI means positions are closing. The labels say who is entering or leaving; they do not say who is right, and OI does not show who is on the other side.

## What NSE data allows and how StockSense computes it

Daily from the F&O bhavcopy since 2024-07 (`flow/oi.py`, dataset `flow_oi_daily`). Intraday while the server runs: NSE's OI spurts list every 180 seconds and, with Angel credentials, near-month futures quotes for the top 50 F&O names (OI is in each quote; NFO tokens are parsed from the scrip master). Rollover percentage and basis are shown beside the label.

## Evidence

Bessembinder and Seguin (1993) and later work relate futures OI to volume and volatility; academic evidence for the four-quadrant labels as stock-return signals is limited. `flow_oi_long_buildup` and `flow_short_covering_rev` are pre-registered, so their forward returns are measured with costs.

## How traders misuse it

Reading intraday OI changes without rollover effects, which dominate near expiry; applying the labels to illiquid stocks; treating a long buildup as a certain continuation.

## Where it appears in the dashboard

Money Flow: quad chip and OI change columns, and the OI quadrant scatter.

## Sources

- https://www.nseindia.com/all-reports-derivatives
- https://scholar.google.com/scholar?q=Bessembinder+Seguin+price+volatility+trading+volume+and+market+depth+futures+markets
