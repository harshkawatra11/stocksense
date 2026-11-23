# Order-book heatmaps and liquidity

## What it is

A liquidity heatmap draws resting limit-order quantity by price over time: price on the vertical axis, time on the horizontal axis, colour for the quantity waiting at each price. Bright bands are large resting orders; thin gaps are places price can move quickly.

## How it is computed

Take snapshots of the order book at a regular interval (here every 5 seconds). For each time bucket (30 seconds) and price bin, keep the maximum resting quantity seen. Plot the matrix with a fixed colour scale.

## What it says about flow and positioning

Resting size marks where participants are willing to trade, and price often pauses at big bids and offers. But resting orders are intentions, not trades: they can be cancelled in milliseconds, and displayed size can be used to mislead other traders (spoofing, which regulators prohibit).

## What NSE data allows and how StockSense computes it

Angel One's full quote gives best-5 levels per side. A deeper 20-level mode is to be probed in card M4.6 and recorded in DECISIONS. History does not exist: depth snapshots are captured only while the server runs in market hours, for at most 25 symbols (top bull and bear names plus NIFTY and BANKNIFTY futures), kept two hours in memory and flushed to the `depth_snaps` dataset each minute. For dates with no capture the page shows "depth not captured for this date". Five levels cover a small price range, so the heatmap is a close-in view, not the whole book.

## Evidence

Cont, Kukanov and Stoikov (2014) show that changes in the best levels explain short-horizon price moves; deeper-book heatmaps are not validated as signals in the literature we know of. Any strategy based on them would have to be tested on our own captured data, which will take months to accumulate.

## How traders misuse it

Treating a large visible order as support or resistance that will hold; ignoring that the other side of a heatmap (hidden or iceberg orders) is invisible; extrapolating from five levels to the whole book.

## Where it appears in the dashboard

Order Flow workbench heatmap panel.

## Sources

- https://arxiv.org/abs/1011.6402
- https://doi.org/10.1017/9781316659335
- https://smartapi.angelone.in/docs
