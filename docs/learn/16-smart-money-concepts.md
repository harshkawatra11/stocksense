# Smart-money concepts (BOS, CHoCH, order blocks, fair-value gaps, sweeps)

## What it is

A vocabulary used by some chart traders: a break of structure (BOS) is a close beyond the last confirmed swing high or low in the trend's direction; a change of character (CHoCH) is the first break against the trend; an order block (OB) is the last opposite candle before a strong move; a fair-value gap (FVG) is a three-candle pattern whose first and third candles leave a price range untouched by the second; a liquidity sweep is a wick through a prior high or low that closes back inside.

## How it is computed

Exactly as in Market Lab's `sm/smc.py` and `sm/structure.py`, which `marketlab/smc.py` and `marketlab/structure.py` port to numpy. Swings are fractal highs and lows with n bars on each side (confirmed n bars later, so a swing is never used before it is known); structure events compare closes with the latest confirmed swing; an FVG exists when the low of bar i+1 is above the high of bar i-1 (bullish) or the mirror; order blocks are tracked until a close breaks them. The port is checked against Market Lab's own output on synthetic bars to within 1e-9.

## What it says about flow and positioning

The idea is that institutions leave footprints at structure breaks. That is a story about unobserved orders: nothing in these definitions uses volume or order data, only candle shapes.

## What NSE data allows and how StockSense computes it

Any bar series works (1m, 5m, 15m, 60m, daily). The tape calculates SMC from 5-minute bars built from 1-minute closes, so tape order blocks and gaps are approximations; the chart popup uses true OHLCV. The ICT unicorn rule `tier_f_ict_unicorn` in the tier list uses order blocks and FVGs.

## Evidence

We know of no peer-reviewed evidence that these patterns predict returns. General evidence on pattern-based technical analysis is mixed (Lo, Mamaysky and Wang 2000 find some information in patterns; Park and Irwin 2007 survey mixed profitability). The patterns are tested only through the pre-registered unicorn rule.

## How traders misuse it

Labelling swings and blocks after the move; drawing many zones so something always "works"; ignoring costs and the number of patterns that failed.

## Where it appears in the dashboard

Smart Money tape tiles and the chart popup layers (order blocks, FVGs, structure events, liquidity levels).

## Sources

- https://github.com/heysanjeevv/market-lab
- https://doi.org/10.1111/0022-1082.00265
- https://doi.org/10.1111/j.1467-6419.2007.00519.x
