# What Indian market data can and cannot show

## What it is

Order-flow tools (footprint, delta, CVD, heatmaps) were built for markets where every trade is published with its aggressor side and the full order book is sold. This note says which of those inputs a retail trader in India actually gets, and what StockSense substitutes. Every estimated tool in the dashboard is labelled as an estimate for this reason.

## How it is computed

Nothing is computed here; this note is the map the other notes refer to. The rule is: if a measure needs a field we do not have, we use a named estimator (BVC, CLV, tick rule on snapshots) and show its name next to the number, for example "delta (BVC estimate from 1-minute bars)".

## What it says about flow and positioning

True order flow means knowing who hit whom: a trade at the ask is a buyer lifting an offer, a trade at the bid is a seller hitting a bid. That is the aggressor side. Without it, "buying volume" and "selling volume" are guesses from price movement. Positioning data we do have is exact but slow: open interest, participant positions, delivery and deals arrive after the close.

## What NSE data allows and how StockSense computes it

| Concept | Needs | We have | Method here | History |
|---|---|---|---|---|
| Footprint, delta, CVD | every trade with aggressor side | not sold to retail; Angel gives LTP, last traded quantity, volume, best-5 depth, total buy/sell quantity | estimates: BVC on bars (`orderflow/delta.py`), CLV, tick rule on streamed LTP | 5m bars from 2024-09, 1m from 2026-03, ticks forward only |
| Big trades | prints with size | volume per bar | bar volume above 5x its same-minute median; price reaction after | 5m from 2024-09 |
| Heatmap | order book over time | best-5 depth snapshots | snapshots every 5 s for up to 25 symbols while the server runs (`orderflow/heatmap.py`) | forward only |
| Absorption | volume at a level without progress | bars | high volume, small range at a day extreme, VWAP band or value-area edge | 5m |
| VWAP, volume profile, market profile | intraday OHLCV | yes | `orderflow/vwap.py`, `profile.py` | 5m / 1m |
| Gamma exposure | option OI by strike plus implied volatility | F&O bhavcopy end of day from 2024-07 (OI in shares); live option chain | Black-Scholes gamma, `flow/options.py` | 2024-07 onward |
| OI buildup | futures OI and price | bhavcopy 2024-07; live OI spurts | four-quadrant label, `flow/oi.py` | 2024-07 onward |
| Delivery | delivery quantity | bhavcopy delivery from 2019-08 | z-score, `flow/delivery.py` | 2019-08 onward |
| FII/DII and participants | NSE files | participant OI from 2019-12; FII/DII cash only about 17 days | long ratios and changes | 2019-12 / forward |
| Pre-open | indicative price and buy/sell quantities | stored from 2026-09 with a gap | `flow/preopen.py` | forward, short |

Why aggressor-tagged ticks are not available: exchanges sell the full tick-by-tick feed with trade direction to institutions and data vendors under paid licences; the free and broker retail feeds are snapshots. A snapshot every second or two cannot recover which side initiated each trade.

## Evidence

Estimating trade direction from bars works only statistically. Easley, Lopez de Prado and O'Hara (2012) show that classifying bar volume by the size of the price change (bulk volume classification) tracks tick-level classification well on liquid futures; it is weaker on thin stocks and short bars. Lee and Ready (1991) is the tick-level benchmark we cannot run without quotes at every trade.

## How traders misuse it

Reading a footprint or delta number built from snapshots as if it were exchange-reported aggressor volume; trusting a "heatmap" of resting orders without knowing that large resting orders can be cancelled before they trade; treating a single bar's estimated delta as a signal.

## Where it appears in the dashboard

Every Order Flow panel title carries its method label, and every API response under `/api/of/` has `estimated: true` and a `data_source` field.

## Sources

- https://doi.org/10.1093/rfs/hhs053
- https://doi.org/10.1111/j.1540-6261.1991.tb02683.x
- https://smartapi.angelone.in/docs
- https://www.nseindia.com/all-reports
