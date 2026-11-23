# Order flow

## What it is

Order flow is the stream of buy and sell orders that arrive at the exchange and the way they change the order book. A market order takes liquidity: it trades immediately against a resting limit order. A limit order provides liquidity: it waits at its price. The side that initiates a trade (the aggressor) is the buyer if the trade happens at the ask and the seller if it happens at the bid.

## How it is computed

Order-flow imbalance (Cont, Kukanov and Stoikov 2014) measures the net change in resting size at the best bid and best ask between two snapshots. If the best bid price rises or its size grows, that counts as buying pressure; if the best ask price falls or its size grows on the ask side that counts as selling pressure. In plain terms: increases in bid size and decreases in ask size are positive flow. They show that over short intervals the price change is close to linearly related to this imbalance, with slope inversely proportional to market depth.

## What it says about flow and positioning

Persistent positive imbalance means buyers are consuming offers faster than sellers replenish them. It says who is more urgent right now, not who is right. A large imbalance with no price progress is the setup for absorption (see note 04).

## What NSE data allows and how StockSense computes it

Angel One full quotes give five bid and ask levels and total buy and sell quantity for each symbol; from H1 the stored `spreads` dataset keeps `depth_imbalance_5`, which is (sum of bid size minus sum of ask size) divided by their total over five levels. That is a snapshot ratio, not the change-based imbalance of the paper; the change-based version needs snapshots every few seconds and is captured forward only while the server runs (`orderflow/heatmap.py`). The tape's flow card shows `depth_imb5` and `buy_sell_ratio` from the live quote.

## Evidence

Cont, Kukanov and Stoikov (2014) find the price impact of order-book events is mostly explained by order-flow imbalance, with a roughly linear relationship that holds across stocks. Their data are tick-level order books of US equities; the relation on NSE retail snapshots has not been measured by us, so StockSense only uses it descriptively.

## How traders misuse it

Treating one snapshot's bid-ask size difference as a forecast, when displayed size can be pulled before it trades; ignoring that imbalance at five levels can be dominated by one large resting order.

## Where it appears in the dashboard

Money Flow page: depth imbalance and buy/sell ratio columns in the stocks table. Order Flow workbench: delta and heatmap panels.

## Sources

- https://arxiv.org/abs/1011.6402
- https://global.oup.com/academic/product/trading-and-exchanges-9780195144703
- https://doi.org/10.1017/9781316659335
