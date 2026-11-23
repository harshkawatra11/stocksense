# Options positioning and gamma exposure

## What it is

Option open interest (OI) by strike shows where traders have positioned. Put-call ratio (PCR) compares put and call OI. Max pain is the expiry price at which option buyers' total payoff is lowest. Gamma exposure (GEX) estimates how much stock or futures market makers must buy or sell as price moves, assuming they hedge their option books.

## How it is computed

- PCR (OI) = total put OI divided by total call OI. PCR of change in OI uses the day's change.
- Max pain: the strike that minimises the sum of call and put payoffs over all open contracts.
- Black-Scholes gamma per unit of spot: phi(d1) / (S x sigma x sqrt(t)).
- GEX at a strike = gamma x OI (in shares) x S squared x 0.01, positive for calls and negative for puts, in rupees per 1% move; total GEX is the sum. The zero-gamma level is the spot price at which total GEX crosses zero.
- Call wall and put wall: the strikes with the largest call OI at or above spot and put OI at or below spot.
- Implied volatility comes from the end-of-day settle price (`features/derivs.py:implied_vol`); option rows on their own expiry day are excluded because the settle price equals the underlying then.

## What it says about flow and positioning

The sign convention assumes dealers are long calls and short puts (customers sell calls and buy puts). Positive total GEX then means dealers buy dips and sell rallies, damping moves; negative means they sell into falls and buy into rallies, amplifying moves. Walls and max pain mark strikes where option positioning is concentrated and where pinning near expiry is claimed.

## What NSE data allows and how StockSense computes it

`flow/options.py` builds these from the F&O bhavcopy (OI in shares, so no lot size is needed; history from 2024-07) and from the live option chain while the server runs (live OI is in contracts and is multiplied by the lot size, which is probed and recorded in DECISIONS). Total GEX and its regime are shown for NIFTY and BANKNIFTY. The dealer-position assumption is only an assumption: true dealer inventory is not published.

## Evidence

Garleanu, Pedersen and Poteshman (2009) show option prices depend on end-user demand when dealers cannot hedge perfectly. Ni, Pearson, Poteshman and White (2021) find option-related hedging affects underlying prices. Barbon and Buraschi (2021) link dealer gamma imbalance to intraday momentum and reversal ("gamma fragility"). All are mostly US studies and indexes; the strategy `flow_gex_regime` tests the idea on NIFTY in the pre-registered library.

## How traders misuse it

Treating GEX as exact when the dealer position is assumed; trading walls as hard levels; ignoring that expiry-day mechanics differ.

## Where it appears in the dashboard

Money Flow gamma cards (regime, zero-gamma, walls, max pain, PCR) and the Options positioning page (`/flow/options`).

## Sources

- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3725454
- https://scholar.google.com/scholar?q=Garleanu+Pedersen+Poteshman+Demand-based+option+pricing
- https://scholar.google.com/scholar?q=Ni+Pearson+Poteshman+White+Does+option+trading+have+a+pervasive+impact+on+underlying+stock+prices
- https://www.nseindia.com/option-chain
