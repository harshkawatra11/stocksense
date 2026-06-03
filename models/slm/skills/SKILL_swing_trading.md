# SKILL: NSE Swing Trading Playbook

## Overview
Swing trading on NSE targets multi-day price moves, typically holding positions for **2–10 trading sessions**. The goal is to capture 3–8% moves in large-caps and 5–15% in mid/small-caps while managing overnight risk through disciplined stop placement and position sizing.

---

## 1. Multi-Day Setup Framework

### Trade Duration by Category
| Stock Category | Typical Hold | Target Move | Stop Width |
|---|---|---|---|
| Nifty 50 large-cap | 2–5 days | 3–5% | 1.5–2% |
| Nifty Midcap 150 | 3–7 days | 5–10% | 2–3% |
| Nifty Smallcap 250 | 5–10 days | 8–15% | 3–5% |

### Core Swing Philosophy
- Enter at inflection points (support/resistance), not mid-range
- Always define stop BEFORE entry; position size derives from stop width
- Ride winners; cut losers quickly — asymmetric R:R is the edge
- Prefer stocks in strong uptrending sectors (sector tailwind = higher hit rate)

---

## 2. Support and Resistance Identification

### Pivot Points (Daily Chart)
Standard pivot formula used across NSE derivatives desks:

```
Pivot (P)  = (High + Low + Close) / 3
R1 = 2P − Low        S1 = 2P − High
R2 = P + (High−Low)  S2 = P − (High−Low)
R3 = High + 2(P−Low) S3 = Low − 2(High−P)
```

- Use **weekly pivots** for swing entries; daily pivots for intraday confirmation
- Nifty and BankNifty pivots are published pre-open by all major brokers — reference these as market-wide S/R

### Previous Highs and Lows
- **52-week high breakout**: High-conviction swing entry if accompanied by volume >1.5× 20-day average. Target: 5–8% above breakout. Stop: close below 52W high level.
- **Prior swing high reclaim**: Stock breaks above a swing high formed 3–8 weeks ago — signals institutional accumulation complete.
- **Prior swing low as stop**: For BUY entries, place stop 1–2% below the most recent significant swing low visible on daily chart.
- **Monthly/Quarterly highs-lows**: Institutional desks track these; breakouts above quarterly high = strong swing signal.

### Confluence Zones
A support/resistance zone is stronger when 3+ factors align:
- Pivot level
- Previous swing high/low
- Round number (e.g., ₹500, ₹1000, ₹2500)
- 200-DMA or 50-DMA
- Fibonacci retracement (38.2%, 50%, 61.8% of prior swing)

---

## 3. Gap Analysis

### Gap Classification
| Gap Type | Definition | Probability of Fill | Action |
|---|---|---|---|
| Common Gap | <1% gap, low volume | 70–80% fill within 3 days | Fade the gap (sell gap-up, buy gap-down) |
| Breakaway Gap | >2% gap, high volume, breaks key level | 15–25% fill | Trade in gap direction |
| Runaway/Continuation Gap | Mid-trend, moderate volume | 30–40% fill | Hold existing position, add on retest |
| Exhaustion Gap | End of trend, extreme volume spike | 85–95% fill | Reversal trade |

### NSE-Specific Gap Behavior
- **SGX Nifty / Gift Nifty** (pre-market): Gaps >0.5% on Gift Nifty at 8:45 AM IST are reliable directional indicators for the first 30-minute session
- **F&O expiry gaps**: Gap-ups on expiry Thursdays (weekly) often get faded by 11:30 AM as writers take profits
- **Result-day gaps**: Earnings gap-ups >5% on high volume — do NOT chase. Wait for a 1–2 day pullback/consolidation before entering swing
- **Gap-fill trades**: For common gaps in large-caps (Reliance, HDFC Bank, Infosys), 70%+ fill within 2–3 sessions — valid mean-reversion swing setup

### Gap-Up Classification Decision Tree
```
Gap > 2% + Volume > 1.5× avg → Breakaway → Trade direction
Gap > 2% + Volume < avg       → Potential Exhaustion → Watch for reversal signals
Gap 0.5–2% + No news          → Common Gap → Expect fill, fade or wait
Gap < 0.5%                    → Noise, ignore for swing
```

---

## 4. Bollinger Band Squeeze (Volatility Contraction)

### Setup Rules
- **Squeeze**: BB Width (Upper − Lower / Middle) drops to lowest in 6 months
- **Trigger**: Price closes outside the band after 5+ days of squeeze
- **Volume confirmation**: Volume on breakout day > 1.3× 20-day average

### Entry Protocol
1. Identify stocks with BB Width in bottom 10th percentile (use screener)
2. Set alert for price closing above upper band (bullish) or below lower band (bearish)
3. Enter next morning's open (or on retest of band if gap-up occurs)
4. Stop: opposite band at time of entry
5. Target: 1.5–2× the width of the BB at time of squeeze

### NSE Application
- BB squeezes in **Nifty IT index stocks** (TCS, Infosys, HCL Tech) before US earnings season — trade the break
- **Banking stocks** often squeeze ahead of RBI policy — direction determined by rate decision
- Small-cap BB squeezes carry higher false-breakout risk; require delivery volume confirmation

---

## 5. Inside Bar Setups

### Definition
An inside bar is a candle whose High and Low are both within the previous day's High-Low range. It signals indecision and compression before a directional move.

### Setup Rules
- Look for inside bars after a strong trending move (not in choppy sideways markets)
- **Double inside bar** (two consecutive inside bars) = stronger setup
- Enter on breakout of the mother bar's High (for BUY) or Low (for SELL)
- Stop: opposite side of mother bar
- Target: 1.5–2× mother bar range projected from breakout point

### NSE Notes
- Inside bars on **weekly charts** are more reliable than daily
- Inside bars in **F&O stocks** can be traded with OTM options for defined risk (buy calls on breakout above mother high)
- Avoid inside bar setups in stocks with results within 5 days (event risk distorts pattern)

---

## 6. VWAP Reclaim Plays

### Definition
VWAP Reclaim = stock was trading below VWAP, then reclaims it with conviction (price holds above VWAP for 2+ candles on meaningful volume).

### Swing Application (using Anchored VWAP)
- **Anchor VWAP to**: last major swing low, earnings date, or 52-week low date
- If stock reclaims anchored VWAP after extended period below it → institutional re-accumulation signal
- Entry: first close above anchored VWAP
- Stop: back below anchored VWAP (on closing basis)
- Target: prior resistance / previous swing high

### Intraday VWAP for Swing Entry Timing
- Even for swing trades, use 15-minute chart VWAP to time entry
- Best entries: price dips to VWAP in first 90 minutes, holds it, then resumes uptrend
- If SPX (via GIFT Nifty) is positive and stock holds VWAP — high-probability swing entry

---

## 7. Sector Rotation Entries

### Rotation Sequence (Bull Market)
```
IT → Banking → Auto → FMCG → Pharma → Metal → Realty
(Leadership rotates roughly every 4–8 weeks in trending markets)
```

### Entry Strategy
1. Identify which sector is beginning to outperform (rising relative strength vs Nifty 50)
2. Find the 2–3 strongest stocks within that sector (highest RS, cleanest chart)
3. Enter when individual stock breaks out or pulls back to support while sector is still outperforming
4. Exit when sector RS begins declining (sector rotation into new leader)

### NSE Sector Signals
- **FII buying concentrated in sector**: Visible in SEBI F&O data and bulk deals — follow FII sector preference
- **Nifty sector indices** (Bank Nifty, Nifty IT, Nifty Pharma): When a sector index breaks 52-week high, constituent stocks offer swing entries
- **PSU stocks** outperform during election run-up (government spending narrative)

---

## 8. F&O Expiry Week Behavior

### Weekly Expiry (Every Thursday — Bank Nifty, Nifty, and stocks)
- **Monday–Tuesday**: Max pain drives price; market makers push toward max pain strike
- **Wednesday**: Volatility increase as traders roll or square positions
- **Thursday AM**: Pin to max pain level is common in morning session
- **Thursday PM (after 1 PM)**: Sharp moves as positions unwound; momentum can accelerate
- **Strategy**: Avoid entering new swing trades on Wednesday–Thursday; exit existing swings by Wednesday close if target not hit

### Monthly Expiry (Last Thursday of month)
- Monthly expiry week sees **higher-than-normal volatility**
- **Short squeeze potential**: Stocks with high short OI can spike as positions covered
- **Rollover data**: If >70% positions rolled to next month on Tuesday/Wednesday, indicates institutional conviction in direction
- **FII index positions**: FII net long/short on index futures is a directional tell — check NSE F&O data daily

### Max Pain Calculation
```
Max Pain = Strike where total open interest (calls + puts) loss is minimized for option sellers
```
- Available on NSE website and tools like Sensibull, Opstra
- Nifty/BankNifty tend to close within 1–2% of max pain on expiry Thursdays (80% of weeks historically)
- Use max pain as a swing trade exit zone if approaching expiry

---

## 9. Delivery Volume Confirmation

### Why Delivery Volume Matters
Delivery volume (shares actually delivered, not squared intraday) = genuine investor commitment. It filters out noise from intraday speculation.

### NSE Delivery Data
- Available on **NSE Bhavcopy** (end-of-day, free download)
- Key metric: **Delivery %** = Delivery Volume / Total Volume × 100
- High delivery % = informed, longer-horizon buyers (strong signal)
- Low delivery % = intraday noise, pattern less reliable

### Interpretation Rules
| Delivery % | Signal |
|---|---|
| >60% on breakout day | Strong institutional buying; high-conviction swing entry |
| 40–60% | Moderate; confirm with price action next day |
| <30% on breakout | Weak hands; wait for delivery to confirm before entry |
| Rising delivery % over 3 days | Accumulation phase; swing buy setup forming |

### Application
- Before taking any swing trade, check delivery % for the last 2–3 days
- For breakouts: delivery % on breakout day must be >50% (ideally >60%)
- For pullback entries: rising delivery % on down days = institutional buying the dip

---

## 10. Position Sizing for Swing Trades

### Core Rule: Maximum 15% per Position
- No single swing trade should risk more than **2% of portfolio capital**
- Position size = (Portfolio Capital × Risk %) / (Entry Price − Stop Price)
- Maximum position value capped at **15% of total portfolio** regardless of stop width

### Sizing Formula
```
Risk Amount = Portfolio Value × 0.02 (max 2% risk per trade)
Position Size (shares) = Risk Amount / (Entry − Stop)
Position Value = Position Size × Entry Price
If Position Value > 0.15 × Portfolio Value → reduce shares until ≤ 15%
```

### Example
- Portfolio: ₹10,00,000
- Max risk per trade: ₹20,000 (2%)
- Entry: ₹500, Stop: ₹480 (₹20 risk per share)
- Shares: ₹20,000 / ₹20 = 1,000 shares
- Position value: ₹5,00,000 = 50% of portfolio → EXCEEDS 15% cap
- Adjusted: ₹10,00,000 × 15% = ₹1,50,000 / ₹500 = 300 shares (actual risk = ₹6,000 = 0.6% of portfolio)

---

## 11. Stop Loss Placement

### For BUY (Long) Trades
- Stop = 1–3% **below the most recent swing low** on daily chart
- For breakout trades: stop = below the breakout level (or below the pre-breakout base)
- For pullback entries: stop = below the low of the pullback candle

### For SELL (Short) Trades — F&O Only (NSE short selling requires F&O or same-day delivery only)
- Stop = 1–3% **above the most recent swing high** on daily chart
- For breakdown shorts: stop = above the breakdown level
- In cash market: short selling only allowed intraday; use PUT options or futures for multi-day shorts

### Stop Discipline
- **Never widen a stop** — if the reason for the trade is broken, exit
- Once position is up 1.5× your initial risk (1.5R), trail stop to breakeven
- Once position is up 2R, trail stop to 1R profit
- Hard stop: if stock hits stop on intraday basis + closes past it → exit without exception

---

## 12. Overnight Hold vs. Intraday Exit Decision

### Hold Overnight If:
- Trade is in profit and trend is intact (above key moving averages)
- No binary event overnight (results, RBI meeting, US Fed meeting, geopolitical risk)
- Volume pattern supports continuation (delivery volume rising)
- Broader market (Nifty) is in uptrend; GIFT Nifty flat to positive
- Position size is within swing sizing limits (overnight risk manageable)

### Exit Intraday (Do NOT Hold) If:
- Stock is at resistance and stalling (volume drying up near target)
- Binary event tonight or pre-market (results, FDA approval, court ruling)
- Broader market breaking down sharply (Nifty down >1.5% intraday)
- Position is at loss and the trade thesis has not played out as expected
- F&O expiry is next day and position is options-based

---

## 13. NSE-Specific Notes

### Settlement: T+1
- Since January 2023, NSE equity trades settle T+1 (next trading day)
- Delivery-based swing trades: shares arrive in demat account next day
- Short selling beyond intraday requires F&O (futures/put options) — cannot deliver short in cash market
- MTM (mark-to-market) gains/losses settle next day; plan liquidity accordingly

### Illiquidity in Small-Caps After Hours
- Small-cap stocks (daily volume < ₹5 crore) can have **wide bid-ask spreads** at market open
- Use **limit orders** only for small-cap swing entries; never market orders
- Impact cost (slippage) can be 0.5–2% in illiquid small-caps — factor into target calculations
- Avoid holding illiquid small-caps over weekends (Monday gap risk + no exit options)
- ASM (Additional Surveillance Measure) / GSM (Graded Surveillance Measure) stocks: extreme volatility and liquidity risk; avoid for swing trading

### FII Activity Patterns
- **FII buying phase** (typically Oct–Dec, post-monsoon): Broad market uptrend; swing success rate high
- **FII selling phase** (typically Jan–Mar, US rate fears): Higher whipsaw; tighten stops, reduce position size
- **FII data source**: NSE website publishes daily FII/DII equity buy-sell data by 7 PM
- When FIIs are net buyers >₹3,000 crore/day for 3+ consecutive days: enter aggressive swing longs
- When FIIs net sell >₹5,000 crore/day: reduce swing book, avoid new longs until stabilization
- **Sectoral FII preference**: FIIs favor Financials, IT, Consumer Staples — swing trades in these sectors have higher liquidity and smoother exits
