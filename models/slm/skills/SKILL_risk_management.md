# SKILL: NSE Portfolio Risk Management

## Overview
Risk management on NSE requires adapting global frameworks (Kelly criterion, R-multiples) to Indian market specifics: circuit breaker behavior, T+1 settlement, illiquid mid/small caps, SEBI regulations on short selling, and the outsized impact of FII flows on portfolio correlation. This playbook governs position sizing, drawdown rules, and trade-level stop discipline.

---

## 1. Position Sizing

### Kelly Criterion — NSE Adaptation
The full Kelly formula is too aggressive for retail NSE trading. Use **Quarter Kelly** to account for estimation errors and fat-tail risk:

```
Full Kelly f* = (bp − q) / b
  where: b = odds (reward/risk ratio), p = win probability, q = 1 − p

Quarter Kelly (NSE Safe) = f* / 4
```

**Practical Example**:
- Win rate (p) = 0.55, Reward:Risk (b) = 2:1
- Full Kelly = (2 × 0.55 − 0.45) / 2 = 0.325 → 32.5% of capital per trade (DANGEROUS)
- Quarter Kelly = 32.5% / 4 = **8.1%** of capital per trade (acceptable)

### Hard Caps (Override Kelly)
| Rule | Limit |
|------|-------|
| Single stock maximum | 15% of portfolio |
| Single sector maximum | 40% of portfolio |
| Single F&O position | 10% of portfolio (notional) |
| Maximum simultaneous positions | 8–12 (6 for portfolios < ₹5L) |
| Cash reserve (minimum) | 20% always available |

### Why 15% Single Stock Cap?
- NSE stocks can hit 10–20% circuits in a single session on regulatory/promoter news
- 15% cap ensures a single circuit-down event = maximum 3% portfolio loss
- Beyond 15%, concentration risk exceeds the benefit of conviction

---

## 2. Portfolio Heat

### Definition
**Portfolio Heat** = Sum of all open risk (distance from entry to stop × position size) across all open positions, expressed as % of total portfolio.

```
Portfolio Heat = Σ (Risk per Position) / Portfolio Value × 100

Risk per Position = (Entry Price − Stop Price) × Number of Shares
```

### Heat Rules
| Portfolio Heat | Action |
|---------------|--------|
| 0–4% | Normal operations, can add new positions |
| 4–6% | Caution — no new positions until existing risk reduces |
| > 6% | STOP new entries; tighten stops on existing positions |
| > 8% | Reduce at least 2 positions immediately |

### Practical Heat Management
- Check portfolio heat every morning before NSE open (9:00–9:15 AM)
- After a gap-down open that breaches stops, recalculate heat before adding hedges
- F&O positions count at **notional value × delta** for heat calculation, not just premium paid

---

## 3. Maximum Drawdown Rules

### Drawdown Tiers and Response
| Portfolio DD from Peak | Response |
|----------------------|---------|
| 5% drawdown | Review all open positions; no new positions in losing sectors |
| 10% drawdown | **Reduce all position sizes by 50%** immediately |
| 12% drawdown | Close all positions with unrealized losses > 15%; trade paper for 5 days |
| 15% drawdown | **Stop all live trading**; full capital preservation mode |
| 15% DD recovery | Restart with 25% of normal position sizes for first 2 weeks |

### Peak Calculation
- Peak = highest portfolio value over rolling 3-month period (not all-time high)
- Reset peak reference every quarter to avoid permanent "peak anchoring"
- Use MTM (Mark-to-Market) portfolio value including unrealized P&L for drawdown calculation

### Psychological Circuit Breakers
- After 3 consecutive losing trades: Take 1-day break from trading; review trade log
- After 5 consecutive losses: Mandatory 1-week break; strategy review required
- After 2 weeks of red P&L: Reduce trade frequency by 50% for the following month

---

## 4. Correlation Awareness

### Why Correlation Matters on NSE
FII flows affect all large-cap stocks simultaneously. During risk-off events (US Fed decision, INR crisis), correlations across NSE stocks spike toward 0.8–0.9, eliminating diversification benefits.

### Correlation Rules
- **Avoid 3+ correlated positions in the same sector** simultaneously
- Correlation threshold: if two stocks have 30-day rolling correlation > 0.70, treat them as ONE position for heat calculation
- Common high-correlation pairs on NSE:
  - HDFC Bank + ICICI Bank + Kotak (correlation typically 0.75–0.85)
  - TCS + Infosys + HCL Tech (correlation 0.80–0.90)
  - Tata Motors + M&M (EV play correlation spikes > 0.70)
  - ONGC + Oil India (correlation > 0.85)

### Sector Concentration Check
Before entering a new position, verify:
1. Current sector allocation < 40%
2. New position does not create 3rd correlated holding in same sector
3. If market is in high-correlation regime (VIX > 18), treat all BFSI as one sector

### Cross-Asset Correlation Risks
- Gold ETFs: Negative correlation to equities (-0.3 to -0.5) — useful hedge
- Nifty IT vs. USD/INR: Positive (IT earns in USD) — use INR strength as IT hedge signal
- Nifty Metal vs. China PMI: High positive correlation — track monthly China PMI data

---

## 5. Stop Loss Discipline

### Golden Rules (Never Break)
1. **Never widen a stop loss** after entry — it was set for a reason
2. **Never average down** on a losing position without a new valid setup signal
3. **Close-based stops** for swing trades; **intraday hard stops** for F&O and intraday
4. If you miss a stop execution (fast market, circuit), exit at next available price — no "hoping"
5. Stop is placed at order entry time, not "mentally tracked"

### Trailing Stop Protocol
| Trade Progress | Action |
|---------------|--------|
| Entry hit | Initial stop at swing low/high |
| Target 1 reached (1.0R) | Move stop to breakeven |
| Target 1.5R reached | Trail stop to 0.5R profit |
| Target 2R reached | Trail to 1R profit; reduce position 50% |
| Target 3R+ | Trail by ATR (1× daily ATR below current price for longs) |

### Stop Placement Best Practices
- **BUY**: Stop below the most recent swing low minus 0.25% buffer
- **SELL/SHORT**: Stop above the most recent swing high plus 0.25% buffer
- **Pivot-based stop**: Stop below S1 (daily pivot) for long trades entered near PP
- **ATR-based stop**: Stop = Entry − (1.5 × 14-day ATR) for volatile mid-caps
- Minimum stop distance: 0.5% from entry (below this = too tight, noise will trigger it)
- Maximum stop distance: 4% from entry (above this = R:R becomes unfavorable, reduce size)

### What to Do When Stop is Hit
1. Exit immediately — no second-guessing
2. Log the trade in trade journal (entry, exit, reason for stop hit)
3. If stopped out 3 times on the same stock in 10 days: blacklist the stock for 1 month
4. Do NOT re-enter the same position the same day it stopped you out

---

## 6. R-Multiple Tracking

### Definition
R = Risk on the trade (Entry − Stop × Shares = 1R loss)
- Trade outcome expressed as multiples of initial risk
- Target: Average winning trade ≥ 2R; average losing trade = −1R

### R-Multiple Performance Benchmarks
| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Average Win (R) | 1.5R | 2.0R | 3.0R+ |
| Average Loss (R) | −1.0R | −0.8R | −0.6R |
| Win Rate | 40% | 50% | 55%+ |
| Expectancy (R) | 0.3R | 0.6R | 1.0R+ |

```
Expectancy = (Win Rate × Avg Win R) − (Loss Rate × Avg Loss R)
Minimum acceptable: Expectancy ≥ 0.3R per trade
```

### R-Multiple Journal Template
```
Date | Stock | Entry | Stop | Target | Exit | R-Result | Notes
2024-01-15 | RELIANCE | 2450 | 2420 | 2510 | 2508 | +1.93R | Gap-up open, partial exit
```

### Quarterly R-Multiple Review
- If 3-month expectancy < 0.2R: Strategy is broken — pause and review
- If win rate drops below 35%: Adjust entry criteria (stricter volume confirmation)
- If average win < 1.5R: Targets are too conservative — extend to next pivot level

---

## 7. NSE Circuit Breaker Behavior

### Index-Level Circuit Breakers
| Market Halt | Trigger | Duration |
|------------|---------|---------|
| 10% drop in Nifty/Sensex | Before 1 PM: 45 min halt | Review at re-open |
| 10% drop in Nifty/Sensex | After 1 PM, before 2:30 PM: 15 min halt | Review at re-open |
| 10% drop in Nifty/Sensex | After 2:30 PM: No halt | Trade until close |
| 15% drop | Before 2 PM: 1 hour 45 min halt | Resume or close |
| 15% drop | After 2 PM: Close market for the day | — |
| 20% drop | Any time: Close for the day | — |

### Stock-Level Circuits (Individual Stocks)
| Category | Upper/Lower Circuit Limit |
|---------|--------------------------|
| F&O stocks | No daily circuit (but dynamic price bands apply) |
| Non-F&O > ₹20 price | 5%, 10%, or 20% (assigned by NSE) |
| Illiquid/penny stocks | 5% circuit common |

### How to Handle Circuit Situations
**If holding a stock that hits lower circuit:**
1. Do NOT panic sell at circuit — you may be in the queue without execution
2. Check if circuit is temporary (news-driven) or fundamental (fraud, regulatory action)
3. If fundamental: Place sell order at lower circuit limit and wait (may need multiple sessions)
4. If temporary: Monitor next day open; circuit may lift with counter-news

**If circuit hits on a short position (upper circuit):**
1. Immediate risk: Short squeeze in cash segment impossible (cannot borrow easily)
2. For F&O shorts: Close at market immediately if underlying hits upper circuit; option premiums collapse on calls
3. Never hold naked short into a stock that keeps hitting upper circuit for 2+ days

### Portfolio Protection During Market-Wide Circuits
- Keep 20% cash reserve to buy quality large-caps at circuit-induced dislocations
- Nifty 50 stocks rarely hit individual circuits — preferred during market-wide stress
- Have Nifty Put options as hedge (2–5% OTM, 1-month expiry) during high-VIX periods

---

## 8. Illiquid Stock Risks

### Identifying Illiquidity
| Metric | Liquid | Illiquid |
|--------|--------|---------|
| Daily Volume | > 5 lakh shares | < 50,000 shares |
| Bid-Ask Spread | < 0.1% | > 0.5% |
| Impact Cost (NSE) | < 0.15% | > 0.5% |
| Market Cap | > ₹5,000 Cr | < ₹1,000 Cr |

### Bid-Ask Spread Impact on Returns
```
Effective Cost = Entry Spread + Exit Spread + Brokerage
Example: Stock with 1% spread = 2% round-trip cost before profit
NSE STT (0.1% on delivery sell) + Brokerage (0.1–0.3%) + Spread = 2.4%+ total friction
```

### Rules for Illiquid Stocks
1. **Maximum 5% of portfolio** in any stock with spread > 0.3%
2. Use **limit orders only** — never market orders in illiquid stocks
3. Exit strategy must be planned before entry (who will buy at the target?)
4. Avoid illiquid stocks in the last 30 minutes of trading (spread widens)
5. Check if stock is on NSE's trade-to-trade (T2T) segment — T2T stocks cannot be squared intraday, require full delivery

### Impact Cost Formula
```
Impact Cost = (Actual Price − Ideal Price) / Ideal Price × 100
Ideal Price = (Best Ask + Best Bid) / 2
```
NSE publishes impact cost monthly for F&O-eligible stocks — use as liquidity filter.

---

## 9. FII vs. DII Flow as Portfolio Tail Risk Indicator

### Data Sources and Timing
- **Provisional FII/DII data**: Available by 5 PM NSE on trading day
- **Final FII/DII data**: SEBI website by 7 PM
- **Sectoral FII data**: Available monthly via SEBI FPI disclosure

### Interpreting FII/DII Flows
| FII Activity | DII Activity | Market Implication |
|-------------|-------------|-------------------|
| Net Buyer | Net Buyer | Strong bull — go long, high conviction |
| Net Buyer | Net Seller | Moderate bull — DII profit-taking, FII driving rally |
| Net Seller | Net Buyer | Tug of war — DII supports dips; range-bound |
| Net Seller | Net Seller | Bear signal — reduce portfolio heat immediately |

### Tail Risk Triggers from FII Flows
- **FII sell > ₹5,000 Cr/day**: Significant outflow — reduce equities by 20% same day
- **FII sell > ₹10,000 Cr/day**: Panic signal — move to 50% cash, hedge remaining with Nifty Puts
- **FII sell > ₹15,000 Cr/day**: Extreme event (occurs during global crises) — full defense mode
- **Consecutive FII sell days (7+)**: Historical correlation with Nifty 5–10% decline; activate drawdown protocol

### USD/INR as FII Flow Predictor
- INR depreciates (USD/INR rises): FII USD-denominated returns shrink → accelerated selling
- Monitor USD/INR at 84.50: Historical FII selling threshold
- Rule: If USD/INR > 85 for 3 consecutive days: Reduce portfolio heat to < 4%, avoid fresh longs in FII-heavy sectors (BFSI, IT)

---

## 10. Earnings Season Blackout Rules

### Blackout Periods
- **Individual stock blackout**: No new positions in a stock **2 trading days before** its quarterly result announcement
- **Sector blackout**: When 30% of sector stocks are in results week, avoid new sector entries
- **Market-wide results season** (April, July, October, January): Reduce swing trade frequency by 30% — higher volatility, unpredictable gaps

### Why Blackout?
- Post-results gaps can be 5–15% in either direction — a 2% stop becomes irrelevant
- IV (implied volatility) inflates options prices pre-results; stocks often gap through any technical level
- Blackout protects from binary event risk that is fundamentally unforecastable

### How to Use Results Season
1. **Pre-results screening**: Identify stocks with strong technical setup but avoid entry until results pass
2. **Post-results entry**: Wait 1 full trading day after results for dust to settle; enter on established new trend
3. **Earnings momentum plays**: For stocks that beat significantly (EPS > 10% above consensus) + gap-up > 2% + high delivery: Enter on first pullback to VWAP on results day — valid swing setup
4. **Earnings disappointment shorts**: Stock gaps down > 3% on miss + guidance cut + high volume: Short on first bounce toward gap (VWAP resistance) with stop above gap-fill level

### Results Calendar Sources
- NSE website: Corporate announcements section (updated live)
- Trendlyne / Screener.in: Bulk results calendar with analyst estimates
- Check at start of each week for the coming 5 days

---

## Quick Risk Checklist (Before Every Trade)

- [ ] Portfolio heat after this trade < 6%?
- [ ] Position size ≤ 15% of portfolio?
- [ ] Sector allocation ≤ 40% after this trade?
- [ ] Stop loss placed at order entry?
- [ ] R:R ratio ≥ 2:1?
- [ ] No earnings for this stock within 2 trading days?
- [ ] Portfolio not in drawdown > 10% from peak?
- [ ] Correlation check: Not creating 3rd correlated position in same sector?

*All 8 must be YES — if any is NO, do not enter the trade.*

---

*Last Updated: NSE Risk Management Playbook v2.1 | Framework applies to equity cash + F&O swing positions*
