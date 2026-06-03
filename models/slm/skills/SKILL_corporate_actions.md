# SKILL: NSE Corporate Action Impact Playbook

## Overview
Corporate actions on NSE create predictable short-term price patterns driven by institutional repositioning, retail sentiment, arbitrage mechanics, and mandatory SEBI disclosure requirements. This playbook covers the complete lifecycle of each corporate action type — from announcement to post-event behavior — with specific NSE data sources and execution guidelines.

---

## 1. Quarterly Results — Earnings Impact Playbook

### Earnings Surprise Formula
```
Earnings Surprise % = ((Actual EPS − Street Estimate EPS) / |Street Estimate EPS|) × 100

Positive Surprise: Actual > Estimate → Stock likely to gap up
Negative Surprise: Actual < Estimate → Stock likely to gap down

Street Estimate: Consensus from Bloomberg, Refinitiv, or Indian platforms (Trendlyne, Screener)
```

### Pre-Results Strategy (2 Weeks Before)
- **Buy the Rumor**: For stocks with strong price momentum (52-week high proximity) AND improving sector fundamentals, initiate position 10–14 days before results
- Entry: Buy 60% position at T-14, add 40% at T-7 if setup still intact
- Stop: Below the swing low formed just before entry
- Exit plan: If stock surges 5%+ before results day → exit 50–75% and hold balance into results
- Do NOT enter if stock already rallied > 15% in the preceding 3 weeks — risk of "sell the news" on even a beat

### Results Day — "Sell the News" Identification
Signs that a beat is already priced in (SELL on results):
1. Stock up > 20% in 60 days before results
2. Analyst upgrades already published in prior 2 weeks
3. FII bulk buying already visible in delivery data
4. Management guidance language: "In line with expectations" = no positive surprise

Signs of genuine surprise (HOLD or ADD on results):
1. Management upgrades guidance significantly (revenue growth upgraded by 3%+)
2. Margin expansion beats by > 100 bps vs. estimates
3. New order wins / contract announcements simultaneous with results
4. Stock was NOT a recent momentum darling — was ignored by market

### Earnings Gap Behavior
| Surprise Level | Expected Gap | Trade Strategy |
|---------------|-------------|----------------|
| Beat > 15% EPS | Gap-up 5–10% | Wait for 30-min VWAP establishment; buy only on VWAP hold |
| Beat 5–15% | Gap-up 2–5% | Wait; gap may partially fill in first hour |
| In-line (±5%) | Gap ±1% | Trade with general market direction |
| Miss 5–15% | Gap-down 2–5% | Do NOT buy; potential for continued selling |
| Miss > 15% | Gap-down > 5% | SELL immediately; earnings downgrades follow |

### Post-Results Monitoring
- **Analyst rating revisions** (next 3–5 days): Check for upgrades — institutional accumulation follows
- **Management commentary on concall**: Revenue guidance more important than headline EPS for stock direction
- **4-quarter EPS trend**: If Q4 beats but full-year EPS is lower than 2 years ago → not a genuine turnaround

---

## 2. Stock Splits

### How Splits Work on NSE
- **Board approval**: Stock split announced with ratio (e.g., 1:2, 1:5, 1:10)
- **Record date**: Shareholders holding on this date receive split shares (T+1 settlement consideration)
- **Ex-date**: One trading day before record date; price adjusts on ex-date

### Historical Price Patterns
| Event | Average Price Reaction | Notes |
|-------|----------------------|-------|
| Announcement day | +5% to +8% | Retail euphoria, liquidity expectations |
| Between announcement and record date | +3% to +7% cumulative | Continued momentum |
| Ex-date / Record date | −3% to −5% | Profit-taking, "sell the news" |
| 1 month post-split | +2% to +8% | Improved retail participation, wider ownership |

### Why Splits Matter Operationally
1. **Liquidity improvement**: Lower face value attracts retail traders unable to afford high-price stocks (e.g., MRF at ₹1,50,000 pre-split vs. ₹15,000 post 1:10 split)
2. **F&O lot size**: Post-split, new F&O lot sizes are adjusted (NSE adjusts lot sizes for derivative contracts)
3. **Index weight**: Split does not change market cap or index weight — purely cosmetic for index

### Split Play Strategy
- Buy 1–3 days after announcement on first pullback
- Target: 5–8% from announcement level before record date
- Exit: On or before ex-date (capture full pre-event rally)
- Avoid: Re-entering after ex-date unless fundamental catalyst exists

---

## 3. Rights Issues

### Mechanics
- Company offers existing shareholders the right to buy new shares at a **discount to market price**
- Example: Market price ₹500, Rights issue price ₹400 = 20% discount
- Rights entitlement traded on NSE for a specified period (check NSE circular for dates)

### Why Rights Issues Are Typically Bearish Short-Term
1. **Dilution signal**: Company needs capital — often implies stress or heavy capex that may not deliver near-term
2. **Discount creates arbitrage**: Shareholders can sell in market and subscribe at discounted rights price
3. **Supply overhang**: Large issuances add share supply; market cap-neutral but creates selling pressure

### Rights Issue Price as Arbitrage Anchor
```
Theoretical Ex-Rights Price (TERP) = (Market Price × Old Shares + Rights Price × New Shares) / Total Shares Post-Rights

Arbitrage: Buy rights entitlement at discount, subscribe, and sell post-allotment
```

### Trading Strategy Around Rights Issues
- **On announcement (bearish)**: Sell 30–50% of existing position; rights issues rarely signal positive fundamental surprise
- **During rights trading window**: Check if rights entitlement trades at significant discount to intrinsic value — potential arbitrage
- **Post-allotment**: Hold only if: company has clear capital deployment plan (capacity expansion, debt reduction) + management credibility intact
- **Red flag**: Rights issue to fund working capital gaps or operating losses → EXIT all positions

---

## 4. Bonus Issues

### Mechanics
- Company issues free additional shares in ratio to existing holdings (e.g., 1:1 bonus = double shares)
- No cash exchange; company transfers from free reserves to share capital
- Ex-bonus date: Price adjusts downward proportionately (1:1 bonus = price halved)
- NSE adjusts historical price charts (backward-adjusted) on ex-date

### Historical Price Patterns
| Event | Average Price Reaction | Notes |
|-------|----------------------|-------|
| Announcement day | +2% to +4% | Positive sentiment signal |
| Until ex-date | +1% to +3% additional | Retail buying on "cheaper price" expectation |
| Ex-date | Price adjusts by bonus ratio | Not a real loss — shares double |
| 3-month post-bonus | Flat to −2% vs. pre-announcement | No long-term value creation |

### Key Insight
**Bonus shares do not create shareholder value** — merely cosmetic restatement of reserves into share capital. The company's fundamentals, earnings, and growth outlook are unchanged.

### When Bonus IS a Signal
- Bonus issue after strong earnings growth = management confidence signal
- Company with clean balance sheet + no debt + consistent profits + bonus = quality sign
- Do NOT treat bonus as "free money" — it is a capital restructuring, not a gift

### Trading Strategy
- Small speculative buy on announcement day if stock is technically breaking out
- Hold until ex-date; exit before if rally exceeds 8%
- Do NOT hold purely for bonus — fundamentals matter more

---

## 5. Dividends

### Dividend Types on NSE
- **Interim dividend**: Declared between annual reports; quick cash return
- **Final dividend**: Declared at AGM; confirmed in annual report
- **Special dividend**: One-time, often from asset sale proceeds

### Ex-Date Mechanics
```
Stock price on ex-date = Previous close − Dividend amount (exactly)
Example: Stock at ₹500, ₹15 dividend → Ex-date opens at ₹485
The ₹15 drop is NOT a loss — you receive ₹15 in cash (post TDS)
```

### Dividend TDS on NSE (Important)
- Dividend > ₹5,000/year from one company: 10% TDS deducted (for resident Indians)
- NRI investors: 20% TDS on dividends
- Net dividend yield = Gross yield × (1 − TDS rate)

### Dividend Yield as Valuation Anchor
| Dividend Yield | Market Behavior |
|---------------|----------------|
| > 3% | Attracts delivery-based institutional buyers (insurance, pension funds) |
| > 5% | Strong floor support — institutional mandate buying creates price support |
| < 1% | Growth stock — dividend is token; not a valuation anchor |

### Dividend Capture Strategy (NSE)
- Buy 5–7 trading days before ex-date (T+1 settlement means you need to buy before record date)
- Stock often rallies 2–5% in run-up as yield seekers accumulate
- Sell on ex-date morning after gap-down (you capture dividend + any pre-ex run-up)
- **Risk**: If market corrects significantly before ex-date, dividend income does not compensate stock loss

### High-Yield NSE Stocks (Structural Dividend Payers)
- Coal India, ONGC, NTPC: > 4–6% yield consistently
- HDFC Bank, Infosys: Lower yield but growing dividend trajectory
- PSU stocks often used by domestic institutions for dividend income — creates natural demand floor

---

## 6. Promoter Buying and Selling

### SEBI Disclosure Requirements
- Promoters must disclose any purchase/sale within **2 trading days** to NSE/BSE
- Insider trading regulations apply — promoters cannot trade 60 days before and 48 hours after board meetings
- Disclosures available on: NSE > Listed Companies > Shareholding Pattern + Bulk/Block Deal sections

### Promoter BUY Signal
| Promoter Purchase | Signal Strength | Trade Action |
|------------------|----------------|-------------|
| Purchase > 1% of equity in open market | STRONG BULLISH | Buy within 1 day of disclosure |
| Purchase 0.5–1% | Moderate bullish | Buy with confirmation from technicals |
| Regular small creep (0.1–0.2%) | Mild positive | Monitor; directionally positive |

**Why promoter buying > 1% is a high-conviction signal:**
- Promoters have full insider knowledge of business
- Buying in open market (not ESOP) uses personal capital — high conviction
- Triggers mandatory SEBI disclosure — public knowledge creates institutional attention
- After-effect: FII/DII often follow promoter buys within 2–4 weeks

### Promoter SELL Signal
| Promoter Sale | Signal Strength | Trade Action |
|--------------|----------------|-------------|
| Promoter pledge INCREASE > 5% of promoter holding | RED FLAG — SELL IMMEDIATELY | Exit position same day of disclosure |
| Open market sale > 1% (non-ESOP) | BEARISH | Reduce position significantly |
| Block deal sale to institutions (at minor discount) | Neutral to mild negative | Institutional buyers at discount = support |
| Gradual pledge build-up over 6 months | WARNING SIGN | Monitor; be ready to exit |

### Pledge Escalation — Critical Warning
```
Promoter Pledge Risk = (Total Pledged Shares / Total Promoter Holding) × 100

> 30% pledged: Caution zone
> 50% pledged: High risk — margin call cycle potential
> 70% pledged: AVOID — one market correction = forced selling cascade
```
**Examples from NSE history**: Zee Entertainment (pledge unwind crisis), DHFL (pledge + corporate governance failure) — stocks dropped 50–90%.

---

## 7. FII/DII Bulk and Block Deal Disclosures

### SEBI Bulk Deal Definition
- A **bulk deal** = single transaction or series of transactions in a stock exceeding **0.5% of the company's equity** in one trading day
- Must be reported to NSE within 1 hour of trade and disclosed publicly by end of day

### Block Deal
- Block deals happen in a **separate 15-minute window** (8:45–9:00 AM on NSE, called the "pre-open block window")
- Minimum size: ₹10 crore
- Price range: Within ±1% of previous close or VWAP

### Reading Bulk/Block Deals
| Type | Implication | Action |
|------|------------|--------|
| FII buys > 0.5% equity at market or above VWAP | STRONG BULLISH — institutional conviction | Buy on next day open |
| FII sells > 0.5% equity | BEARISH — review fundamentals | Exit or reduce positions |
| Domestic MF buys > 0.5% | Positive — MF accumulation signals long-term view | Hold/add |
| PE/VC block sale to public markets | Supply pressure — discount priced in | Wait for dust to settle (2–3 days) |
| Promoter to FII block transfer | Often positive — marquee FII entering = quality signal | Buy after confirmation |

### Data Source
- NSE website: Market Data → Bulk Deals / Block Deals (updated live during trading)
- Time of check: 9:30 AM daily (block deals from pre-open); 4:00 PM for full day bulk deals
- Third-party tools: Trendlyne "Bulk Deals" section; StockEdge "Corporate Actions"

---

## 8. Open Offers and Delistings

### Open Offer Mechanics (Takeover Code — SEBI SAST Regulations)
- Triggered when acquirer reaches **25% stake** (creeping acquisition limit) OR acquires control
- **Open offer price = minimum of**:
  - Highest price paid in last 26 weeks
  - Volume-weighted average price of last 60 trading days
  - Negotiated deal price
- Duration: Open offer stays open for 10 trading days (tendering window)

### Open Offer Arbitrage Strategy
```
Arbitrage Spread = Open Offer Price − Current Market Price

If market price < offer price: Buy in market, tender in offer
Annualized return = (Spread / Market Price) × (365 / Days to offer close) × 100
```
- Open offer price = **price floor** for the stock during the offer period
- Stocks trade at 2–5% discount to open offer price (time value + deal failure risk premium)
- Risk: Open offer withdrawal (rare) or SEBI objection (check for ongoing regulatory issues)

### Delisting Offer
- Company initiates delisting; minimum public shareholding threshold must be reached
- **Reverse book building** determines final delisting price (shareholders bid at what price they'll sell)
- Delisting floor price = Book value or formula price; final price often 30–100% above market
- Strategy: Buy before delisting announcement if you get intelligence of high probability; exit at book-building premium

### Delisting Failure
- If sufficient shares not tendered in reverse book building: Delisting fails; stock remains listed
- Price impact of failed delisting: −15% to −30% (the delisting premium evaporates)
- Risk management: Never put > 3% portfolio in a delisting play; binary outcome

---

## 9. Merger Arbitrage on NSE

### How Mergers Work in India (NCLT Process)
1. **Board approval** (both companies announce) → Merger ratio disclosed
2. **SEBI/NCLT filing** → 4–12 months process
3. **Shareholder voting** (EGM) → Usually approved unless promoter conflict
4. **NCLT approval** → Final; shares are swapped at merger ratio
5. **New shares listed** on NSE; old shares suspended

### Merger Arbitrage Spread
```
Spread = (Acquiree Price × Swap Ratio) − Acquirer Price

Example: Company A (acquirer) at ₹1000; Company B (target, 0.5 swap ratio) should be at ₹500
If B is at ₹480: Arbitrage spread = ₹20 (4%)
Buy B, Short A (or just long B if you cannot short)
```

### Typical Spread Dynamics on NSE
| Deal Stage | Typical Discount to Deal Price |
|-----------|-------------------------------|
| Just announced | 8–12% discount (high uncertainty) |
| SEBI approved | 5–8% discount |
| Shareholder approval done | 2–5% discount |
| NCLT hearing scheduled | 1–3% discount |
| NCLT approved | Near zero |

### Risks in Merger Arbitrage
1. **Deal failure**: NCLT rejection (rare) or regulatory block (CCI — Competition Commission of India)
2. **Delays**: Each delay extends holding period, reducing annualized return
3. **Acquirer stock decline**: If holding acquirer short, any short squeeze hurts
4. **Liquidity**: Target stock often becomes illiquid during deal period

### Practical Rules
- Never size merger arb > 5% of portfolio per deal
- Minimum expected annualized return: 15% (else not worth the complexity)
- Monitor: NSE Corporate Filings for NCLT hearing dates
- Exit: If deal is delayed > 6 months from original timeline without good reason — exit; deal risk rising

---

## 10. NSE Corporate Action Data Sources

| Action Type | Primary Source | Update Frequency |
|-------------|---------------|-----------------|
| Quarterly results | NSE corporate filings / Screener.in | Live during results |
| Dividends, splits, bonus | NSE corporate actions calendar | Daily |
| Promoter buying/selling | NSE shareholding pattern + bulk deals | 2-day lag (SEBI deadline) |
| Bulk/block deals | NSE market data → bulk deals | Real-time + EOD |
| Open offer details | SEBI website → Takeover panel | Filing day |
| Merger documents | NCLT website + SEBI SCORES | As filed |
| Pledge data | NSE/BSE quarterly shareholding pattern | Quarterly |
| FII sector data | SEBI FPI statistics | Monthly |

---

## Quick Reference: Corporate Action Checklist

### Before Earnings (T-14 to T-1)
- [ ] Has stock already rallied > 15% in last 60 days? (Yes = risk of sell-the-news)
- [ ] What is street consensus EPS estimate? (Source: Trendlyne/Bloomberg)
- [ ] Is management known for sandbagging estimates? (Consistent beats historically?)
- [ ] Any recent promoter selling in last 30 days? (Red flag if yes)

### Bulk Deal Alert
- [ ] Who bought/sold? (FII > DII > domestic MF = pecking order of bullishness)
- [ ] Size vs. equity? (> 1% of equity = material; > 0.5% = significant)
- [ ] Price vs. VWAP? (Bought above VWAP = urgency; at discount = negotiated)

### Promoter Signal Assessment
- [ ] Is this open market buy or ESOP exercise? (Open market = stronger signal)
- [ ] Is promoter pledge increasing simultaneously? (Buy + pledge increase = contradictory, caution)
- [ ] Promoter holding trend over last 4 quarters (consistent increase = accumulation thesis)

---

*Last Updated: NSE Corporate Actions Playbook v2.0 | SEBI regulations reference: SAST (Takeover), LODR (disclosures), Insider Trading Regulations 2015*
