---
name: nse-market-structure
description: NSE segments, product types, settlement, circuit limits, and session structure. Use whenever reasoning about what a trade record means, what's tradeable when, or how a position settles.
allowed-tools: []
---

# NSE Market Structure

Reference for the mechanics of how the NSE actually operates, so
statement analysis and research code interpret trade records correctly
rather than assuming US-market conventions (this codebase's earlier
research explicitly rejected bulk-installing third-party finance skills
because none were NSE-native — this skill exists to close that gap for
market-structure basics).

## Segments

- **EQ (Equity Cash)** — regular delivery/intraday equity trading.
- **NFO / BFO (F&O)** — equity derivatives (futures, options) on NSE/BSE.
- **CDS (Currency Derivatives)**, **MCX (Commodity Derivatives)** —
  separate segments with their own contract specs and cost structure.

## Product types (how a trade is margined and settled)

- **CNC (Cash and Carry)** — delivery. Full payment, shares move to
  demat. Settlement is **T+1** (trade date plus one working day).
- **MIS (Margin Intraday Square-off)** — must be closed the same day;
  broker auto-squares-off near session close (typically 3:20pm, broker-
  specific) if not closed manually. This is what makes a position
  "intraday" for STT purposes (`india-cost-model` skill) — not the
  holding duration alone, the *product type* on the order.
- **NRML (Normal)** — for carrying F&O positions overnight, margined
  under SPAN/exposure rules, not applicable to cash equity.

A statement row's `product_type` field (MIS/CNC/NRML) is the ground
truth for delivery-vs-intraday classification — do not infer it from
same-day buy/sell timing alone, since a CNC position bought and sold the
same day is rare but possible and taxed differently than an MIS trade.

## Session structure

- **Pre-open auction**: 09:00–09:08 (order collection), 09:08–09:12
  (price discovery), 09:12–09:15 (buffer). Sets the opening price via
  auction, not continuous matching.
- **Normal market**: 09:15–15:30.
- **Closing session / post-close**: brief window after 15:30 for
  closing-price-referenced orders.
- Muhurat trading (Diwali) is a special one-hour evening session, NSE
  holiday-calendar-specific, not a regular occurrence.

## Circuit limits and surveillance

- **Circuit filters**: price bands (2%/5%/10%/20% depending on the
  stock) beyond which trading halts for that name that session. A
  position that "should have" moved further but didn't may simply have
  hit its circuit — check before assuming a signal failed.
- **ASM (Additional Surveillance Measure) / GSM (Graded Surveillance
  Measure)**: NSE/SEBI flags for stocks showing unusual price/volume
  patterns, which impose extra margin or trading restrictions. A name
  under ASM/GSM behaves differently (wider effective spreads, margin
  changes) than an unflagged one — relevant when a backtest or live
  screen doesn't account for surveillance status, since historical
  data alone won't show it.

## Settlement cycle

**T+1** for equity cash since 2023 (India moved from T+2). Funds/shares
from a sale are available the working day after trade date. This matters
for statement reconciliation: a sale's cash impact and a buy's margin
requirement land one working day later, not same-day.
