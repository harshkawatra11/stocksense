"""
Transaction cost model — product-aware (MIS intraday vs. CNC delivery).

India-specific round-trip cost components, current as of 2026, for a discount
broker (Upstox-style flat/low-cost schedule — see docs/UPSTOX_API_NOTES.md;
the docs don't publish a brokerage table, so we use the standard discount-
broker flat rate, which is the conservative/typical figure across Zerodha/
Upstox/Groww):

  - Brokerage:  min(₹20, 0.03% of turnover) per executed order, both legs.
                Applies to both MIS and CNC on most discount brokers.
  - STT:        MIS (intraday) — 0.025% on the SELL leg only.
                CNC (delivery) — 0.1% on BOTH legs (buy + sell).
  - Exchange transaction charges: ~0.00325% of turnover, NSE, both legs.
  - SEBI turnover fee: ~0.0001% of turnover, both legs (₹10 per crore).
  - Stamp duty: ~0.003% of turnover, BUY leg only (state-levied, buyer pays).
  - GST: 18% on (brokerage + exchange transaction charges + SEBI fee) —
         NOT on STT or stamp duty, which are already taxes.
  - Slippage: a flat conservative estimate (0.05% per leg = 0.10% round trip).
              This is NOT a depth-based model — just a placeholder floor.
              A proper order-book-depth slippage model is future work.

Worked example — MIS, buy+sell ₹10,000 turnover each leg (₹20,000 total):
  brokerage        = min(20, 0.0003*10000) * 2 legs = min(20,3)=3 -> ₹3 * 2 = ₹6.00
  STT (sell only)  = 0.00025 * 10000                              = ₹2.50
  exch txn charges = 0.0000325 * 20000                            = ₹0.65
  SEBI fee         = 0.000001 * 20000                              = ₹0.02
  stamp duty (buy) = 0.00003 * 10000                               = ₹0.30
  GST (18% of brokerage+exch+SEBI = 6.00+0.65+0.02=6.67)           = ₹1.20
  slippage (0.05% * 2 legs on 20000)                                = ₹20.00
  ------------------------------------------------------------------------
  total cost                                                        = ₹30.67
  as % of one-leg trade value (₹10,000)                             = 0.307%
This is close to the old flat ROUND_TRIP_COST_PCT=0.30% estimate for MIS —
consistent by design. CNC costs more (STT on both legs at 4x the rate),
landing near ~0.45-0.55% round trip depending on turnover.
"""
from __future__ import annotations

# ------------------------------------------------------------------ #
# Legacy flat constant — kept as the "simple mode" fallback/default so
# existing callers that don't pass product/price/qty still get a sane
# conservative number. New callers should prefer the full product-aware
# net_return() signature below.
# ------------------------------------------------------------------ #
ROUND_TRIP_COST_PCT = 0.0030

# ------------------------------------------------------------------ #
# Cost constants (fractions, not percentages: 0.0003 = 0.03%)
# ------------------------------------------------------------------ #
BROKERAGE_PCT = 0.0003          # 0.03% of turnover per order...
BROKERAGE_FLAT_CAP = 20.0       # ...capped at ₹20/order, whichever is lower

STT_INTRADAY_SELL_PCT = 0.00025   # 0.025% on sell leg only (MIS)
STT_DELIVERY_PCT = 0.001          # 0.1% on both legs (CNC)

EXCHANGE_TXN_PCT = 0.0000325      # ~0.00325% NSE, both legs
SEBI_FEE_PCT = 0.000001           # ~0.0001% (₹10/crore), both legs
STAMP_DUTY_BUY_PCT = 0.00003      # ~0.003% of turnover, buy leg only

GST_PCT = 0.18                    # 18% GST on (brokerage + exchange + SEBI fee)

SLIPPAGE_PCT_PER_LEG = 0.0005     # 0.05% per leg, flat estimate (not depth-based)


def _brokerage(turnover: float) -> float:
    """Per-order brokerage: min(flat ₹20, 0.03% of turnover)."""
    if turnover <= 0:
        return 0.0
    return min(BROKERAGE_FLAT_CAP, BROKERAGE_PCT * turnover)


def round_trip_cost_pct(product: str = "MIS", price: float = 0, qty: int = 0) -> float:
    """
    Total round-trip transaction cost as a fraction of one-leg trade value
    (i.e. price*qty for a single buy or sell), for the given product.

    product: "MIS" (intraday) or "CNC" (delivery). Unknown values fall back
    to MIS (the target product for this build — see WHAT_TO_DO_NEXT.txt 2.10).

    If price/qty aren't provided (price*qty <= 0), falls back to a
    representative ₹10,000-per-leg trade so this still returns a sane
    percentage for cost-agnostic callers.
    """
    product = (product or "MIS").upper()
    turnover_one_leg = price * qty if price and qty else 10_000.0
    turnover_round_trip = turnover_one_leg * 2  # buy leg + sell leg

    brokerage_total = _brokerage(turnover_one_leg) * 2  # both legs

    if product == "CNC":
        stt = STT_DELIVERY_PCT * turnover_round_trip
    else:
        stt = STT_INTRADAY_SELL_PCT * turnover_one_leg  # sell leg only

    exch_txn = EXCHANGE_TXN_PCT * turnover_round_trip
    sebi_fee = SEBI_FEE_PCT * turnover_round_trip
    stamp_duty = STAMP_DUTY_BUY_PCT * turnover_one_leg  # buy leg only

    gst = GST_PCT * (brokerage_total + exch_txn + sebi_fee)

    slippage = SLIPPAGE_PCT_PER_LEG * turnover_round_trip

    total_cost = brokerage_total + stt + exch_txn + sebi_fee + stamp_duty + gst + slippage

    # Express as a fraction of one-leg trade value (matches how gross_return_pct
    # is defined elsewhere: (exit-entry)/entry, i.e. per one-leg notional).
    return total_cost / turnover_one_leg if turnover_one_leg > 0 else ROUND_TRIP_COST_PCT


def net_return(gross_return_pct: float, product: str = "MIS", price: float = 0, qty: int = 0) -> float:
    """
    Subtract the round-trip transaction cost from a gross return.

    `gross_return_pct` is a fractional return (e.g. 0.012 for +1.2%), not a
    percentage-points integer. Returns the cost-adjusted (net) fractional
    return.

    `product` distinguishes MIS (intraday, default — this build's target
    product) from CNC (delivery), which carries a materially higher STT.
    `price`/`qty` let the caller give the real trade size for a more precise
    brokerage calc (brokerage is capped at ₹20/order, so small trades pay a
    higher effective %); omitted, a representative ₹10,000 leg is assumed —
    this keeps existing call sites (accuracy_tracker.py, trading_mode.py)
    working unchanged with a conservative default.
    """
    cost_pct = round_trip_cost_pct(product=product, price=price, qty=qty)
    return gross_return_pct - cost_pct
