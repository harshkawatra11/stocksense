"""
Transaction cost model.

Per docs/06-retraining-rigor.md, the full Indian equity cost stack must be
modeled explicitly, not approximated by one round-trip number pulled from
nowhere. `indian_delivery_cost_bps` computes that stack from first
principles so the Phase 0 cost grid can be checked against it rather than
assumed.

The Phase 0 sweep (research/phase0_sweep.py) still varies cost as a single
swept round-trip-bps parameter — that is a deliberate simplification for
finding the viability surface quickly; `indian_delivery_cost_bps` tells us
which grid points are actually realistic once the sweep is done.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostBreakdown:
    brokerage_bps: float
    stt_bps: float
    exchange_txn_bps: float
    sebi_fee_bps: float
    stamp_duty_bps: float
    gst_bps: float
    slippage_bps: float

    @property
    def total_bps(self) -> float:
        return (
            self.brokerage_bps
            + self.stt_bps
            + self.exchange_txn_bps
            + self.sebi_fee_bps
            + self.stamp_duty_bps
            + self.gst_bps
            + self.slippage_bps
        )


def indian_delivery_cost_bps(
    brokerage_bps: float = 0.0,  # many discount brokers (incl. Upstox) charge 0 for delivery
    slippage_bps: float = 5.0,  # modeled, not replayed — see docs/10-evaluation.md fidelity tiers
) -> CostBreakdown:
    """One-way costs, doubled by the caller for round-trip where relevant.

    Rates as of the docs written for this project (2026); these are
    statutory/exchange rates and should be reconciled against real
    contract notes per docs/09-open-questions.md OQ-7 before being trusted
    for anything beyond Phase 0 research.

    STT on delivery: 0.1% (10 bps) on BOTH buy and sell legs.
    Exchange transaction charge (NSE, equity): ~0.00297% (~0.3 bps).
    SEBI turnover fee: ~0.0001% (~0.01 bps), negligible.
    Stamp duty: 0.015% (1.5 bps) on the BUY leg only.
    GST: 18% on (brokerage + exchange txn charge).
    """
    stt_bps = 10.0  # per leg
    exchange_txn_bps = 0.3
    sebi_fee_bps = 0.01
    stamp_duty_bps = 1.5  # buy leg only; caller decides whether to include
    gst_bps = 0.18 * (brokerage_bps + exchange_txn_bps)

    return CostBreakdown(
        brokerage_bps=brokerage_bps,
        stt_bps=stt_bps,
        exchange_txn_bps=exchange_txn_bps,
        sebi_fee_bps=sebi_fee_bps,
        stamp_duty_bps=stamp_duty_bps,
        gst_bps=gst_bps,
        slippage_bps=slippage_bps,
    )


def realistic_round_trip_bps(slippage_bps: float = 5.0) -> float:
    """One buy leg (with stamp duty) + one sell leg (no stamp duty),
    each carrying STT/exchange/SEBI/GST, plus modeled slippage on both
    legs. This is the reference number Phase 0's swept cost grid should
    be checked against.
    """
    buy = indian_delivery_cost_bps(slippage_bps=slippage_bps)
    sell = indian_delivery_cost_bps(slippage_bps=slippage_bps)
    buy_total = buy.total_bps  # includes stamp duty
    sell_total = sell.total_bps - sell.stamp_duty_bps  # no stamp duty on sell
    return buy_total + sell_total


@dataclass(frozen=True)
class Charges:
    """Exact per-trade charge breakdown in rupees, not bps — this is
    applied to real fills (docs/12-statement-forensics.md), so the caller
    needs absolute amounts to compare against a statement's own totals."""

    brokerage: float
    stt: float
    exchange_txn: float
    sebi_fee: float
    stamp_duty: float
    gst: float

    @property
    def total_charges(self) -> float:
        return self.brokerage + self.stt + self.exchange_txn + self.sebi_fee + self.stamp_duty + self.gst


# Verified 2026-08-16 against Zerodha's published charge sheet (the
# lowest-common-denominator discount broker rate; see
# research/phase0_verdict.md's intraday-cost-correction section for the
# ₹100,000-position worked example this table reproduces exactly).
_EXCHANGE_TXN_RATE = {"NSE": 0.0000307, "BSE": 0.0000375}  # of trade value, both legs
_SEBI_FEE_RATE = 10.0 / 1e7  # ₹10 per crore of trade value, both legs
_GST_RATE = 0.18


def compute_charges(
    segment: str,       # 'equity_delivery' | 'equity_intraday' | 'fno_futures' | 'fno_options'
    side: str,           # 'buy' | 'sell'
    quantity: float,
    price: float,
    exchange: str = "NSE",
    brokerage_flat: float = 20.0,   # ₹20/order flat, or 0.03% whichever lower (discount broker default)
    brokerage_pct: float = 0.0003,
) -> Charges:
    """Exact Indian equity/F&O charge stack for one trade leg, in rupees.

    STT is the load-bearing term this whole cost correction turns on:
    - equity_delivery: 0.1% (10 bps) on BOTH buy and sell legs
    - equity_intraday: 0.025% (2.5 bps), SELL LEG ONLY
    - fno_futures: 0.0125% (1.25 bps), SELL LEG ONLY, on trade value
    - fno_options: 0.0625% (6.25 bps), SELL LEG ONLY, on premium value

    Stamp duty is charged on the BUY leg only, at 0.015% (delivery) or
    0.003% (intraday/F&O) — this is a state-government levy, not brokerage.

    GST is 18% on (brokerage + exchange transaction charge + SEBI fee),
    not on STT or stamp duty (those are already government levies).
    """
    value = quantity * price

    if segment == "equity_delivery":
        brokerage = 0.0  # most discount brokers, incl. Upstox/Zerodha, charge zero for delivery
    else:
        brokerage = min(brokerage_flat, brokerage_pct * value)

    if segment == "equity_delivery":
        stt = 0.001 * value  # both legs
    elif segment == "equity_intraday":
        stt = 0.00025 * value if side == "sell" else 0.0
    elif segment == "fno_futures":
        stt = 0.000125 * value if side == "sell" else 0.0
    elif segment == "fno_options":
        stt = 0.000625 * value if side == "sell" else 0.0
    else:
        raise ValueError(f"unknown segment: {segment}")

    exchange_txn = _EXCHANGE_TXN_RATE.get(exchange, _EXCHANGE_TXN_RATE["NSE"]) * value
    sebi_fee = _SEBI_FEE_RATE * value

    if side == "buy":
        stamp_rate = 0.00015 if segment == "equity_delivery" else 0.00003
        stamp_duty = stamp_rate * value
    else:
        stamp_duty = 0.0

    gst = _GST_RATE * (brokerage + exchange_txn + sebi_fee)

    return Charges(
        brokerage=brokerage, stt=stt, exchange_txn=exchange_txn,
        sebi_fee=sebi_fee, stamp_duty=stamp_duty, gst=gst,
    )


def apply_turnover_cost(turnover_fraction: float, round_trip_cost_bps: float) -> float:
    """Cost incurred, as a fraction of portfolio value, for rebalancing
    `turnover_fraction` of the book at `round_trip_cost_bps` round-trip.

    turnover_fraction: one-way turnover, sum(|target_weight - current_weight|) / 2
    (1.0 = the entire book was sold and replaced). Each unit of one-way
    turnover corresponds to exactly one sell-leg-plus-buy-leg round trip
    on that fraction of the book, so cost scales linearly with it at the
    full round-trip rate — no double-counting.
    """
    return turnover_fraction * (round_trip_cost_bps / 10_000.0)
