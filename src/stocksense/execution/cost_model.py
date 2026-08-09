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
