"""
Execution realism for intraday backtesting (Phase E3). Closes A3/A4/A5/
A10 from the Phase E audit: a backtest that fills at the signal bar's
own close, assumes a flat cost in bps, never checks whether an order
would exceed the bar's real volume, and assumes uniform 5x leverage
everywhere is not a backtest of a strategy -- it's a fantasy with a P&L
column attached.

This module does not replace execution.cost_model (still the source of
truth for STT/exchange/SEBI/stamp-duty/GST -- the regulatory cost
stack). It adds the layer cost_model always assumed away: WHETHER a
fill can happen at all, and WHERE it actually happens.

Four checks, applied in this order (cheapest/most-disqualifying first):
1. Circuit-lock filter (A5) -- a stock frozen at its band cannot be
   bought or sold at all, regardless of size.
2. MIS-leverage lookup (A4) -- sizing must never assume leverage the
   broker doesn't actually offer for that symbol.
3. Participation cap (A3) -- an order that would be a large share of a
   bar's own volume could not realistically have filled at the modeled
   price; reject rather than silently pretend it filled.
4. Fill price (A10) -- next-bar open plus half-spread, never the signal
   bar's own close (which had already happened by the time a strategy
   could act on it).

Honest limitation, stated once here rather than re-discovered later:
bhavcopy_eq carries no explicit circuit-band percentage (verified: the
ingested schema is symbol/series/date/OHLC/prev_close/volume/turnover/
era -- no band column). Circuit-lock detection here is therefore
behavioural, not a lookup: a bar with zero intrabar range (high == low)
is what a genuinely locked stock produces, since every print in that bar
sat at the exact same price. This can occasionally also flag a bar with
one single print (extremely thin liquidity), which is arguably still
correct to reject -- a backtest cannot realistically fill against a bar
that traded at only one price either.
"""

from __future__ import annotations

from dataclasses import dataclass

CIRCUIT_LOCKED = "circuit_locked"
LEVERAGE_UNAVAILABLE = "leverage_unavailable"
PARTICIPATION_EXCEEDED = "participation_exceeded"
NO_NEXT_BAR = "no_next_bar"

# Conservative by construction: a symbol not explicitly listed is assumed
# to have NO intraday leverage (1.0x), never assumed to have 5x. Angel
# One's actual MIS-eligible list is broker-maintained and changes --
# this table is a mechanism to load/override it, not a claim that any
# specific symbol is on it today. Populating it with the real current
# list is an operational step, not something fabricated here.
DEFAULT_LEVERAGE = 1.0


@dataclass(frozen=True)
class FillResult:
    filled: bool
    fill_price: float | None
    rejection_reason: str | None
    leverage_applied: float | None = None


def is_circuit_locked(bar_high: float, bar_low: float) -> bool:
    """True if a bar shows zero intrabar range -- see module docstring
    for why this is the available signal, not a direct band lookup."""
    return bar_high == bar_low


def get_mis_leverage(symbol: str, leverage_table: dict[str, float] | None = None) -> float:
    """Looks up allowed intraday leverage for `symbol`. Unlisted symbols
    default to DEFAULT_LEVERAGE (1.0x, i.e. no leverage) -- the safe
    direction to be wrong in, since assuming leverage that doesn't exist
    understates risk, while assuming none that does exist only
    understates opportunity."""
    table = leverage_table or {}
    return table.get(symbol, DEFAULT_LEVERAGE)


def check_participation(order_qty: float, bar_volume: float, max_participation_pct: float = 0.1) -> bool:
    """True if `order_qty` stays within `max_participation_pct` of the
    bar's own traded volume. A zero/negative bar_volume never passes --
    there is nothing to have filled against."""
    if bar_volume <= 0:
        return False
    return order_qty <= max_participation_pct * bar_volume


def compute_fill_price(next_bar_open: float, direction: str, half_spread_bps: float = 2.5) -> float:
    """Next-bar open, walked by half the modeled spread AWAY from the
    trader (buys fill slightly above open, sells slightly below) -- the
    direction a real spread always costs you, never in your favor. Never
    the signal bar's own close: that price had already happened and
    finished printing by the time a strategy could act on it.

    half_spread_bps default (2.5bps) is a modeled estimate, not replayed
    from real quotes -- this data has no bid/ask, only OHLCV -- matching
    execution.cost_model's own documented practice of naming slippage as
    modeled rather than pretending it was measured.
    """
    if direction not in ("buy", "sell"):
        raise ValueError(f"direction must be 'buy' or 'sell', got {direction!r}")
    factor = 1 + (half_spread_bps / 10_000.0) if direction == "buy" else 1 - (half_spread_bps / 10_000.0)
    return next_bar_open * factor


def simulate_fill(
    symbol: str,
    direction: str,
    order_qty: float,
    next_bar_open: float | None,
    next_bar_high: float,
    next_bar_low: float,
    next_bar_volume: float,
    leverage_table: dict[str, float] | None = None,
    max_participation_pct: float = 0.1,
    half_spread_bps: float = 2.5,
) -> FillResult:
    """The single entry point combining all four checks in order. Every
    rejection carries a reason -- a caller (the E4 backtest loop) must
    be able to report WHY a signal never became a trade, not just that
    it didn't."""
    if next_bar_open is None:
        return FillResult(filled=False, fill_price=None, rejection_reason=NO_NEXT_BAR)

    if is_circuit_locked(next_bar_high, next_bar_low):
        return FillResult(filled=False, fill_price=None, rejection_reason=CIRCUIT_LOCKED)

    leverage = get_mis_leverage(symbol, leverage_table)
    if leverage <= 0:
        return FillResult(filled=False, fill_price=None, rejection_reason=LEVERAGE_UNAVAILABLE, leverage_applied=leverage)

    if not check_participation(order_qty, next_bar_volume, max_participation_pct):
        return FillResult(filled=False, fill_price=None, rejection_reason=PARTICIPATION_EXCEEDED, leverage_applied=leverage)

    fill_price = compute_fill_price(next_bar_open, direction, half_spread_bps)
    return FillResult(filled=True, fill_price=fill_price, rejection_reason=None, leverage_applied=leverage)
