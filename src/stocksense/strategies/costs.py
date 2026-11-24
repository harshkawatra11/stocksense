"""Round-trip trade cost as a fraction of notional, from StockSense's own cost schedule.

Statutory costs come from `costs.india.trade_cost` (buy and sell), and each side adds the slippage model's
adverse move: half-spread by liquidity rank, impact, and an open/close auction penalty.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from stocksense.core.config import Settings
from stocksense.costs.india import trade_cost
from stocksense.costs.slippage import SlippageTable, slip_frac

DEFAULT_RANK = 200  # an index, or a stock with no turnover rank
When = Literal["open", "close", "now"]


def trade_cost_frac(
    product: Literal["delivery", "intraday"],
    notional: float,
    settings: Settings,
    slip: SlippageTable,
    turnover_rank: int | None,
    vol_21: float,
    med_turn_20: float,
    when_in: When,
    when_out: When,
    as_of: date | None = None,
) -> float:
    """Round-trip cost as a fraction of notional: statutory buy and sell costs plus slippage on both sides.

    `when_in` is "open" for a next-bar or next-day open fill; `when_out` is "now" for a stop or target,
    "open" for a gap stop, "close" for time and moving-average exits.
    """
    rank = turnover_rank or DEFAULT_RANK
    statutory = (
        trade_cost("BUY", product, notional, settings.costs).total
        + trade_cost("SELL", product, notional, settings.costs).total
    ) / notional
    s_in = slip_frac(rank, vol_21, notional, med_turn_20, when_in, slip, as_of)
    s_out = slip_frac(rank, vol_21, notional, med_turn_20, when_out, slip, as_of)
    return statutory + s_in + s_out
