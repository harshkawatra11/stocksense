"""Round-trip trade cost as a fraction of notional, from StockSense's own cost schedule.

Statutory costs come from `costs.india.trade_cost` (buy and sell), and each side adds the slippage model's
adverse move: half-spread by liquidity rank, impact, and an open/close auction penalty.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
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


# ---- futures and options (config/costs_derivs.toml) --------------------------------------------------

_DERIV_TOML = Path(__file__).resolve().parents[3] / "config" / "costs_derivs.toml"


@dataclass(frozen=True)
class DerivCosts:
    """Parsed `config/costs_derivs.toml`: option and futures charge rates and lot sizes."""

    options: dict[str, float]
    futures: dict[str, float]
    lots: dict[str, float]


@lru_cache(maxsize=4)
def load_deriv_costs(path: Path | None = None) -> DerivCosts:
    """Read the derivatives cost schedule (default `config/costs_derivs.toml` of this checkout)."""
    with (path or _DERIV_TOML).open("rb") as f:
        d = tomllib.load(f)
    return DerivCosts(dict(d["options"]), dict(d["futures"]), dict(d["lots"]))


@dataclass(frozen=True)
class LegFill:
    """One leg's fills in index points per unit: `side` +1 buy / -1 sell at entry, `qty` in lots.

    `exit_price` is the settlement intrinsic value when `closed_by_trade` is False (held to expiry) and the
    traded price when True (a leg closed before its own expiry, e.g. the long leg of a calendar spread).
    """

    kind: Literal["CE", "PE", "FUT"]
    side: int
    qty: float
    entry_price: float
    exit_price: float
    closed_by_trade: bool = False


def futures_cost_rs(side: int, value_inr: float, c: DerivCosts) -> float:
    """Rupee charges of one futures order (side +1 buy, -1 sell) of `value_inr`."""
    f = c.futures
    brok = min(f["brokerage_cap_inr"], f["brokerage_rate"] * value_inr)
    stt = value_inr * f["stt_sell"] if side < 0 else 0.0
    exch, sebi = value_inr * f["exchange"], value_inr * f["sebi"]
    stamp = value_inr * f["stamp_buy"] if side > 0 else 0.0
    return brok + stt + exch + sebi + stamp + f["gst"] * (brok + exch + sebi)


def _option_order_rs(side: int, value_inr: float, c: DerivCosts) -> float:
    o = c.options
    brok = o["brokerage_flat_inr"]
    stt = value_inr * o["stt_sell_premium"] if side < 0 else 0.0
    exch, sebi = value_inr * o["exchange"], value_inr * o["sebi"]
    stamp = value_inr * o["stamp_buy"] if side > 0 else 0.0
    return brok + stt + exch + sebi + stamp + o["gst"] * (brok + exch + sebi)


def options_cost_frac(
    legs_premia: Sequence[LegFill],
    spot: float,
    settings: Settings | None = None,
    lot: float = 65.0,
    schedule: DerivCosts | None = None,
) -> float:
    """Round-trip charges of a combo held to expiry as a fraction of `lot * spot` (0.001 = 0.1%).

    Entry: every leg is one order (flat brokerage; STT on sold premium; exchange fee on premium; stamp on
    bought premium; SEBI fee; GST on brokerage, exchange and SEBI fees); futures legs use the futures
    schedule. Exit: legs closed by trade pay the opposite order; long option legs finishing in the money
    pay STT on intrinsic value (exercise); short legs and worthless legs cost nothing; a long futures leg
    pays STT on its settlement value. `settings` is accepted for call-site symmetry; the rates come from
    `config/costs_derivs.toml`. Slippage is not modelled (stated in DECISIONS).
    """
    del settings
    c = schedule or load_deriv_costs()
    total = 0.0
    for lf in legs_premia:
        n = lf.qty * lot
        if lf.kind == "FUT":
            total += futures_cost_rs(lf.side, lf.entry_price * n, c)
            if lf.closed_by_trade or lf.side > 0:  # settlement is a sale for a long future
                total += futures_cost_rs(-lf.side, lf.exit_price * n, c)
            continue
        total += _option_order_rs(lf.side, lf.entry_price * n, c)
        if lf.closed_by_trade:
            total += _option_order_rs(-lf.side, lf.exit_price * n, c)
        elif lf.side > 0 and lf.exit_price > 0:
            total += lf.exit_price * n * c.options["stt_exercise"]
    return total / (spot * lot)
