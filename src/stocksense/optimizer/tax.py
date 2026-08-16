"""
Tax-aware analysis for Indian equity (Section 111A/112A). Two genuinely
different things live here, and this module is careful not to conflate
them:

1. Classifying and computing tax on ALREADY-REALIZED gains (from
   statements.positions, i.e. the Kundli's reconstructed round trips) --
   this is retrospective arithmetic, useful for "what did my trading
   actually cost in tax," same spirit as counterfactuals.py.
2. Loss-harvesting analysis on CURRENTLY-OPEN positions, which needs
   unrealized P&L (current market price vs entry) -- something
   statement data alone cannot provide, so harvest_candidates() takes an
   explicit open-positions-with-current-price input rather than trying
   to infer one.

Rates verified 2026-08-16 (Budget 2026 kept the framework from the prior
year unchanged): STCG (equity, held < 12 months) = 20% flat. LTCG
(equity, held >= 12 months) = 12.5% on gains above a Rs 1.25 lakh
exemption, applied ONCE per financial year across all LTCG gains
combined, not per position. A 4% cess applies on top of the tax amount
in both cases. These are statutory rates, not modeling assumptions --
they belong in one place so a future rate change is a one-line edit,
same reasoning as execution.cost_model's rate constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

STCG_RATE = 0.20
LTCG_RATE = 0.125
LTCG_EXEMPTION_PER_FY = 125_000.0
CESS_RATE = 0.04
LTCG_HOLDING_DAYS = 365


@dataclass(frozen=True)
class TaxLot:
    symbol: str
    gain_type: str  # 'STCG' | 'LTCG'
    gross_gain: float
    holding_days: int


@dataclass(frozen=True)
class TaxSummary:
    total_stcg: float
    total_ltcg: float
    ltcg_exemption_used: float
    taxable_ltcg: float
    stcg_tax: float
    ltcg_tax: float
    cess: float
    total_tax: float


def classify_gain(open_date, close_date) -> str:
    holding_days = (pd.Timestamp(close_date) - pd.Timestamp(open_date)).days
    return "LTCG" if holding_days >= LTCG_HOLDING_DAYS else "STCG"


def classify_positions(positions: pd.DataFrame) -> pd.DataFrame:
    """Adds gain_type and holding_days columns to a positions frame
    (stocksense.statements.positions' output schema: open_date,
    close_date, net_pnl, ...). Only equity_delivery positions are
    classified -- intraday and F&O gains are always short-term/business
    income under Indian tax law regardless of holding period, a
    different regime this module doesn't attempt to model."""
    df = positions[positions["segment"] == "equity_delivery"].copy()
    df["holding_days"] = (pd.to_datetime(df["close_date"]) - pd.to_datetime(df["open_date"])).dt.days
    df["gain_type"] = df["holding_days"].apply(lambda d: "LTCG" if d >= LTCG_HOLDING_DAYS else "STCG")
    return df


def compute_tax_liability(positions: pd.DataFrame, ltcg_exemption_used_this_fy: float = 0.0) -> TaxSummary:
    """Tax on realized delivery gains only, for one financial year's
    worth of `positions` (caller filters the date range). The LTCG
    exemption is applied ONCE across the total LTCG gain, not per
    position -- getting this wrong (applying it per-trade) would
    understate tax owed for anyone with multiple LTCG positions in a
    year, which is the actual case for most real portfolios.

    `ltcg_exemption_used_this_fy` lets a caller account for exemption
    already consumed elsewhere (e.g. mutual fund LTCG) in the same FY.
    """
    classified = classify_positions(positions)
    stcg_gains = classified[classified["gain_type"] == "STCG"]["net_pnl"]
    ltcg_gains = classified[classified["gain_type"] == "LTCG"]["net_pnl"]

    total_stcg_gain = float(stcg_gains[stcg_gains > 0].sum())  # losses don't generate tax; netting is a separate concern
    total_ltcg_gain = float(ltcg_gains[ltcg_gains > 0].sum())

    remaining_exemption = max(0.0, LTCG_EXEMPTION_PER_FY - ltcg_exemption_used_this_fy)
    exemption_applied = min(total_ltcg_gain, remaining_exemption)
    taxable_ltcg = max(0.0, total_ltcg_gain - exemption_applied)

    stcg_tax = total_stcg_gain * STCG_RATE
    ltcg_tax = taxable_ltcg * LTCG_RATE
    cess = (stcg_tax + ltcg_tax) * CESS_RATE

    return TaxSummary(
        total_stcg=total_stcg_gain, total_ltcg=total_ltcg_gain,
        ltcg_exemption_used=exemption_applied, taxable_ltcg=taxable_ltcg,
        stcg_tax=stcg_tax, ltcg_tax=ltcg_tax, cess=cess,
        total_tax=stcg_tax + ltcg_tax + cess,
    )


def days_to_ltcg(open_date, as_of_date=None) -> int:
    """For a currently-open position: trading days (calendar days, for
    simplicity -- tax holding period counts calendar days, not trading
    days, unlike everything else in this codebase's horizon logic) until
    it crosses the 12-month LTCG threshold. Zero or negative means it
    has already crossed."""
    as_of = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp(date.today())
    threshold = pd.Timestamp(open_date) + timedelta(days=LTCG_HOLDING_DAYS)
    return int((threshold - as_of).days)


def harvest_candidates(open_positions: pd.DataFrame, as_of_date=None) -> pd.DataFrame:
    """Loss-harvesting review on CURRENTLY OPEN positions. Expects
    columns: symbol, open_date, quantity, entry_price, current_price
    (the caller supplies current_price -- this module has no live market
    data access, by design, per docs/12's parser/analysis-only scope).

    Returns each open position's unrealized P&L, its holding-period
    status, and days-to-LTCG, so a genuine harvesting decision (sell an
    unrealized loser to realize a loss that offsets a gain elsewhere)
    can be made with the actual numbers in front of it -- this function
    computes the numbers; it does not recommend selling anything."""
    df = open_positions.copy()
    df["unrealized_pnl"] = (df["current_price"] - df["entry_price"]) * df["quantity"]
    df["holding_days"] = (pd.Timestamp(as_of_date or date.today()) - pd.to_datetime(df["open_date"])).dt.days
    df["gain_type_if_sold_today"] = df["holding_days"].apply(lambda d: "LTCG" if d >= LTCG_HOLDING_DAYS else "STCG")
    df["days_to_ltcg"] = df["open_date"].apply(lambda d: days_to_ltcg(d, as_of_date))
    return df.sort_values("unrealized_pnl")
