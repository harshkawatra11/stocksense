"""Strategy family 1: the overnight/intraday tug of war.

Lou, Polk & Skouras (JFE 2019): sorting on the past overnight return, the
long-winner/short-loser portfolio earns +3.47%/month overnight (t=16.83) and
-3.02%/month intraday (t=-9.74) -- profits concentrate in one leg or the
other, typically with opposite signs. The intraday-tradeable form is the
mirror: long the overnight LOSERS, short the overnight WINNERS, hold
open-to-close only, harvesting the negative intraday leg as a positive.

Mechanism (required by strategies.base.require_hypothesis): investor
clienteles. Institutional ownership rises with intraday returns -- funds
trade during the session, individuals cluster around the open -- so the two
clienteles push price in opposite directions at different times of day.
Retail participation on NSE is unusually high, which is exactly the
condition that should sharpen this clientele separation.

This is the cheapest-to-test lead the project owns: the signal needs only
`adj_open`, `prev_adj_close` and `prev_gap_sessions` -- already produced by
data.adjust.with_prev_adjusted_close -- no minute bars, no Upstox token, no
static IP. It runs first for that reason as well as the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stocksense.strategies.base import require_hypothesis

DEFAULT_HYPOTHESIS = (
    "Overnight moves are driven by order imbalance accumulated while the "
    "market is shut, and are absorbed and partly reversed once continuous "
    "trading resumes (investor-clientele separation: institutions trade "
    "intraday, individuals cluster at the open)."
)

# Verified 2026 statutory round-trip cost at realistic small-account position
# sizes (THE COST WALL). Fixed, never swept -- costs are a measured fact
# about the market, not a strategy parameter to search over.
DEFAULT_CHARGES_BPS = 10.62


@dataclass(frozen=True)
class OvernightReversalConfig:
    # --- signal -------------------------------------------------------
    demean: bool = True
    winsorise_pct: float = 0.01
    # --- selection ------------------------------------------------------
    side: str = "long"  # "long" | "short" | "long_short"
    n_positions: int = 5
    cap_band: str = "mid"  # applied upstream via data.universe_pit, not here
    min_overnight_move: float = 0.01
    # --- exits ------------------------------------------------------------
    exit_rule: str = "session_close"  # only rule implemented so far
    trail_pct: float = 0.015
    giveback_pct: float = 0.30
    hard_stop_pct: float = 0.015
    # --- costs (not swept) ------------------------------------------------
    charges_bps: float = DEFAULT_CHARGES_BPS
    hypothesis: str = DEFAULT_HYPOTHESIS

    def __post_init__(self) -> None:
        require_hypothesis(self.hypothesis)
        if self.side not in ("long", "short", "long_short"):
            raise ValueError("side must be 'long', 'short' or 'long_short'")
        if self.n_positions < 1:
            raise ValueError("n_positions must be >= 1")
        if self.exit_rule != "session_close":
            raise ValueError("only exit_rule='session_close' is implemented so far")


def compute_overnight_signal(
    panel: pd.DataFrame, demean: bool = True, winsorise_pct: float = 0.01
) -> pd.DataFrame:
    """signal = adj_open / prev_adj_close - 1, restricted to a genuine one-
    session gap. Rows with prev_gap_sessions != 1 (multi-week gaps on
    illiquid names) or a missing prev_adj_close (first observation) are
    dropped, never treated as a zero move.
    """
    out = panel[panel["prev_gap_sessions"] == 1].copy()
    out["signal"] = out["adj_open"] / out["prev_adj_close"] - 1.0

    if winsorise_pct and winsorise_pct > 0:
        lo = out.groupby("date")["signal"].transform(lambda s: s.quantile(winsorise_pct))
        hi = out.groupby("date")["signal"].transform(lambda s: s.quantile(1 - winsorise_pct))
        out["signal"] = out["signal"].clip(lower=lo, upper=hi)

    if demean:
        out["signal"] = out["signal"] - out.groupby("date")["signal"].transform("mean")

    return out.reset_index(drop=True)


def select_positions(signal_panel: pd.DataFrame, cfg: OvernightReversalConfig) -> pd.DataFrame:
    """Per date, pick the extreme `n_positions` names by signal.

    side="long"  -> the most NEGATIVE signal (overnight losers), bought.
    side="short" -> the most POSITIVE signal (overnight winners), shorted.
    side="long_short" -> both tails, n_positions per leg.

    `min_overnight_move` filters out near-zero gaps before ranking -- a
    fraction-of-a-percent gap is noise, not the effect this family targets.
    """
    eligible = signal_panel[signal_panel["signal"].abs() >= cfg.min_overnight_move]

    picks = []
    for _, group in eligible.groupby("date"):
        if cfg.side in ("long", "long_short"):
            longs = group.nsmallest(cfg.n_positions, "signal").copy()
            longs["side"] = 1
            picks.append(longs)
        if cfg.side in ("short", "long_short"):
            shorts = group.nlargest(cfg.n_positions, "signal").copy()
            shorts["side"] = -1
            picks.append(shorts)

    if not picks:
        return pd.DataFrame(columns=[*signal_panel.columns, "side"])
    return pd.concat(picks, ignore_index=True)


def daily_pnl(positions: pd.DataFrame, prices: pd.DataFrame, charges_bps: float) -> pd.Series:
    """Equally-weighted open-to-close return of the selected book, net of a
    fixed round-trip charge per position, per date.

    Entry at the open, exit at the close (session_close exit -- unconditional
    at 15:10 since these are daily bars). A date with no positions is simply
    absent from the result, not zero -- zero would be a claim that the book
    was flat, which "no signal fired" is not (evaluation.gate.evaluate_gate
    already treats a missing/None fold the same way).
    """
    if positions.empty:
        return pd.Series(dtype=float)

    # positions may carry the full upstream signal panel (select_positions
    # passes its input columns through) -- keep only the identity + side so
    # the merge below can't collide with prices' own adj_open/adj_close.
    book = positions[["date", "symbol", "side"]]
    merged = book.merge(prices, on=["date", "symbol"], how="left")
    merged["leg_return"] = merged["side"] * (merged["adj_close"] / merged["adj_open"] - 1.0)
    merged["leg_return_net"] = merged["leg_return"] - charges_bps / 10_000.0

    return merged.groupby("date")["leg_return_net"].mean()
