from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Literal

from stocksense.core.config import Settings
from stocksense.core.paths import Paths

When = Literal["open", "close", "now"]
_BEYOND_TABLE_BPS = 20.0


@dataclass(frozen=True)
class SlippageRow:
    """Slippage parameters valid from `date` (bps in the fields ending _bps)."""

    date: date
    impact_k: float
    open_penalty_bps: float
    close_penalty_bps: float
    half_spread_bps_by_rank: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class SlippageTable:
    """Dated slippage rows; the newest row with date <= as-of is used."""

    rows: list[SlippageRow]
    scale: float = 1.0

    @classmethod
    def load_latest(cls, settings: Settings) -> SlippageTable:
        """Config defaults (dated costs.verified_on) plus rows appended in state/slippage_table.json."""
        s = settings.costs.slippage
        rows = [
            SlippageRow(
                settings.costs.verified_on,
                s.impact_k,
                s.open_penalty_bps,
                s.close_penalty_bps,
                tuple((int(r), float(b)) for r, b in s.half_spread_bps_by_rank),
            )
        ]
        path = Paths(settings).state / "slippage_table.json"
        if path.is_file():
            raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
            for r in raw:
                rows.append(
                    SlippageRow(
                        date.fromisoformat(r["date"]),
                        float(r["impact_k"]),
                        float(r["open_penalty_bps"]),
                        float(r["close_penalty_bps"]),
                        tuple((int(a), float(b)) for a, b in r["half_spread_bps_by_rank"]),
                    )
                )
        return cls(sorted(rows, key=lambda r: r.date))

    def row_at(self, as_of: date | None = None) -> SlippageRow:
        """Newest row dated on or before as_of (the newest row overall when as_of is None)."""
        if as_of is None:
            return self.rows[-1]
        ok = [r for r in self.rows if r.date <= as_of]
        return ok[-1] if ok else self.rows[0]

    def scaled(self, factor: float) -> SlippageTable:
        """The same table with every slippage component multiplied by `factor` (2x-slippage gate line)."""
        return replace(self, scale=self.scale * factor)


def _half_spread_frac(rank: int, buckets: tuple[tuple[int, float], ...]) -> float:
    for upper, bps in buckets:
        if upper >= rank:
            return bps / 1e4
    return _BEYOND_TABLE_BPS / 1e4


def slip_frac(
    turnover_rank: int,
    vol_21: float,
    order_value: float,
    med_turn_20: float,
    when: When,
    table: SlippageTable,
    as_of: date | None = None,
) -> float:
    """Adverse price move as a fraction: half-spread + impact + open/close penalty."""
    row = table.row_at(as_of)
    half_spread = _half_spread_frac(turnover_rank, row.half_spread_bps_by_rank)
    impact = row.impact_k * vol_21 * math.sqrt(max(order_value, 0.0) / med_turn_20)
    penalty = {"open": row.open_penalty_bps, "close": row.close_penalty_bps, "now": 0.0}[when] / 1e4
    return (half_spread + impact + penalty) * table.scale


def estimate_open_penalty(settings: Settings) -> dict[str, float]:
    """Median open/close auction penalties in bps from intraday bars."""
    # implemented after Phase 3 (needs real intraday bars and the security master)
    raise NotImplementedError("implemented after Phase 3")
