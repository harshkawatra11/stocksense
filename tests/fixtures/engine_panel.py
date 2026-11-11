"""Hand-made panel/daily-shaped frames for exact engine tests (no data_store needed)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from stocksense.core.config import Settings, load_settings
from stocksense.costs.slippage import SlippageRow, SlippageTable
from tests.fixtures.handmade import weekdays

_REPO = Path(__file__).resolve().parents[2]


def settings() -> Settings:
    """Real config/ with no .env."""
    return load_settings(_REPO / "config", _REPO / "no.env")


def zero_slippage() -> SlippageTable:
    """A table whose slippage is exactly 0 so fills happen at the reference price."""
    return SlippageTable([SlippageRow(date(2000, 1, 1), 0.0, 0.0, 0.0, ((500, 0.0),))])


def dates(n: int) -> list[date]:
    return weekdays(n)


def stock_rows(
    sid: int,
    ds: list[date],
    opens: list[float | None],
    closes: list[float | None],
    *,
    gap: dict[int, float] | None = None,
    div: dict[int, float] | None = None,
    one_price: set[int] | None = None,
    turnover: float = 1e10,
) -> list[dict]:
    """Panel rows for one sid; index i with open None has no row. prev_close = previous raw close."""
    gap, div, one_price = gap or {}, div or {}, one_price or set()
    rows: list[dict] = []
    last_close: float | None = None
    for i, d in enumerate(ds):
        o, c = opens[i], closes[i]
        if o is None or c is None:
            continue
        prev = last_close if last_close is not None else o
        g = gap.get(i, 1.0)
        denom = prev * g
        hi = o if i in one_price else max(o, c) + 0.5
        lo = o if i in one_price else min(o, c) - 0.5
        if i in one_price:
            c = o
        rows.append(
            {
                "sid": sid,
                "date": d,
                "open": float(o),
                "high": float(hi),
                "low": float(lo),
                "close": float(c),
                "prev_close": float(prev),
                "turnover_inr": turnover,
                "gap_factor": g,
                "div_ret": div.get(i, 0.0) / prev,
                "ret": c / denom - 1.0,
                "ret_co": o / denom - 1.0,
                "flag_one_price": i in one_price,
                "flag_locked_up": False,
                "flag_locked_dn": False,
                "adj_open": float(o),
                "adj_close": float(c),
            }
        )
        last_close = c
    return rows


def frame(*stocks: list[dict]) -> pl.DataFrame:
    """Concatenate per-stock row lists into a panel frame with the engine's dtypes."""
    rows = [r for s in stocks for r in s]
    return pl.DataFrame(rows, schema_overrides={"sid": pl.Int32})
