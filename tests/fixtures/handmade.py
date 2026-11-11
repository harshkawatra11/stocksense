"""Hand-made raw frames for exact-arithmetic panel tests (small, explicit prices)."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from stocksense.core.config import Settings
from stocksense.data import io

_START = date(2020, 1, 6)  # a Monday


def weekdays(n: int, start: date = _START) -> list[date]:
    """n consecutive weekdays from `start`."""
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def eq_rows(
    symbol: str,
    dates: list[date],
    closes: list[float],
    *,
    opens: list[float] | None = None,
    prev_closes: list[float | None] | None = None,
    series: str | list[str] = "EQ",
    volume: int | list[int] = 100_000,
    isin: str | None = None,
    name: str | None = None,
) -> pl.DataFrame:
    """bhavcopy_eq rows; prev_close defaults to the previous raw close (first row: the open)."""
    n = len(dates)
    opens = opens if opens is not None else list(closes)
    if prev_closes is None:
        prev_closes = [opens[0]] + list(closes[:-1])
    ser = series if isinstance(series, list) else [series] * n
    vols = volume if isinstance(volume, list) else [volume] * n
    return pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "series": ser,
            "date": dates,
            "open": [float(x) for x in opens],
            "high": [max(o, c) + 0.5 for o, c in zip(opens, closes, strict=True)],
            "low": [min(o, c) - 0.5 for o, c in zip(opens, closes, strict=True)],
            "close": [float(x) for x in closes],
            "prev_close": prev_closes,
            "last_price": [float(x) for x in closes],
            "volume": vols,
            "turnover_inr": [v * c for v, c in zip(vols, closes, strict=True)],
            "n_trades": [100.0] * n,
            "era": ["legacy"] * n,
            "isin": [isin] * n,
            "name": [name or f"{symbol} Ltd"] * n,
        },
        schema_overrides={"prev_close": pl.Float64, "isin": pl.Utf8},
    )


def write_raw(settings: Settings, **frames: pl.DataFrame) -> None:
    """Upsert the given raw frames (keyword = catalog dataset name) into the temp data root."""
    for name, df in frames.items():
        if df.height:
            io.upsert(settings, name, df)
