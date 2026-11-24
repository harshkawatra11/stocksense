"""Builders shared by the M5.3 and M5.5 strategy tests: constructed bars, feature dicts, truncation check."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from stocksense.marketlab.bars import Bars
from stocksense.strategies import indicators as I  # noqa: E741

IST = timezone(timedelta(hours=5, minutes=30))
BASE = int(datetime(2026, 1, 5, 9, 15, tzinfo=IST).timestamp())


def day_rows(
    closes: list[float],
    vol: float | list[float] = 1000.0,
    wick: float = 0.1,
    o0: float | None = None,
) -> list[tuple[float, float, float, float, float]]:
    """(o, h, l, c, v) rows: each open is the previous close, wicks of `wick` beyond the body."""
    rows = []
    prev = closes[0] if o0 is None else o0
    for i, c in enumerate(closes):
        v = vol[i] if isinstance(vol, list) else vol
        rows.append((prev, max(prev, c) + wick, min(prev, c) - wick, c, v))
        prev = c
    return rows


def make_bars(days: list[list[tuple[float, float, float, float, float]]]) -> Bars:
    """5-minute bars from 09:15 on consecutive days (86400 s apart) from per-day (o, h, l, c, v) rows."""
    ts, o, h, l, c, v = ([] for _ in range(6))  # noqa: E741
    for d, rows in enumerate(days):
        for k, (oo, hh, ll, cc, vv) in enumerate(rows):
            ts.append(BASE + d * 86400 + k * 300)
            o.append(oo)
            h.append(hh)
            l.append(ll)
            c.append(cc)
            v.append(vv)
    return Bars.from_arrays(ts, o, h, l, c, v)


def feats(days: list[list[tuple[float, float, float, float, float]]], sign: int = 1) -> dict[str, Any]:
    """Intraday feature dict of the constructed days; sign -1 builds it on the mirrored chart."""
    b = make_bars(days)
    return I.intraday_features(I.mirror(b) if sign == -1 else b, None, 5, None, sign)


def cut(f: dict[str, Any], m: int) -> dict[str, Any]:
    """The same feature dict with every row after m-1 removed (caches dropped)."""
    n = f["n"]
    g: dict[str, Any] = {}
    for k, x in f.items():
        if k.startswith("_"):
            continue
        g[k] = x[:m] if isinstance(x, np.ndarray) and x.shape[:1] == (n,) else x
    g["n"] = m
    return g


def assert_past_only(fn: Any, f: dict[str, Any], cuts: list[int]) -> None:
    """The signal and stop at every bar below m are unchanged when the rows from m on are removed."""
    sig, stop = fn(f)
    for m in cuts:
        s2, st2 = fn(cut(f, m))
        assert np.array_equal(s2, sig[:m]), f"signal changed after cutting to {m}"
        assert np.allclose(st2, stop[:m], equal_nan=True), f"stop changed after cutting to {m}"


def fired(sig: np.ndarray) -> list[int]:
    """Indices where a signal is True."""
    return [int(i) for i in np.flatnonzero(np.asarray(sig, bool))]
