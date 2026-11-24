"""Footprint ladders estimated from 1-minute bars (bulk-volume classification spread over price bins)."""

from __future__ import annotations

from typing import Any

import numpy as np

from stocksense.marketlab.bars import Bars
from stocksense.orderflow.delta import delta_bvc

METHOD = "BVC estimate from 1-minute bars"
_EPS = 1e-9
_OPEN_MINUTE = 9 * 60 + 15


def stacked_imbalances(
    rows: list[dict[str, Any]], ratio: float = 3.0, min_run: int = 3
) -> list[dict[str, Any]]:
    """Runs of >= `min_run` adjacent rows with buy >= ratio*sell below, or sell >= ratio*buy above."""
    n = len(rows)
    buy_flag = [
        i > 0 and rows[i]["buy"] > 0 and rows[i]["buy"] >= ratio * rows[i - 1]["sell"] for i in range(n)
    ]
    sell_flag = [
        i < n - 1 and rows[i]["sell"] > 0 and rows[i]["sell"] >= ratio * rows[i + 1]["buy"] for i in range(n)
    ]
    out: list[dict[str, Any]] = []
    for side, flags in (("buy", buy_flag), ("sell", sell_flag)):
        i = 0
        while i < n:
            if not flags[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            if j - i + 1 >= min_run:
                out.append(
                    {
                        "side": side,
                        "from_price": rows[i]["price"],
                        "to_price": rows[j]["price"],
                        "n": j - i + 1,
                    }
                )
            i = j + 1
    return out


def footprint(b1m: Bars, tf_minutes: int, bin: float) -> list[dict[str, Any]]:  # noqa: A002
    """Per display bar: ladder rows (price=bin lower edge, shares), bar delta, POC, stacked imbalances."""
    n = len(b1m)
    if n == 0:
        return []
    delta = delta_bvc(b1m)
    buy_v = (b1m.volume + delta) / 2.0
    sell_v = (b1m.volume - delta) / 2.0
    slot = (b1m.day_index() * 1440 + np.maximum(b1m.minute_of_day() - _OPEN_MINUTE, 0)) // tf_minutes
    out: list[dict[str, Any]] = []
    starts = np.r_[0, np.flatnonzero(slot[1:] != slot[:-1]) + 1, n]
    for a, z in zip(starts[:-1], starts[1:], strict=True):
        acc: dict[int, list[float]] = {}
        for i in range(a, z):
            k0 = int(np.floor(b1m.low[i] / bin + _EPS))
            k1 = max(k0, int(np.ceil(b1m.high[i] / bin - _EPS)) - 1)
            ks = np.arange(k0, k1 + 1)
            rng = float(b1m.high[i] - b1m.low[i])
            if rng <= 0 or len(ks) == 1:
                w = np.zeros(len(ks))
                w[0] = 1.0
            else:
                left = ks * bin
                w = np.maximum(np.minimum(left + bin, b1m.high[i]) - np.maximum(left, b1m.low[i]), 0.0)
                w = w / w.sum()
            for k, wk in zip(ks, w, strict=True):
                cell = acc.setdefault(int(k), [0.0, 0.0])
                cell[0] += buy_v[i] * wk
                cell[1] += sell_v[i] * wk
        rows = [
            {"price": round(k * bin, 10), "buy": v[0], "sell": v[1], "delta": v[0] - v[1]}
            for k, v in sorted(acc.items())
        ]
        poc = max(rows, key=lambda r: (r["buy"] + r["sell"], -r["price"]))["price"]
        out.append(
            {
                "ts": int(b1m.ts[a]),
                "open": float(b1m.open[a]),
                "high": float(b1m.high[a:z].max()),
                "low": float(b1m.low[a:z].min()),
                "close": float(b1m.close[z - 1]),
                "volume": float(b1m.volume[a:z].sum()),
                "rows": rows,
                "delta": float(sum(r["delta"] for r in rows)),
                "poc": poc,
                "stacked_imbalances": stacked_imbalances(rows),
                "method": METHOD,
            }
        )
    return out
