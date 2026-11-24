"""M4.4 footprint (BVC estimate from 1-minute bars)."""

from __future__ import annotations

import numpy as np

from stocksense.marketlab.bars import IST_OFFSET_S, Bars
from stocksense.orderflow.footprint import footprint, stacked_imbalances

DAY = 20000


def _rows(buy: list[float], sell: list[float], p0: float = 100.0, bin_: float = 1.0) -> list[dict]:
    return [
        {"price": p0 + i * bin_, "buy": b, "sell": s, "delta": b - s}
        for i, (b, s) in enumerate(zip(buy, sell, strict=True))
    ]


def test_volume_conserved_and_grouping() -> None:
    rng = np.random.default_rng(7)
    n = 90
    c = 100 + np.cumsum(rng.normal(0, 0.15, n))
    ts = DAY * 86400 - IST_OFFSET_S + (555 + np.arange(n)) * 60
    b = Bars.from_arrays(ts, c, c + 0.2, c - 0.2, c, rng.integers(100, 500, n).astype(float))
    fp = footprint(b, 5, 0.1)
    assert len(fp) == 18
    tot = sum(r["buy"] + r["sell"] for bar in fp for r in bar["rows"])
    assert abs(tot - b.volume.sum()) < 1e-6
    for bar in fp:
        assert abs(bar["delta"] - sum(r["delta"] for r in bar["rows"])) < 1e-9
        assert bar["method"] == "BVC estimate from 1-minute bars"
    assert fp[0]["ts"] == int(ts[0]) and fp[1]["ts"] == int(ts[5])


def test_stacked_buy_imbalance_on_ladder() -> None:
    # buy[i] >= 3*sell[i-1] for i = 1,2,3 only
    rows = _rows(buy=[1, 30, 40, 50, 2], sell=[10, 10, 12, 15, 9])
    out = stacked_imbalances(rows)
    assert len(out) == 1
    assert out[0]["side"] == "buy" and out[0]["n"] == 3
    assert (out[0]["from_price"], out[0]["to_price"]) == (101.0, 103.0)


def test_stacked_sell_imbalance_and_no_pair() -> None:
    rows = _rows(buy=[10, 12, 15, 1, 9], sell=[50, 50, 50, 10, 2])
    out = stacked_imbalances(rows)
    assert [(o["side"], o["n"]) for o in out] == [("sell", 3)]
    assert stacked_imbalances(_rows([5, 5, 5], [5, 5, 5])) == []
