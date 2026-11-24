"""M4.2 volume profile, TPO, day types, open types."""

from __future__ import annotations

import numpy as np

from stocksense.marketlab.bars import IST_OFFSET_S, Bars
from stocksense.orderflow.profile import bin_size, day_type, open_type, tpo_profile, volume_profile

DAY = 20000


def _ts(minute: int) -> int:
    return DAY * 86400 - IST_OFFSET_S + minute * 60


def _bars(rows: list[tuple[float, float, float, float, float]], start: int = 555, step: int = 5) -> Bars:
    """rows of (open, high, low, close, volume), one bar every `step` minutes from `start` (IST minute)."""
    ts = [_ts(start + i * step) for i in range(len(rows))]
    o, h, lo, c, v = (np.array(x, dtype=float) for x in zip(*rows, strict=True))
    return Bars.from_arrays(ts, o, h, lo, c, v)


def _periods(lh: list[tuple[float, float]], close_last: float | None = None) -> Bars:
    """One 30-minute bar per period (A, B, ...) with (low, high)."""
    rows = []
    for i, (lo, hi) in enumerate(lh):
        c = (lo + hi) / 2
        if close_last is not None and i == len(lh) - 1:
            c = close_last
        rows.append((c, hi, lo, c, 100.0))
    return _bars(rows, 555, 30)


def test_bin_size() -> None:
    assert bin_size(100.0) == 0.05
    assert bin_size(1000.0) == 0.5
    assert bin_size(2468.0) == 1.23


def test_one_bar_uniform_four_bins() -> None:
    p = volume_profile(_bars([(100, 101, 100, 100.5, 100)]), bin=0.25)
    assert np.allclose(p["vol"], [25, 25, 25, 25])
    assert p["poc"] == p["bins"][0] + 0.125  # ties go to the lowest bin
    assert p["total"] == 100.0
    assert p["vah"] - p["val"] >= 0.75 - 1e-9  # 3 bins = 75 >= 70


def test_bell_distribution_value_area() -> None:
    b = _bars(
        [
            (100, 101, 100, 100.5, 10),  # bin 0
            (101, 102, 101, 101.5, 20),
            (102, 103, 102, 102.5, 40),
            (103, 104, 103, 103.5, 20),
            (104, 105, 104, 104.5, 10),
        ]
    )
    p = volume_profile(b, bin=1.0)
    assert np.allclose(p["vol"], [10, 20, 40, 20, 10])
    assert p["poc"] == 102.5
    assert (p["val"], p["vah"]) == (101.0, 104.0)


def test_volume_conserved() -> None:
    rng = np.random.default_rng(1)
    base = 100 + np.cumsum(rng.normal(0, 0.2, 60))
    rows = [(x, x + 0.4, x - 0.3, x + 0.1, 50.0 + i) for i, x in enumerate(base)]
    p = volume_profile(_bars(rows, step=1), bin=0.1)
    assert abs(p["vol"].sum() - sum(r[4] for r in rows)) < 1e-9


def test_tpo_letters_and_ib() -> None:
    lh = [(100 + i, 102 + i) for i in range(13)]
    t = tpo_profile(_periods(lh), bin=1.0)
    assert t["periods"] == 13
    assert t["letters"][0] == "A"  # lowest bin 100 touched only by A
    assert t["letters"][1] == "AB"  # 101: A (100-102) and B (101-103)
    assert t["letters"][-1] == "M"
    assert (t["ib_low"], t["ib_high"]) == (100.0, 103.0)
    assert t["high"] == 114.0 and t["low"] == 100.0


def test_tpo_single_prints_and_poor_extremes() -> None:
    # A,B,C trade 100-104, D alone 104-108, E,F,G trade 108-112 -> singles at 104..107
    lh = [(100, 104)] * 3 + [(104, 108)] + [(108, 112)] * 3
    t = tpo_profile(_periods(lh), bin=1.0)
    assert t["single_prints"] == [104.0, 105.0, 106.0, 107.0]
    assert t["poor_high"] is True and t["poor_low"] is True
    assert t["poc_tpo"] in (103.5, 108.5)


def test_day_types() -> None:
    normal = _periods([(100, 110), (101, 109), (102, 108), (103, 107)])
    assert day_type(tpo_profile(normal, bin=1.0), 10.0) == "normal"
    assert day_type(tpo_profile(normal, bin=1.0), 100.0) == "non_trend"
    trend = _periods([(100, 102), (101, 103), (103, 106), (106, 109), (109, 112), (111, 113)], 113)
    assert day_type(tpo_profile(trend, bin=1.0), 10.0) == "trend"
    neutral = _periods([(100, 104), (101, 105), (98, 103), (103, 108)], 104)
    assert day_type(tpo_profile(neutral, bin=1.0), 10.0) == "neutral"
    nv = _periods([(100, 106), (101, 107), (102, 110)], 105)
    assert day_type(tpo_profile(nv, bin=1.0), 10.0) == "normal_variation"
    dd = _periods([(100, 104)] * 3 + [(104, 108)] + [(108, 112)] * 3)
    assert day_type(tpo_profile(dd, bin=1.0), 10.0) == "double_distribution"


def _first30(rows: list[tuple[float, float, float, float]]) -> Bars:
    return _bars([(o, h, lo, c, 100.0) for o, h, lo, c in rows])


def test_open_types() -> None:
    drive = _first30(
        [(100, 100.2, 100, 100.2), (100.2, 100.4, 100.1, 100.4), (100.4, 100.6, 100.3, 100.5)]
        + [(100.5, 100.6, 100.4, 100.5)] * 3
    )
    assert open_type(101, 99, drive) == "open_drive"
    test_drive = _first30(
        [(100, 100.1, 99.8, 99.9), (99.9, 100.1, 99.85, 100.0), (100.0, 100.5, 99.95, 100.45)]
        + [(100.45, 100.5, 100.3, 100.4)] * 3
    )
    assert open_type(101, 99, test_drive) == "open_test_drive"
    reject = _first30(
        [
            (100, 100.5, 100, 100.4),
            (100.4, 100.45, 100.2, 100.25),
            (100.25, 100.3, 100.1, 100.15),
            (100.15, 100.2, 99.95, 100.0),
            (100.0, 100.0, 99.75, 99.8),
            (99.8, 99.9, 99.75, 99.8),
        ]
    )
    assert open_type(101, 99, reject) == "open_rejection_reverse"
    calm = _first30([(100, 100.1, 99.9, 100.0)] * 6)
    assert open_type(101, 99, calm) == "open_auction_in_range"
    assert open_type(99, 98, calm) == "open_auction_out_of_range"
