"""M4.1 session and anchored VWAP."""

from __future__ import annotations

import numpy as np

from stocksense.marketlab.bars import IST_OFFSET_S, Bars
from stocksense.orderflow.vwap import anchored_vwap, bands, session_vwap

DAY0 = 20000  # an IST day number


def _ts(day: int, minute: int) -> int:
    return day * 86400 - IST_OFFSET_S + minute * 60


def _flat(tp: list[float], vol: list[float], day: int = DAY0) -> Bars:
    ts = [_ts(day, 555 + i) for i in range(len(tp))]
    return Bars.from_arrays(ts, tp, tp, tp, tp, vol)


def test_vwap_and_sigma_hand_numbers() -> None:
    v, s = session_vwap(_flat([10, 11, 12], [1, 1, 2]))
    assert np.allclose(v, [10, 10.5, 11.25])
    assert abs(s[2] - np.sqrt(0.6875)) < 1e-12
    assert abs(s[2] - 0.8292) < 1e-4
    assert s[0] == 0.0


def test_zero_volume_weight_is_one() -> None:
    v, _ = session_vwap(_flat([10, 12], [0, 0]))
    assert np.allclose(v, [10, 11])


def test_resets_each_session() -> None:
    a, b = _flat([10, 11], [1, 1], DAY0), _flat([20, 22], [1, 1], DAY0 + 1)
    both = Bars.from_arrays(
        np.r_[a.ts, b.ts], np.r_[a.open, b.open], np.r_[a.high, b.high], np.r_[a.low, b.low],
        np.r_[a.close, b.close], np.r_[a.volume, b.volume],
    )  # fmt: skip
    v, _ = session_vwap(both)
    assert np.allclose(v, [10, 10.5, 20, 21])


def test_bands() -> None:
    out = bands(np.array([10.0]), np.array([0.5]))
    assert set(out) == {"u1", "l1", "u2", "l2", "u3", "l3"}
    assert out["u2"][0] == 11.0 and out["l3"][0] == 8.5


def test_anchored_equals_session_at_open_and_nan_before() -> None:
    b = _flat([10, 11, 12], [1, 1, 2])
    v, _ = session_vwap(b)
    assert np.allclose(anchored_vwap(b, int(b.ts[0])), v)
    a = anchored_vwap(b, int(b.ts[1]))
    assert np.isnan(a[0]) and a[1] == 11 and abs(a[2] - (11 + 24) / 3) < 1e-12
