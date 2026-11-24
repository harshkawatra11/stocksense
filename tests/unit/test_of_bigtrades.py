"""M4.5 volume-spike bars (big trades) and absorption."""

from __future__ import annotations

import numpy as np

from stocksense.marketlab.bars import IST_OFFSET_S, Bars
from stocksense.orderflow.absorption import absorption
from stocksense.orderflow.bigtrades import big_bars

DAY = 20000
NAN = float("nan")


def _session(day: int, vol: np.ndarray, close: np.ndarray | None = None) -> Bars:
    n = len(vol)
    c = 100 + 0.1 * np.arange(n) if close is None else close
    ts = day * 86400 - IST_OFFSET_S + (555 + np.arange(n)) * 60
    return Bars.from_arrays(ts, c - 0.02, c + 0.05, c - 0.05, c, vol)


def _concat(parts: list[Bars]) -> Bars:
    return Bars.from_arrays(
        *(
            np.concatenate([getattr(p, f) for p in parts])
            for f in ("ts", "open", "high", "low", "close", "volume")
        )
    )


def _hist(volumes: list[float], n: int = 40) -> Bars:
    first = DAY - len(volumes)
    return _concat([_session(first + k, np.full(n, v)) for k, v in enumerate(volumes)])


def _today(spike: float = 600.0, n: int = 40) -> Bars:
    vol = np.full(n, 100.0)
    vol[2] = spike
    return _session(DAY, vol)


def test_flags_spike_with_forward_stats() -> None:
    out = big_bars(_today(), _hist([100.0] * 12))
    assert [o["i"] for o in out] == [2]
    o = out[0]
    assert o["baseline"] == 100.0 and abs(o["ratio"] - 6.0) < 1e-12
    assert o["side"] == 1
    c = 100 + 0.1 * np.arange(40)
    assert abs(o["ret_5m"] - (c[7] / c[2] - 1)) < 1e-12
    assert abs(o["ret_15m"] - (c[17] / c[2] - 1)) < 1e-12
    assert abs(o["ret_30m"] - (c[32] / c[2] - 1)) < 1e-12
    assert abs(o["mfe"] - ((c[32] + 0.05) / c[2] - 1)) < 1e-12
    assert o["mae"] == 0.0  # price never traded below the close


def test_threshold_is_strict_and_needs_history() -> None:
    assert big_bars(_today(500.0), _hist([100.0] * 12)) == []
    assert big_bars(_today(), _hist([100.0] * 9)) == []  # fewer than min_sessions


def test_baseline_uses_previous_20_sessions_only() -> None:
    hist = _hist([1000.0] * 25 + [100.0] * 20)
    assert [o["i"] for o in big_bars(_today(), hist)] == [2]


def test_past_only() -> None:
    hist = _hist([100.0] * 12)
    full = big_bars(_today(), hist)
    cut = big_bars(_today().take(slice(0, 10)), hist)
    assert [(o["ts"], o["baseline"], o["ratio"]) for o in cut] == [
        (o["ts"], o["baseline"], o["ratio"]) for o in full
    ]
    assert cut[0]["ret_30m"] is None
    future = _session(DAY + 1, np.full(40, 9999.0))
    again = big_bars(_today(), _concat([hist, future]))
    assert [o["baseline"] for o in again] == [100.0]


def _flat_series(n: int = 30) -> dict[str, np.ndarray]:
    c = 100.0 - 0.1 * np.arange(n)
    return {"open": c.copy(), "high": c + 0.5, "low": c - 0.5, "close": c.copy(), "volume": np.full(n, 100.0)}


def _mk(d: dict[str, np.ndarray]) -> Bars:
    n = len(d["close"])
    ts = DAY * 86400 - IST_OFFSET_S + (555 + np.arange(n)) * 60
    return Bars.from_arrays(ts, d["open"], d["high"], d["low"], d["close"], d["volume"])


def test_absorption_bid_at_day_low() -> None:
    d = _flat_series()
    d["volume"][25] = 400.0
    d["high"][25] = d["low"][24] - 0.05 + 0.2
    d["low"][25] = d["low"][24] - 0.05
    d["open"][25] = d["close"][25] = d["low"][25] + 0.1
    nan = np.full(30, NAN)
    out = absorption(_mk(d), nan, nan, None, None)
    assert [(o["i"], o["side"], o["level"]) for o in out] == [(25, "bid", "day_low")]
    assert abs(out[0]["volume_ratio"] - 4.0) < 1e-12


def test_absorption_offer_at_day_high() -> None:
    d = _flat_series()
    for k in ("open", "high", "low", "close"):
        d[k] = 200.0 - d[k]  # mirror: rising prices
    d["high"], d["low"] = d["low"], d["high"]
    d["volume"][25] = 400.0
    d["low"][25] = d["high"][24] + 0.05 - 0.2
    d["high"][25] = d["high"][24] + 0.05
    d["open"][25] = d["close"][25] = d["high"][25] - 0.1
    nan = np.full(30, NAN)
    out = absorption(_mk(d), nan, nan, None, None)
    assert [(o["i"], o["side"], o["level"]) for o in out] == [(25, "offer", "day_high")]


def test_absorption_rejects_wide_range_and_low_volume() -> None:
    d = _flat_series()
    nan = np.full(30, NAN)
    d["volume"][25] = 400.0  # range stays 1.0 > 0.5 x median
    assert absorption(_mk(d), nan, nan, None, None) == []
    d = _flat_series()
    d["volume"][25] = 250.0
    d["high"][25], d["low"][25] = d["close"][25] + 0.1, d["close"][25] - 0.1
    assert absorption(_mk(d), nan, nan, None, None) == []


def test_absorption_at_prev_val_mid_range() -> None:
    n = 30
    c = np.r_[np.full(20, 100.0), np.full(5, 103.0), 101.0, np.full(4, 102.0)]
    d = {"open": c.copy(), "high": c + 0.5, "low": c - 0.5, "close": c.copy(), "volume": np.full(n, 100.0)}
    d["volume"][25] = 400.0
    d["high"][25], d["low"][25] = 101.1, 100.9
    nan = np.full(n, NAN)
    out = absorption(_mk(d), nan, nan, 105.0, 101.0)
    assert [(o["i"], o["side"], o["level"]) for o in out] == [(25, "bid", "val")]


def test_absorption_is_past_only() -> None:
    d = _flat_series()
    d["volume"][25] = 400.0
    d["high"][25] = d["low"][24] - 0.05 + 0.2
    d["low"][25] = d["low"][24] - 0.05
    d["open"][25] = d["close"][25] = d["low"][25] + 0.1
    full_b = _mk(d)
    nan = np.full(30, NAN)
    full = absorption(full_b, nan, nan, None, None)
    cut = absorption(full_b.take(slice(0, 26)), nan[:26], nan[:26], None, None)
    assert [(o["i"], o["side"]) for o in cut] == [(o["i"], o["side"]) for o in full]
