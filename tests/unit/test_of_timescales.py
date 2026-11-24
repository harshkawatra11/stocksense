"""M4.7 multi-timeframe alignment."""

from __future__ import annotations

import numpy as np

from stocksense.marketlab.bars import IST_OFFSET_S, Bars
from stocksense.orderflow.timescales import alignment, resample

DAY = 20000
INTRADAY = {"5m": 5, "15m": 15, "60m": 60}


def _closes(n: int, sign: float = 1.0) -> np.ndarray:
    start = (5 - (n - 1)) % 10  # the last step is the end of an up-leg
    steps = np.tile([2.0] * 6 + [-0.5] * 4, n // 10 + 2)[start : start + n]
    return 300.0 + sign * np.cumsum(steps)


def _bars(ts: np.ndarray, close: np.ndarray) -> Bars:
    return Bars.from_arrays(ts, close, close + 0.02, close - 0.02, close, np.full(len(close), 100.0))


def _intraday(tf: int, sessions: int, sign: float) -> Bars:
    per = -(-375 // tf)
    ts = np.concatenate(
        [(DAY + s) * 86400 - IST_OFFSET_S + (555 + np.arange(per) * tf) * 60 for s in range(sessions)]
    )
    return _bars(ts, _closes(len(ts), sign))


def _daily(n: int, step_days: int, sign: float) -> Bars:
    ts = (DAY + np.arange(n) * step_days) * 86400 - IST_OFFSET_S + 555 * 60
    return _bars(ts, _closes(n, sign))


def _all(sign: float) -> dict[str, Bars]:
    out = {k: _intraday(m, 4, sign) for k, m in INTRADAY.items()}
    out["1d"] = _daily(80, 1, sign)
    out["1w"] = _daily(60, 7, sign)
    return out


def test_all_rising_scores_plus_one() -> None:
    res = alignment(_all(1.0))
    assert [r["tf"] for r in res["rows"]] == ["5m", "15m", "60m", "1d", "1w"]
    for r in res["rows"]:
        assert (r["trend"], r["vs_value"], r["vs_va"], r["roc"]) == (1, 1, 1, 1), r
    assert res["score"] == 1.0


def test_all_falling_scores_minus_one() -> None:
    assert alignment(_all(-1.0))["score"] == -1.0


def test_resample_from_5m_and_daily() -> None:
    b5 = _intraday(5, 2, 1.0)
    b15 = resample(b5, "15m")
    assert len(b15) == 50
    assert b15.open[0] == b5.open[0] and b15.close[0] == b5.close[2]
    assert b15.high[0] == b5.high[:3].max() and b15.low[0] == b5.low[:3].min()
    assert b15.volume[0] == 300.0 and int(b15.ts[1]) == int(b5.ts[3])
    assert len(resample(b5, "60m")) == 14  # 7 per session
    d = _daily(14, 1, 1.0)  # 14 consecutive days = exactly 2 or 3 Monday-based weeks
    w = resample(d, "1w")
    assert w.volume.sum() == d.volume.sum() and len(w) in (2, 3)


def test_alignment_builds_missing_timeframes() -> None:
    res = alignment({"5m": _intraday(5, 4, 1.0), "1d": _daily(80, 1, 1.0)})
    assert [r["tf"] for r in res["rows"]] == ["5m", "15m", "60m", "1d", "1w"]
    assert -1.0 <= res["score"] <= 1.0
