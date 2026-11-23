"""Write deterministic synthetic bar inputs for the Market Lab golden tests (numpy only).

Run: .venv\\Scripts\\python.exe scripts\\golden\\make_inputs.py tests\\fixtures\\marketlab
Outputs inputs_5m_stock.json, inputs_5m_index.json (volume 0), inputs_1d_stock.json, inputs_15m_stock.json and
inputs_1d_from5m.json. Each is a dict of equal-length lists: ts, open, high, low, close, volume
with ts in epoch seconds (UTC) at real IST session times.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

IST_OFFSET = 19800
OPEN_MIN = 9 * 60 + 15


def sessions(start: date, n: int) -> list[date]:
    """First n weekdays from `start`."""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def epoch(d: date, minute: int) -> int:
    """Epoch second of IST date `d` at `minute` minutes after IST midnight."""
    return (d - date(1970, 1, 1)).days * 86400 + minute * 60 - IST_OFFSET


def make_bars(
    ts: list[int], sigma: float, base: float, seed: int, with_volume: bool
) -> dict[str, list[float]]:
    """GBM closes; high/low extend the open-close range by |N(0, .0008)|; lognormal volume."""
    rng = np.random.default_rng(seed)
    n = len(ts)
    close = base * np.exp(np.cumsum(rng.normal(0.0, sigma, n)))
    open_ = np.r_[base, close[:-1]]
    wick = 0.0008 if sigma < 0.01 else 0.004
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, wick, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, wick, n)))
    vol = np.exp(rng.normal(10.0, 0.6, n)) if with_volume else np.zeros(n)
    return {
        "ts": [int(t) for t in ts],
        "open": [round(float(x), 6) for x in open_],
        "high": [round(float(x), 6) for x in high],
        "low": [round(float(x), 6) for x in low],
        "close": [round(float(x), 6) for x in close],
        "volume": [round(float(x), 2) for x in vol],
    }


def aggregate(b: dict[str, list[float]], per: int) -> dict[str, list[float]]:
    """Merge every `per` consecutive bars (bars of one day must divide evenly)."""
    n = len(b["ts"]) // per
    out: dict[str, list[float]] = {k: [] for k in b}
    for i in range(n):
        a, z = i * per, (i + 1) * per
        out["ts"].append(b["ts"][a])
        out["open"].append(b["open"][a])
        out["high"].append(max(b["high"][a:z]))
        out["low"].append(min(b["low"][a:z]))
        out["close"].append(b["close"][z - 1])
        out["volume"].append(round(sum(b["volume"][a:z]), 2))
    return out


def daily_from(b: dict[str, list[float]]) -> dict[str, list[float]]:
    """One bar per IST date from intraday bars (open of first, high, low, close of last, volume sum)."""
    days = [(t + IST_OFFSET) // 86400 for t in b["ts"]]
    out: dict[str, list[float]] = {k: [] for k in b}
    for d in sorted(set(days)):
        idx = [i for i, x in enumerate(days) if x == d]
        out["ts"].append(b["ts"][idx[0]])
        out["open"].append(b["open"][idx[0]])
        out["high"].append(max(b["high"][i] for i in idx))
        out["low"].append(min(b["low"][i] for i in idx))
        out["close"].append(b["close"][idx[-1]])
        out["volume"].append(round(sum(b["volume"][i] for i in idx), 2))
    return out


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    s5 = sessions(date(2025, 1, 1), 40)
    ts5 = [epoch(d, OPEN_MIN + 5 * k) for d in s5 for k in range(75)]
    stock5 = make_bars(ts5, 0.0015, 1000.0, 11, True)
    index5 = make_bars(ts5, 0.0010, 22000.0, 13, False)
    s1 = sessions(date(2023, 1, 2), 600)
    stock1 = make_bars([epoch(d, OPEN_MIN) for d in s1], 0.018, 500.0, 12, True)
    files = {
        "inputs_5m_stock": stock5,
        "inputs_5m_index": index5,
        "inputs_1d_stock": stock1,
        "inputs_15m_stock": aggregate(stock5, 3),
        "inputs_1d_from5m": daily_from(stock5),
    }
    for name, b in files.items():
        (out_dir / f"{name}.json").write_text(json.dumps(b), encoding="utf-8")
        print(name, len(b["ts"]))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
