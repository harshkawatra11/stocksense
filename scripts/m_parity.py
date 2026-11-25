"""Parity check between Market Lab (port 8810) and StockSense (port 8765); both are started by hand.

Takes N samples SECONDS apart in market hours, compares /api/sm/live and /api/horizon?range=today, prints a
table and a pass/fail against fixed thresholds. Read-only: it only issues GET requests.

    .venv\\Scripts\\python.exe scripts/m_parity.py [--samples 10] [--interval 30] [--ml 8810] [--ours 8765]
"""

from __future__ import annotations

import argparse
import sys
import threading
from typing import Any

import httpx
import numpy as np
from scipy.stats import kendalltau

THRESHOLDS = {"jaccard": 0.85, "spearman": 0.98, "mad_pp": 0.05, "tau": 0.9}
TOP = 20


def jaccard(a: list[str], b: list[str]) -> float:
    """|A and B| / |A or B| of two symbol lists (1.0 when both are empty)."""
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):  # average the ranks of ties
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties; NaN when fewer than 3 points or no spread."""
    if len(x) < 3:
        return float("nan")
    rx, ry = _ranks(np.asarray(x, float)), _ranks(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def mad(a: dict[str, float], b: dict[str, float]) -> float:
    """Mean absolute difference over the keys both dicts have (percentage points); NaN when none."""
    keys = sorted(set(a) & set(b))
    return float(np.mean([abs(a[k] - b[k]) for k in keys])) if keys else float("nan")


def tau(a: dict[str, float], b: dict[str, float]) -> float:
    """Kendall tau between two name -> value maps over their common names; NaN when fewer than 3."""
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return float("nan")
    return float(kendalltau([a[k] for k in keys], [b[k] for k in keys]).statistic)


def sample(
    ml: dict[str, Any], ours: dict[str, Any], hz_ml: dict[str, Any], hz_ours: dict[str, Any]
) -> dict[str, float]:
    """One parity sample from the two /api/sm/live and the two /api/horizon payloads."""
    disp_a = {s["sym"]: float(s["disp"]) for s in ml.get("stocks", []) if s.get("disp") is not None}
    disp_b = {s["sym"]: float(s["disp"]) for s in ours.get("stocks", []) if s.get("disp") is not None}
    common = sorted(set(disp_a) & set(disp_b))
    chips_a = {s["name"]: float(s["pct"]) for s in ml.get("sectors", []) if s.get("pct") is not None}
    chips_b = {s["name"]: float(s["pct"]) for s in ours.get("sectors", []) if s.get("pct") is not None}
    order_a = {s["name"]: -float(s["v"]) for s in hz_ml.get("sectors", []) if s.get("v") is not None}
    order_b = {s["name"]: -float(s["v"]) for s in hz_ours.get("sectors", []) if s.get("v") is not None}
    return {
        "jaccard_bull": jaccard(ml.get("bull", [])[:TOP], ours.get("bull", [])[:TOP]),
        "jaccard_bear": jaccard(ml.get("bear", [])[:TOP], ours.get("bear", [])[:TOP]),
        "spearman": spearman(np.array([disp_a[s] for s in common]), np.array([disp_b[s] for s in common])),
        "mad_pp": mad(chips_a, chips_b),
        "tau": tau(order_a, order_b),
    }


def verdict(rows: list[dict[str, float]]) -> dict[str, tuple[float, bool]]:
    """Mean of each metric over the samples (NaNs ignored) and whether it meets its threshold."""

    def mean(key: str) -> float:
        v = [r[key] for r in rows if not np.isnan(r[key])]
        return float(np.mean(v)) if v else float("nan")

    jac = float(np.nanmean([mean("jaccard_bull"), mean("jaccard_bear")]))
    vals = {"jaccard": jac, "spearman": mean("spearman"), "mad_pp": mean("mad_pp"), "tau": mean("tau")}
    ok = {
        "jaccard": vals["jaccard"] >= THRESHOLDS["jaccard"],
        "spearman": vals["spearman"] >= THRESHOLDS["spearman"],
        "mad_pp": vals["mad_pp"] <= THRESHOLDS["mad_pp"],
        "tau": vals["tau"] >= THRESHOLDS["tau"],
    }
    return {k: (vals[k], bool(ok[k])) for k in vals}


def _get(client: httpx.Client, port: int, path: str) -> dict[str, Any]:
    r = client.get(f"http://127.0.0.1:{port}{path}")
    r.raise_for_status()
    return r.json()


def main(argv: list[str] | None = None) -> int:
    """Run the samples, print the table, and return 0 when every threshold passes, else 1."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--ml", type=int, default=8810)
    ap.add_argument("--ours", type=int, default=8765)
    args = ap.parse_args(argv)
    rows: list[dict[str, float]] = []
    wait = threading.Event()  # Event.wait is the project's sleep
    with httpx.Client(timeout=20) as client:
        for i in range(args.samples):
            try:
                row = sample(
                    _get(client, args.ml, "/api/sm/live"),
                    _get(client, args.ours, "/api/sm/live"),
                    _get(client, args.ml, "/api/horizon?range=today"),
                    _get(client, args.ours, "/api/horizon?range=today"),
                )
            except (httpx.HTTPError, ValueError) as e:
                print(f"sample {i + 1}: failed ({type(e).__name__}); continuing")
            else:
                rows.append(row)
                print(
                    f"sample {i + 1:>2}: jac {row['jaccard_bull']:.2f}/{row['jaccard_bear']:.2f}"
                    f"  rho {row['spearman']:.3f}  mad {row['mad_pp']:.3f}pp  tau {row['tau']:.2f}"
                )
            if i + 1 < args.samples:
                wait.wait(args.interval)
    if not rows:
        print("no successful sample: are both servers up and the market open?")
        return 1
    res = verdict(rows)
    print(f"\n{'metric':<10}{'mean':>10}{'threshold':>12}  result")
    marks = {"jaccard": ">=", "spearman": ">=", "mad_pp": "<=", "tau": ">="}
    for k, (v, ok) in res.items():
        print(f"{k:<10}{v:>10.3f}{marks[k] + ' ' + str(THRESHOLDS[k]):>12}  {'PASS' if ok else 'FAIL'}")
    return 0 if all(ok for _, ok in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
