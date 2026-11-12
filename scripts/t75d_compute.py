# ruff: noqa: E501
"""T7.5d Kronos zero-shot computation under the pre-registered reduced plan (docs/DECISIONS.md).

Usage: python scripts/t75d_compute.py time            # time the first 3 dates (tag ...-timing)
       python scripts/t75d_compute.py run N_PATHS EVERY  # resumable full run
"""

from __future__ import annotations

import sys
import time
from datetime import date

from stocksense.core.config import load_settings
from stocksense.features import kronos_feats as kf
from stocksense.features.base import FeatureContext
from stocksense.models import kronos_model as km

START, END = date(2016, 1, 4), date(2025, 9, 30)
TAG = "small-zs-r150"


def decision_dates(settings, every: int) -> list[date]:
    ss = FeatureContext(settings, as_of=END).sessions().collect()["date"].to_list()
    ss = [d for d in ss if d >= START]
    return ss[::every]


def main() -> None:
    s = load_settings()
    mode = sys.argv[1]
    bundle = km.load_bundle("small", "cuda")
    if mode == "time":
        n_paths, every = int(sys.argv[2]), int(sys.argv[3])
        ds = decision_dates(s, every)
        cfg = kf.KronosConfig(lookback=128, n_paths=n_paths, max_rank=150)
        per = []
        for d in ds[:3]:
            t = time.time()
            df = kf.compute_kronos_day(s, d, "x", bundle, config=cfg)
            per.append(time.time() - t)
            print(f"{d} rows={len(df)} sec={per[-1]:.1f}", flush=True)
        avg = sum(per) / len(per)
        print(
            f"n_paths={n_paths} every={every} dates={len(ds)} avg_sec={avg:.1f} projected_hours={avg * len(ds) / 3600:.2f}"
        )
    else:
        n_paths, every = int(sys.argv[2]), int(sys.argv[3])
        ds = decision_dates(s, every)
        cfg = kf.KronosConfig(lookback=128, n_paths=n_paths, max_rank=150)
        print(f"run: {len(ds)} dates, n_paths={n_paths}, every={every}", flush=True)
        for i, d in enumerate(ds):
            t = time.time()
            done = kf.compute_kronos_range(s, [d], TAG, bundle, config=cfg)
            print(f"{i + 1}/{len(ds)} {d} computed={bool(done)} sec={time.time() - t:.1f}", flush=True)
        print("DONE", flush=True)


if __name__ == "__main__":
    main()
