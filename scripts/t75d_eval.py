# ruff: noqa: E501
"""T7.5d evaluation of the stored Kronos zero-shot features under the pre-registered design
(docs/DECISIONS.md, "T7.5d pre-registration"). Prints markdown tables; research period only."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from stocksense.core.config import load_settings
from stocksense.core.paths import Paths
from stocksense.features.build import cross_section_transform, current_version
from stocksense.features.build import load_features as load_features
from stocksense.labels.forward import load_labels
from stocksense.models import kronos_eval as ke

TAG = "small-zs-r150"
NAMES = ["kr_ret5_mean", "kr_ret5_std", "kr_up_prob", "kr_oc1_mean", "kr_co1_mean", "kr_range5", "kr_volchg5"]
TARGETS = {"y_oc": ("z_y_oc", "end_oc"), "y_c2o_5": ("z_y_c2o_5", "end_5")}
STANDALONE = ["y_oc", "y_c2o_5", "y_c2o_20"]
CORR_WITH = ["mom_5", "ret_1", "vol_21"]
ALPHA = 100.0
START, END = date(2016, 1, 1), date(2025, 9, 30)


def main() -> None:
    s = load_settings()
    folder = Paths(s).features / "kronos" / TAG
    kr = pl.concat([pl.read_parquet(f) for f in sorted(folder.glob("*.parquet"))]).filter(
        pl.col("date") <= END
    )
    kr = kr.with_columns(pl.col("sid").cast(pl.Int32))
    dates = sorted(kr["date"].unique().to_list())
    print(f"kr rows {kr.height}, dates {len(dates)} ({dates[0]} .. {dates[-1]})")
    man = json.loads((Paths(s).features / f"v{current_version(s)}" / "manifest.json").read_text())
    xs = [f["name"] for f in man["features"] if f["kind"] == "xs" and not f["name"].startswith("kr_")]
    feats = (
        load_features(s, "research", start=START, end=END, columns=xs)
        .filter(pl.col("date").is_in(dates))
        .collect()
    )
    ycols = [c for v in TARGETS.values() for c in v] + STANDALONE
    labs = load_labels(s, "research", start=START, end=END, columns=list(dict.fromkeys(ycols))).collect()
    krr = cross_section_transform(kr, NAMES, min_valid=50).select("date", "sid", *NAMES)
    kr_raw = kr.rename({n: f"raw_{n}" for n in NAMES})
    df = (
        feats.join(krr, on=["date", "sid"], how="inner")
        .join(kr_raw, on=["date", "sid"])
        .join(labs, on=["date", "sid"], how="inner")
    )
    df = df.with_columns([pl.col(c).fill_null(0.0) for c in [*xs, *NAMES]])
    print(f"joined rows {df.height}, dates {df['date'].n_unique()}, base features {len(xs)}")
    print("\n### Standalone mean daily Spearman IC (naive t over dates)\n")
    print("| feature | " + " | ".join(STANDALONE) + " |\n|---|" + "---|" * len(STANDALONE))
    for n in NAMES:
        cells = []
        for y in STANDALONE:
            ic = ke.daily_rank_ic(df, f"raw_{n}", y)["ic"].to_numpy()
            m, t, _ = ke.paired_t(ic)
            cells.append(f"{m:+.4f} (t {t:+.2f})")
        print(f"| {n} | " + " | ".join(cells) + " |")
    print("\n### Mean daily Spearman correlation of kr_ret5_mean with ranked features\n")
    for c in CORR_WITH:
        d = df.select("date", pl.col("raw_kr_ret5_mean").alias("a"), pl.col(c).alias("b"))
        m, t, _ = ke.paired_t(ke.daily_rank_ic(d, "a", "b")["ic"].to_numpy())
        print(f"- {c}: {m:+.3f}")
    print("\n### Incremental walk-forward (ridge alpha=100; without vs with the seven kr_* features)\n")
    print(
        "| target | dates | rows | IC without | IC with | mean diff | paired t | admitted |\n|---|---|---|---|---|---|---|---|"
    )
    out = {}
    for y, (tgt, end) in TARGETS.items():
        r = ke.incremental_walk_forward(df, xs, NAMES, y, end, target_col=tgt, alpha=ALPHA)
        ok = r["mean_diff"] >= 0.002 and r["t"] >= 2
        out[y] = (r, ok)
        print(f"| {y} | {r['n_dates']} | {r['n_test_rows']} | {r['mean_ic_without']:+.4f} | {r['mean_ic_with']:+.4f} | "
              f"{r['mean_diff']:+.4f} | {r['t']:+.2f} | {'YES' if ok else 'no'} |")  # fmt: skip
    print(
        "\nADMITTED"
        if any(ok for _, ok in out.values())
        else "\nNOT ADMITTED: no incremental value established"
    )
    if "--dry" in sys.argv:
        return
    try:  # trial ledger: family kronos-zs
        from stocksense.core.state import State
        from stocksense.validation.trials import finish_trial, register_trial

        state = State(s)
        rep = Path(Paths(s).root) / "reports" / "t75d"
        rep.mkdir(parents=True, exist_ok=True)
        for y, (r, _) in out.items():
            spec = {"model": "ridge", "alpha": ALPHA, "arms": "base vs base+kr7", "tag": TAG, "lookback": 128,
                    "n_paths": 10, "universe": "turnover_rank<=150", "every": 10}  # fmt: skip
            tid = register_trial(state, "kronos-zs", y, spec)
            p = rep / f"{y}_ic_diff.parquet"
            r["per_date"].write_parquet(p)
            finish_trial(state, tid, {"mean_diff": r["mean_diff"], "t": r["t"], "n_dates": r["n_dates"],
                                      "sharpe_daily": float(r["mean_diff"] / max(np.std(r["per_date"]["diff"].to_numpy(), ddof=1), 1e-12))}, p)  # fmt: skip
            print("trial", tid)
    except Exception as e:  # noqa: BLE001
        print("trial ledger not updated:", repr(e)[:200], file=sys.stderr)


if __name__ == "__main__":
    main()
