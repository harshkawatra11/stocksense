"""T7.2 diagnostics for a finished GBM trial: gain importance by feature group (from the per-refit files
`models/_wf/<trial>/gain_*.json`) and the with/without ablation of a feature group.

The ablation is a MASKING ablation, not a retrain: the stored fold booster (the first seed's file per fold)
scores the fold's out-of-fold rows once as trained and once with the group's columns set to NaN (LightGBM
then follows each split's missing-value direction); the mean per-date rank IC against the raw target is
compared over the test dates on which the group exists (date >= the latest first_valid_date of its
features). A retrain without the group would be a new registered trial, so it is not done here.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from stocksense.core.config import Settings
from stocksense.core.paths import Paths
from stocksense.data import io
from stocksense.features.build import load_features
from stocksense.labels.forward import load_labels
from stocksense.validation.ic_report import group_gain_share
from stocksense.validation.splits import Fold

LATE_GROUPS = ("delivery", "derivs", "events", "intraday", "auction")
_TAG = re.compile(r"gain_(\d{4}-\d{2}-\d{2})_s(\d+)\.json$")


def _manifest(settings: Settings) -> dict[str, Any]:
    v = (Paths(settings).features / "CURRENT").read_text().strip()
    return json.loads((Paths(settings).features / f"v{v}" / "manifest.json").read_text(encoding="utf-8"))


def _per_date_ic(df: pl.DataFrame, score: str, label: str) -> pl.DataFrame:
    return (
        df.select("date", score, label)
        .drop_nulls()
        .with_columns(
            pl.col(score).rank("average").over("date").alias("_a"),
            pl.col(label).rank("average").over("date").alias("_b"),
            pl.len().over("date").alias("_n"),
        )
        .filter(pl.col("_n") >= 50)
        .group_by("date")
        .agg(pl.corr("_a", "_b").alias("ic"))
        .drop_nulls("ic")
        .sort("date")
    )


def _fold_of(train_end: date, folds: list[Fold]) -> Fold:
    return min((f for f in folds if f.test_start > train_end), key=lambda f: f.test_start)


def diagnose(
    settings: Settings, trial_id: str, spec: dict[str, Any], folds: list[Fold], groups: tuple[str, ...]
) -> dict[str, Any]:
    wf = Paths(settings).models / "_wf" / trial_id
    files = sorted(wf.glob("gain_*.json"))
    if not files:
        raise FileNotFoundError(f"no gain files under {wf} (the trial was not run through DiagGbm)")
    man = _manifest(settings)
    group_of = {f["name"]: f["group"] for f in man["features"]}
    first_valid = {f["name"]: f.get("first_valid_date") for f in man["features"]}
    # gain: normalise each refit to shares, average over folds and seeds
    acc: dict[str, float] = {}
    for p in files:
        g = json.loads(p.read_text(encoding="utf-8"))["gain"]
        tot = sum(g.values()) or 1.0
        for k, v in g.items():
            acc[k] = acc.get(k, 0.0) + v / tot / len(files)
            group_of.setdefault(k, "auction" if k.startswith("auc_") else "other")
    out: dict[str, Any] = {
        "n_refits": len(files),
        "group_gain": group_gain_share(acc, group_of),
        "top_features": sorted(acc.items(), key=lambda kv: -kv[1])[:15],
        "ablation": {},
    }
    # one booster per fold: the first seed's file
    by_fold: dict[str, tuple[Fold, Path, Path]] = {}
    for p in files:
        m = _TAG.search(p.name)
        assert m is not None
        f = _fold_of(date.fromisoformat(m.group(1)), folds)
        by_fold.setdefault(f.name, (f, wf / p.name.replace("gain_", "booster_").replace(".json", ".txt"), p))
    target = spec["target"]
    use_auction = bool(spec.get("params", {}).get("auction"))
    rows: dict[str, list[pl.DataFrame]] = {g: [] for g in groups}
    lab = load_labels(settings, "research", columns=[target]).collect()
    for f, bpath, gpath in by_fold.values():
        booster = lgb.Booster(model_file=str(bpath))
        names = list(json.loads(gpath.read_text(encoding="utf-8"))["gain"])
        base = [n for n in names if not n.startswith("auc_")]
        preds = io.read(settings, "preds", start=f.test_start, end=f.test_end, sub=trial_id).collect()
        feats = load_features(settings, "research", f.test_start, f.test_end, columns=base).collect()
        df = preds.select("date", "sid").join(feats, on=["date", "sid"], how="inner")
        if use_auction:
            from stocksense.features.auction_matrix import auction_columns

            df = df.join(auction_columns(settings, "research")[0].collect(), on=["date", "sid"], how="left")
        df = df.join(lab.select("date", "sid", target), on=["date", "sid"], how="inner")
        X = df.select([pl.col(n).cast(pl.Float32) for n in names]).to_numpy().astype(np.float32)
        df = df.select("date", target).with_columns(pl.Series("full", booster.predict(X)))
        for grp in groups:
            cols = [i for i, n in enumerate(names) if group_of.get(n) == grp]
            if not cols:
                continue
            starts = [first_valid[names[i]] for i in cols if first_valid.get(names[i])]
            since = date.fromisoformat(max(starts)) if starts else date.min
            Xm = X.copy()
            Xm[:, cols] = np.nan
            part = df.with_columns(pl.Series("masked", booster.predict(Xm))).filter(pl.col("date") >= since)
            rows[grp].append(part)
    for grp, parts in rows.items():
        if not parts:
            continue
        df = pl.concat(parts)
        a, b = _per_date_ic(df, "full", target), _per_date_ic(df, "masked", target)
        j = a.join(b, on="date", suffix="_m")
        d = (j["ic"] - j["ic_m"]).to_numpy()
        sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        t = float(d.mean() / sd * np.sqrt(len(d))) if sd > 0 else float("nan")
        out["ablation"][grp] = {
            "n_dates": int(j.height),
            "ic_with": float(j["ic"].mean()),
            "ic_without": float(j["ic_m"].mean()),
            "diff_tstat": t,
        }
    (wf / "diagnostics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
