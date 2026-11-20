"""Hyperparameter selection for a G2 model family (EP of PLAN-G2 Part 4.2, card D0.7).

At most 6 configurations are fitted on the inner-train rows of one fold (default `2016`: training data up
to 2015-12-31 minus embargo) and scored by the mean daily rank IC on that fold's inner-validation rows. No
test fold is touched and no trial is registered. The table and the best configuration go to
`models/_selection/g2-<family>.json`.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import Clock, SystemClock
from stocksense.core.config import Settings
from stocksense.core.paths import Paths
from stocksense.data import io
from stocksense.features.build import load_features
from stocksense.labels.forward import load_labels
from stocksense.models.base import Model, ModelSpec, build_model, mean_daily_rank_ic
from stocksense.validation.splits import (
    Fold,
    inner_validation_split,
    sample_weights,
    train_mask,
    walk_forward_folds,
)
from stocksense.validation.walkforward import END_COLUMN, _manifest, _release_memory

MAX_CONFIGS = 6
SELECT_FOLD = "2016"


def selection_path(settings: Settings, key: str) -> Path:
    return Paths(settings).models / "_selection" / f"g2-{key}.json"


def select_nn(
    key: str,
    target: str,
    configs: Sequence[dict[str, Any]],
    settings: Settings,
    *,
    fold_name: str = SELECT_FOLD,
    folds: Sequence[Fold] | None = None,
    calendar: TradingCalendar | None = None,
    clock: Clock | None = None,
    seed: int = 1,
    inner_months: int = 12,
    embargo_sessions: int = 5,
    universe: str = "u500",
    entry: str | None = None,
    purpose: str = "research",
    model_factory: Callable[[ModelSpec], Model] | None = None,
    min_pairs: int = 10,
    should_stop: Callable[[], bool] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Fit every config of `configs` (1..6 parameter dicts for `ModelSpec.params`) on the inner-train rows
    of `fold_name` and score the inner-validation mean IC. Returns {family, target, fold, table, best,
    best_val_ic}; a config not reached because `should_stop()` fired is listed with `skipped`."""
    if not 1 <= len(configs) <= MAX_CONFIGS:
        raise ValueError(f"1 to {MAX_CONFIGS} configurations are allowed, got {len(configs)}")
    clk = clock if clock is not None else SystemClock()
    cal = calendar if calendar is not None else TradingCalendar.from_store(settings, clk)
    fold_list = list(folds) if folds is not None else walk_forward_folds(settings, cal)
    fold = next((f for f in fold_list if f.name == fold_name), None)
    if fold is None:
        raise ValueError(f"no fold named {fold_name!r}")
    factory = model_factory if model_factory is not None else build_model
    spec0 = ModelSpec(
        family=f"s1-{key}", target=target, universe=universe,
        entry=entry or ("open" if target == "y_oc" else "close"),
    )  # fmt: skip
    label_col = f"{factory(spec0).target_prefix}{target}"
    end_col = END_COLUMN[target]
    blocked = f"entry_blocked_{spec0.entry}"
    lab = (
        load_labels(settings, purpose, columns=[label_col, end_col, blocked])
        .collect()
        .filter(pl.col(label_col).is_not_null() & ~pl.col(blocked).fill_null(False))
    )
    if universe != "u500":
        uni = io.read(settings, "panel_universe", columns=["date", "sid", universe]).collect()
        lab = lab.join(uni.filter(pl.col(universe)).select("date", "sid"), on=["date", "sid"])
    man = _manifest(settings)
    names = [f["name"] for f in man["features"]]
    kinds = [f["kind"] for f in man["features"]]
    light = bool(getattr(factory(spec0), "needs_cube", False))  # cube models: X holds 'mkt' columns only
    x_names = [n for n, k in zip(names, kinds, strict=True) if k == "mkt"] if light else names
    feats = load_features(settings, purpose, columns=x_names).collect()
    df = lab.join(feats, on=["date", "sid"], how="inner").sort("date", "sid")
    dates, end = df["date"], df[end_col]
    dn = dates.to_numpy()
    y = df[label_col].to_numpy().astype(np.float64)
    X = df.select([pl.col(n).cast(pl.Float32) for n in x_names]).to_numpy().astype(np.float32)
    del feats
    if factory(spec0).needs_keys:
        from stocksense.models.seqnet import add_keys

        X = add_keys(X, df["sid"].to_numpy(), dn)
    tr = train_mask(
        end, fold, embargo_sessions, calendar=cal, dates=dates, dev_start=settings.research.dev_start
    ).to_numpy()
    if not tr.any():
        raise ValueError(f"fold {fold_name!r} has no training rows")
    itr = np.flatnonzero(tr)
    d_tr, end_tr = dates.filter(pl.Series(tr)), end.filter(pl.Series(tr))
    inner_tr, inner_va = inner_validation_split(d_tr, end_tr, inner_months, embargo_sessions, calendar=cal)
    a, b = inner_tr.to_numpy(), inner_va.to_numpy()
    X_tr, y_tr, dn_tr = X[itr], y[itr], dn[itr]
    w_tr = sample_weights(d_tr, fold, cal)
    table: list[dict[str, Any]] = []
    for cfg in configs:
        if should_stop is not None and should_stop():
            table.append({"config": dict(cfg), "skipped": "wall cap"})
            continue
        spec = replace(spec0, params=dict(cfg))
        model = factory(spec)
        model.configure(names, kinds)
        t0 = time.perf_counter()
        info = model.fit(X_tr[a], y_tr[a], w_tr[a], dn_tr[a], X_tr[b], y_tr[b], dn_tr[b], seed=seed)
        ic = mean_daily_rank_ic(np.asarray(model.predict(X_tr[b])), y_tr[b], dn_tr[b], min_pairs=min_pairs)
        row: dict[str, Any] = {
            "config": dict(cfg), "val_ic": float(ic) if math.isfinite(ic) else None,
            "best_iteration": info.best_iteration, "seconds": time.perf_counter() - t0,
        }  # fmt: skip
        if info.params.get("stopped"):
            row["stopped"] = info.params["stopped"]
        table.append(row)
        del model
        _release_memory()
    scored = [r for r in table if r.get("val_ic") is not None]
    best = max(scored, key=lambda r: r["val_ic"]) if scored else None
    out: dict[str, Any] = {
        "family": f"g2-{key}", "target": target, "fold": fold.name, "seed": seed,
        "n_inner_train": int(a.sum()), "n_inner_val": int(b.sum()), "table": table,
        "best": None if best is None else best["config"],
        "best_val_ic": None if best is None else best["val_ic"],
    }  # fmt: skip
    if write:
        p = selection_path(settings, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
        os.replace(tmp, p)
    return out
