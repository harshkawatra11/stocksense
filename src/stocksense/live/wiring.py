"""Production wiring of the daemon (Program G2, phase G): live scores, the S1 plan and entries, learning.

Everything here is paper only: orders are rows in the paper ledger, never broker calls. The hooks built by
`production_hooks` are what `live.jobs` calls through `ctx.hook(...)`; tests inject their own.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date

import numpy as np
import polars as pl

from stocksense.core.config import Settings
from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.core.state import State
from stocksense.data import io
from stocksense.features.build import load_features
from stocksense.labels.forward import load_labels
from stocksense.learn.nightly import load_bag
from stocksense.models import store
from stocksense.models.base import ModelSpec, build_model
from stocksense.models.ensemble import expected_return_map, expected_return_map_decile, rank_scores
from stocksense.portfolio.construct import (
    DayScores,
)
from stocksense.validation.walkforward import END_COLUMN

log = get_logger(__name__)

S1_TARGET = "y_c2o_10"
DEFAULT_LIVE = {
    "live_model:s1": "s1-gbm-y_c2o_10-120055de-20250930",
    "live_model:s2": "s2-gbm-y_oc-4fd8ff9b-20250930",
}
ER_WINDOW_DATES = 520  # walk-forward dates kept for the expected-return map (its window is 504)


# ---- which model is live ------------------------------------------------------------------------------


def live_ids(state: State) -> dict[str, str | None]:
    """The kv ids of the live models and the walk-forward trial that calibrates S1's expected-return map.

    Missing keys are initialised with the production defaults (the S1 trial is looked up among the finished
    `s1-gbm` walk-forward trials of `y_c2o_10`; None when there is none)."""
    out: dict[str, str | None] = {}
    for key, default in DEFAULT_LIVE.items():
        cur = state.kv_get(key)
        if cur is None:
            state.kv_set(key, default)
            cur = default
        out[key] = cur
    wf = state.kv_get("live_wf_trial:s1")
    if wf is None:
        from stocksense.validation.g1 import surviving_models

        cands = surviving_models(state, "s1", S1_TARGET)
        gbm = [c for c in cands if c.family == "s1-gbm"] or cands
        if gbm:
            wf = gbm[0].trial_id
            state.kv_set("live_wf_trial:s1", wf)
    out["live_wf_trial:s1"] = wf
    wf2 = state.kv_get("live_wf_trial:s2")
    if wf2 is None:
        from stocksense.validation.g1 import surviving_models

        c2 = surviving_models(state, "s2", "y_oc")
        g2 = [c for c in c2 if c.family == "s2-gbm"] or c2
        if g2:
            wf2 = g2[0].trial_id
            state.kv_set("live_wf_trial:s2", wf2)
    out["live_wf_trial:s2"] = wf2
    return out


def live_policy(state: State) -> dict:
    """The live S1 policy (kv JSON `live_policy:s1`): k_max, overlay, S1Config overrides under `policy`,
    optional `er_map` ('linear' | 'decile'). Default: the g1-s1-y_c2o_10-cce12fb757 spec (k_max 10, no
    overlay) with every other option at its default."""
    raw = state.kv_get("live_policy:s1")
    base = {"k_max": 10, "overlay": False, "policy": {}, "er_map": "linear"}
    if raw:
        base.update(json.loads(raw))
    return base


# ---- the predict stage --------------------------------------------------------------------------------


def predict_frame(settings: Settings, model_id: str, d: date) -> pl.DataFrame:
    """date, sid, score of the production model `model_id` for the u500 members of d (one cross-section)."""
    mdir = store.model_dir(settings, model_id)
    names = list(json.loads((mdir / "feature_manifest.json").read_text(encoding="utf-8"))["features"])
    spec = ModelSpec.from_dict(json.loads((mdir / "spec.json").read_text(encoding="utf-8")))
    uni = io.read(settings, "panel_universe", start=d, end=d, columns=["date", "sid", "u500"]).collect()
    uni = uni.filter(pl.col("u500").fill_null(False))
    if uni.is_empty():
        raise DataNotAvailable(f"no u500 members for {d}")
    feats = load_features(settings, "production", start=d, end=d, columns=names).collect()
    if feats.is_empty():
        raise DataNotAvailable(f"no features for {d}")
    f = uni.select("date", "sid").join(feats, on=["date", "sid"], how="inner").sort("sid")
    if f.is_empty():
        raise DataNotAvailable(f"no feature rows for the u500 members of {d}")
    X = f.select([pl.col(n).cast(pl.Float32) for n in names]).to_numpy().astype(np.float32)
    if build_model(spec).needs_keys:  # sequence / cube models look rows up by (sid, day)
        from stocksense.models.seqnet import add_keys

        X = add_keys(X, f["sid"].to_numpy(), f["date"].to_numpy())
    score = load_bag(settings, model_id).predict(X)
    return pl.DataFrame({"date": f["date"], "sid": f["sid"], "score": score})


def stage_predict(ctx, d: date) -> str:  # type: ignore[no-untyped-def]
    ids = live_ids(ctx.state)
    mid = str(ids["live_model:s1"])
    frame = predict_frame(ctx.settings, mid, d)
    store.append_preds(ctx.settings, mid, frame, "live", clock=ctx.clock)
    return f"s1:{frame.height} names"


# ---- the live score source ----------------------------------------------------------------------------


class LiveScoreSource:
    """`ScoreSource` over the live predictions of `live_model:<strategy>`.

    score: the day's live score ranked to [-0.5, 0.5]. er: `max(0, b) * score` with b the through-the-origin
    slope of the raw label on the score over the trailing 504 dates whose label ended before d, computed on
    the union of the calibration walk-forward trial's stored predictions (ranked per date) and the live
    predictions; with `er_map == 'decile'` the isotonic decile map is used instead. Cached per date."""

    def __init__(
        self, settings: Settings, state: State, strategy: str = "s1", target: str = S1_TARGET,
        er_map: str | None = None,
    ) -> None:  # fmt: skip
        self.settings, self.state, self.strategy, self.target = settings, state, strategy, target
        self.er_map = er_map
        self._cache: dict[date, DayScores | None] = {}

    def _frames(self, d: date) -> tuple[pl.DataFrame, pl.DataFrame]:
        ids = live_ids(self.state)
        mid = ids[f"live_model:{self.strategy}"]
        live = (
            io.read(self.settings, "preds", end=d, columns=["date", "sid", "score"], sub=mid)
            .collect()
            .sort("date", "sid")
        )
        wf_id = ids.get(f"live_wf_trial:{self.strategy}")
        if wf_id:
            wf = io.read(self.settings, "preds", columns=["date", "sid", "score"], sub=wf_id).collect()
            keep = wf["date"].unique().sort().tail(ER_WINDOW_DATES)
            wf = wf.filter(pl.col("date").is_in(keep.implode()) & (pl.col("date") <= d))
            wf = wf.filter(~pl.col("date").is_in(live["date"].unique().implode()))
        else:
            wf = live.clear()
        return live, wf

    def scores(self, d: date) -> DayScores | None:
        if d in self._cache:
            return self._cache[d]
        try:
            live, wf = self._frames(d)
        except (DataNotAvailable, FileNotFoundError, pl.exceptions.PolarsError):
            self._cache[d] = None
            return None
        day = live.filter(pl.col("date") == d)
        if day.is_empty():
            self._cache[d] = None
            return None
        blend = rank_scores(
            pl.concat([wf.select("date", "sid", "score"), live.select("date", "sid", "score")])
        )
        lab = load_labels(
            self.settings,
            "production",
            start=blend["date"].min(),
            columns=[self.target, END_COLUMN[self.target]],
        ).collect()
        er_map = self.er_map or live_policy(self.state).get("er_map", "linear")
        if er_map == "decile":
            er = expected_return_map_decile(blend, lab, self.target)
        else:
            er = expected_return_map(blend, lab, self.target)
        er = er.filter(pl.col("date") == d)
        sids = er["sid"].to_list()
        out = DayScores(
            dict(zip(sids, er["score"].to_list(), strict=True)),
            dict(
                zip(sids, [max(0.0, float(x)) for x in er["exp_ret"].fill_null(0.0).to_list()], strict=True)
            ),
        )
        self._cache[d] = out
        return out


def production_hooks(settings: Settings) -> Mapping[str, object]:
    """The hooks `build_context` installs when none are injected."""
    from stocksense.live.jobs import u500_symbols

    _ = u500_symbols
    return {
        "predict": stage_predict,
    }
