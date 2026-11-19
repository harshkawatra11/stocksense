"""Nightly learning (PLAN T11.1): only the production fit exists so far.

`fit_production(spec, settings, state, as_of)` is a walk-forward fold whose test period starts at `as_of`:
the training rows are every row whose label ended strictly before the session five sessions before `as_of`
(and dated on or after dev_start), the last 12 months of those rows pick the model's complexity
(`Model.fit` on inner-train with early stopping on inner-validation), the model is refit on all training rows
(`refit_full`), once per seed, and the seeds are saved as one model through the model store.

Production data may include dates after the research end, but only on request: `as_of` beyond the research
end needs `allow_beyond_research=True` (the Gate G1 run passes it only after the lockbox has passed).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import Clock, SystemClock
from stocksense.core.config import Settings
from stocksense.core.errors import LockboxError
from stocksense.core.ids import model_id as make_model_id
from stocksense.core.log import get_logger
from stocksense.core.state import State
from stocksense.data import io
from stocksense.features.build import current_version, load_features
from stocksense.labels.forward import load_labels
from stocksense.models import store
from stocksense.models.base import FitInfo, Model, ModelSpec, build_model
from stocksense.validation.lockbox import research_end
from stocksense.validation.splits import (
    Fold,
    _first_session_on_or_after,
    inner_validation_split,
    sample_weights,
)
from stocksense.validation.walkforward import END_COLUMN, _manifest

log = get_logger(__name__)

DEFAULT_SEEDS = (1, 2, 3)
EMBARGO_SESSIONS = 5
INNER_MONTHS = 12
BAG_FILE = "bag.json"


class ModelBag(Model):
    """The seeds of one production fit saved as one model.

    The first seed's files sit at the top of the model directory, so `models.store.load_model` returns a
    working single-seed model; `load_bag` restores every seed (files under `seeds/<i>/`). `predict` averages
    the members' per-row ranks in [-0.5, 0.5], so `X` must hold one cross-section (one date) per call."""

    def __init__(
        self, members: Sequence[Model] | None = None, factory: Callable[[], Model] | None = None
    ) -> None:
        self.members: list[Model] = list(members or [])
        self._factory = factory
        if self.members:
            self.target_prefix = self.members[0].target_prefix

    def fit(self, X, y, w, dates, X_val=None, y_val=None, dates_val=None, seed=0) -> FitInfo:  # type: ignore[no-untyped-def]
        raise NotImplementedError("a ModelBag is assembled from fitted members by fit_production")

    def refit_full(self, X, y, w, dates, fit_info, seed) -> None:  # type: ignore[no-untyped-def]
        raise NotImplementedError("a ModelBag is assembled from fitted members by fit_production")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.members:
            raise RuntimeError("empty ModelBag")
        ranks = []
        for m in self.members:
            p = np.asarray(m.predict(X), dtype=float)
            p = np.where(np.isfinite(p), p, np.nanmedian(p) if np.isfinite(p).any() else 0.0)
            r = pl.Series(p).rank("average").to_numpy()
            ranks.append((r - 1.0) / max(len(p) - 1, 1) - 0.5)
        return np.mean(ranks, axis=0)

    def save(self, dir: Path) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        self.members[0].save(dir)
        for i, m in enumerate(self.members):
            m.save(dir / "seeds" / str(i))
        (dir / BAG_FILE).write_text(json.dumps({"n": len(self.members)}), encoding="utf-8")

    def load(self, dir: Path) -> None:
        if self._factory is None:
            raise RuntimeError("ModelBag.load needs a member factory (use load_bag)")
        n = int(json.loads((dir / BAG_FILE).read_text(encoding="utf-8"))["n"])
        self.members = []
        for i in range(n):
            m = self._factory()
            m.load(dir / "seeds" / str(i))
            self.members.append(m)
        if self.members:
            self.target_prefix = self.members[0].target_prefix

    def explain(self, X: np.ndarray, feature_names: Sequence[str]) -> np.ndarray | None:
        outs = [m.explain(X, feature_names) for m in self.members]
        return None if any(o is None for o in outs) else np.mean(outs, axis=0)


def load_bag(settings: Settings, model_id: str) -> ModelBag:
    """Every seed of a production model saved by `fit_production`."""
    d = store.model_dir(settings, model_id)
    spec = ModelSpec.from_dict(json.loads((d / "spec.json").read_text(encoding="utf-8")))
    bag = ModelBag(factory=lambda: build_model(spec))
    bag.load(d)
    return bag


# ---- the training rows ------------------------------------------------------------------------------


def production_cutoff(
    calendar: TradingCalendar, as_of: date, embargo_sessions: int = EMBARGO_SESSIONS
) -> date:
    """The session `embargo_sessions` before `as_of`: a label ending on or after it is not trained on."""
    return calendar.shift(_first_session_on_or_after(calendar, as_of), -embargo_sessions)


def production_training_rows(
    spec: ModelSpec,
    settings: Settings,
    as_of: date,
    calendar: TradingCalendar,
    *,
    model_factory: Callable[[ModelSpec], Model] | None = None,
    embargo_sessions: int = EMBARGO_SESSIONS,
    purpose: str = "production",
) -> tuple[pl.DataFrame, list[str], list[str], date]:
    """(rows, feature names, feature kinds, cutoff): every row of the spec's universe with a label whose end
    date is strictly before `cutoff` = as_of - 5 sessions, dated on or after dev_start, with no
    `entry_blocked_<entry>` flag, sorted by date and sid. Columns: date, sid, the end column, the label
    (`<prefix><target>`) and the features."""
    factory = model_factory if model_factory is not None else build_model
    label_col = f"{factory(spec).target_prefix}{spec.target}"
    end_col = END_COLUMN[spec.target]
    blocked = f"entry_blocked_{spec.entry}"
    cutoff = production_cutoff(calendar, as_of, embargo_sessions)
    lab = (
        load_labels(settings, purpose, end=as_of, columns=[label_col, end_col, blocked])
        .collect()
        .filter(pl.col(label_col).is_not_null() & ~pl.col(blocked).fill_null(False))
        .filter((pl.col(end_col) < cutoff) & (pl.col("date") >= settings.research.dev_start))
    )
    if spec.universe != "u500":
        uni = io.read(settings, "panel_universe", end=as_of, columns=["date", "sid", spec.universe]).collect()
        lab = lab.join(uni.filter(pl.col(spec.universe)).select("date", "sid"), on=["date", "sid"])
    man = _manifest(settings)
    kind_of = {f["name"]: f["kind"] for f in man["features"]}
    names = [f["name"] for f in man["features"]] if spec.features == "all" else list(spec.features)
    kinds = [kind_of[n] for n in names]
    feats = load_features(settings, purpose, end=as_of, columns=names).collect()
    rows = lab.join(feats, on=["date", "sid"], how="inner").sort("date", "sid")
    return rows, names, kinds, cutoff


# ---- fit_production ---------------------------------------------------------------------------------


def fit_production(
    spec: ModelSpec,
    settings: Settings,
    state: State,
    as_of: date,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    calendar: TradingCalendar | None = None,
    clock: Clock | None = None,
    inner_months: int = INNER_MONTHS,
    embargo_sessions: int = EMBARGO_SESSIONS,
    model_factory: Callable[[ModelSpec], Model] | None = None,
    allow_beyond_research: bool = False,
    note: str | None = None,
) -> str:
    """Fit `spec` for decisions from `as_of` on and save it; returns the model id (the existing one when this
    (spec, as_of) was already fitted: the job is idempotent for the same day).

    The saved `train_end` is `as_of`; the newest label used ended before as_of - 5 sessions."""
    clk = clock if clock is not None else SystemClock()
    if as_of > research_end(settings) and not allow_beyond_research:
        raise LockboxError(
            f"as_of {as_of} is after the research end {research_end(settings)}: a production fit would read "
            "lockbox-period data (pass allow_beyond_research=True only once the lockbox condition holds)"
        )
    mid = make_model_id(spec.family, spec.target, spec.to_dict(), as_of)
    con = state.connect()
    try:
        if con.execute("SELECT 1 FROM models WHERE model_id=?", (mid,)).fetchone() is not None:
            return mid
    finally:
        con.close()
    cal = calendar if calendar is not None else TradingCalendar.from_store(settings, clk)
    factory = model_factory if model_factory is not None else build_model
    rows, names, kinds, cutoff = production_training_rows(
        spec, settings, as_of, cal, model_factory=factory, embargo_sessions=embargo_sessions
    )
    if rows.is_empty():
        raise ValueError(f"no training rows for {spec.family}/{spec.target} before {cutoff}")
    label_col = f"{factory(spec).target_prefix}{spec.target}"
    end_col = END_COLUMN[spec.target]
    dates, end = rows["date"], rows[end_col]
    assert end.max() < cutoff  # type: ignore[operator]
    y = rows[label_col].to_numpy().astype(np.float64)
    X = rows.select([pl.col(n).cast(pl.Float32) for n in names]).to_numpy().astype(np.float32)
    dn = dates.to_numpy()
    if factory(spec).needs_keys:  # sequence / cube models look rows up by (sid, day)
        from stocksense.models.seqnet import add_keys

        X = add_keys(X, rows["sid"].to_numpy(), dn)
    fold = Fold("production", cal.prev_session(as_of), as_of, as_of)
    w = sample_weights(dates, fold, cal)
    inner_tr, inner_va = inner_validation_split(dates, end, inner_months, embargo_sessions, calendar=cal)
    a, b = inner_tr.to_numpy(), inner_va.to_numpy()
    members: list[Model] = []
    info = FitInfo()
    for s in seeds:
        m = factory(spec)
        m.configure(names, kinds)
        info = m.fit(X[a], y[a], w[a], dn[a], X[b], y[b], dn[b], seed=int(s))
        m.refit_full(X, y, w, dn, info, seed=int(s))
        members.append(m)
    bag = ModelBag(members)
    first = dates.min()
    meta: dict[str, Any] = {
        "feature_names": names, "feature_kinds": kinds, "feature_version": current_version(settings),
        "fit_info": {**info.to_dict(), "seeds": list(seeds), "n_train": int(rows.height),
                     "cutoff": cutoff.isoformat(), "inner_months": inner_months},
        "train_start": first.isoformat() if isinstance(first, date) else None,
        "note": note or f"production fit as_of {as_of}",
    }  # fmt: skip
    saved = store.save_model(settings, state, bag, spec, as_of, meta)
    log.info("fit_production_done", model=saved, n_train=rows.height, seeds=len(members))
    return saved


def last_research_session(settings: Settings, calendar: TradingCalendar) -> date:
    """The last session on or before the research end."""
    end = research_end(settings)
    return calendar.sessions(end - timedelta(days=14), end)[-1]
