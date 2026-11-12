"""A synthetic market for the Gate G1 pipeline: the walk-forward world of wf_world plus planted features.

`planted_c2o_<h>` is the realised h-session close-to-open label rank plus noise (the synthetic market's own
alpha only drives the open-to-close label, so the delivery targets would otherwise have no signal at all);
it exists only here, to give the S1 models a measurable out-of-fold IC. The lockbox window is the last
weeks of the synthetic period: research ends 2020-03-01, the lockbox runs 2020-03-02 onwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from stocksense import panel as panel_pkg
from stocksense.core.calendar import TradingCalendar
from stocksense.core.config import Settings
from stocksense.core.state import State
from stocksense.features.base import FeatureContext, FeatureSpec, feature
from stocksense.features.build import build_features
from stocksense.labels.forward import build_labels, load_labels
from stocksense.models.base import ModelSpec
from stocksense.validation.splits import Fold
from stocksense.validation.walkforward import backtest_walk_forward, run_walk_forward
from tests.fixtures.synth import SynthMarket, _settings_for, make_market, write_market
from tests.fixtures.wf_world import WINDOWS, make_registry

LOCKBOX_START = date(2020, 3, 2)
RESEARCH_WINDOWS = tuple(w for w in WINDOWS if w[0] != "f6")


@dataclass(frozen=True)
class G1World:
    settings: Settings  # lockbox_start moved to LOCKBOX_START
    market: SynthMarket
    calendar: TradingCalendar
    registry: dict[str, FeatureSpec]
    folds: list[Fold]  # five research folds, none inside the lockbox window


def _planted_registry(settings: Settings, market: SynthMarket, noise: float) -> dict[str, FeatureSpec]:
    reg = make_registry(settings, market)
    lab = load_labels(settings, "production", columns=[f"r_y_c2o_{h}" for h in (5, 10, 20)]).collect()
    rng = np.random.default_rng(11)
    for h in (5, 10, 20):
        col = f"planted_c2o_{h}"
        noisy = lab.select(
            "date", "sid", (pl.col(f"r_y_c2o_{h}") + pl.Series(rng.normal(0.0, noise, lab.height))).alias(col)
        ).drop_nulls(col)

        def make(col: str = col, noisy: pl.DataFrame = noisy) -> None:
            @feature(col, group="test", kind="xs", registry=reg, label=f"planted {col}")
            def _f(ctx: FeatureContext) -> pl.LazyFrame:
                return (
                    ctx.panel().select("date", "sid").join(noisy.lazy(), on=["date", "sid"], how="left")
                    .select("date", "sid", col)
                )  # fmt: skip

        make()
    return reg


def build_g1_world(root: Path, planted_ic: float = 0.10, n: int = 150, noise: float = 1.0) -> G1World:
    m = make_market(n=n, planted_ic=planted_ic)
    write_market(m, root)
    s = _settings_for(root)
    s = s.model_copy(
        update={"research": s.research.model_copy(update={"lockbox_start": LOCKBOX_START})}
    )
    panel_pkg.build_all(s)
    build_labels(s)
    reg = _planted_registry(s, m, noise)
    build_features(s, min_valid=10, registry=reg)
    cal = TradingCalendar.from_dates(m.sessions)
    folds = [Fold(name, cal.prev_session(a), a, b) for name, a, b in RESEARCH_WINDOWS]
    return G1World(s, m, cal, reg, folds)


def run_model_trials(w: G1World, state: State, *, families: dict[str, str] | None = None) -> dict[str, str]:
    """One finished ridge walk-forward trial per target (s1-ridge for the three delivery targets, s2-ridge for
    y_oc), each with its top-k backtest recorded (so its metrics carry the out-of-fold IC)."""
    out: dict[str, str] = {}
    for target in ("y_c2o_5", "y_c2o_10", "y_c2o_20", "y_oc"):
        fam = (families or {}).get(target) or ("s2-ridge" if target == "y_oc" else "s1-ridge")
        spec = ModelSpec(
            family=fam, target=target, universe="u200" if target == "y_oc" else "u500",
            entry="open" if target == "y_oc" else "close",
        )  # fmt: skip
        res = run_walk_forward(
            spec, w.settings, state, folds=w.folds, calendar=w.calendar, registry=w.registry,
            seeds=(1, 2), inner_months=2,
        )  # fmt: skip
        backtest_walk_forward(res, spec, w.settings, state, calendar=w.calendar, k=10)
        out[target] = res.trial_id
    return out
