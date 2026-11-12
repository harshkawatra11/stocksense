from __future__ import annotations

import polars as pl
import pytest

from stocksense.core.paths import Paths
from stocksense.core.state import State
from stocksense.data import io
from stocksense.features.auction import OUT_COLUMNS
from stocksense.features.auction_matrix import AUCTION_PREFIX, auction_columns
from stocksense.models.base import Model, ModelSpec, build_model
from stocksense.validation.walkforward import run_walk_forward
from tests.fixtures.wf_world import WFWorld, build_world

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> WFWorld:
    w = build_world(tmp_path_factory.mktemp("wf_auc"), planted_ic=0.06, n=80)
    # a hand-made auction store: next-session pairs of the u500 rows, constant numeric columns
    lab = io.read(w.settings, "panel_universe", columns=["date", "sid"]).collect().sort("date", "sid")
    sess = sorted(lab["date"].unique().to_list())
    nxt = pl.DataFrame({"decision_date": sess[:-1], "trade_date": sess[1:]})
    df = lab.rename({"date": "decision_date"}).join(nxt, on="decision_date")
    cols = [c for c in OUT_COLUMNS if c not in ("trade_date", "sid", "decision_date", "gap_source")]
    df = df.with_columns(*[pl.lit(0.01).alias(c) for c in cols], pl.lit("official_open").alias("gap_source"))
    d = Paths(w.settings).features / "auction"
    d.mkdir(parents=True, exist_ok=True)
    df.select(OUT_COLUMNS).write_parquet(d / "part-2019.parquet")
    return w


def test_auction_columns_keyed_by_decision_date(world):
    lf, names, kinds = auction_columns(world.settings, "research")
    df = lf.collect()
    assert df.columns[:2] == ["date", "sid"]
    assert (
        all(n.startswith(AUCTION_PREFIX) for n in names) and len(names) == len(kinds) == len(df.columns) - 2
    )
    assert kinds[names.index(AUCTION_PREFIX + "gap")] == "xs"
    assert kinds[names.index(AUCTION_PREFIX + "mkt_gap")] == "mkt"
    assert kinds[names.index(AUCTION_PREFIX + "spx_overnight")] == "mkt"


def test_runner_appends_auction_columns_only_when_asked(world):
    st = State(world.settings)
    st.migrate()
    seen: dict[str, list[str]] = {}

    def spy(spec: ModelSpec) -> Model:
        m = build_model(spec)
        orig = m.configure

        def configure(names, kinds):
            seen[spec.params.get("tag", "")] = list(names)
            orig(names, kinds)

        m.configure = configure  # type: ignore[method-assign]
        return m

    for tag, extra in (("plain", {}), ("auc", {"auction": True})):
        spec = ModelSpec("s2-ridge", "y_oc", features="all", params={"tag": tag, **extra}, entry="open")
        run_walk_forward(
            spec, world.settings, st, folds=world.folds[:1], calendar=world.calendar, seeds=(1,),
            inner_months=2, model_factory=spy,
        )  # fmt: skip
    assert not any(n.startswith(AUCTION_PREFIX) for n in seen["plain"])
    assert [n for n in seen["auc"] if n.startswith(AUCTION_PREFIX)]


def test_selection_can_include_auction_columns(world, monkeypatch):
    from stocksense.models import gbm as gbm_mod

    seen: list[int] = []
    orig = gbm_mod.GbmModel.fit

    def spy(self, X, y, w, dates, X_val=None, y_val=None, dates_val=None, seed=0):
        seen.append(X.shape[1])
        return orig(self, X, y, w, dates, X_val, y_val, dates_val, seed)

    monkeypatch.setattr(gbm_mod.GbmModel, "fit", spy)
    st = State(world.settings)
    st.migrate()
    anchors = [world.folds[3]]
    for auc in (False, True):
        gbm_mod.select_hyperparams(
            "y_oc", "regression", world.settings, st, anchors=anchors, date_frac=0.5, folds=anchors,
            calendar=world.calendar, inner_months=2, grid=gbm_mod.GRID[:1],
            base_params={"n_estimators": 20, "num_leaves": 7, "min_child_samples": 20}, entry="open",
            auction=auc,
        )  # fmt: skip
    assert seen[1] > seen[0]
