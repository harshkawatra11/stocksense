from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from stocksense.portfolio.meta import (
    META_FEATURES,
    MetaModel,
    build_meta_samples,
    oof_meta_predictions,
    size_multiplier,
    veto,
)
from stocksense.validation.splits import Fold

D0 = date(2020, 1, 6)


def samples(n: int = 1500, seed: int = 0, signal: float = 2.5, hold: int = 6) -> pl.DataFrame:
    """Entries on consecutive days (about 7 a day); label depends on `score` through a logistic link."""
    rng = np.random.default_rng(seed)
    score = rng.uniform(-0.5, 0.5, n)
    p = 1.0 / (1.0 + np.exp(-signal * score * 2))
    y = (rng.uniform(size=n) < p).astype(int)
    entry = [D0 + timedelta(days=int(i // 7)) for i in range(n)]
    return pl.DataFrame(
        {
            "sid": np.arange(n) % 40,
            "entry_date": entry,
            "exit_date": [e + timedelta(days=hold) for e in entry],
            "ret": np.where(y == 1, 0.02, -0.02),
            "label": y,
            "score": score,
            "score_rank": rng.integers(1, 20, n).astype(float),
            "disagreement": rng.uniform(0, 0.3, n),
            "vol_21": rng.uniform(0.01, 0.03, n),
            "turnover_rank": rng.integers(1, 500, n).astype(float),
            "mkt_ret_21": rng.normal(0, 0.03, n),
            "breadth_200": rng.uniform(0.2, 0.8, n),
            "days_since_result": rng.integers(0, 90, n).astype(float),
            "gross_target": rng.uniform(0.5, 1.0, n),
        }
    )


def model() -> MetaModel:
    return MetaModel(min_child_samples=20)  # the card's 200 needs far more rows than a unit test has


# ---- multiplier and veto ---------------------------------------------------------------------------------


@given(st.floats(0.0, 1.0))
def test_multiplier_is_clipped_to_unit_interval(p):
    m = size_multiplier(p)
    assert 0.0 <= m <= 1.0
    assert m == pytest.approx(min(max(2 * p - 1, 0.0), 1.0))


def test_multiplier_and_veto_values():
    assert size_multiplier(0.5) == 0.0 and size_multiplier(0.75) == pytest.approx(0.5)
    assert size_multiplier(1.0) == 1.0 and size_multiplier(0.0) == 0.0
    assert size_multiplier(1.2) == 1.0 and size_multiplier(-0.3) == 0.0  # outside [0, 1] is clipped
    assert veto(0.4999) and not veto(0.5) and not veto(0.9)


def test_unavailable_probability_passes_through():
    assert size_multiplier(float("nan")) == 1.0 and veto(float("nan")) is False


# ---- training only on closed trades ----------------------------------------------------------------------


def test_training_rows_are_trades_closed_strictly_before_the_test_start():
    s = pl.DataFrame(
        {
            "exit_date": [date(2020, 3, 1), date(2020, 3, 31), date(2020, 4, 1), date(2020, 4, 2)],
            "entry_date": [date(2020, 2, 1), date(2020, 3, 20), date(2020, 3, 25), date(2020, 4, 1)],
        }
    )
    tr = MetaModel.training_rows(s, date(2020, 4, 1))
    # a trade exiting on the test start is not closed yet
    assert tr["exit_date"].to_list() == [date(2020, 3, 1), date(2020, 3, 31)]


def test_fit_never_sees_a_trade_closing_on_or_after_the_test_start():
    s = samples()
    start = D0 + timedelta(days=100)
    m = model()
    info = m.fit(s, start, seed=1)
    assert info.n_train > 0 and info.max_exit_date < start
    assert info.n_train == MetaModel.training_rows(s, start).height
    assert info.fitted


def test_fit_with_too_few_or_one_class_samples_stays_unfitted():
    s = samples(60)
    m = MetaModel()  # the card's min_child_samples = 200
    info = m.fit(s, D0 + timedelta(days=100))
    assert not info.fitted and not m.fitted
    assert np.isnan(m.predict(s)).all()
    one = samples(1500).with_columns(pl.lit(1).alias("label"))
    m2 = model()
    assert not m2.fit(one, D0 + timedelta(days=300)).fitted


def test_the_card_parameters_are_the_defaults():
    m = MetaModel()
    assert (m.num_leaves, m.min_child_samples, m.learning_rate) == (7, 200, 0.05)


# ---- learning ---------------------------------------------------------------------------------------------


def test_probability_rises_with_the_score_when_the_label_depends_on_it():
    s = samples(3000, seed=1)
    m = model()
    assert m.fit(s, D0 + timedelta(days=380)).fitted
    base = {k: 0.0 for k in META_FEATURES}
    lo = float(m.predict_rows([{**base, "score": -0.45}])[0])
    hi = float(m.predict_rows([{**base, "score": 0.45}])[0])
    assert 0.0 < lo < hi < 1.0 and hi - lo > 0.2
    assert hi > 0.5 > lo


def test_predict_rows_accepts_dict_rows_with_missing_values():
    s = samples(3000, seed=2)
    m = model()
    m.fit(s, D0 + timedelta(days=380))
    p = m.predict_rows([{"score": 0.3, "disagreement": float("nan")}])
    assert p.shape == (1,) and 0.0 <= p[0] <= 1.0


# ---- walk-forward -----------------------------------------------------------------------------------------


class Spy(MetaModel):
    seen: list[tuple[date, date, int]] = []

    def fit(self, samples, test_start, seed=0):  # type: ignore[no-untyped-def]
        tr = MetaModel.training_rows(samples, test_start)
        Spy.seen.append((test_start, tr["exit_date"].max(), tr.height))
        return super().fit(samples, test_start, seed)


def test_walk_forward_fits_each_fold_on_closed_trades_and_scores_only_that_folds_entries():
    s = samples(3000, seed=3)
    f1 = Fold("f1", D0 + timedelta(days=199), D0 + timedelta(days=200), D0 + timedelta(days=259))
    f2 = Fold("f2", D0 + timedelta(days=259), D0 + timedelta(days=260), D0 + timedelta(days=319))
    Spy.seen = []
    out = oof_meta_predictions(s, [f1, f2], factory=lambda: Spy(min_child_samples=20))
    for start, last_exit, n in Spy.seen:
        assert last_exit < start and n > 0
    assert [x[0] for x in Spy.seen] == [f1.test_start, f2.test_start]
    inside = s.filter(
        ((pl.col("entry_date") >= f1.test_start) & (pl.col("entry_date") <= f1.test_end))
        | ((pl.col("entry_date") >= f2.test_start) & (pl.col("entry_date") <= f2.test_end))
    )
    assert out.height == inside.height
    assert out["p"].is_between(0.0, 1.0).all()
    assert set(out.columns) >= {"entry_date", "sid", "p", "fold"}


def test_walk_forward_scores_are_nan_before_there_is_enough_closed_history():
    s = samples(3000, seed=4)
    f = Fold("early", D0 + timedelta(days=3), D0 + timedelta(days=4), D0 + timedelta(days=20))
    out = oof_meta_predictions(s, [f], factory=model)
    assert out.height > 0 and out["p"].is_nan().all()  # no trade has closed yet: meta stays neutral


# ---- samples ----------------------------------------------------------------------------------------------


def test_build_meta_samples_joins_entries_to_closed_trades_and_labels_by_net_return():
    log = pl.DataFrame(
        {
            "sid": [1, 2, 3],
            "date": [date(2020, 3, 3)] * 3,
            **{k: [0.1, 0.2, 0.3] for k in META_FEATURES},
        }
    )
    trades = pl.DataFrame(
        {
            "sid": [1, 2, 2, 9],
            "entry_date": [date(2020, 3, 3)] * 4,
            "exit_date": [date(2020, 3, 9), date(2020, 3, 9), date(2020, 3, 10), date(2020, 3, 9)],
            "qty": [10, 10, 10, 5],
            "entry_price": [100.0] * 4,
            "net_pnl_inr": [50.0, -30.0, -10.0, 1.0],
        }
    )
    out = build_meta_samples(log, trades).sort("sid")
    assert out["sid"].to_list() == [1, 2]  # sid 3 never closed; sid 9 was never an entry
    assert out["label"].to_list() == [1, 0]
    assert out["ret"].to_list() == pytest.approx([0.05, -0.02])  # partial exits are pooled
    assert out["exit_date"].to_list() == [date(2020, 3, 9), date(2020, 3, 10)]
    assert set(META_FEATURES) <= set(out.columns)
    assert math.isclose(out["score"][0], 0.1)


# ---- use inside the S1 planner ----------------------------------------------------------------------------


class StubMeta:
    """p = 0.2 for sid 0 (vetoed), 0.75 for sid 1 (half size), 1.0 otherwise."""

    def predict_rows(self, rows):
        return np.array([{0.4: 0.2, 0.35: 0.75}.get(round(r["score"], 2), 1.0) for r in rows])

    def size_multiplier(self, p):
        return size_multiplier(p)

    def veto(self, p):
        return veto(p)


def test_meta_veto_and_size_multiplier_inside_plan_s1():
    from stocksense.portfolio.construct import NameRow, PanelDay, S1Book, S1Config, plan_s1
    from tests.fixtures.engine_panel import settings, zero_slippage

    rows = {i: NameRow(100.0, 0.005, 10, 1e10, f"I{i}") for i in range(14)}
    sc = {i: 0.4 - 0.05 * i for i in range(14)}
    er = {i: 0.05 for i in range(14)}
    cfg = S1Config.from_settings(settings(), 5, zero_slippage(), k_max=10)
    day = PanelDay(date(2020, 3, 2), date(2020, 3, 3), rows)
    plain = plan_s1(date(2020, 3, 2), sc, er, S1Book({}, 1e9), 1e8, day, cfg)
    meta = plan_s1(date(2020, 3, 2), sc, er, S1Book({}, 1e9), 1e8, day, cfg, meta=StubMeta())
    assert 0 in plain.weights and 0 not in meta.weights  # vetoed: p < 0.5
    assert len(meta.weights) == 10 and 10 in meta.weights  # the freed slot goes to the next name
    # sid 1 (p = 0.75, multiplier 0.5) gets half the raw weight of an otherwise identical name
    raw1 = (0.5 + sc[1]) / 0.005 * 0.5
    raw2 = (0.5 + sc[2]) / 0.005
    assert meta.weights[1] / meta.weights[2] == pytest.approx(raw1 / raw2, rel=1e-6)
    assert {e["sid"] for e in meta.entries} == set(meta.weights)
