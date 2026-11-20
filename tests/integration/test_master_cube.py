from __future__ import annotations

import pytest

pytest.importorskip("torch")

from stocksense.models.master import _CUBES, MasterModel  # noqa: E402
from tests.fixtures.wf_world import build_world  # noqa: E402

pytestmark = pytest.mark.integration


def test_need_cube_loads_from_the_real_settings(tmp_path, monkeypatch):
    w = build_world(tmp_path, planted_ic=0.05, n=40)
    monkeypatch.setattr("stocksense.core.config.load_settings", lambda *a, **k: w.settings)
    _CUBES.clear()
    m = MasterModel(d_model=16, T=5, device="cpu")
    m.configure(["alpha_t", "ret_1", "vol_21", "mkt_ret_1"], ["xs", "xs", "xs", "mkt"])
    cube = m._need_cube()
    assert cube.columns == ["alpha_t", "ret_1", "vol_21"]
    assert m._need_cube() is cube
    _CUBES.clear()


def test_walkforward_passes_a_light_x_to_cube_models(tmp_path, monkeypatch):
    from stocksense.core.state import State
    from stocksense.models.base import ModelSpec, build_model
    from stocksense.validation.walkforward import run_walk_forward

    w = build_world(tmp_path, planted_ic=0.08, n=60)
    monkeypatch.setattr("stocksense.core.config.load_settings", lambda *a, **k: w.settings)
    _CUBES.clear()
    seen: list[int] = []

    def factory(spec):
        m = build_model(spec)
        orig = m.fit

        def fit(X, *a, **k):
            seen.append(X.shape[1])
            return orig(X, *a, **k)

        m.fit = fit
        return m

    st = State(w.settings)
    st.migrate()
    spec = ModelSpec(
        family="s1-master", target="y_oc", entry="open",
        params={"d_model": 16, "T": 5, "device": "cpu", "min_names": 20, "max_epochs": 2, "dropout": 0.1},
    )  # fmt: skip
    res = run_walk_forward(
        spec, w.settings, st, folds=w.folds, calendar=w.calendar, registry=w.registry, seeds=(1,),
        inner_months=2, fold_names=["f1"], model_factory=factory, trial_family="g2-master-test",
    )  # fmt: skip
    assert res.preds.height > 0
    assert seen and set(seen) == {1 + 2}  # one 'mkt' column plus the (sid, day) keys, no 'xs' columns
    _CUBES.clear()
