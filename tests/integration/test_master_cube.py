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
