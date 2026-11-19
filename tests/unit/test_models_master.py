from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from stocksense.features.cube import FeatureCube  # noqa: E402
from stocksense.models.base import ModelSpec, build_model  # noqa: E402
from stocksense.models.master import MasterModel, MasterNet  # noqa: E402
from stocksense.models.seqnet import add_keys  # noqa: E402

N_DATES, N = 70, 40
NAMES = ["a", "b", "c", "m1"]
KINDS = ["xs", "xs", "xs", "mkt"]


def _world(seed=0, signal=0.25):
    rng = np.random.default_rng(seed)
    d0 = date(2021, 1, 4)
    days = [d0 + timedelta(days=i) for i in range(N_DATES)]
    dates = np.repeat(np.array(days, dtype="datetime64[D]"), N)
    sids = np.tile(np.arange(1, N + 1), N_DATES)
    X = rng.normal(size=(N_DATES * N, 4)).astype(np.float32)
    X[:, 3] = np.repeat(rng.normal(size=N_DATES), N)  # market state: one value per date
    y = signal * X[:, 0] + rng.normal(size=N_DATES * N)
    X[rng.random(X.shape) < 0.03] = np.nan
    df = pl.DataFrame(
        {"date": dates, "sid": sids.astype(np.int64), **{n: X[:, i] for i, n in enumerate(NAMES[:3])}}
    )
    cube = FeatureCube.__new__(FeatureCube)
    cube.columns = NAMES[:3]
    cube._fill(df.sort("date", "sid"))
    return add_keys(X, sids, dates), y, dates, cube


def _model(cube, **kw):
    kw.setdefault("d_model", 16)
    kw.setdefault("T", 5)
    kw.setdefault("device", "cpu")
    kw.setdefault("min_names", 20)
    m = MasterModel(**kw)
    m.cube = cube
    m.configure(NAMES, KINDS)
    return m


def _split(X, y, dates, cut=55):
    m = np.repeat(np.arange(N_DATES), N) < cut
    w = np.ones(len(y))
    return (X[m], y[m], w[m], dates[m]), (X[~m], y[~m], dates[~m])


def test_forward_shape():
    net = MasterNet(n_xs=3, n_mkt=1, d_model=16, t_heads=4, s_heads=2, dropout=0.1, beta=5.0)
    out = net(torch.randn(7, 5, 3), torch.randn(7, 1))
    assert out.shape == (7,) and torch.isfinite(out).all()


def test_forward_shape_without_market_columns():
    net = MasterNet(n_xs=3, n_mkt=0, d_model=16, t_heads=4, s_heads=2, dropout=0.1, beta=5.0)
    assert net(torch.randn(7, 5, 3), torch.zeros(7, 0)).shape == (7,)


def test_gate_is_a_scaled_softmax():
    net = MasterNet(n_xs=6, n_mkt=2, d_model=16, t_heads=4, s_heads=2, dropout=0.0, beta=5.0)
    g = net.gate(torch.randn(3, 2))
    assert torch.allclose(g.sum(-1), torch.full((3,), 6.0), atol=1e-5)


def test_finite_with_nan_inputs():
    X, y, dates, cube = _world()
    m = _model(cube, max_epochs=1)
    (Xt, yt, wt, dt), (Xv, yv, dv) = _split(X, y, dates)
    m.fit(Xt, yt, wt, dt, Xv, yv, dv, seed=1)
    Xn = Xv.copy()
    Xn[:, :4][np.arange(len(Xn)) % 3 == 0] = np.nan
    p = m.predict(Xn)
    assert p.shape == (len(Xn),) and np.isfinite(p).all()


def test_learns_planted_signal_in_three_epochs():
    X, y, dates, cube = _world()
    (Xt, yt, wt, dt), (Xv, yv, dv) = _split(X, y, dates)
    m = _model(cube, max_epochs=3, lr=1e-2, patience=3, loss="mse", dropout=0.1)
    info = m.fit(Xt, yt, wt, dt, Xv, yv, dv, seed=1)
    assert info.val_ic is not None and info.val_ic > 0.03


def test_window_is_causal_and_zero_filled():
    X, y, dates, cube = _world()
    m = _model(cube, T=5)
    m._bind(X)
    rows = np.flatnonzero(np.repeat(np.arange(N_DATES), N) == 1)  # second date: only 1 prior session
    inp = m.batch_inputs(rows)
    x = inp["x"].numpy()
    assert x.shape == (N, 5, 3)
    assert np.all(x[:, :3] == 0)  # sessions before the cube starts


def test_save_load_identical_predictions(tmp_path):
    X, y, dates, cube = _world()
    (Xt, yt, wt, dt), (Xv, yv, dv) = _split(X, y, dates)
    m = _model(cube, max_epochs=2)
    m.fit(Xt, yt, wt, dt, Xv, yv, dv, seed=1)
    m.save(tmp_path / "m")
    m2 = MasterModel(d_model=16, T=5, device="cpu")
    m2.cube = cube
    m2.load(tmp_path / "m")
    assert np.allclose(m.predict(Xv), m2.predict(Xv), atol=1e-6)


def test_registry_builds_master_from_params():
    m = build_model(
        ModelSpec(family="s1-master", target="y_c2o_10", params={"d_model": 128, "dropout": 0.3, "lr": 3e-5})
    )
    assert isinstance(m, MasterModel) and m.d_model == 128 and m.lr == 3e-5 and m.needs_keys


@pytest.mark.gpu
def test_gpu_smoke():
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    X, y, dates, cube = _world()
    (Xt, yt, wt, dt), (Xv, yv, dv) = _split(X, y, dates)
    m = _model(cube, max_epochs=1, device="cuda", amp=True)
    m.fit(Xt, yt, wt, dt, Xv, yv, dv, seed=1)
    assert np.isfinite(m.predict(Xv)).all()
