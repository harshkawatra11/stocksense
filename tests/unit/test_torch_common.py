from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from stocksense.models import seqnet  # noqa: E402
from stocksense.models.torch_common import (  # noqa: E402
    DateBatches,
    TorchCSModel,
    ic_loss,
    listnet_loss,
    mse_loss,
)

N_DATES, N = 80, 60


def _data(seed=0, n_dates=N_DATES, n=N, n_feat=4, signal=0.25):
    rng = np.random.default_rng(seed)
    d0 = date(2021, 1, 4)
    dates = np.repeat(np.array([np.datetime64(d0 + timedelta(days=i), "D") for i in range(n_dates)]), n)
    X = rng.normal(size=(n_dates * n, n_feat)).astype(np.float32)
    y = signal * X[:, 0] + rng.normal(size=n_dates * n)  # per-date rank IC around 0.2
    X[rng.random(X.shape) < 0.03] = np.nan
    return X, y, dates


class Mlp(TorchCSModel):
    """2-layer MLP on the tabular columns."""

    def __init__(self, hidden=16, **kw):
        kw.setdefault("device", "cpu")
        kw.setdefault("min_names", 20)
        super().__init__(**kw)
        self.hidden = hidden

    def build_net(self, n_feat, n_mkt):
        return nn.Sequential(nn.Linear(n_feat, self.hidden), nn.GELU(), nn.Linear(self.hidden, 1))

    def batch_inputs(self, idx):
        return {"x": self.tab_tensor(idx)}

    def forward_net(self, inputs):
        return self.net(inputs["x"]).squeeze(-1)


def test_ic_loss_is_the_seqnet_one():
    assert ic_loss is seqnet.ic_loss


def test_mse_loss_hand_value():
    assert float(mse_loss(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0]))) == pytest.approx(2.5)


def test_listnet_loss_matches_hand_computation_for_three_names():
    pred = torch.tensor([0.2, -0.1, 0.5])
    y = torch.tensor([1.0, 0.0, 2.0])
    for tau in (1.0, 0.5):
        p = np.exp(y.numpy() / tau)
        p /= p.sum()
        lq = pred.numpy() - np.log(np.exp(pred.numpy()).sum())
        want = -(p * lq).sum()
        assert float(listnet_loss(pred, y, tau=tau)) == pytest.approx(want, rel=1e-5)


def test_date_batches_one_per_date_min_names_and_seeded_shuffle():
    days = np.array(["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"], dtype="datetime64[D]")
    dates = np.repeat(days, [60, 10, 55, 70])
    db = DateBatches(dates, min_names=50)
    assert len(db) == 3 and sorted(len(b) for b in db.batches) == [55, 60, 70]
    for b in db.batches:
        assert len(set(dates[b])) == 1
    a1 = [len(b) for b in db.epoch(1)]
    a2 = [len(b) for b in db.epoch(1)]
    assert a1 == a2  # seeded
    orders = {tuple(len(b) for b in db.epoch(s)) for s in range(10)}
    assert len(orders) > 1  # shuffled across epochs
    cover = np.sort(np.concatenate(db.epoch(3)))
    assert np.array_equal(cover, np.sort(np.concatenate(db.batches)))


def _split(X, y, dates, cut=60):
    m = np.repeat(np.arange(N_DATES), N) < cut
    w = np.ones(len(y))
    return (X[m], y[m], w[m], dates[m]), (X[~m], y[~m], dates[~m])


def test_mlp_reaches_inner_ic_on_planted_signal():
    X, y, dates = _data()
    (Xt, yt, wt, dt), (Xv, yv, dv) = _split(X, y, dates)
    m = Mlp(max_epochs=3, lr=1e-2, patience=3)
    m.configure(["a", "b", "c", "d"], ["xs", "xs", "xs", "mkt"])
    info = m.fit(Xt, yt, wt, dt, Xv, yv, dv, seed=1)
    assert info.val_ic is not None and info.val_ic > 0.03
    assert info.best_iteration is not None and 1 <= info.best_iteration <= 3
    assert m.n_mkt == 1
    p = m.predict(Xv)
    assert p.shape == (len(Xv),) and np.isfinite(p).all()


@pytest.mark.parametrize("loss", ["ic", "mse", "listnet"])
def test_all_losses_train(loss):
    X, y, dates = _data()
    (Xt, yt, wt, dt), (Xv, yv, dv) = _split(X, y, dates)
    m = Mlp(max_epochs=2, lr=1e-2, loss=loss)
    info = m.fit(Xt, yt, wt, dt, Xv, yv, dv, seed=1)
    assert info.val_ic is not None and info.val_ic > 0.0


def test_refit_full_uses_ceil_1p1_best_epoch(monkeypatch):
    X, y, dates = _data()
    (Xt, yt, wt, dt), (Xv, yv, dv) = _split(X, y, dates)
    m = Mlp(max_epochs=3, lr=1e-2)
    info = m.fit(Xt, yt, wt, dt, Xv, yv, dv, seed=1)
    seen = []
    m.on_epoch_end = seen.append
    m.refit_full(Xt, yt, wt, dt, info, seed=1)
    import math

    assert seen == list(range(1, math.ceil(1.1 * info.best_iteration) + 1))


def test_deterministic_for_a_seed():
    X, y, dates = _data()
    (Xt, yt, wt, dt), (Xv, yv, dv) = _split(X, y, dates)
    outs = []
    for _ in range(2):
        m = Mlp(max_epochs=2)
        m.fit(Xt, yt, wt, dt, Xv, yv, dv, seed=4)
        outs.append(m.predict(Xv))
    assert np.array_equal(outs[0], outs[1])


class _Halt(Exception):
    pass


def test_checkpoint_resume_reproduces_final_weights(tmp_path):
    X, y, dates = _data()
    (Xt, yt, wt, dt), (Xv, yv, dv) = _split(X, y, dates)
    full = Mlp(max_epochs=4, patience=10, work_dir=tmp_path / "a")
    full.fit(Xt, yt, wt, dt, Xv, yv, dv, seed=2)
    ref = {k: v.clone() for k, v in full.net.state_dict().items()}

    def halt(ep):
        if ep == 2:
            raise _Halt

    part = Mlp(max_epochs=4, patience=10, work_dir=tmp_path / "b")
    part.on_epoch_end = halt
    with pytest.raises(_Halt):
        part.fit(Xt, yt, wt, dt, Xv, yv, dv, seed=2)
    assert (tmp_path / "b" / "fit" / "epoch_2.pt").is_file()
    resumed = Mlp(max_epochs=4, patience=10, work_dir=tmp_path / "b")
    resumed.fit(Xt, yt, wt, dt, Xv, yv, dv, seed=2)
    for k, v in resumed.net.state_dict().items():
        assert torch.allclose(v, ref[k], atol=1e-6), k


def test_wall_time_guard_stops_and_records():
    X, y, dates = _data()
    (Xt, yt, wt, dt), (Xv, yv, dv) = _split(X, y, dates)
    m = Mlp(max_epochs=30, patience=30, max_minutes=0.0)
    info = m.fit(Xt, yt, wt, dt, Xv, yv, dv, seed=1)
    assert info.params["stopped"] == "wall" and info.best_iteration == 1


def test_oom_splits_the_date_batch(monkeypatch):
    X, y, dates = _data()
    (Xt, yt, wt, dt), (Xv, yv, dv) = _split(X, y, dates)
    m = Mlp(max_epochs=1)
    calls = {"n": 0}
    real = m.forward_net

    def flaky(inputs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.cuda.OutOfMemoryError("boom")
        return real(inputs)

    m.forward_net = flaky  # type: ignore[method-assign]
    info = m.fit(Xt, yt, wt, dt, Xv, yv, dv, seed=1)
    assert m._chunk_div == 2 and info.val_ic is not None


def test_save_load_roundtrip(tmp_path):
    X, y, dates = _data()
    (Xt, yt, wt, dt), (Xv, yv, dv) = _split(X, y, dates)
    m = Mlp(max_epochs=2)
    m.fit(Xt, yt, wt, dt, Xv, yv, dv, seed=1)
    m.save(tmp_path / "m")
    assert (tmp_path / "m" / "weights.pt").is_file()
    m2 = Mlp()
    m2.load(tmp_path / "m")
    assert np.allclose(m.predict(Xv), m2.predict(Xv), atol=1e-6)
