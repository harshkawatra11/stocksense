from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stocksense.models.base import ModelSpec, build_model  # noqa: E402
from stocksense.models.seqnet import SeqNet, SeqNetModel, add_keys, date_to_days, ic_loss  # noqa: E402

N_DATES, N = 70, 64


class ArrayProvider:
    """Random noise sequences keyed by (days, sid); the planted signal lives in the tabular features."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.cache: dict[tuple[int, int], np.ndarray] = {}

    def get(self, days, sids):
        out = np.zeros((len(days), 60, 12), dtype=np.float32)
        for i, k in enumerate(zip(days.tolist(), sids.tolist(), strict=True)):
            if k not in self.cache:
                self.cache[k] = (
                    np.random.default_rng(hash(k) % (2**32)).normal(size=(60, 12)).astype(np.float32)
                )
            out[i] = self.cache[k]
        return out, np.ones(len(days), dtype=bool)


def _data(seed=0, n_dates=N_DATES, n=N, n_feat=4):
    rng = np.random.default_rng(seed)
    d0 = date(2021, 1, 4)
    dates = np.repeat(np.array([np.datetime64(d0 + timedelta(days=i), "D") for i in range(n_dates)]), n)
    sids = np.tile(np.arange(n), n_dates)
    X = rng.normal(size=(n_dates * n, n_feat)).astype(np.float32)
    y = 0.5 * X[:, 0] + rng.normal(size=n_dates * n)
    X[rng.random(X.shape) < 0.05] = np.nan
    return add_keys(X, sids, dates), y, dates


def _model(**kw):
    kw.setdefault("device", "cpu")
    kw.setdefault("min_batch", 20)
    kw.setdefault("max_epochs", 3)
    kw.setdefault("hidden", 16)
    return SeqNetModel(ArrayProvider(), **kw)


def test_forward_shape_and_nan_inputs_are_finite():
    net = SeqNet(n_tab=5, hidden=16)
    seq = torch.randn(7, 60, 12)
    tab = torch.randn(7, 5)
    assert net(seq, tab).shape == (7,)
    seq[0, 3, 2] = float("nan")
    tab[1, 0] = float("nan")
    tab[2] = float("nan")
    assert torch.isfinite(net(seq, tab)).all()


def test_ic_loss_extremes():
    a = torch.randn(100)
    assert float(ic_loss(2 * a + 1, a)) == pytest.approx(0.0, abs=1e-5)
    assert float(ic_loss(-3 * a, a)) == pytest.approx(2.0, abs=1e-5)


def test_three_epochs_reach_inner_ic_on_planted_tabular_signal():
    X, y, d = _data()
    cut = 50 * N
    m = _model(max_epochs=3)
    info = m.fit(X[:cut], y[:cut], np.ones(cut), d[:cut], X[cut:], y[cut:], d[cut:], seed=1)
    print(f"\nseqnet synthetic inner IC {info.val_ic:.4f}")
    assert info.val_ic > 0.03
    m.refit_full(X, y, np.ones(len(y)), d, info, seed=1)
    p = m.predict(X)
    assert p.shape == (len(y),) and np.isfinite(p).all()
    assert m.explain(X, list("abcd")) is None


def test_checkpoint_resume_equals_uninterrupted_run(tmp_path):
    X, y, d = _data(n_dates=40)
    cut = 30 * N
    args = (X[:cut], y[:cut], np.ones(cut), d[:cut], X[cut:], y[cut:], d[cut:])

    full = _model(max_epochs=4, patience=99, work_dir=tmp_path / "a")
    full.fit(*args, seed=3)

    def boom(ep):
        if ep == 2:
            raise RuntimeError("crash")

    broken = _model(max_epochs=4, patience=99, work_dir=tmp_path / "b")
    broken.on_epoch_end = boom
    with pytest.raises(RuntimeError):
        broken.fit(*args, seed=3)
    assert sorted(p.name for p in (tmp_path / "b" / "fit").glob("*.pt")) == ["epoch_1.pt", "epoch_2.pt"]
    resumed = _model(max_epochs=4, patience=99, work_dir=tmp_path / "b")
    resumed.fit(*args, seed=3)
    for (k, a), (_, b) in zip(full.net.state_dict().items(), resumed.net.state_dict().items(), strict=True):
        assert torch.equal(a, b), k
    assert (tmp_path / "b" / "fit" / "epoch_4.pt").exists()


def test_out_of_memory_splits_the_batch_and_continues(monkeypatch):
    X, y, d = _data(n_dates=12)
    m = _model(max_epochs=1)
    orig = SeqNet.forward
    calls = {"oom": 0}

    def fwd(self, seq, tab):
        if seq.shape[0] > 20 and self.training:
            calls["oom"] += 1
            raise torch.cuda.OutOfMemoryError("fake")
        return orig(self, seq, tab)

    monkeypatch.setattr(SeqNet, "forward", fwd)
    m.fit(X, y, np.ones(len(y)), d, seed=0)
    assert calls["oom"] >= 1 and m._chunk_div >= 4


def test_save_load_roundtrip_and_registry(tmp_path):
    X, y, d = _data(n_dates=8)
    m = _model(max_epochs=1)
    m.fit(X, y, np.ones(len(y)), d, seed=0)
    m.save(tmp_path)
    m2 = _model()
    m2.load(tmp_path)
    assert np.allclose(m.predict(X), m2.predict(X), atol=1e-6)
    assert isinstance(build_model(ModelSpec("s1-seqnet", "y_c2o_5")), SeqNetModel)
    assert date_to_days(np.array([np.datetime64("1970-01-03", "D")]))[0] == 2


@pytest.mark.gpu
def test_gpu_smoke_two_epochs():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    X, y, d = _data(n_dates=20)
    cut = 14 * N
    try:
        m = _model(device="cuda", max_epochs=2)
        info = m.fit(X[:cut], y[:cut], np.ones(cut), d[:cut], X[cut:], y[cut:], d[cut:], seed=1)
        p = m.predict(X)
    except torch.cuda.OutOfMemoryError:
        pytest.skip("CUDA out of memory (the GPU is in use by another job)")
    assert np.isfinite(p).all() and info.best_iteration >= 1
