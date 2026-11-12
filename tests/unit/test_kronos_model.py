"""Kronos wrapper (PLAN T7.5b): sample paths equal upstream's per-path draws, preprocessing equals
`KronosPredictor.predict`, de-normalization inverts normalization, zero-variance columns stay finite,
OOM halving works. CPU tests use a tiny random-weight model; the `gpu` test uses the real weights."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import torch
from kronos.model import KronosPredictor
from kronos.model.kronos import auto_regressive_inference, calc_time_stamps

from stocksense.models import kronos_model as km

PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]


@pytest.fixture(scope="module")
def tiny():
    return km.build_test_bundle(seed=0)


def _dates(n: int, start: date = date(2024, 1, 1)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _frame(rng: np.random.Generator, lookback: int, pred_len: int, scale: float = 100.0) -> km.KronosInput:
    close = scale * np.exp(np.cumsum(rng.normal(0, 0.01, lookback)))
    open_ = close * (1 + rng.normal(0, 0.003, lookback))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, lookback)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, lookback)))
    vol = rng.uniform(1e5, 2e5, lookback)
    amount = vol * close
    vals = np.stack([open_, high, low, close, vol, amount], axis=1)
    ds = _dates(lookback + pred_len)
    return km.KronosInput(values=vals, dates=ds[:lookback], future_dates=ds[lookback:])


def _tensors(rng, n: int, lookback: int, pred_len: int, bundle: km.KronosBundle):
    frames = [_frame(rng, lookback, pred_len) for _ in range(n)]
    x = np.stack([km.normalize(f.values, clip=5.0)[0] for f in frames])
    xs = np.stack([km.time_features(f.dates) for f in frames])
    ys = np.stack([km.time_features(f.future_dates) for f in frames])
    return (torch.from_numpy(x), torch.from_numpy(xs), torch.from_numpy(ys))


def test_time_features_match_upstream_calc_time_stamps():
    ds = _dates(40)
    ours = km.time_features(ds)
    theirs = calc_time_stamps(pd.Series(pd.to_datetime(ds))).values.astype(np.float32)
    assert ours.dtype == np.float32 and ours.shape == (40, 5)
    np.testing.assert_array_equal(ours, theirs)
    assert (ours[:, 0] == 0).all() and (ours[:, 1] == 0).all()  # minute, hour of a daily bar


def test_sample_paths_mean_equals_upstream_same_seed(tiny):
    rng = np.random.default_rng(1)
    x, xs, ys = _tensors(rng, 3, 40, 5, tiny)
    n_paths = 7
    paths = km.sample_paths(
        tiny.tokenizer,
        tiny.model,
        x,
        xs,
        ys,
        tiny.max_context,
        5,
        n_paths,
        T=1.0,
        top_k=0,
        top_p=0.9,
        seed=11,
    )
    assert paths.shape == (3, n_paths, 5, 6)
    torch.manual_seed(11)
    up = auto_regressive_inference(
        tiny.tokenizer, tiny.model, x, xs, ys, tiny.max_context, 5, 5, 1.0, 0, 0.9, n_paths
    )
    np.testing.assert_allclose(paths.mean(dim=1).numpy(), up[:, -5:, :], atol=1e-4)


def test_sample_paths_are_distinct_and_deterministic(tiny):
    rng = np.random.default_rng(2)
    x, xs, ys = _tensors(rng, 2, 40, 4, tiny)
    kw = dict(max_context=tiny.max_context, pred_len=4, n_paths=6, T=1.0, top_p=0.9)
    a = km.sample_paths(tiny.tokenizer, tiny.model, x, xs, ys, seed=5, **kw)
    b = km.sample_paths(tiny.tokenizer, tiny.model, x, xs, ys, seed=5, **kw)
    c = km.sample_paths(tiny.tokenizer, tiny.model, x, xs, ys, seed=6, **kw)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert (a[:, 0] != a[:, 1]).any()  # different paths of one series differ


def test_normalize_denormalize_roundtrip_on_constant_plus_noise():
    rng = np.random.default_rng(3)
    base = np.full((256, 6), 500.0) + rng.normal(0, 1.0, (256, 6))
    z, mean, std = km.normalize(base, clip=1e9)
    assert z.dtype == np.float32
    np.testing.assert_allclose(z.mean(0), 0.0, atol=1e-3)
    back = km.denormalize(z, mean, std)
    np.testing.assert_allclose(back, base, rtol=1e-5)


def test_normalize_matches_upstream_formula_and_clips():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, (50, 6)).astype(np.float32)
    x[3, 0] = 80.0
    z, mean, std = km.normalize(x, clip=5.0)
    ref = np.clip((x - x.mean(0)) / (x.std(0) + 1e-5), -5, 5)
    np.testing.assert_allclose(z, ref, rtol=1e-6)
    assert z.max() <= 5.0 and np.allclose(std, x.std(0))


def test_forecast_panel_matches_upstream_predictor_single_series(tiny):
    rng = np.random.default_rng(5)
    f = _frame(rng, 60, 5)
    out = km.forecast_panel([f], tiny, pred_len=5, n_paths=1, T=1.0, top_k=0, top_p=0.9, seed=3)
    assert out.shape == (1, 1, 5, 6) and out.dtype == np.float32
    pred = KronosPredictor(tiny.model, tiny.tokenizer, device="cpu", max_context=tiny.max_context)
    df = pd.DataFrame(f.values, columns=PRICE_COLS)
    torch.manual_seed(3)
    ref = pred.predict(
        df,
        pd.Series(pd.to_datetime(f.dates)),
        pd.Series(pd.to_datetime(f.future_dates)),
        pred_len=5,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
    )
    np.testing.assert_allclose(out[0, 0], ref.to_numpy(), rtol=1e-4, atol=1e-3)


def test_forecast_panel_shapes_and_zero_variance_column_is_finite(tiny):
    rng = np.random.default_rng(6)
    frames = [_frame(rng, 48, 5) for _ in range(5)]
    for f in frames:
        f.values[:, 4] = 1000.0  # constant volume -> zero std
    out = km.forecast_panel(frames, tiny, pred_len=5, n_paths=4, seed=1, batch_series=2)
    assert out.shape == (5, 4, 5, 6)
    assert np.isfinite(out).all()


def test_forecast_panel_rejects_nan_and_ragged_input(tiny):
    rng = np.random.default_rng(7)
    f = _frame(rng, 48, 5)
    bad = km.KronosInput(values=f.values.copy(), dates=f.dates, future_dates=f.future_dates)
    bad.values[2, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        km.forecast_panel([bad], tiny, pred_len=5, n_paths=2)
    short = _frame(rng, 30, 5)
    with pytest.raises(ValueError, match="lookback"):
        km.forecast_panel([f, short], tiny, pred_len=5, n_paths=2)
    with pytest.raises(ValueError, match="future_dates"):
        km.forecast_panel([f], tiny, pred_len=7, n_paths=2)


def test_forecast_panel_halves_batch_on_oom(tiny, monkeypatch):
    rng = np.random.default_rng(8)
    frames = [_frame(rng, 40, 3) for _ in range(6)]
    real = km.sample_paths
    seen: list[int] = []

    def flaky(tok, mod, x, xs, ys, *a, **k):
        seen.append(x.shape[0])
        if x.shape[0] > 2:
            raise torch.cuda.OutOfMemoryError("simulated")
        return real(tok, mod, x, xs, ys, *a, **k)

    monkeypatch.setattr(km, "sample_paths", flaky)
    out = km.forecast_panel(frames, tiny, pred_len=3, n_paths=2, batch_series=64, seed=2)
    assert out.shape == (6, 2, 3, 6) and np.isfinite(out).all()
    assert seen == [6, 3] + [1] * 6  # 6 and 3 fail, the halved batch of 1 series fits


def test_oom_at_one_series_raises(tiny, monkeypatch):
    rng = np.random.default_rng(9)
    frames = [_frame(rng, 40, 3)]

    def always(*a, **k):
        raise torch.cuda.OutOfMemoryError("simulated")

    monkeypatch.setattr(km, "sample_paths", always)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        km.forecast_panel(frames, tiny, pred_len=3, n_paths=2)


def test_test_model_env_flag(monkeypatch):
    monkeypatch.setenv("STOCKSENSE_KRONOS_TEST_MODEL", "1")
    b = km.load_bundle("small")
    assert b.size == "test" and b.device == "cpu" and not b.model.training
    monkeypatch.delenv("STOCKSENSE_KRONOS_TEST_MODEL")


@pytest.mark.gpu
def test_real_weights_sample_paths_mean_equals_upstream_on_cuda():
    assert torch.cuda.is_available()
    tok, model = km.load_kronos("small", device="cuda")
    rng = np.random.default_rng(10)
    bundle = km.KronosBundle(tok, model, 512, "cuda", "small")
    x, xs, ys = _tensors(rng, 4, 256, 5, bundle)
    x, xs, ys = x.cuda(), xs.cuda(), ys.cuda()
    paths = km.sample_paths(tok, model, x, xs, ys, 512, 5, 8, T=1.0, top_k=0, top_p=0.9, seed=123)
    torch.manual_seed(123)
    up = auto_regressive_inference(tok, model, x, xs, ys, 512, 5, 5, 1.0, 0, 0.9, 8)
    diff = np.abs(paths.mean(dim=1).cpu().numpy() - up[:, -5:, :]).max()
    assert diff < 1e-4, diff
    assert paths.shape == (4, 8, 5, 6)


@pytest.mark.gpu
def test_vendored_regression_on_cuda():
    """The upstream regression test (512 context), run on CUDA; PLAN T7.5a records the max diffs."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "third_party/kronos/tests/test_kronos_regression.py"
    spec = importlib.util.spec_from_file_location("kronos_regression_cuda", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.DEVICE = "cuda"
    mod.test_kronos_predictor_regression(512)
