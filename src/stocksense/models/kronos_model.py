# ruff: noqa: E501  (sample_paths is a verbatim copy of upstream lines)
"""Kronos wrapper (PLAN T7.5b): load, sample paths (all of them, not the mean), batched panel forecasts.

`sample_paths` is upstream's `auto_regressive_inference` body copied verbatim except for its first lines
(no `verbose`) and its last three statements, which return every path instead of their mean.
`forecast_panel` reproduces `KronosPredictor.predict` pre-processing exactly. No pandas is needed: the
five time features are computed directly (the pandas allowance of PLAN D.2 is not used).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import torch
from kronos.model import Kronos, KronosTokenizer
from kronos.model.kronos import sample_from_logits

from stocksense.core.log import get_logger

log = get_logger(__name__)

TEST_MODEL_ENV = "STOCKSENSE_KRONOS_TEST_MODEL"
FEATURES = ("open", "high", "low", "close", "volume", "amount")
EPS = 1e-5
_REPOS = {
    "mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
}


@dataclass(frozen=True)
class KronosInput:
    """One series: `values` [lookback, 6] in FEATURES order, its session dates, and the future sessions."""

    values: np.ndarray
    dates: Sequence[date]
    future_dates: Sequence[date]


@dataclass(frozen=True)
class KronosBundle:
    tokenizer: KronosTokenizer
    model: Kronos
    max_context: int
    device: str
    size: str


# On Windows (WDDM) a CUDA out of memory can surface as torch.AcceleratorError('CUDA error: out of memory')
# instead of torch.cuda.OutOfMemoryError; both are handled.
_AcceleratorError = getattr(torch, "AcceleratorError", torch.cuda.OutOfMemoryError)


# ---- loading -----------------------------------------------------------------------------------


def load_kronos(size: str = "small", device: str = "cuda") -> tuple[KronosTokenizer, Kronos]:
    """(tokenizer, model) in eval() on `device`; weights come from the Hugging Face cache."""
    if size not in _REPOS:
        raise ValueError(f"size must be one of {sorted(_REPOS)}, got {size!r}")
    model_repo, tok_repo, _ = _REPOS[size]
    tok = KronosTokenizer.from_pretrained(tok_repo).to(device).eval()
    model = Kronos.from_pretrained(model_repo).to(device).eval()
    return tok, model


def build_test_bundle(seed: int = 0) -> KronosBundle:
    """A tiny random-weight 2-layer Kronos with its tokenizer (CPU, fixed seed) for fast tests."""
    cur = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        tok = KronosTokenizer(
            d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=2, n_dec_layers=2,
            ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, s1_bits=4, s2_bits=4,
            beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4,
        )  # fmt: skip
        model = Kronos(
            s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=2, ff_dim=64, ffn_dropout_p=0.0,
            attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0, learn_te=True,
        )  # fmt: skip
    finally:
        torch.random.set_rng_state(cur)
    return KronosBundle(tok.eval(), model.eval(), 512, "cpu", "test")


def load_bundle(size: str = "small", device: str | None = None) -> KronosBundle:
    """The model bundle; with STOCKSENSE_KRONOS_TEST_MODEL=1 a tiny random-weight CPU model instead."""
    if os.environ.get(TEST_MODEL_ENV) == "1":
        return build_test_bundle()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_kronos(size, dev)
    return KronosBundle(tok, model, _REPOS[size][2], dev, size)


# ---- pre/post processing (upstream KronosPredictor.predict) --------------------------------------


def time_features(dates: Sequence[date]) -> np.ndarray:
    """[n, 5] float32 columns minute, hour, weekday, day, month of daily bars (minute = hour = 0)."""
    return np.array([[0, 0, d.weekday(), d.day, d.month] for d in dates], dtype=np.float32).reshape(-1, 5)


def normalize(values: np.ndarray, clip: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(clip((x - mean)/(std + 1e-5)), mean, std) over the rows, float32, as upstream."""
    x = np.asarray(values).astype(np.float32)
    mean, std = np.mean(x, axis=0), np.std(x, axis=0)
    z = np.clip((x - mean) / (std + EPS), -clip, clip)
    return z, mean, std


def denormalize(z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Inverse of `normalize` (before clipping): z * (std + 1e-5) + mean."""
    return z * (std + EPS) + mean


# ---- sampling ----------------------------------------------------------------------------------


def sample_paths(
    tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len, n_paths,
    T=1.0, top_k=0, top_p=0.9, clip=5.0, seed=0,
) -> torch.Tensor:  # fmt: skip
    """All sampled paths, normalized units: Tensor[n_series, n_paths, pred_len, 6].

    Body of upstream `auto_regressive_inference` (sample_count = n_paths) up to the final reshape.
    """
    torch.manual_seed(seed)
    sample_count = n_paths
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)

        device = x.device
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = (
            x_stamp.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, x_stamp.size(1), x_stamp.size(2))
            .to(device)
        )
        y_stamp = (
            y_stamp.unsqueeze(1)
            .repeat(1, sample_count, 1, 1)
            .reshape(-1, y_stamp.size(1), y_stamp.size(2))
            .to(device)
        )

        x_token = tokenizer.encode(x, half=True)

        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx : start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx : start_idx + buffer_len]

        ran = range
        for i in ran(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(
                s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
            )

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(
                s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
            )

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous(),
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        # STOCKSENSE: upstream averages over paths here; we keep every path.
        return z[:, :, -pred_len:, :]


# ---- panel forecast ----------------------------------------------------------------------------


def _prepare(frames: Sequence[KronosInput], pred_len: int, clip: float):
    lookback = len(frames[0].dates)
    xs, stamps_x, stamps_y, means, stds = [], [], [], [], []
    for f in frames:
        v = np.asarray(f.values)
        if v.shape != (len(f.dates), 6):
            raise ValueError(f"values must be [lookback, 6], got {v.shape}")
        if len(f.dates) != lookback:
            raise ValueError(f"all series must share one lookback length ({lookback}, got {len(f.dates)})")
        if len(f.future_dates) < pred_len:
            raise ValueError(f"future_dates has {len(f.future_dates)} sessions, need {pred_len}")
        if not np.isfinite(v).all():
            raise ValueError("input contains NaN or infinite values")
        z, m, s = normalize(v, clip)
        xs.append(z)
        means.append(m)
        stds.append(s)
        stamps_x.append(time_features(f.dates))
        stamps_y.append(time_features(list(f.future_dates)[:pred_len]))
    return np.stack(xs), np.stack(stamps_x), np.stack(stamps_y), np.stack(means), np.stack(stds)


def forecast_panel(
    frames: Sequence[KronosInput],
    bundle: KronosBundle,
    pred_len: int,
    n_paths: int,
    *,
    T: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    clip: float = 5.0,
    seed: int = 0,
    batch_series: int = 8,
    autocast: bool = False,
) -> np.ndarray:
    """Sampled paths in price units: float32 [n_series, n_paths, pred_len, 6] (FEATURES order).

    Series are processed `batch_series` at a time (each batch is seeded with `seed`); on a CUDA out of
    memory error the batch is halved and retried. `autocast=True` runs float16 mixed precision.
    """
    if not frames:
        return np.zeros((0, n_paths, pred_len, 6), dtype=np.float32)
    x, xs, ys, mean, std = _prepare(frames, pred_len, clip)
    n = len(frames)
    dev = bundle.device
    out = np.zeros((n, n_paths, pred_len, 6), dtype=np.float32)
    bs = max(1, min(batch_series, n))
    i = 0
    while i < n:
        j = min(n, i + bs)
        try:
            tx = torch.from_numpy(x[i:j]).to(dev)
            tsx = torch.from_numpy(xs[i:j]).to(dev)
            tsy = torch.from_numpy(ys[i:j]).to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=autocast and dev.startswith("cuda")):
                p = sample_paths(
                    bundle.tokenizer, bundle.model, tx, tsx, tsy, bundle.max_context, pred_len, n_paths,
                    T=T, top_k=top_k, top_p=top_p, clip=clip, seed=seed,
                )  # fmt: skip
            out[i:j] = p.float().cpu().numpy()
        except (torch.cuda.OutOfMemoryError, _AcceleratorError) as e:
            if isinstance(e, _AcceleratorError) and "out of memory" not in str(e).lower():
                raise
            if bs == 1:
                raise
            if dev.startswith("cuda"):
                with contextlib.suppress(torch.cuda.OutOfMemoryError, _AcceleratorError):
                    torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            log.warning("kronos_oom_halving", batch_series=bs)
            continue
        i = j
    log.info("kronos_forecast_panel", n_series=n, batch_series=bs, n_paths=n_paths, pred_len=pred_len)
    out = denormalize(out, mean[:, None, None, :], std[:, None, None, :])
    return out.astype(np.float32)


def autocast_max_diff(
    bundle: KronosBundle, frames: Sequence[KronosInput], pred_len: int, n_paths: int
) -> float:
    """Max abs difference (normalized units) of the mean path, float16 autocast vs float32, same seed."""
    x, xs, ys, _, _ = _prepare(frames, pred_len, 5.0)
    args = (torch.from_numpy(a).to(bundle.device) for a in (x, xs, ys))
    tx, tsx, tsy = args
    kw = dict(T=1.0, top_k=0, top_p=0.9, seed=0)
    a = sample_paths(
        bundle.tokenizer, bundle.model, tx, tsx, tsy, bundle.max_context, pred_len, n_paths, **kw
    )
    with torch.autocast("cuda", dtype=torch.float16):
        b = sample_paths(
            bundle.tokenizer, bundle.model, tx, tsx, tsy, bundle.max_context, pred_len, n_paths, **kw
        )
    return float((a.mean(1) - b.float().mean(1)).abs().max())
