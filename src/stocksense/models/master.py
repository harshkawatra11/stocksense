"""MASTER, the market-guided stock transformer (G2 card D1; Li et al., AAAI 2024).

Reimplemented in the `TorchCSModel` contract from the reference `github.com/SJTU-DMTai/MASTER` (MIT).
Per stock a window of T sessions of the 'xs' features (read from a `FeatureCube`) is scaled feature-wise by
a market gate `F * softmax(W m_t / beta)` (m_t: the 'mkt' features of the day), projected to `d_model`,
given a positional encoding, passed through intra-stock temporal self-attention, then inter-stock
attention across all names of the date at each time step, then a temporal attention aggregation whose
query is the last step, and a linear score. One batch is one date. Loss MSE on the `z_` label by default.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from stocksense.models.base import Model, ModelSpec, register_model
from stocksense.models.torch_common import TorchCSModel

_CUBES: dict[tuple[Any, ...], Any] = {}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]].to(x.dtype)


class _Block(nn.Module):
    """Pre-norm multi-head attention + FFN; `axis` 'time' attends over T (per stock), 'stock' over N."""

    def __init__(self, d_model: int, heads: int, dropout: float, axis: str) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by the number of heads")
        self.h, self.axis, self.p = heads, axis, dropout
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x [N, T, D]
        x = self.n1(x)
        z = x if self.axis == "time" else x.transpose(0, 1)  # [B, L, D]
        b, n, d = z.shape

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, n, self.h, d // self.h).transpose(1, 2)

        a = F.scaled_dot_product_attention(
            split(self.q(z)), split(self.k(z)), split(self.v(z)), dropout_p=self.p if self.training else 0.0
        )
        a = a.transpose(1, 2).reshape(b, n, d)
        if self.axis == "stock":
            a = a.transpose(0, 1)
        xt = self.n2(x + a)
        return xt + self.ffn(xt)


class Gate(nn.Module):
    def __init__(self, d_in: int, d_out: int, beta: float) -> None:
        super().__init__()
        self.trans = nn.Linear(d_in, d_out)
        self.d_out, self.t = d_out, float(beta)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.d_out * torch.softmax(self.trans(m) / self.t, dim=-1)


class TemporalAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # [N, T, D] -> [N, D]
        h = self.trans(z)
        lam = torch.softmax(torch.matmul(h, h[:, -1, :].unsqueeze(-1)).squeeze(-1), dim=1)
        return torch.matmul(lam.unsqueeze(1), z).squeeze(1)


class MasterNet(nn.Module):
    def __init__(
        self, n_xs: int, n_mkt: int, d_model: int, t_heads: int, s_heads: int, dropout: float, beta: float
    ) -> None:
        super().__init__()
        self.n_mkt = n_mkt
        self.gate = Gate(n_mkt, n_xs, beta) if n_mkt else None
        self.inp = nn.Linear(n_xs, d_model)
        self.pos = PositionalEncoding(d_model)
        self.tatt = _Block(d_model, t_heads, dropout, "time")
        self.satt = _Block(d_model, s_heads, dropout, "stock")
        self.agg = TemporalAttention(d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:  # x [N,T,F], m [N,M] -> [N]
        if self.gate is not None:
            x = x * self.gate(m).unsqueeze(1)
        h = self.satt(self.tatt(self.pos(self.inp(x))))
        return self.out(self.agg(h)).squeeze(-1)


class MasterModel(TorchCSModel):
    needs_keys = True
    needs_cube = True

    def __init__(
        self,
        d_model: int = 256,
        dropout: float = 0.5,
        beta: float = 5.0,
        T: int = 20,
        t_heads: int = 4,
        s_heads: int = 2,
        feature_version: int | None = None,
        **kw: Any,
    ) -> None:
        kw.setdefault("loss", "mse")
        kw.setdefault("lr", 1e-4)
        super().__init__(**kw)
        self.d_model, self.dropout, self.beta, self.T = int(d_model), float(dropout), float(beta), int(T)
        self.t_heads, self.s_heads = int(t_heads), int(s_heads)
        self.feature_version = feature_version
        self.cube: Any = None
        self.purpose = "research"

    # ---- cube ----------------------------------------------------------------------------------
    def _xs_names(self) -> list[str]:
        return [n for n, k in zip(self.feature_names, self.kinds, strict=True) if k == "xs"]

    def _need_cube(self) -> Any:
        if self.cube is None:
            from stocksense.core.config import load_settings
            from stocksense.features.build import current_version
            from stocksense.features.cube import FeatureCube

            settings = load_settings()
            v = self.feature_version or current_version(settings)
            cols = self._xs_names()
            key = (str(settings.paths.data_root), v, tuple(cols), self.purpose)
            if key not in _CUBES:
                _CUBES.clear()  # one cube at a time (about 0.5 GB)
                _CUBES[key] = FeatureCube(settings, int(v), cols, self.purpose)  # type: ignore[arg-type]
                gc.collect()
            self.cube = _CUBES[key]
        return self.cube

    def _bind(self, X: np.ndarray) -> None:
        super()._bind(X)
        self.n_feat = len(self.kinds)  # X may hold only the 'mkt' columns (light X) plus the two keys

    def _mkt_columns(self) -> list[int]:
        """Columns of X holding the 'mkt' features: all columns in kind order, or the first n_mkt ones when
        the runner passed a light X (needs_cube: 'mkt' columns and keys only)."""
        if self._X.shape[1] - 2 == len(self.kinds):
            return [i for i, k in enumerate(self.kinds) if k == "mkt"]
        return list(range(self.n_mkt))

    # ---- TorchCSModel hooks --------------------------------------------------------------------
    def build_net(self, n_feat: int, n_mkt: int) -> nn.Module:
        return MasterNet(
            n_feat - n_mkt, n_mkt, self.d_model, self.t_heads, self.s_heads, self.dropout, self.beta
        )

    def batch_inputs(self, idx: np.ndarray) -> dict[str, torch.Tensor]:
        cube = self._need_cube()
        sid, days = self.key_columns(idx)
        x, _ = cube.window(cube.day_idx(days), sid, self.T)
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        m = np.nan_to_num(self._X[idx][:, self._mkt_columns()], nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "x": torch.as_tensor(x, dtype=torch.float32, device=self.device),
            "m": torch.as_tensor(m, dtype=torch.float32, device=self.device),
        }

    def forward_net(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        assert self.net is not None
        return self.net(inputs["x"], inputs["m"])  # type: ignore[no-any-return]


_KEYS = (
    "d_model", "dropout", "beta", "T", "t_heads", "s_heads", "lr", "weight_decay", "max_epochs", "patience",
    "loss", "device", "amp", "min_names", "max_minutes", "feature_version",
)  # fmt: skip


def _factory(spec: ModelSpec) -> Model:
    p = dict(spec.params)
    work = p.get("work_dir")
    kw = {k: p[k] for k in _KEYS if k in p}
    return MasterModel(work_dir=Path(work) if work else None, **kw)


register_model("master", _factory)
