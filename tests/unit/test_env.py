from __future__ import annotations

import importlib

import pytest


def test_imports() -> None:
    for name in ["polars", "duckdb", "lightgbm", "torch", "einops", "httpx", "typer"]:
        assert importlib.import_module(name) is not None


@pytest.mark.gpu
def test_torch_cuda() -> None:
    import torch

    assert torch.cuda.is_available()
    assert "3050" in torch.cuda.get_device_name(0)
