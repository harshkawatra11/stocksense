from __future__ import annotations

import pytest

from stocksense.models import base
from stocksense.models.base import MODEL_MODULES, Model, ModelSpec, build_model


def test_model_modules_listed():
    assert MODEL_MODULES == (
        "gbm", "linear", "seqnet", "master", "hist", "gats", "tra", "tabm", "ipca", "cae",
        "factorvae", "dens",
    )  # fmt: skip


def test_defaults_on_model_class():
    assert Model.needs_keys is False and Model.needs_cube is False


def test_unknown_key_is_keyerror():
    with pytest.raises(KeyError):
        build_model(ModelSpec(family="s1-nosuchfamily", target="y_c2o_10"))


def test_listed_but_unwritten_module_is_keyerror():
    with pytest.raises(KeyError):
        build_model(ModelSpec(family="g2-factorvae", target="y_c2o_10"))


def test_only_the_named_module_is_imported(monkeypatch):
    seen: list[str] = []
    real = base.importlib.import_module

    def spy(name, *a, **k):
        seen.append(name)
        return real(name, *a, **k)

    monkeypatch.setattr(base.importlib, "import_module", spy)
    build_model(ModelSpec(family="s1-ridge", target="y_c2o_10"))
    assert "stocksense.models.master" not in seen and "stocksense.models.seqnet" not in seen


def test_seqnet_needs_keys():
    pytest.importorskip("torch")
    from stocksense.models.seqnet import SeqNetModel

    assert SeqNetModel.needs_keys is True
