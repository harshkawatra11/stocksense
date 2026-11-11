"""GbmModel that keeps what the T7.2 diagnostics need: after each refit, the gain importance by feature and
the booster itself, keyed by the last training date and the seed (the runner does not hand a model its fold
name, and the last training date identifies the fold)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from stocksense.models.base import FitInfo
from stocksense.models.gbm import GbmModel


class DiagGbm(GbmModel):
    def __init__(self, out_dir: Path, **kw: Any) -> None:
        super().__init__(**kw)
        self.out_dir = Path(out_dir)
        self._names: list[str] = []

    def configure(self, feature_names: Sequence[str], kinds: Sequence[str]) -> None:
        self._names = list(feature_names)

    def refit_full(self, X, y, w, dates, fit_info: FitInfo, seed: int) -> None:  # type: ignore[no-untyped-def]
        super().refit_full(X, y, w, dates, fit_info, seed)
        tag = f"{np.max(np.asarray(dates)).astype('datetime64[D]')}_s{int(seed)}"
        booster = self._need()
        gain = booster.feature_importance("gain")
        names = self._names or [f"f{i}" for i in range(len(gain))]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "seed": int(seed), "best_iteration": fit_info.best_iteration, "val_ic": fit_info.val_ic,
            "n_trees": booster.num_trees(), "gain": {n: float(g) for n, g in zip(names, gain, strict=True)},
        }  # fmt: skip
        (self.out_dir / f"gain_{tag}.json").write_text(json.dumps(meta), encoding="utf-8")
        booster.save_model(str(self.out_dir / f"booster_{tag}.txt"))
