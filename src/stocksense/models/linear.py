"""Floors (PLAN T7.1): a single signed feature and a weighted ridge regression.

RidgeModel: NaN -> 0.0 (the cross-sectional median once features are rank-scaled), market-state ('mkt')
features dropped (a constant per date carries no cross-sectional information), closed-form solution
`solve(X'WX + alpha I, X'Wy)` without an intercept (features and the rank target are centred already),
alpha in {1, 10, 100, 1000} chosen by the mean daily rank IC on the inner-validation rows.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from stocksense.models.base import FitInfo, Model, ModelSpec, mean_daily_rank_ic, register_model

ALPHAS = (1.0, 10.0, 100.0, 1000.0)
DEFAULT_ALPHA = 100.0  # used only when no validation rows are supplied


class SingleFactor(Model):
    """score = sign * feature (a baseline with nothing to fit)."""

    def __init__(self, feature: str, sign: float = 1.0) -> None:
        self.feature = feature
        self.sign = float(sign)
        self._col: int | None = None
        self._n: int | None = None

    def configure(self, feature_names: Sequence[str], kinds: Sequence[str]) -> None:
        names = list(feature_names)
        if self.feature not in names:
            raise KeyError(f"feature {self.feature!r} not among the model's columns")
        self._col, self._n = names.index(self.feature), len(names)

    def _need_col(self) -> int:
        if self._col is None:
            raise RuntimeError("SingleFactor.configure(feature_names, kinds) was not called")
        return self._col

    def fit(self, X, y, w, dates, X_val=None, y_val=None, dates_val=None, seed: int = 0) -> FitInfo:  # type: ignore[no-untyped-def]
        self._need_col()
        return FitInfo(params={"feature": self.feature, "sign": self.sign})

    def refit_full(self, X, y, w, dates, fit_info: FitInfo, seed: int) -> None:  # type: ignore[no-untyped-def]
        self._need_col()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.sign * np.asarray(X[:, self._need_col()], dtype=float)

    def explain(self, X: np.ndarray, feature_names: Sequence[str]) -> np.ndarray | None:
        out = np.zeros((X.shape[0], X.shape[1]))
        out[:, self._need_col()] = self.predict(X)
        return out

    def save(self, dir: Path) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        (dir / "weights.json").write_text(
            json.dumps({"feature": self.feature, "sign": self.sign, "col": self._col, "n": self._n}),
            encoding="utf-8",
        )

    def load(self, dir: Path) -> None:
        d = json.loads((dir / "weights.json").read_text(encoding="utf-8"))
        self.feature, self.sign, self._col, self._n = d["feature"], d["sign"], d["col"], d["n"]


class RidgeModel(Model):
    """Weighted closed-form ridge on the cross-sectional features (NaN -> 0, mkt features excluded)."""

    def __init__(self, alphas: Sequence[float] = ALPHAS) -> None:
        self.alphas = tuple(float(a) for a in alphas)
        self._use: np.ndarray | None = None  # boolean mask over the columns the model sees
        self._coef: np.ndarray | None = None  # full width, 0 for excluded columns
        self.alpha: float | None = None

    def configure(self, feature_names: Sequence[str], kinds: Sequence[str]) -> None:
        self._use = np.array([k == "xs" for k in kinds], dtype=bool)

    def _prep(self, X: np.ndarray) -> np.ndarray:
        Z = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if self._use is not None:
            Z = Z * self._use[None, :]
        return Z

    @staticmethod
    def _solve(Z: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float) -> np.ndarray:
        zw = Z * w[:, None]
        a = Z.T @ zw + alpha * np.eye(Z.shape[1])
        return np.linalg.solve(a, zw.T @ y)

    def fit(self, X, y, w, dates, X_val=None, y_val=None, dates_val=None, seed: int = 0) -> FitInfo:  # type: ignore[no-untyped-def]
        Z, yy, ww = self._prep(X), np.asarray(y, dtype=np.float64), np.asarray(w, dtype=np.float64)
        if X_val is None or y_val is None or dates_val is None:
            self.alpha = DEFAULT_ALPHA
            self._coef = self._solve(Z, yy, ww, self.alpha)
            return FitInfo(params={"alpha": self.alpha})
        Zv = self._prep(X_val)
        best_a, best_ic, by_alpha = self.alphas[0], -np.inf, {}
        for a in self.alphas:
            ic = mean_daily_rank_ic(Zv @ self._solve(Z, yy, ww, a), y_val, dates_val)
            by_alpha[a] = ic
            if np.isfinite(ic) and ic > best_ic:
                best_a, best_ic = a, ic
        self.alpha = best_a
        self._coef = self._solve(Z, yy, ww, best_a)
        return FitInfo(val_ic=float(best_ic), params={"alpha": best_a, "ic_by_alpha": by_alpha})

    def refit_full(self, X, y, w, dates, fit_info: FitInfo, seed: int) -> None:  # type: ignore[no-untyped-def]
        self.alpha = float(fit_info.params["alpha"])
        self._coef = self._solve(
            self._prep(X), np.asarray(y, dtype=np.float64), np.asarray(w, dtype=np.float64), self.alpha
        )

    def _need_coef(self) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("RidgeModel is not fitted")
        return self._coef

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._prep(X) @ self._need_coef()

    def explain(self, X: np.ndarray, feature_names: Sequence[str]) -> np.ndarray | None:
        return self._prep(X) * self._need_coef()[None, :]

    def save(self, dir: Path) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        use = np.ones_like(self._need_coef(), dtype=bool) if self._use is None else self._use
        np.savez(dir / "weights.npz", coef=self._need_coef(), use=use, alpha=float(self.alpha or 0.0))

    def load(self, dir: Path) -> None:
        z = np.load(dir / "weights.npz")
        self._coef, self._use, self.alpha = z["coef"], z["use"].astype(bool), float(z["alpha"])


def _ridge_factory(spec: ModelSpec) -> Model:
    alphas = spec.params.get("alphas", ALPHAS)
    return RidgeModel(alphas=alphas)


def _single_factory(spec: ModelSpec) -> Model:
    return SingleFactor(spec.params["feature"], spec.params.get("sign", 1.0))


register_model("ridge", _ridge_factory)
register_model("single", _single_factory)
