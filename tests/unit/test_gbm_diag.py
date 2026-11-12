from __future__ import annotations

import json

import numpy as np

from stocksense.models.gbm_diag import DiagGbm


def test_refit_writes_gain_and_booster_keyed_by_last_training_date(tmp_path):
    rng = np.random.default_rng(0)
    n_dates, n = 40, 30
    dates = np.repeat(np.arange("2019-01-01", "2019-02-10", dtype="datetime64[D]")[:n_dates], n)
    X = rng.normal(size=(len(dates), 3)).astype(np.float32)
    y = X[:, 1] + 0.1 * rng.normal(size=len(dates))
    m = DiagGbm(tmp_path, n_estimators=20, num_leaves=7, min_child_samples=20)
    m.configure(["a", "b", "c"], ["xs", "xs", "xs"])
    info = m.fit(X, y, np.ones(len(y)), dates, X, y, dates, seed=1)
    m.refit_full(X, y, np.ones(len(y)), dates, info, seed=1)
    meta = json.loads((tmp_path / "gain_2019-02-09_s1.json").read_text())
    assert meta["gain"]["b"] > meta["gain"]["a"] and set(meta["gain"]) == {"a", "b", "c"}
    assert (tmp_path / "booster_2019-02-09_s1.txt").is_file()
    assert m.predict(X).shape == (len(y),)
