from __future__ import annotations

import math
import time

import numpy as np
import pytest

from stocksense.validation.pbo import pbo_cscv


def test_hand_example_two_blocks():
    # 2 blocks of 2 rows, 2 trials.  Per-period Sharpe = mean / std(ddof=1):
    #   A: block0 [.02,.04] -> .03/.014142 = 2.1213 ; block1 [.01,.03] -> .02/.014142 = 1.4142
    #   B: block0 [.00,.02] -> .01/.014142 = 0.7071 ; block1 [.04,.06] -> .05/.014142 = 3.5355
    # IS = block0: winner A, OOS (block1) A 1.4142 < B 3.5355 -> A is worst, rank 1 -> w = 1/3
    # IS = block1: winner B, OOS (block0) B 0.7071 < A 2.1213 -> rank 1 -> w = 1/3
    # logit = ln((1/3)/(2/3)) = -0.693147 for both -> PBO = 1.0
    r = np.array([[0.02, 0.00], [0.04, 0.02], [0.01, 0.04], [0.03, 0.06]])
    out = pbo_cscv(r, n_blocks=2)
    assert out["pbo"] == pytest.approx(1.0)
    assert sorted(out["logits"]) == [pytest.approx(math.log(0.5)), pytest.approx(math.log(0.5))]


def test_noise_trials_give_pbo_near_half():
    rng = np.random.default_rng(11)
    out = pbo_cscv(rng.normal(0, 0.01, (1600, 50)), n_blocks=16)
    assert len(out["logits"]) == 12870
    assert 0.35 <= out["pbo"] <= 0.65


def test_real_edge_gives_low_pbo():
    rng = np.random.default_rng(12)
    r = rng.normal(0, 0.01, (3000, 50))
    r[:, 7] += 0.1 * 0.01  # daily Sharpe 0.1: t-stat about 3.9 per half vs noise maxima near 2.3
    assert pbo_cscv(r, n_blocks=16)["pbo"] < 0.10


def test_needs_two_trials():
    out = pbo_cscv(np.random.default_rng(1).normal(size=(160, 1)))
    assert math.isnan(out["pbo"])


def test_runtime_3000_by_50():
    r = np.random.default_rng(2).normal(0, 0.01, (3000, 50))
    t0 = time.perf_counter()
    pbo_cscv(r, n_blocks=16)
    assert time.perf_counter() - t0 < 20.0
