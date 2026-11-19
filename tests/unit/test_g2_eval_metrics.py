from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.validation import g2_eval as ev

N_D, N = 120, 80


def _frames(seed=0, signal_base=0.15, signal_model=0.15, shared=0.0):
    """Label y; base = a*y + noise; model = b*y + independent noise + shared * base."""
    rng = np.random.default_rng(seed)
    d0 = date(2019, 1, 7)
    dates = [d0 + timedelta(days=i) for i in range(N_D)]
    date_col = np.repeat(np.array(dates, dtype="datetime64[D]"), N)
    sid = np.tile(np.arange(N), N_D)
    y = rng.normal(size=N_D * N)
    base = signal_base * y + rng.normal(size=y.size)
    model = signal_model * y + rng.normal(size=y.size) + shared * base

    def mk(v):
        return pl.DataFrame({"date": date_col, "sid": sid, "score": v})

    lab = pl.DataFrame({"date": date_col, "sid": sid, "r_t": y})
    return mk(model), mk(base), lab


def test_horizon():
    assert ev.horizon("y_c2o_10") == 10 and ev.horizon("y_oc") == 1 and ev.horizon("y_o2o_5") == 5


def test_orth_ic_of_a_copy_of_the_base_is_zero_and_blend_gain_zero():
    _, base, lab = _frames()
    d = ev.daily_series(base, base, lab, "r_t")
    assert d.height == N_D
    assert np.abs(d["orth_ic"].to_numpy()).max() < 1e-9
    assert np.abs((d["ic_blend"] - d["ic_base"]).to_numpy()).max() < 1e-9
    assert np.allclose(d["corr_base"].to_numpy(), 1.0)
    m = ev.summarise(d, "y_c2o_10")
    assert abs(m["orth_ic"]) < 1e-9 and abs(m["blend_gain"]) < 1e-9
    assert m["corr_with_base"] == pytest.approx(1.0)
    assert m["mean_ic"] == pytest.approx(m["base_mean_ic"])
    assert not ev.admission(m)["admitted"]


def test_independent_signal_has_orth_ic_and_blend_gain():
    model, base, lab = _frames(signal_base=0.15, signal_model=0.15)
    m = ev.summarise(ev.daily_series(model, base, lab, "r_t"), "y_c2o_5")
    assert m["orth_ic"] > 0.05 and m["orth_ic_t"] > 5
    assert m["blend_gain"] > 0.01 and m["blend_gain_t"] > 3
    assert abs(m["corr_with_base"]) < 0.1
    assert set(m["ic_by_year"]) == {"2019"}
    assert ev.admission({**m, "mean_ic": 0.05})["admitted"]


def test_orth_ic_hand_check_on_one_date():
    model, base, lab = _frames(shared=1.0)
    d = ev.daily_series(model, base, lab, "r_t")
    day = d["date"][3]
    j = (
        model.rename({"score": "m"})
        .join(base.rename({"score": "b"}), on=["date", "sid"])
        .join(lab, on=["date", "sid"])
        .filter(pl.col("date") == day)
    )
    from scipy.stats import rankdata, spearmanr

    rm, rb = rankdata(j["m"].to_numpy()), rankdata(j["b"].to_numpy())
    A = np.column_stack([np.ones_like(rb), rb])
    coef, *_ = np.linalg.lstsq(A, rm, rcond=None)
    res = rm - A @ coef
    want = spearmanr(res, j["r_t"].to_numpy()).statistic
    row = d.filter(pl.col("date") == day)
    assert row["orth_ic"][0] == pytest.approx(want, abs=1e-9)
    ic = spearmanr(j["m"].to_numpy(), j["r_t"].to_numpy()).statistic
    assert row["ic"][0] == pytest.approx(ic, abs=1e-9)


def test_dates_with_fewer_than_50_names_are_skipped():
    model, base, lab = _frames()
    keep = pl.col("sid") < 49
    d = ev.daily_series(model.filter(keep), base.filter(keep), lab, "r_t")
    assert d.is_empty() and ev.summarise(d, "y_oc")["n_dates"] == 0


def test_admission_rule_branches():
    base = {"mean_ic": 0.05, "orth_ic": 0.0, "orth_ic_t": 0.0, "blend_gain": 0.0, "blend_gain_t": 0.0}
    assert not ev.admission(base)["admitted"]
    assert ev.admission({**base, "orth_ic": 0.006, "orth_ic_t": 2.0})["admitted"]
    assert not ev.admission({**base, "orth_ic": 0.006, "orth_ic_t": 1.9})["admitted"]
    assert ev.admission({**base, "blend_gain": 0.003, "blend_gain_t": 2.5})["admitted"]
    assert not ev.admission({**base, "mean_ic": 0.039, "orth_ic": 0.01, "orth_ic_t": 9})["admitted"]
    nan = float("nan")
    assert not ev.admission({**base, "orth_ic": nan, "orth_ic_t": nan})["admitted"]


def test_table_renders_rows():
    row = {
        "trial_id": "g2-x-y_c2o_10-abc", "family": "g2-x", "mean_ic": 0.05, "ic_tstat": 9.0,
        "corr_with_base": 0.4, "orth_ic": 0.01, "orth_ic_t": 2.5, "blend_gain": 0.004, "blend_gain_t": 2.1,
        "net_sharpe_s1": {"sharpe": 1.1, "net_2x": 0.7, "mean_turnover": 0.12}, "admitted": True,
    }  # fmt: skip
    txt = ev.table([row, {"trial_id": "g2-y", "family": "g2-y", "net_sharpe_s1": None}])
    assert "g2-x-y_c2o_10-abc" in txt and "YES" in txt and "+1.10" in txt and "no" in txt
