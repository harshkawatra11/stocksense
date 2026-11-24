from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stocksense.strategies import base
from stocksense.strategies import indicators as I  # noqa: E741
from stocksense.strategies import intraday as T
from stocksense.strategies import swing as S
from stocksense.strategies.data import DailyFrame, NiftyTable
from tests.unit.test_strat_indicators_golden import bars, daily

FX = Path(__file__).resolve().parents[1] / "fixtures" / "marketlab"
GOLD = json.loads((FX / "golden_signals.json").read_text(encoding="utf-8"))


def rows(fn, f):
    sig, stop = fn(f)
    idx = np.flatnonzero(np.asarray(sig, bool))
    st = np.asarray(stop, float)[idx]
    return [int(i) for i in idx], [None if not np.isfinite(x) else float(x) for x in st]


def check(got, want, label):
    gi, gs = got
    assert gi == want["idx"], label
    for a, b in zip(gs, want["stop"], strict=True):
        assert (a is None and b is None) or a == pytest.approx(b, rel=1e-9, abs=1e-9), label


@pytest.fixture(scope="module")
def f1():
    d = daily("1d_nifty")
    return I.daily_features(daily("1d_stock"), NiftyTable.from_close(d.date, d.close))


@pytest.fixture(scope="module")
def f5():
    s5, ix5, dctx = bars("5m_stock"), bars("5m_index"), daily("1d_ctx")
    ctx = I.index_context(I.intraday_features(ix5, None, 5, None, 1))
    ctxm = I.index_context(I.intraday_features(I.mirror(ix5), None, 5, None, -1))
    mir = DailyFrame(dctx.date, -dctx.open, -dctx.low, -dctx.high, -dctx.close, dctx.volume)
    return (
        I.intraday_features(s5, dctx, 5, ctx, 1),
        I.intraday_features(I.mirror(s5), mir, 5, ctxm, -1),
    )


@pytest.mark.parametrize("name", list(GOLD["swing"]))
def test_swing_signals_match_market_lab(f1, name):
    check(rows(S.STRATS[name][0], f1), GOLD["swing"][name], name)


@pytest.mark.parametrize("name", list(GOLD["intra"]))
def test_intraday_signals_match_market_lab(f5, name):
    check(rows(T.STRATEGIES[name], f5[0]), GOLD["intra"][name], name)
    check(rows(T.STRATEGIES[name], f5[1]), GOLD["intra_mirror"][name], name + " (mirror)")


def test_counts_and_kinds():
    assert len(S.SWING) == 23 and len(T.INTRADAY) == 39
    assert set(GOLD["swing"]) == set(S.STRATS) and set(GOLD["intra"]) == set(T.STRATEGIES)
    assert {n: k for n, (_, k) in S.STRATS.items()} == GOLD["swing_kinds"]


def test_registry_entries():
    reg = base.load_library()
    sw = [s for s in reg.values() if s.source == "marketlab_swing"]
    it = [s for s in reg.values() if s.source == "marketlab_intraday"]
    assert len(sw) == 23 and len(it) == 39
    assert all(s.family == "m-swing" and s.timeframe == "1d" and s.direction == "long" for s in sw)
    assert all(s.family == "m-intra" and s.direction == "both" and s.default_exit == "rr2" for s in it)
    rsi = reg["ml_swing_rsi2"]
    assert rsi.kind == "mr" and "mr5" in rsi.exits and rsi.default_exit == "mr5"
    vcp = reg["ml_swing_vcp"]
    assert "mr5" not in vcp.exits and vcp.default_exit == "rr3_40" and vcp.forming is not None
    assert reg["ml_swing_breakout20"].default_exit == "trail3"
    assert base.by_name_or_key("VCP tightening break") is vcp
    assert reg["ml_intra_orb_15m"].name == "ORB 15m"
    assert reg["ml_intra_bull_flag"].name == "Bull flag (5m)"  # a swing strategy owns the plain name


def test_trend_filter_trials_are_117():
    assert len(T.INTRADAY) * len(T.TRENDS) == 117


def test_signals_do_not_look_ahead(f1):
    """A swing signal at day i is unchanged when later days are removed."""
    d, n = daily("1d_nifty"), daily("1d_stock")
    cut = 450
    part = DailyFrame(n.date[:cut], n.open[:cut], n.high[:cut], n.low[:cut], n.close[:cut], n.volume[:cut])
    short = I.daily_features(part, NiftyTable.from_close(d.date, d.close))
    for name, (fn, _) in S.STRATS.items():
        a, b = fn(short)[0], fn(f1)[0][:cut]
        assert np.array_equal(a, b), name


def test_vcp_forming_state(f1):
    st = S.setup_state(f1)
    assert st["state"] in ("FORMING", "NONE") and "close" in st and "mkt_up" in st
