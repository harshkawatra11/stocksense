from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stocksense.marketlab.bars import Bars
from stocksense.strategies import exits as X
from stocksense.strategies import indicators as I  # noqa: E741
from stocksense.strategies.data import DailyFrame, NiftyTable

FX = Path(__file__).resolve().parents[1] / "fixtures" / "marketlab"
GOLD = json.loads((FX / "golden_features.json").read_text(encoding="utf-8"))


def raw(name: str) -> dict[str, Any]:
    return json.loads((FX / f"inputs_{name}.json").read_text(encoding="utf-8"))


def bars(name: str) -> Bars:
    b = raw(name)
    return Bars.from_arrays(b["ts"], b["open"], b["high"], b["low"], b["close"], b["volume"])


def daily(name: str) -> DailyFrame:
    b = raw(name)
    d = ((np.asarray(b["ts"], dtype=np.int64) + 19800) // 86400).astype("datetime64[D]")
    return DailyFrame(d, *(np.asarray(b[k], dtype=float) for k in ("open", "high", "low", "close", "volume")))


def digest(a: Any) -> dict[str, Any]:
    x = np.asarray(a).astype(float)
    ok = np.isfinite(x)
    idx = np.unique(np.linspace(0, len(x) - 1, 40).round().astype(int)) if len(x) else []
    return {
        "n": int(len(x)),
        "nan": int((~ok).sum()),
        "sum": float(x[ok].sum()),
        "abs": float(np.abs(x[ok]).sum()),
        "at": [None if not ok[i] else float(x[i]) for i in idx],
    }


def digest_all(f: dict[str, Any]) -> dict[str, Any]:
    return {k: digest(v) for k, v in f.items() if isinstance(v, np.ndarray) and v.dtype.kind in "fiub"}


def same(got: Any, want: Any, path: str = "") -> None:
    if isinstance(want, dict):
        assert set(got) == set(want), f"{path}: keys differ: {set(got) ^ set(want)}"
        for k in want:
            same(got[k], want[k], f"{path}.{k}")
    elif isinstance(want, list):
        assert len(got) == len(want), path
        for i, (a, b) in enumerate(zip(got, want, strict=True)):
            same(a, b, f"{path}[{i}]")
    elif want is None:
        assert got is None, path
    elif isinstance(want, (int, float)) and not isinstance(want, bool):
        assert got == pytest.approx(want, rel=1e-9, abs=1e-9), f"{path}: {got} != {want}"
    else:
        assert got == want, path


@pytest.fixture(scope="module")
def nifty() -> NiftyTable:
    d = daily("1d_nifty")
    return NiftyTable.from_close(d.date, d.close)


def test_daily_features_match_market_lab(nifty):
    f = I.daily_features(daily("1d_stock"), nifty)
    same(digest_all(f), GOLD["d1"])


def test_daily_exits_match_market_lab(nifty):
    f = I.daily_features(daily("1d_stock"), nifty)
    for ie, stop, mode, j, px in GOLD["follow_1d"]:
        gj, gpx, _ = X.follow_1d(f, ie, stop, mode)
        assert gj == j and gpx == pytest.approx(px, rel=1e-9), (ie, mode)


@pytest.fixture(scope="module")
def intraday():
    s5, ix5, dctx = bars("5m_stock"), bars("5m_index"), daily("1d_ctx")
    ixf = I.intraday_features(ix5, None, 5, None, 1)
    ixm = I.intraday_features(I.mirror(ix5), None, 5, None, -1)
    ctx, ctxm = I.index_context(ixf), I.index_context(ixm)
    mirrored = DailyFrame(dctx.date, -dctx.open, -dctx.low, -dctx.high, -dctx.close, dctx.volume)
    return {
        "ix": ixf,
        "ctx": ctx,
        "m5": I.intraday_features(s5, dctx, 5, ctx, 1),
        "m5_mirror": I.intraday_features(I.mirror(s5), mirrored, 5, ctxm, -1),
        "m5_nodaily": I.intraday_features(s5, None, 5, None, 1),
    }


@pytest.mark.parametrize("key", ["ix", "m5", "m5_mirror", "m5_nodaily"])
def test_intraday_features_match_market_lab(intraday, key):
    same(digest_all(intraday[key]), GOLD[key])


def test_index_context_matches_market_lab(intraday):
    same({k: digest(v) for k, v in intraday["ctx"].items()}, GOLD["ix_ctx"])


def test_windows_and_trend_masks_match_market_lab(intraday):
    f = intraday["m5"]
    entry_ok, exit_idx = X.entry_window(f)
    same({"entry_ok": digest(entry_ok), "exit_idx": digest(exit_idx)}, GOLD["windows"])
    same({k: digest(X.trend_mask(f, k)) for k in ("D", "I", "DI", "DIX")}, GOLD["trend"])


def test_intraday_exits_match_market_lab(intraday):
    f = intraday["m5"]
    _, exit_idx = X.entry_window(f)
    assert GOLD["follow_intra"]
    for ie, s, e, xe, want in GOLD["follow_intra"]:
        assert int(exit_idx[f["day"][ie]]) == xe
        got = X.follow_intra(f, ie, s, e, e - s, xe)
        same(got, want)


def test_features_do_not_look_ahead(nifty):
    """A feature value at bar i is unchanged when every later bar is removed (the truncation property)."""
    s5 = bars("5m_stock")
    cut = 1500
    part = Bars(s5.ts[:cut], s5.open[:cut], s5.high[:cut], s5.low[:cut], s5.close[:cut], s5.volume[:cut])
    full, short = I.intraday_features(s5, None, 5, None, 1), I.intraday_features(part, None, 5, None, 1)
    bad = [
        k
        for k, v in short.items()
        if isinstance(v, np.ndarray)
        and v.dtype.kind in "fiub"
        and len(v) == cut
        and k
        not in {"hod", "lod", "or15h", "or15l", "or30h", "or30l"}  # day-complete values are not point-in-time
        and not np.allclose(np.asarray(v, float), np.asarray(full[k][:cut], float), equal_nan=True)
    ]
    assert bad == []
    d = daily("1d_stock")
    dp = DailyFrame(*(getattr(d, k)[:400] for k in ("date", "open", "high", "low", "close", "volume")))
    nt = NiftyTable.from_close(dp.date, dp.close)
    a, b = I.daily_features(d, nifty), I.daily_features(dp, nt)
    for k, v in b.items():
        if (
            isinstance(v, np.ndarray)
            and v.dtype.kind in "fiub"
            and len(v) == 400
            and k not in {"mkt_up", "mkt_roc63", "rs63"}
        ):
            assert np.allclose(np.asarray(v, float), np.asarray(a[k][:400], float), equal_nan=True), k
