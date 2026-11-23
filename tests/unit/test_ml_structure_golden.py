"""Our numpy port of Market Lab's structure/SMC code against outputs recorded from Market Lab itself.

The golden files come from scripts/golden/run_marketlab.py (Market Lab's own functions on synthetic bars); the
default suite never runs Market Lab, it only compares with the committed files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from stocksense.marketlab import smc
from stocksense.marketlab import structure as S
from stocksense.marketlab.bars import Bars

FX = Path(__file__).resolve().parents[1] / "fixtures" / "marketlab"


def bars(name: str) -> Bars:
    b = json.loads((FX / f"inputs_{name}.json").read_text(encoding="utf-8"))
    return Bars.from_arrays(b["ts"], b["open"], b["high"], b["low"], b["close"], b["volume"])


def golden(name: str) -> Any:
    return json.loads((FX / f"golden_{name}.json").read_text(encoding="utf-8"))


def same(a: Any, b: Any, path: str = "") -> None:
    """Deep equality: exact except floats (1e-9, NaN equals NaN); key sets must be identical."""
    if isinstance(b, dict):
        assert isinstance(a, dict) and set(a) == set(b), f"{path}: keys {sorted(set(a) ^ set(b))}"
        for k in b:
            same(a[k], b[k], f"{path}.{k}")
    elif isinstance(b, list):
        assert isinstance(a, list) and len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            same(x, y, f"{path}[{i}]")
    elif isinstance(b, float) and not isinstance(b, bool):
        if math.isnan(b):
            assert isinstance(a, float) and math.isnan(a), f"{path}: {a} != NaN"
        else:
            assert a == pytest.approx(b, rel=1e-9, abs=1e-9), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.mark.parametrize(
    ("golden_name", "input_name", "n"),
    [("smc_5m_stock", "5m_stock", 2), ("smc_5m_index", "5m_index", 2), ("smc_1d_stock", "1d_stock", 3)],
)
def test_read_matches_market_lab(golden_name: str, input_name: str, n: int) -> None:
    same(smc.read(bars(input_name), n), golden(golden_name))


def test_read_with_daily_frame_matches_market_lab() -> None:
    same(smc.read(bars("5m_stock"), 2, daily=bars("1d_from5m")), golden("smc_5m_stock_daily"))


@pytest.mark.parametrize(
    ("name", "input_name"), [("5m_stock", "5m_stock"), ("1d_stock", "1d_stock"), ("5m_index", "5m_index")]
)
def test_analyse_matches_market_lab(name: str, input_name: str) -> None:
    b = bars(input_name)
    a = S.analyse(b, 3)
    g = golden(f"analyse_{name}")
    mine = {
        k: a[k] for k in ("bias", "last_event", "recent_swings", "zones", "regime", "price", "atr", "rsi")
    }
    mine["swings"] = [
        {k: s[k] for k in ("time", "kind", "price", "confirmed_at", "label")} for s in a["swings"]
    ]
    mine["events"] = a["events"]
    mine["df_atr_last"] = float(a["ind"]["atr"][-1])
    mine["df_rsi_last"] = float(a["ind"]["rsi"][-1])
    mine["df_ema200_last"] = float(a["ind"]["ema200"][-1])
    same(mine, g)


def test_session_levels_match_market_lab() -> None:
    same(S.session_levels(bars("15m_stock")), golden("session_levels_15m"))


@pytest.mark.parametrize("input_name", ["5m_stock", "1d_stock"])
def test_swings_are_confirmed_n_bars_later_and_never_used_early(input_name: str) -> None:
    b = bars(input_name)
    n = 3
    sw = S.label_swings(S.swings(b, n))
    assert sw
    for s in sw:
        assert s["ci"] == s["i"] + n and s["confirmed_at"] == int(b.ts[s["i"] + n])
    events, _ = S.structure_events(b, sw)
    assert events
    by_level = {(s["price"], s["kind"]): s for s in sw}
    for e in events:
        kind = "H" if e["dir"] == "bull" else "L"
        s = by_level[(e["level"], kind)]
        assert e["time"] >= s["confirmed_at"], "an event used a swing before it was confirmed"


def test_read_needs_thirty_bars() -> None:
    assert smc.read(bars("5m_stock").tail(29), 2) is None
