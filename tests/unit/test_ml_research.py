from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stocksense.api.app import create_app
from stocksense.core.clock import IST, SteppingClock
from stocksense.marketlab import research as R
from stocksense.marketlab.sources import SourceError
from tests.unit.test_ml_sectors import build_engine
from tests.unit.test_ml_structure_golden import same

FX = Path(__file__).resolve().parents[1] / "fixtures" / "marketlab"
RIN = json.loads((FX / "inputs_research.json").read_text(encoding="utf-8"))
RGOLD = json.loads((FX / "golden_research.json").read_text(encoding="utf-8"))


class FakeSources:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.chart_closes: dict[str, list[float | None]] = {}

    def _n(self, k: str) -> None:
        self.calls[k] = self.calls.get(k, 0) + 1

    def nse_symbol_data(self, sym: str) -> dict[str, Any]:
        self._n("quote")
        if sym not in RIN["quotes"]:
            raise SourceError("nse", "no quote")
        return dict(RIN["quotes"][sym])

    def nse_integrated_filing(self, sym: str) -> list[dict[str, Any]]:
        self._n("filing")
        return copy.deepcopy(RIN["filings"][sym])

    def nse_results_comparison(self, sym: str) -> dict[str, Any]:
        self._n("cmp")
        if RIN["cmp"][sym] is None:
            raise SourceError("nse", "no filing")
        return copy.deepcopy(RIN["cmp"][sym])

    def wiki_summary(self, name: str) -> dict[str, Any] | None:
        self._n("wiki")
        return RIN["wiki"].get(name)

    def yahoo_chart(self, ysym: str, interval: str, rng: str) -> dict[str, Any]:
        self._n("chart")
        if ysym not in self.chart_closes:
            raise SourceError("yahoo", "down")
        return {"indicators": {"quote": [{"close": self.chart_closes[ysym]}]}}


@pytest.fixture
def rig(tmp_data_root):
    sec = build_engine(tmp_data_root)
    clock = SteppingClock(datetime(2026, 11, 19, 10, 0, tzinfo=IST))
    src = FakeSources()
    eng = R.ResearchEngine(tmp_data_root, clock, src, sec.store)
    eng.macro = lambda: copy.deepcopy(RIN["macro"])  # the golden fixes the economy readings
    return eng, src, clock


@pytest.mark.parametrize("sym", ["ENG03", "BNK02", "ENG14"])
def test_note_matches_market_lab(rig, sym):
    eng, _, _ = rig
    got = eng.build(sym)
    got.pop("built")
    same(got, RGOLD[sym])


def test_lin_maps_and_clips():
    assert R.lin(5, 0, 10) == 5.0
    assert R.lin(-1, 0, 10) == 0
    assert R.lin(11, 0, 10) == 10
    assert R.lin(None, 0, 1) is None


@pytest.mark.parametrize(
    "total,grade",
    [
        (75, "Strong"),
        (74, "Good"),
        (62, "Good"),
        (61, "Average"),
        (48, "Average"),
        (47, "Weak"),
        (35, "Weak"),
        (34, "Poor"),
    ],
)
def test_grade_bands(total, grade):
    assert R.grade_of(total) == grade
    assert R.grade_of(None) is None


@pytest.mark.parametrize(
    "pe,spe,g,expected",
    [
        (10, 20, None, 9),
        (15, 20, None, 7.5),
        (20, 20, None, 6),
        (28, 20, 0, 4.5),
        (30, 20, None, 3),
        (45, 20, None, 1.5),
        (30, 20, 30, 4.5),
        (10, 20, 30, 10),
        (None, 20, None, None),
        (0, 20, None, None),
    ],
)
def test_valuation_score(pe, spe, g, expected):
    assert R.valuation_score(pe, spe, g, None) == expected


def test_valuation_for_a_loss_maker_is_two():
    assert R.valuation_score(None, 20, None, -5.0) == 2


def test_quarters_prefer_consolidated_with_a_profit_and_convert_lakh_to_crore():
    rows = [
        {
            "gfrQuaterEnded": "30 Sep 2026",
            "gfrConsolidated": "Standalone",
            "gfrTotalIncome": "1,000,000",
            "gfrProBefTax": "200000",
            "gfrNetProLoss": "150000",
            "gfrErnPerShare": "2",
        },
        {
            "gfrQuaterEnded": "30 Sep 2026",
            "gfrConsolidated": "Consolidated",
            "gfrTotalIncome": "2000000",
            "gfrProBefTax": "300000",
            "gfrNetProLoss": "250000",
            "gfrErnPerShare": "3",
        },
        {
            "gfrQuaterEnded": "30 Jun 2026",
            "gfrConsolidated": "Standalone",
            "gfrTotalIncome": "500000",
            "gfrProBefTax": "1",
            "gfrNetProLoss": "100000",
        },
        {
            "gfrQuaterEnded": "30 Jun 2026",
            "gfrConsolidated": "Consolidated",
            "gfrTotalIncome": "900000",
            "gfrNetProLoss": None,
        },
        {"gfrQuaterEnded": None},
        {"gfrQuaterEnded": "garbage"},
    ]
    q = R.parse_quarters(rows)
    assert [x["date"] for x in q] == ["2026-06-30", "2026-09-30"]
    assert q[1]["kind"] == "Consolidated" and q[1]["income"] == 20000.0 and q[1]["profit"] == 2500.0
    assert q[1]["label"] == "Sep 26" and q[0]["kind"] == "Standalone" and q[0]["income"] == 5000.0


def test_quarters_keep_only_the_last_eight():
    rows = [
        {
            "gfrQuaterEnded": f"30 Sep {y}",
            "gfrConsolidated": "Standalone",
            "gfrTotalIncome": "100",
            "gfrNetProLoss": "10",
        }
        for y in range(2010, 2026)
    ]
    assert len(R.parse_quarters(rows)) == 8


def test_macro_three_month_change_from_yahoo_closes(rig):
    eng, src, _ = rig
    del eng.macro  # use the real method
    closes: list[float | None] = [None] + [100.0 + k for k in range(70)] + [0.0]
    src.chart_closes = {"BZ=F": closes}
    out = eng.macro()
    kept = [x for x in closes if x]
    assert out["oil"]["now"] == kept[-1] and out["oil"]["chg3m"] == round((kept[-1] / kept[-64] - 1) * 100, 1)
    assert set(out) == {"oil"}  # the other three feeds failed and are skipped
    assert eng.macro() is out and src.calls["chart"] == 4  # cached for six hours


def test_report_caches_twelve_hours_and_fresh_rebuilds(rig):
    eng, src, clock = rig
    first = eng.report("ENG03")
    eng.report("ENG03")
    assert src.calls["quote"] == 1
    clock.advance(timedelta(hours=11))
    eng.report("ENG03")
    assert src.calls["quote"] == 1
    clock.advance(timedelta(hours=2))
    assert eng.report("ENG03")["built"] > first["built"] and src.calls["quote"] == 2
    eng.report("ENG03", fresh=True)
    assert src.calls["quote"] == 3


def test_filing_failure_leaves_an_empty_results_section(rig):
    eng, src, _ = rig

    def down(sym: str) -> list[dict[str, Any]]:
        raise SourceError("nse", "down")

    src.nse_integrated_filing = down
    r = eng.build("ENG03")
    assert r["results"] == [] and r["ttm"]["income"] is None and r["swot"]["Strengths"]


def test_tech_needs_more_than_260_sessions(rig):
    eng, _, _ = rig
    assert eng.build("ENG14")["tech"] == {}  # 45 sessions
    t = eng.build("ENG03")["tech"]
    assert set(t) == {"uptrend", "rs63", "from_high", "atr_pct", "ret1y", "ret3m"}


def test_route_serves_the_note_and_refuses_indices(tmp_data_root, tmp_path, web_fixture):
    sec = build_engine(tmp_data_root)
    eng = R.ResearchEngine(tmp_data_root, sec.clock, FakeSources(), sec.store)
    eng.macro = lambda: copy.deepcopy(RIN["macro"])
    app = create_app(
        tmp_data_root,
        start_feeds=False,
        clock=sec.clock,
        store=sec.store,
        sector=sec,
        research=eng,
        web_out=tmp_path / "n",
    )
    with TestClient(app) as c:
        r = c.get("/api/research?sym=eng03")
        assert r.status_code == 200 and r.json()["sym"] == "ENG03"
        web_fixture("research", r.json())
        assert c.get("/api/research?sym=NIFTY").json() == {"error": "indices have no company report"}
        assert c.get("/api/research?sym=NOSUCH").status_code == 502
        assert c.get("/api/research").json() == {"error": "indices have no company report"}


def test_no_engine_gives_503(tmp_data_root, fixed_clock, tmp_path):
    app = create_app(tmp_data_root, start_feeds=False, clock=fixed_clock, web_out=tmp_path / "n")
    with TestClient(app) as c:
        assert c.get("/api/research?sym=TCS").status_code == 503


def test_numpy_scalars_do_not_leak_into_json():
    out = R.plain({"a": np.float64(1.5), "b": [np.int64(2), np.bool_(True)]})
    assert json.dumps(out) == '{"a": 1.5, "b": [2, true]}'
