from __future__ import annotations

import copy
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock
from stocksense.data import io
from stocksense.marketlab import sectors as S
from stocksense.marketlab.sources import SourceError
from stocksense.marketlab.store import LabStore
from tests.unit.test_ml_structure_golden import same

FX = Path(__file__).resolve().parents[1] / "fixtures" / "marketlab"
INP = json.loads((FX / "inputs_sector.json").read_text(encoding="utf-8"))
GOLD = json.loads((FX / "golden_sector.json").read_text(encoding="utf-8"))
CAL = TradingCalendar.from_dates([date(2026, 11, 18), date(2026, 11, 19)])


class FakeNews:
    """The slice of NewsService the engine uses, serving the recorded headlines."""

    def sector(self, name: str, wait: bool = True) -> list[dict[str, Any]]:
        return copy.deepcopy(INP["news"][name])

    def stock(self, sym: str, company: str | None) -> list[dict[str, Any]]:
        return copy.deepcopy(INP["stock_news"])


class FakeSources:
    def nse_symbol_data(self, sym: str) -> dict[str, Any]:
        q = INP["quotes"].get(sym)
        if not q:
            raise SourceError("nse", "no quote")
        return dict(q)


def hist_of(close, high=None, volume=None, start: str = "2026-01-01") -> S.Hist:
    n = len(close)
    d = np.datetime64(start) + np.arange(n)
    c = np.asarray(close, dtype=float)
    return S.Hist(
        d,
        c,
        c + 1 if high is None else np.asarray(high, float),
        np.full(n, 10.0) if volume is None else np.asarray(volume, float),
    )


def build_engine(tmp_data_root) -> S.SectorEngine:
    store = LabStore()
    clock = FixedClock(datetime(2026, 11, 19, 10, 0, tzinfo=IST))
    dates = np.array(INP["dates"], dtype="datetime64[D]")
    hist = {
        sym: S.Hist(dates[-h["n"] :], np.array(h["close"]), np.array(h["high"]), np.array(h["volume"]))
        for sym, h in INP["hist"].items()
    }
    for name, h in INP["idx_hist"].items():
        c = np.array(h["close"])
        hist["IDX:" + name] = S.Hist(dates, c, c, np.zeros(len(c)))
    store.univ.update(list=copy.deepcopy(INP["list"]), cls=copy.deepcopy(INP["cls"]), hist=hist, progress="")
    store.sx.update(
        indices=copy.deepcopy(INP["indices"]),
        stocks=copy.deepcopy(INP["stocks"]),
        members=copy.deepcopy(INP["members"]),
        spark={},
        focus=[],
        focus_until=0.0,
        day={"snaps": copy.deepcopy(INP["snaps"])},
        time="2026-11-19T09:00:00+00:00",
        errors=0,
        mk=copy.deepcopy(INP["mk"]),
    )
    return S.SectorEngine(tmp_data_root, clock, CAL, FakeSources(), store, FakeNews())


@pytest.fixture
def eng(tmp_data_root):
    return build_engine(tmp_data_root)


# ---- golden: every payload equals Market Lab's own output on the same recorded state ------------------


@pytest.mark.parametrize("rng", ["today", "w1", "m1", "m3"])
def test_horizon_matches_market_lab(eng, rng):
    same(eng.horizon(rng), GOLD[f"horizon_{rng}"])


@pytest.mark.parametrize(
    "name,rng",
    [
        ("Energy", "today"),
        ("Pharma", "w1"),
        ("Bank", "m3"),
        ("Fin Services", "today"),
        ("Healthcare", "today"),
    ],
)
def test_sector_view_matches_market_lab(eng, name, rng):
    same(eng.sector_view(name, rng), GOLD[f"sector_{name.lower().replace(' ', '_')}_{rng}"])


@pytest.mark.parametrize("sym", ["ENG03", "BNK02", "ZZZ"])
def test_stock_view_matches_market_lab(eng, sym):
    same(eng.stock_view(sym), GOLD[f"stock_{sym.lower()}"])


def test_search_and_tick_match_market_lab(eng):
    same(eng.search("EN"), GOLD["search_en"])
    same(eng.search("bank"), GOLD["search_bank"])
    same(eng.tick(["ENG01", "ENG02", "NOPE"]), GOLD["tick"])


def test_unknown_sector_is_an_error_payload(eng):
    assert eng.sector_view("Nope", "today") == {"error": "unknown sector"}


# ---- pure helpers with hand numbers ------------------------------------------------------------------


def test_stats_from_daily_bars():
    h = hist_of(np.arange(1.0, 101.0))
    st = S.stats(h)
    assert st["w1"] == pytest.approx((100 / 95 - 1) * 100)
    assert st["m1"] == pytest.approx((100 / 79 - 1) * 100)
    assert st["m3"] == pytest.approx((100 / 37 - 1) * 100)
    assert st["dma50"] == pytest.approx(75.5)
    assert st["vol60"] == 10.0 and st["high52"] == 101.0 and st["last_close"] == 100.0
    assert S.stats(hist_of(np.arange(1.0, 30.0))) == {}
    assert S.stats(None) == {}


def test_stats_short_history_has_no_long_moves():
    st = S.stats(hist_of(np.arange(1.0, 41.0)))
    assert st["w1"] is not None and st["m1"] is not None and st["m3"] is None


def test_sector_series_is_the_median_rebased_to_100():
    k = np.arange(72.0)
    a, b = hist_of(100 * (1 + 0.01 * k)), hist_of(50 * (1 + 0.02 * k))
    s = S.sector_series({"A": a, "B": b}, ["A", "B"], days=70)
    ra = a.close[-70:] / a.close[-70] * 100
    rb = b.close[-70:] / b.close[-70] * 100
    assert len(s.values) == 70 and s.values[0] == pytest.approx(100.0)
    assert s.values[-1] == pytest.approx((ra[-1] + rb[-1]) / 2)
    assert S.sector_series({"A": a}, ["A", "Z"], days=72) is None  # needs more than `days` rows
    assert S.sector_series({}, ["A"]) is None


def test_ret_needs_more_than_n_points():
    s = S.Ser(np.arange(7).astype("datetime64[D]"), np.array([1.0, 2, 3, 4, 5, 6, 8]))
    assert S.ret(s, 5) == pytest.approx((8 / 2 - 1) * 100)
    assert S.ret(s, 6) == pytest.approx((8 / 1 - 1) * 100)
    assert S.ret(s, 7) is None and S.ret(None, 1) is None


def test_members_fixed_list_classification_and_not_basic():
    universe = {"IRCTC": {}, "SBIN": {}, "HDFCBANK": {}, "BAJFINANCE": {}, "X": {}}
    cls = {
        "SBIN": {"sector": "Financial Services", "industry": "Banks", "basic": "Public Sector Bank"},
        "HDFCBANK": {"sector": "Financial Services", "industry": "Banks", "basic": "Private Sector Bank"},
        "BAJFINANCE": {
            "sector": "Financial Services",
            "industry": "Finance",
            "basic": "Non Banking Financial Company (NBFC)",
        },
        "GHOST": {"sector": "Financial Services", "industry": "Finance", "basic": "Other"},
    }
    sec = {s["name"]: s for s in S.SECTORS}
    assert S.members(sec["Railways"], cls, universe) == [
        "IRCTC"
    ]  # fixed names present in the list or the classification
    assert S.members(sec["Fin Services"], cls, universe) == [
        "BAJFINANCE"
    ]  # banks excluded, GHOST not in the Nifty 500
    assert S.members(sec["Bank"], cls, universe) == ["SBIN", "HDFCBANK"]
    assert S.members(sec["PSU Bank"], cls, universe) == ["SBIN"]
    assert S.members(sec["Bank"], cls, universe, ["BAJFINANCE", "NOTLISTED"]) == [
        "BAJFINANCE",
        "SBIN",
        "HDFCBANK",
    ]


def test_pins_bucket_headlines_onto_the_last_chart_day_not_after_them():
    dates = ["2026-11-16", "2026-11-17", "2026-11-19"]
    items = [
        {"time": "2026-11-18T01:00:00+00:00", "title": "a", "source": "s", "link": "l", "tag": "RESULTS"},
        {"time": "2026-11-19T05:00:00+00:00", "title": "b", "source": "s", "link": "l", "tag": "NEWS"},
        {"time": "2026-11-10T05:00:00+00:00", "title": "too old", "source": "s", "link": "l", "tag": "NEWS"},
    ]
    out = S.pins(items, dates)
    assert list(out) == ["2026-11-17", "2026-11-19"]
    assert out["2026-11-17"][0]["tag"] == "R" and out["2026-11-19"][0]["tag"] == "N"


def test_pins_use_the_indian_calendar_day():
    # 20:00 UTC on the 18th is 01:30 IST on the 19th
    out = S.pins(
        [{"time": "2026-11-18T20:00:00+00:00", "title": "t", "source": "s", "link": "l", "tag": "POLICY"}],
        ["2026-11-18", "2026-11-19"],
    )
    assert list(out) == ["2026-11-19"]


def test_rel_chart_rebases_to_100_and_needs_five_days():
    d = np.datetime64("2026-01-01") + np.arange(60)
    a = S.Ser(d, np.arange(1.0, 61.0))
    b = S.Ser(d, np.arange(10.0, 70.0))
    c = S.rel_chart(a, b, [])
    assert len(c["dates"]) == 55 and c["a"][0] == 100.0 and c["b"][0] == 100.0
    assert c["a"][-1] == pytest.approx(60 / 6 * 100) and c["pins"] == {}
    assert S.rel_chart(S.Ser(d[-4:], np.arange(1.0, 5.0)), b, []) is None
    assert S.rel_chart(None, b, []) is None


def test_sector_move_today_uses_pct_and_week_prefers_nse(eng):
    sec = next(s for s in S.SECTORS if s["name"] == "Energy")
    ix = eng.store.sx["indices"]["NIFTY ENERGY"]
    assert eng.sector_move(sec, "today") == ix["pct"]
    ix["w1"] = 1.25
    assert eng.sector_move(sec, "w1") == 1.25
    ix["w1"] = None
    ser = eng.all_series()["Energy"]
    assert eng.sector_move(sec, "w1") == pytest.approx(S.ret(ser, 5))


def test_live_spark_overrides_the_nse_snapshot(eng):
    now = eng.clock.now().timestamp()
    eng.store.sx["spark"]["ENG01.NS"] = {"ltp": 110.0, "prev": 100.0, "time": now - 30, "ts": [], "close": []}
    eng.store.sx["stocks"]["ENG01"]["prev"] = 100.0
    p = eng.price("ENG01")
    assert p["ltp"] == 110.0 and p["pct"] == 10.0 and p["live"] is True
    eng.store.sx["spark"]["ENG01.NS"]["time"] = now - 500  # too old: back to NSE
    assert eng.price("ENG01")["ltp"] == INP["stocks"]["ENG01"]["ltp"]


def test_themes_need_five_stocks_and_a_median_above_nifty(eng):
    st = eng.store.sx["stocks"]
    for k, sym in enumerate(["ENG01", "ENG02", "ENG03", "ENG04", "ENG05", "ENG06"]):
        eng.store.univ["cls"][sym]["basic"] = "Testing"
        st[sym]["pct"] = float(k + 1)  # 1..6, median 3.5
    eng.store.sx["indices"]["NIFTY 50"]["pct"] = 0.5
    t = next(x for x in eng.themes() if x["name"] == "Testing")
    assert t["n"] == 6 and t["beat"] == 6 and t["pct"] == 3.5
    eng.store.univ["cls"]["ENG06"]["basic"] = "Other"  # now five: still counted
    assert any(x["name"] == "Testing" for x in eng.themes())
    eng.store.univ["cls"]["ENG05"]["basic"] = "Other"  # four: dropped
    assert not any(x["name"] == "Testing" for x in eng.themes())


def test_sector_view_tolerates_a_stock_without_history(eng):
    eng.store.univ["hist"].pop("ENG01")  # price and class remain, bars missing: above50 is None
    v = eng.sector_view("Energy", "today")
    assert v["inside"]["n"] == 15


# ---- history loading from our stored panel -------------------------------------------------------------


def _panel(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=["sid", "symbol", "date", "series", "adj_close", "adj_high", "adj_volume"],
        orient="row",
    )


def test_load_hist_reads_completed_adjusted_bars(tmp_data_root):
    d = [date(2026, 11, 16), date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 19)]
    rows = [(1, "AAA", x, "EQ", 10.0 + i, 11.0 + i, 100.0 + i) for i, x in enumerate(d)]
    rows += [(3, "AAA", date(2026, 11, 18), "BE", 1.0, 1.0, 1.0), (2, "ZZZ", d[0], "EQ", 5.0, 5.0, 5.0)]
    io.upsert(tmp_data_root, "panel_daily", _panel(rows))
    ix = pl.DataFrame({"index_name": ["NIFTY 50"] * 4, "date": d, "close": [100.0, 101.0, 102.0, 103.0]})
    io.upsert(tmp_data_root, "index_daily", ix)
    h = S.load_hist(tmp_data_root, ["AAA"], ["NIFTY 50"], end=date(2026, 11, 18), days=30)
    assert list(h) == ["AAA", "IDX:NIFTY 50"]  # ZZZ not asked for
    a = h["AAA"]
    assert a.dates[-1] == np.datetime64("2026-11-18") and len(a) == 3  # 19th is after `end`; BE row ignored
    assert a.close.tolist() == [10.0, 11.0, 12.0] and a.volume.tolist() == [100.0, 101.0, 102.0]
    assert h["IDX:NIFTY 50"].close.tolist() == [100.0, 101.0, 102.0]


# ---- background loops ---------------------------------------------------------------------------------


class NoWait:
    """A stop event whose waits return at once (never set unless `set()` is called)."""

    def __init__(self) -> None:
        self.flag = False

    def wait(self, timeout: float | None = None) -> bool:
        return self.flag

    def is_set(self) -> bool:
        return self.flag

    def set(self) -> None:
        self.flag = True


class LiveSources:
    def __init__(self) -> None:
        self.spark_args: list[list[str]] = []
        self.watched: list[str] = []
        self.symbols: list[str] = []
        self.fail_indices = False

    def nse_all_indices(self):
        if self.fail_indices:
            raise SourceError("nse", "down")
        return {"NIFTY 50": {"name": "NIFTY 50", "last": 24000.0, "pct": 0.5}}

    def nse_watch(self, index_name):
        self.watched.append(index_name)
        return {"AAA": {"sym": "AAA", "ltp": 10.0, "pct": 1.0, "ffmc": 5.0}}

    def yahoo_spark(self, ysyms):
        self.spark_args.append(list(ysyms))
        return {"^NSEI": {"ltp": 24100.0, "prev": 24000.0, "time": 1, "ts": [], "close": []}}

    def nse_symbol_data(self, sym):
        self.symbols.append(sym)
        return {"sector": "X", "fetched": 1_800_000_000.0, "name": sym}

    def nifty500_list(self):
        return {"AAA": {"name": "Aaa", "industry": "X"}, "BBB": {"name": "Bbb", "industry": "X"}}


@pytest.fixture
def live(tmp_data_root):
    eng = build_engine(tmp_data_root)
    eng.src = eng.slow_src = LiveSources()
    eng.stop = NoWait()
    return eng


def test_step_pulls_every_source_then_records_the_session_minute(live, tmp_data_root):
    from stocksense.core.paths import Paths

    assert live.step() == 5.0
    sx = live.store.sx
    assert sx["indices"]["NIFTY 50"]["last"] == 24000.0 and sx["stocks"]["AAA"]["ltp"] == 10.0
    assert len(live.src.watched) == 21 and len(sx["members"]) == 20  # NIFTY 500 plus one per sector index
    assert sx["spark"]["^NSEI"]["ltp"] == 24100.0
    assert [s["m"] for s in sx["day"]["snaps"]] == [45]  # 10:00 is 45 minutes after the open
    assert sx["day"]["snaps"][0]["nifty"] == 0.0 + round((24100.0 / 24000.0 - 1) * 100, 2)
    assert (Paths(tmp_data_root).lab / "days" / "2026-11-19.json").exists()
    assert sx["time"].endswith("+00:00")
    assert live.step() == 5.0 and len(sx["day"]["snaps"]) == 1  # nothing new within 5 minutes


def test_step_waits_longer_when_the_market_is_closed(live):
    live.clock = FixedClock(datetime(2026, 11, 19, 20, 0, tzinfo=IST))
    assert live.step() == 30.0


def test_step_skips_yahoo_while_throttled_and_survives_a_failing_source(live):
    live.store.slow_until["yahoo"] = live.clock.now().timestamp() + 100
    live.src.fail_indices = True
    old = dict(live.store.sx["indices"])
    live.step()
    assert live.src.spark_args == [] and live.store.sx["indices"] == old
    assert live.store.sx["errors"] >= 1


def test_spark_requests_the_focused_stocks_for_a_minute(live):
    live.focus(["ENG01", "NOT-IN-LIST"])
    live.step()
    asked = live.src.spark_args[0]
    assert "ENG01.NS" in asked and "NOT-IN-LIST.NS" not in asked and "^NSEI" in asked
    live.sx["focus_until"] = 0.0
    live.last["fast"] = 0.0
    live.step()
    assert "ENG01.NS" not in live.src.spark_args[1]


def test_classification_resumes_and_refreshes_weekly(live, tmp_data_root):
    live.univ["list"] = {"AAA": {}, "BBB": {}, "CCC": {}}
    live.build_classification()
    assert live.src.symbols == ["AAA", "BBB", "CCC"] and sorted(live.univ["cls"]) == ["AAA", "BBB", "CCC"]
    live.src.symbols.clear()
    live.build_classification()  # fetched at 1.8e9, after the fixed clock: nothing is stale
    assert live.src.symbols == []
    live.clock = FixedClock(datetime(2030, 1, 1, tzinfo=IST))  # more than 7 days later: everything is stale
    live.build_classification()
    assert live.src.symbols == ["AAA", "BBB", "CCC"]


def test_classification_stops_when_asked(live):
    live.univ["list"] = {"AAA": {}, "BBB": {}}
    live.stop.set()
    live.build_classification()
    assert live.src.symbols == []


def test_slow_step_loads_the_list_then_history_then_classification(live):
    order: list[str] = []
    live.load_history = lambda: order.append("history")
    live.build_classification = lambda: order.append("classification")
    assert live.slow_step() == 3600.0
    assert sorted(live.univ["list"]) == ["AAA", "BBB"] and order == ["history", "classification"]


def test_news_step_stores_the_market_headlines(live):
    class Warm(FakeNews):
        def warm(self, names, stop):
            assert len(names) == 20
            return [{"title": "m"}]

    live.news = Warm()
    assert live.news_step() == 600.0 and live.sx["mk"] == [{"title": "m"}]
