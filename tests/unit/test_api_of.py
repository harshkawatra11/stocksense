from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from stocksense.api.app import create_app
from stocksense.core.clock import IST, FixedClock
from stocksense.data import io
from stocksense.marketlab.bars import Bars
from stocksense.marketlab.sources import SourceError
from stocksense.orderflow import engine as eng_mod
from stocksense.orderflow.engine import OrderflowEngine

TODAY = date(2026, 11, 12)  # Thursday
PAST = date(2026, 11, 11)
META = {"method", "data_source", "estimated"}


def _day_frame(d: date, seed: int, spike_at: int | None = None) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime.combine(d, time(9, 15), IST)
    close = 100 + np.cumsum(rng.normal(0, 0.1, 375))
    vol = rng.integers(900, 1100, 375).astype(float)
    if spike_at is not None:
        vol[spike_at] *= 20
    return pl.DataFrame(
        {
            "symbol": ["AAA"] * 375,
            "ts": [start + timedelta(minutes=i) for i in range(375)],
            "open": np.r_[100.0, close[:-1]],
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": vol,
        }
    ).with_columns(pl.col("ts").dt.cast_time_unit("us"))


class FakeStore:
    """Replaces load_bars: 1m bars for weekdays in range; optional empty 1m to force the 5m fallback."""

    def __init__(self, one_min=True):
        self.one_min = one_min
        self.calls = []

    def __call__(self, settings, symbols, start, end, interval="5m", basis="adjusted"):
        self.calls.append((start, end, interval))
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        days = [d for d in days if d.weekday() < 5]
        if interval == "1m" and not self.one_min:
            return pl.DataFrame()
        frames = []
        for d in days:
            f = _day_frame(d, d.toordinal(), spike_at=100 if d == PAST else None)
            if interval == "5m":
                f = f.filter(pl.col("ts").dt.minute() % 5 == 0)
            frames.append(f)
        return pl.concat(frames) if frames else pl.DataFrame()


def _yahoo_result(d: date):
    f = _day_frame(d, 99)
    return {
        "timestamp": [int(t.timestamp()) for t in f["ts"]],
        "indicators": {
            "quote": [{k: f[k].to_list() for k in ("open", "high", "low", "close", "volume")}],
        },
        "meta": {},
    }


class FakeSources:
    def __init__(self, fail=False):
        self.fail = fail

    def yahoo_chart(self, ysym, interval, rng):
        if self.fail:
            raise SourceError("yahoo", "down")
        return _yahoo_result(TODAY)


def _client(tmp_data_root, tmp_path, monkeypatch, sources=None, angel=None, store=None, engine=True):
    monkeypatch.setattr(eng_mod, "load_bars", store or FakeStore())
    clock = FixedClock(datetime.combine(TODAY, time(16, 0), IST))
    eng = OrderflowEngine(tmp_data_root, clock, sources, angel) if engine else None
    if eng is not None:
        eng.daily = lambda sym, end: Bars.from_arrays(  # type: ignore[method-assign]
            np.arange(60) * 86400,
            100 + np.arange(60),
            101 + np.arange(60),
            99 + np.arange(60),
            100.5 + np.arange(60),
            np.full(60, 1e6),
        )
    app = create_app(tmp_data_root, start_feeds=False, clock=clock, web_out=tmp_path / "none", orderflow=eng)
    return TestClient(app)


@pytest.fixture
def client(tmp_data_root, tmp_path, monkeypatch):
    with _client(tmp_data_root, tmp_path, monkeypatch) as c:
        yield c


def test_every_route_carries_method_source_and_estimated(client, web_fixture):
    d = PAST.isoformat()
    for name, url in [
        ("of_profile", f"/api/of/profile?sym=aaa&date={d}"),
        ("of_profile_tpo", f"/api/of/profile?sym=AAA&date={d}&kind=tpo"),
        ("of_delta", f"/api/of/delta?sym=AAA&date={d}&method=clv"),
        ("of_footprint", f"/api/of/footprint?sym=AAA&date={d}&tf=5"),
        ("of_bigtrades", f"/api/of/bigtrades?sym=AAA&date={d}"),
        ("of_timescales", "/api/of/timescales?sym=AAA"),
        ("of_bars", f"/api/of/bars?sym=AAA&date={d}&tf=5&vwap=true&bands=true"),
    ]:
        r = client.get(url)
        assert r.status_code == 200, (url, r.text)
        j = r.json()
        web_fixture(name, j)
        assert set(j) >= META and j["estimated"] is True, url


def test_profile_delta_and_bars_content(client):
    d = PAST.isoformat()
    p = client.get(f"/api/of/profile?sym=AAA&date={d}").json()
    assert p["data_source"] == "stored_1m" and p["kind"] == "volume" and p["val"] <= p["poc"] <= p["vah"]
    t = client.get(f"/api/of/profile?sym=AAA&date={d}&kind=tpo").json()
    assert t["kind"] == "tpo" and "letters" in t
    dl = client.get(f"/api/of/delta?sym=AAA&date={d}").json()
    assert len(dl["delta"]) == len(dl["cvd"]) == 375 and dl["method"] == "bulk_volume_classification"
    b = client.get(f"/api/of/bars?sym=AAA&date={d}&tf=5&vwap=true&bands=true&anchor=0").json()
    assert len(b["candles"]) == 75 and len(b["vwap"]) == 75 and {"u1", "l2"} <= set(b["bands"])
    assert len(b["anchored"]) == 75
    assert client.get(f"/api/of/bars?sym=AAA&date={d}&tf=3").status_code == 400


def test_bigtrades_flags_the_planted_spike(client):
    j = client.get(f"/api/of/bigtrades?sym=AAA&date={PAST.isoformat()}").json()
    assert j["history_bars"] > 0 and [t["i"] for t in j["trades"]] == [100]


def test_today_uses_yahoo_then_angel_then_stored(tmp_data_root, tmp_path, monkeypatch):
    url = f"/api/of/delta?sym=AAA&date={TODAY.isoformat()}"
    with _client(tmp_data_root, tmp_path, monkeypatch, sources=FakeSources()) as c:
        assert c.get(url).json()["data_source"] == "yahoo_1m"
    calls = []

    def angel(sym, a, b):
        calls.append(sym)
        return _day_frame(TODAY, 5)

    with _client(tmp_data_root, tmp_path, monkeypatch, sources=FakeSources(fail=True), angel=angel) as c:
        assert c.get(url).json()["data_source"] == "angel_1m" and calls == ["AAA"]
    with _client(tmp_data_root, tmp_path, monkeypatch, sources=FakeSources(fail=True)) as c:
        assert c.get(url).json()["data_source"] == "stored_1m"


def test_falls_back_to_5m_and_404_when_nothing(tmp_data_root, tmp_path, monkeypatch):
    with _client(tmp_data_root, tmp_path, monkeypatch, store=FakeStore(one_min=False)) as c:
        j = c.get(f"/api/of/profile?sym=AAA&date={PAST.isoformat()}").json()
        assert j["data_source"] == "stored_5m"

    def none(*a, **k):
        return pl.DataFrame()

    with _client(tmp_data_root, tmp_path, monkeypatch, store=none) as c:
        r = c.get(f"/api/of/delta?sym=AAA&date={PAST.isoformat()}")
        assert r.status_code == 404 and "no intraday bars" in r.json()["error"]


def test_heatmap_empty_then_filled(client, tmp_data_root):
    url = f"/api/of/heatmap?sym=AAA&date={PAST.isoformat()}"
    j = client.get(url).json()
    assert j["empty"] is True and j["reason"] == "depth not captured for this date" and set(j) >= META
    t0 = datetime.combine(PAST, time(10, 0), IST)
    rows = [
        {
            "symbol": "AAA",
            "ts": t0 + timedelta(seconds=30 * k),
            "side": s,
            "level": lv,
            "price": 100.0 + off,
            "qty": 50.0 + k,
        }
        for k in range(3)
        for s, off in (("bid", -0.1), ("ask", 0.1))
        for lv in range(2)
    ]
    df = pl.DataFrame(rows).with_columns(pl.col("level").cast(pl.Int32))
    io.upsert(tmp_data_root, "depth_snaps", df)
    j = client.get(url).json()
    assert j["empty"] is False and len(j["times"]) == 3 and len(j["qty"]) == 3


def test_503_without_engine_and_get_only(tmp_data_root, tmp_path, monkeypatch):
    with _client(tmp_data_root, tmp_path, monkeypatch, engine=False) as c:
        assert c.get("/api/of/profile?sym=AAA").status_code == 503
        assert c.get("/api/of/heatmap?sym=AAA").status_code == 503
    with _client(tmp_data_root, tmp_path, monkeypatch) as c:
        for m in (c.post, c.put, c.patch, c.delete):
            assert m("/api/of/profile?sym=AAA").status_code == 405
        assert c.get("/api/of/profile").status_code == 422
        assert c.get("/api/of/profile?sym=A%20B").status_code == 422
