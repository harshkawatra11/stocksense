from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from stocksense.core.clock import IST, FixedClock
from stocksense.core.config import Paths, Settings
from stocksense.data import io
from stocksense.strategies import base, scan
from stocksense.strategies.profile import stock_profile

N_SESS = 420
N_SYM = 8
ASOF_IDX = 380


def sessions() -> list[date]:
    out, d = [], date(2024, 1, 1)
    while len(out) < N_SESS:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def write_panel(settings: Settings, upto: date | None = None) -> None:
    """Eight random-walk stocks (planted drift and volume spikes) plus the Nifty and the universe."""
    days = [d for d in sessions() if upto is None or d <= upto]
    rng = np.random.default_rng(11)
    rows, uni = [], []
    for k in range(N_SYM):
        r = 0.0012 + 0.014 * rng.standard_normal(N_SESS)
        c = 100 * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]] * (1 + 0.004 * rng.standard_normal(N_SESS))
        h = np.maximum(o, c) * (1 + np.abs(0.006 * rng.standard_normal(N_SESS)))
        lo = np.minimum(o, c) * (1 - np.abs(0.006 * rng.standard_normal(N_SESS)))
        v = 1e6 * (1 + np.abs(rng.standard_normal(N_SESS))) * (1 + 2 * (rng.random(N_SESS) < 0.05))
        for i, d in enumerate(days):
            rows.append((k + 1, f"S{k:02d}", d, "EQ", int(v[i]), o[i], h[i], lo[i], c[i], v[i]))
            uni.append((d, k + 1, k + 1, 5e8, True, True, True))
    cols = [
        "sid",
        "symbol",
        "date",
        "series",
        "volume",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "adj_volume",
    ]
    io.upsert(settings, "panel_daily", pl.DataFrame(rows, schema=cols, orient="row"))
    ucols = ["date", "sid", "turnover_rank", "med_turn_60", "u500", "u200", "in_fno"]
    io.upsert(settings, "panel_universe", pl.DataFrame(uni, schema=ucols, orient="row"))
    nifty = 10000 * np.exp(np.cumsum(0.0008 + 0.006 * np.random.default_rng(5).standard_normal(N_SESS)))
    idx = [("NIFTY 50", d, x, x, x, x) for d, x in zip(days, nifty[: len(days)], strict=False)]
    io.upsert(
        settings,
        "index_daily",
        pl.DataFrame(idx, schema=["index_name", "date", "open", "high", "low", "close"], orient="row"),
    )


@pytest.fixture
def market(tmp_data_root):
    write_panel(tmp_data_root)
    return tmp_data_root


def test_scan_writes_the_radar_with_the_expected_columns(market):
    asof = sessions()[ASOF_IDX]
    clock = FixedClock(datetime(2026, 11, 20, 18, 0, tzinfo=IST))
    out = scan.run_scan(market, asof, workers=1, clock=clock)
    df = pl.read_parquet(out)
    assert out == market.paths.data_root / "lab" / "scan" / f"{asof}.parquet"
    assert df.columns == list(scan.ROW_SCHEMA) and df.schema == pl.Schema(scan.ROW_SCHEMA)
    assert df.height > 0 and set(df["state"].unique()) <= set(scan.STATES)
    assert set(df["strat"].unique()) <= {s.key for s in scan.scan_specs()}
    sig = df.filter(pl.col("state") == "SIGNAL")
    assert (sig["dist_pct"] == 0).all() and (sig["stop"] < sig["p_in"]).all()
    st = json.loads(scan.stats_path(market, asof).read_text())
    assert st["stocks"] == N_SYM and st["asof"] == str(asof) and st["trades"] > 0
    assert st["peak_rss_mb"] is None or st["peak_rss_mb"] > 0
    s = scan.summary(market)
    assert {"strategies", "profile", "best_stocks", "candidate", "live", "list", "exits", "removed"} <= set(s)
    assert {"signals", "open", "forming", "recent", "mkt_up", "nifty", "nifty_200"} <= set(s["live"])
    assert s["candidate"]["exploratory"] is True and "exploratory candidate" in s["candidate"]["label"]
    assert s["list"] == [x.name for x in base.load_library().values() if x.timeframe == "1d" and x.signal]


def test_summary_without_a_scan(market):
    assert "error" in scan.summary(market)


def test_scan_uses_only_data_up_to_asof(market, tmp_path):
    asof = sessions()[ASOF_IDX]
    full = pl.read_parquet(scan.run_scan(market, asof, workers=1))
    short = Settings.model_copy(market, update={"paths": Paths(data_root=tmp_path / "short_root")})
    write_panel(short, upto=asof)
    cut = pl.read_parquet(scan.run_scan(short, asof, workers=1))
    assert full.height > 0 and full.equals(cut)
    a = json.loads(scan.stats_path(market, asof).read_text())
    b = json.loads(scan.stats_path(short, asof).read_text())
    for k in ("strategies", "removed", "profile", "best_stocks", "candidate_stats", "trades", "stocks"):
        assert a[k] == b[k], k


def test_process_pool_gives_the_same_rows_as_one_process(market):
    asof = sessions()[ASOF_IDX]
    one = pl.read_parquet(scan.run_scan(market, asof, workers=1))
    many = pl.read_parquet(scan.run_scan(market, asof, workers=2, chunk=3))
    assert one.equals(many)
    st = json.loads(scan.stats_path(market, asof).read_text())
    assert st["peak_rss_mb"] is None or st["peak_rss_mb"] > 0


def test_stock_profile_buckets_and_minimum_counts():
    def row(sym, net, risk=5.0, rs=10.0, val=50.0):
        return {"sym": sym, "net": net, "risk": risk, "rs63": rs, "val_cr": val}

    trades = [row("AAA", 4.0)] * 40 + [row("BBB", -2.0, risk=10.0, rs=40.0, val=5.0)] * 40
    trades += [row("CCC", 1.0, risk=6.0, rs=0.0, val=10.0)] * 3
    cls = {"AAA": {"sector": "Banks", "mcap": 6e11}, "BBB": {"sector": "Banks"}, "CCC": {"sector": "Autos"}}
    prof, best = stock_profile(trades, cls)
    assert prof["Sector"] == [
        {"k": "Banks", "n": 80, "avg": 1.0, "win": 50.0}
    ]  # Autos has 3 trades: below 25
    liq = {r["k"]: r for r in prof["Liquidity"]}
    assert (
        liq["₹30–100 cr/day"]["avg"] == 4.0 and liq["Turnover < ₹10 cr/day"]["avg"] == -1.79
    )  # 10.0 falls in (0, 10]
    base_w = {r["k"]: r["n"] for r in prof["Base width"]}
    assert base_w == {"Tight base (risk < 6%)": 43, "Risk 9–13%": 40}  # risk 6.0 is in the (0, 6] bin
    rs = {r["k"]: r["avg"] for r in prof["Strength vs Nifty (3 months)"]}
    assert rs["RS 30+ (leaders)"] == -2.0 and rs["RS 0–15"] == 4.0
    assert [b["sym"] for b in best] == ["AAA", "BBB"]  # CCC has only 3 trades
    assert best[0]["win"] == 100.0 and best[1]["tot"] == -80.0 and best[0]["sector"] == "Banks"
    empty, none = stock_profile([], {})
    assert none == [] and all(v == [] for v in empty.values())


def test_unknown_size_and_sector_fall_back():
    trades = [{"sym": "ZZZ", "net": 1.0, "risk": 5.0, "rs63": 5.0, "val_cr": 20.0}] * 45
    prof, _ = stock_profile(trades, {})
    assert [r["k"] for r in prof["Company size"]] == ["Unknown"]
    assert [r["k"] for r in prof["Sector"]] == ["Other"]


def test_repo_scan_files_are_gitignored():
    assert "data_store/" in (Path(__file__).resolve().parents[2] / ".gitignore").read_text()
