from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.core.clock import IST
from stocksense.core.paths import Paths
from stocksense.data import io
from stocksense.features.auction import (
    auction_features_live,
    build_auction_features,
    gap_features,
    load_auction_features,
)
from stocksense.features.base import FeatureContext
from tests.fixtures.feat58 import write_env
from tests.fixtures.handmade import weekdays

N = 26
DS = weekdays(N, date(2024, 10, 1))
T, D = DS[N - 2], DS[N - 1]  # decision date and trade date


def ctx_for(s, as_of=None):
    return FeatureContext(s, as_of)


def closes(base: float, amp: float) -> list[float]:
    return [base + amp * (i % 2) for i in range(N)]


def world(s, open_d: dict[str, float], industry: dict[str, str] | None = None):
    """Stocks A, B, C (flat-ish alternating closes); the trade-day open is set explicitly."""
    base = {"A": (100.0, 2.0), "B": (50.0, 1.0), "C": (200.0, 4.0)}
    stocks = {}
    for k, (sym, (b, a)) in enumerate(base.items()):
        c = closes(b, a)
        o = list(c)
        o[-1] = open_d[sym]
        stocks[sym] = {"sid": k + 1, "close": c, "open": o, "industry": (industry or {}).get(sym, "UNKNOWN")}
    write_env(s, DS, stocks)
    return stocks


def row(df: pl.DataFrame, sid: int) -> dict:
    return df.filter(pl.col("sid") == sid).to_dicts()[0]


def test_gap_features_by_hand(tmp_data_root):
    s = tmp_data_root
    st = world(s, {"A": 1.0, "B": 1.0, "C": 1.0})
    cl = {k: v["close"][-2] for k, v in st.items()}  # close on t
    opens = {"A": cl["A"] * 1.02, "B": cl["B"] * 0.98, "C": cl["C"]}
    world(s, opens)
    from stocksense.features.auction import auction_frame

    df = auction_frame(ctx_for(s)).filter(pl.col("trade_date") == D)
    assert df.height == 3 and set(df["decision_date"].to_list()) == {T}
    a, b, c = (row(df, i) for i in (1, 2, 3))
    assert (
        a["gap"] == pytest.approx(0.02)
        and b["gap"] == pytest.approx(-0.02)
        and c["gap"] == pytest.approx(0.0)
    )
    assert a["mkt_gap"] == pytest.approx(0.0)  # median of {0.02, -0.02, 0}
    assert a["gap_rel_mkt"] == pytest.approx(0.02)
    assert (a["gap_rank"], b["gap_rank"], c["gap_rank"]) == pytest.approx((0.5, -0.5, 0.0))
    la = np.diff(np.log(closes(100.0, 2.0)))
    vol21 = la[N - 2 - 21 : N - 2].std(ddof=1)  # 21 returns ending on t
    assert a["gap_z"] == pytest.approx(0.02 / vol21)
    assert a["gap_source"] == "official_open"
    assert a["gap_rel_ind"] is None  # UNKNOWN industry / fewer than 5 members


def test_industry_relative_gap_needs_five_members(tmp_data_root):
    s = tmp_data_root
    syms = [f"S{k}" for k in range(5)]
    stocks = {}
    gaps = [0.00, 0.01, 0.02, 0.03, 0.04]
    for k, sym in enumerate(syms):
        c = [100.0 + (i % 2) for i in range(N)]
        o = list(c)
        o[-1] = c[-2] * (1 + gaps[k])
        stocks[sym] = {"sid": k + 1, "close": c, "open": o, "industry": "Banks"}
    write_env(s, DS, stocks)
    from stocksense.features.auction import auction_frame

    df = auction_frame(ctx_for(s)).filter(pl.col("trade_date") == D)
    r = row(df, 5)
    assert r["gap_rel_ind"] == pytest.approx(0.04 - 0.02)  # median gap of the industry is 0.02
    assert r["gap_rel_mkt"] == pytest.approx(0.04 - 0.02)


def test_split_day_gap_is_near_zero_not_minus_half(tmp_data_root):
    s = tmp_data_root
    st = world(s, {"A": 1.0, "B": 1.0, "C": 1.0})
    close_t = st["A"]["close"][-2]
    world(s, {"A": close_t * 0.5, "B": st["B"]["close"][-2], "C": st["C"]["close"][-2]})
    pan = io.read(s, "panel_daily").collect()
    pan = pan.with_columns(
        pl.when((pl.col("symbol") == "A") & (pl.col("date") == D))
        .then(0.5)
        .otherwise(pl.col("gap_factor"))
        .alias("gap_factor")
    )
    io.write_partition(s, "panel_daily", "2024", pan)
    from stocksense.features.auction import auction_frame

    r = row(auction_frame(ctx_for(s)).filter(pl.col("trade_date") == D), 1)
    assert r["gap"] == pytest.approx(0.0, abs=1e-12)  # 50 / (100 * 0.5) - 1, not -50%


def _macro(s, series, vals, hours):
    rows = [
        (series, d, v, datetime.combine(d, datetime.min.time(), tzinfo=IST) + timedelta(hours=hours))
        for d, v in zip(DS, vals, strict=True)
    ]
    io.upsert(
        s, "macro_daily", pl.DataFrame(rows, schema=["series", "date", "close", "available_ts"], orient="row")
    )


def test_overnight_macro_moves(tmp_data_root):
    s = tmp_data_root
    world(s, {"A": 101.0, "B": 50.0, "C": 200.0})
    spx = [4000.0 + 10 * i for i in range(N)]
    _macro(s, "SPX", spx, 29.5)  # US close of date d arrives at 05:30 IST on d+1
    _macro(s, "USDINR", [80.0 + 0.1 * i for i in range(N)], 17)  # same-day, before the cutoff
    from stocksense.features.auction import auction_frame

    r = row(auction_frame(ctx_for(s)).filter(pl.col("trade_date") == D), 1)
    # SPX of date t arrives at 05:30 on the trade date: after t's cutoff, before the 09:07 decision
    assert r["spx_overnight"] == pytest.approx(math.log(spx[N - 2] / spx[N - 3]))
    assert r["usdinr_overnight"] == pytest.approx(0.0)  # nothing new between 20:00 and 09:07
    assert r["ndx_overnight"] is None and r["brent_overnight"] is None


def test_preopen_imbalance_uses_only_snapshots_up_to_0908(tmp_data_root):
    s = tmp_data_root
    world(s, {"A": 101.0, "B": 50.0, "C": 200.0})

    def at(h, m):
        return datetime.combine(D, datetime.min.time(), tzinfo=IST) + timedelta(hours=h, minutes=m)

    rows = [
        ("A", at(9, 5), 100.0, 101.0, None, 100.0, 100.0, "t"),
        ("A", at(9, 7), 100.0, 101.0, None, 300.0, 100.0, "t"),
        ("A", at(9, 20), 100.0, 101.0, 101.0, 1.0, 9.0, "t"),  # after the auction: must be ignored
    ]
    cols = ["symbol", "ts", "prev_close", "indicative_price", "final_price", "buy_qty", "sell_qty", "source"]
    io.upsert(
        s,
        "preopen",
        pl.DataFrame(rows, schema=cols, orient="row", schema_overrides={"final_price": pl.Float64}),
    )
    from stocksense.features.auction import auction_frame

    df = auction_frame(ctx_for(s)).filter(pl.col("trade_date") == D)
    assert row(df, 1)["preopen_imbalance"] == pytest.approx((300 - 100) / (300 + 100))
    assert row(df, 2)["preopen_imbalance"] is None  # no captures for B


def test_uses_nothing_after_the_open(tmp_data_root):
    """Changing everything of trade day d except its open must not change the features for d."""
    s = tmp_data_root
    base_open = {"A": 101.0, "B": 49.5, "C": 200.0}
    world(s, base_open)
    from stocksense.features.auction import auction_frame

    before = auction_frame(ctx_for(s)).filter(pl.col("trade_date") == D).sort("sid")
    pan = io.read(s, "panel_daily").collect()
    hit = pl.col("date") == D
    pan = pan.with_columns(
        pl.when(hit).then(pl.col("close") * 1.25).otherwise(pl.col("close")).alias("close"),
        pl.when(hit).then(pl.col("high") * 1.5).otherwise(pl.col("high")).alias("high"),
        pl.when(hit).then(pl.col("low") * 0.5).otherwise(pl.col("low")).alias("low"),
        pl.when(hit).then(pl.col("turnover_inr") * 9).otherwise(pl.col("turnover_inr")).alias("turnover_inr"),
        pl.when(hit).then(0.3).otherwise(pl.col("ret")).alias("ret"),
        pl.when(hit).then(0.3).otherwise(pl.col("tret")).alias("tret"),
    )
    io.write_partition(s, "panel_daily", "2024", pan)
    after = auction_frame(ctx_for(s)).filter(pl.col("trade_date") == D).sort("sid")
    assert before.equals(after)


def test_truncated_run_equals_full_for_earlier_trade_dates(tmp_data_root):
    s = tmp_data_root
    world(s, {"A": 101.0, "B": 49.5, "C": 200.0})
    from stocksense.features.auction import auction_frame

    full = auction_frame(ctx_for(s))
    cut = DS[N - 4]
    part = auction_frame(ctx_for(s, cut))
    assert part["trade_date"].max() == cut
    ref = full.filter(pl.col("trade_date") <= cut).sort("trade_date", "sid")
    assert ref.equals(part.sort("trade_date", "sid"))


def test_build_writes_and_loads_history(tmp_data_root):
    s = tmp_data_root
    world(s, {"A": 101.0, "B": 49.5, "C": 200.0})
    n = build_auction_features(s, DS[0], D)
    assert n > 0
    out = load_auction_features(s, "production").collect().sort("trade_date", "sid")
    assert out.select("trade_date", "sid").is_duplicated().sum() == 0
    assert (out["gap_source"] == "official_open").all()
    assert out["trade_date"].max() == D and out["trade_date"].min() == DS[1]
    assert (Paths(s).features / "auction" / "part-2024.parquet").is_file()
    # research view stops before the lockbox
    assert load_auction_features(s, "research").collect()["trade_date"].max() <= D
    # rebuilding a sub-range merges by key (no duplicates)
    build_auction_features(s, DS[10], D)
    again = load_auction_features(s, "production").collect()
    assert again.height == out.height


def test_live_features(tmp_data_root):
    s = tmp_data_root
    world(s, {"A": 1.0, "B": 1.0, "C": 1.0})  # the trade-day panel rows exist but live must not use them
    ctx = FeatureContext(s, as_of=T)
    panel_tail = ctx.panel().collect()
    close_a = panel_tail.filter((pl.col("symbol") == "A") & (pl.col("date") == T))["close"][0]
    close_b = panel_tail.filter((pl.col("symbol") == "B") & (pl.col("date") == T))["close"][0]
    ts = datetime.combine(D, datetime.min.time(), tzinfo=IST) + timedelta(hours=9, minutes=7)
    pre = pl.DataFrame(
        {
            "symbol": ["A", "B"],
            "ts": [ts, ts],
            "indicative_price": [close_a * 1.02, close_b * 0.5],
            "buy_qty": [300.0, 10.0],
            "sell_qty": [100.0, 10.0],
            "gap_factor": [1.0, 0.5],  # B has a split effective today
        }
    )
    out = auction_features_live(ctx, pre, panel_tail)
    a, b = row(out, 1), row(out, 2)
    assert a["gap"] == pytest.approx(0.02) and b["gap"] == pytest.approx(0.0, abs=1e-12)
    assert a["preopen_imbalance"] == pytest.approx(0.5) and b["preopen_imbalance"] == pytest.approx(0.0)
    assert a["gap_source"] == "indicative_0907" and (out["trade_date"] == D).all()
    assert (out["decision_date"] == T).all()
    assert out.height == 2  # C has no pre-open row


def test_gap_features_pure_function():
    df = pl.DataFrame(
        {
            "trade_date": [D] * 6,
            "sid": [1, 2, 3, 4, 5, 6],
            "industry": ["I"] * 5 + ["J"],
            "price": [101.0, 98.0, 100.0, 103.0, 99.0, 150.0],
            "close_t": [100.0] * 6,
            "gap_factor": [1.0] * 6,
            "vol_21": [0.01] * 6,
        }
    )
    out = gap_features(df).sort("sid")
    assert out["gap"].to_list() == pytest.approx([0.01, -0.02, 0.0, 0.03, -0.01, 0.5])
    assert out["mkt_gap"][0] == pytest.approx(0.005)  # median of the six gaps
    assert out["gap_rel_ind"].to_list()[:5] == pytest.approx(
        [g - 0.0 for g in [0.01, -0.02, 0.0, 0.03, -0.01]]
    )
    assert out["gap_rel_ind"][5] is None  # industry J has one member
    assert out["gap_z"][0] == pytest.approx(1.0)
    assert out["gap_rank"].to_list() == pytest.approx([(r - 1) / 5 - 0.5 for r in (4, 1, 3, 5, 2, 6)])
