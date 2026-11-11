from __future__ import annotations

import time
from datetime import date

import polars as pl
import pytest

from stocksense.data import io
from stocksense.data.catalog import CATALOG
from tests.fixtures.synth import SynthMarket, make_market, make_tiny_market, write_market

TICK = 0.05


@pytest.fixture(scope="module")
def m() -> SynthMarket:
    return make_market()


def _rows(m: SynthMarket, symbol: str) -> pl.DataFrame:
    return m.eq.filter(pl.col("symbol") == symbol).sort("date")


def _close_at(m: SynthMarket, symbol: str, idx: int) -> float:
    return _rows(m, symbol).filter(pl.col("date") == m.sessions[idx])["close"].item()


def _truth(m: SynthMarket, key: str, idx: int) -> float:
    return m.truth.filter((pl.col("symbol_key") == key) & (pl.col("date") == m.sessions[idx]))[
        "true_ret"
    ].item()


def _ca(m: SynthMarket, symbol: str) -> pl.DataFrame:
    return m.corporate_actions.filter(pl.col("symbol") == symbol)


def test_sessions_and_row_count(m):
    # 600 weekdays + 1 Saturday = 601 sessions; Saturday sits at index 120.
    assert len(m.sessions) == 601
    assert m.sessions[0] == date(2018, 1, 1)
    assert m.sessions[120].weekday() == 5
    assert sum(1 for d in m.sessions if d.weekday() == 5) == 1
    assert m.sessions == sorted(m.sessions)
    # 40 x 601 = 24,040 minus S09 suspension (310..330 = 21) minus S19 before listing (0..349 = 350)
    # minus S11 after delisting (401..600 = 200) = 23,469.
    assert m.eq.height == 24040 - 21 - 350 - 200
    assert m.delivery.height == m.eq.height


@pytest.mark.parametrize(
    "attr,dataset",
    [
        ("eq", "bhavcopy_eq"),
        ("delivery", "bhavcopy_delivery"),
        ("corporate_actions", "corporate_actions"),
        ("symbol_changes", "symbol_changes"),
        ("listed", "listed_equities"),
        ("etf", "etf_list"),
        ("sectors", "sectors"),
    ],
)
def test_frames_conform_to_catalog(m, attr, dataset):
    df = getattr(m, attr)
    schema = CATALOG[dataset].schema
    assert dict(df.schema) == dict(schema)


def test_prices_are_ticks_and_ohlc_consistent(m):
    for c in ("open", "high", "low", "close", "prev_close"):
        assert ((m.eq[c] / TICK).round() * TICK - m.eq[c]).abs().max() < 1e-6
    assert (m.eq["high"] >= m.eq[["open", "close"]].max_horizontal() - 1e-9).all()
    assert (m.eq["low"] <= m.eq[["open", "close"]].min_horizontal() + 1e-9).all()
    # turnover = volume * (o+h+l+c)/4 ; n_trades = volume / 50
    r = m.eq.row(0, named=True)
    assert r["turnover_inr"] == pytest.approx(
        r["volume"] * (r["open"] + r["high"] + r["low"] + r["close"]) / 4
    )
    assert r["n_trades"] == pytest.approx(r["volume"] / 50)


def test_prev_close_is_previous_raw_close(m):
    s = _rows(m, "S00")
    prev = s["close"].shift(1)
    assert s["prev_close"][1:].to_list() == prev[1:].to_list()
    assert s["prev_close"][0] == s["open"][0]


def test_delivery_consistent(m):
    j = m.eq.join(m.delivery, on=["symbol", "series", "date"])
    assert j.height == m.eq.height
    assert j["deliv_pct"].min() >= 5 and j["deliv_pct"].max() <= 95
    assert (j["deliv_qty"] - j["volume"] * j["deliv_pct"] / 100).abs().max() < 1e-6


def test_s03_split(m):
    # raw close ratio across the 1:5 split = 0.2 * (1 + true_ret) within tick rounding (0.05 / ~500)
    ratio = _close_at(m, "S03", 200) / _close_at(m, "S03", 199)
    assert ratio == pytest.approx(0.2 * (1 + _truth(m, "S03", 200)), rel=2e-3)
    row = _ca(m, "S03")
    assert row.height == 1
    assert row["action_type"][0] == "split" and row["factor_price"][0] == 0.2
    assert row["parse_status"][0] == "ok" and row["ex_date"][0] == m.sessions[200]
    # volume x5 across the split: ratio of median volume after/before is about 5
    s = _rows(m, "S03")
    after = s.filter(pl.col("date") >= m.sessions[200])["volume"].median()
    before = s.filter(pl.col("date") < m.sessions[200])["volume"].median()
    assert 3 < after / before < 7


def test_s05_bonus_and_s06_wrong_factor(m):
    r5 = _close_at(m, "S05", 250) / _close_at(m, "S05", 249)
    assert r5 == pytest.approx(0.5 * (1 + _truth(m, "S05", 250)), rel=3e-3)
    assert _ca(m, "S05")["factor_price"][0] == 0.5
    assert _ca(m, "S05")["action_type"][0] == "bonus"
    r6 = _close_at(m, "S06", 260) / _close_at(m, "S06", 259)
    assert r6 == pytest.approx(0.5 * (1 + _truth(m, "S06", 260)), rel=3e-3)  # real price moved by 0.5
    assert _ca(m, "S06")["factor_price"][0] == 0.667  # table is wrong on purpose
    assert _ca(m, "S06")["ex_date"][0] == m.sessions[260]


def test_s07_official_rename(m):
    old = _rows(m, "S07")
    new = _rows(m, "S07N")
    assert old["date"].max() == m.sessions[299]
    assert new["date"].min() == m.sessions[300]
    assert new["prev_close"][0] == old["close"][-1]
    sc = m.symbol_changes
    assert sc.height == 1
    assert (sc["old_symbol"][0], sc["new_symbol"][0], sc["change_date"][0]) == (
        "S07",
        "S07N",
        m.sessions[300],
    )


def test_s08_rename_without_symbol_changes_row(m):
    old, new = _rows(m, "S08"), _rows(m, "S08N")
    assert old["date"].max() == m.sessions[309] and new["date"].min() == m.sessions[310]
    assert new["prev_close"][0] == old["close"][-1]
    assert "S08" not in m.symbol_changes["old_symbol"].to_list()


def test_s09_suspension(m):
    s = _rows(m, "S09")
    d = set(s["date"].to_list())
    assert all(m.sessions[i] not in d for i in range(310, 331))
    assert m.sessions[309] in d and m.sessions[331] in d
    # resumes with prev_close = last close before the suspension
    assert s.filter(pl.col("date") == m.sessions[331])["prev_close"].item() == _close_at(m, "S09", 309)


def test_s11_delisted(m):
    s = _rows(m, "S11")
    assert s["date"].max() == m.sessions[400]
    assert "S11" not in m.listed["symbol"].to_list()


def test_s13_dividend(m):
    row = _ca(m, "S13")
    assert row.height == 1
    assert row["action_type"][0] == "dividend" and row["dividend_amount"][0] == 5.0
    assert row["ex_date"][0] == m.sessions[150]
    # raw price move = total return minus the Rs 5 paid: close_t = prev*(1+r) - 5
    prev = _close_at(m, "S13", 149)
    assert _close_at(m, "S13", 150) == pytest.approx(prev * (1 + _truth(m, "S13", 150)) - 5.0, abs=0.1)


def test_s14_split_without_ca_row(m):
    assert _ca(m, "S14").height == 0
    r = _close_at(m, "S14", 220) / _close_at(m, "S14", 219)
    assert r == pytest.approx(0.5 * (1 + _truth(m, "S14", 220)), rel=3e-3)


def test_s15_locked_upper_circuit(m):
    s = _rows(m, "S15")
    for i in (450, 451, 452):
        r = s.filter(pl.col("date") == m.sessions[i]).row(0, named=True)
        assert r["open"] == r["high"] == r["low"] == r["close"]
        assert r["close"] == pytest.approx(r["prev_close"] * 1.05, abs=TICK)
    r453 = s.filter(pl.col("date") == m.sessions[453]).row(0, named=True)
    assert not (r453["open"] == r453["high"] == r453["low"] == r453["close"])


def test_s16_demerger(m):
    r = _close_at(m, "S16", 330) / _close_at(m, "S16", 329)
    assert r == pytest.approx(0.7 * (1 + _truth(m, "S16", 330)), rel=3e-3)
    row = _ca(m, "S16")
    assert row["action_type"][0] == "demerger" and row["factor_price"][0] == 1.0
    assert row["parse_status"][0] == "unparsed"


def test_s17_etf(m):
    assert m.etf["symbol"].to_list() == ["NIFTYBEES"]
    assert _rows(m, "NIFTYBEES").height == 601
    assert "NIFTYBEES" not in m.sectors["symbol"].to_list()


def test_s19_new_listing(m):
    s = _rows(m, "S19")
    assert s["date"].min() == m.sessions[350]
    assert s["prev_close"][0] == s["open"][0]
    assert 300 < s["open"][0] < 400  # listed around Rs 350
    assert m.listed.filter(pl.col("symbol") == "S19")["listing_date"].item() == m.sessions[350]


def test_s21_penny(m):
    assert _rows(m, "S21")["high"].max() < 20


def test_s23_series_switch(m):
    s = _rows(m, "S23")
    ser = dict(zip(s["date"].to_list(), s["series"].to_list(), strict=True))
    assert ser[m.sessions[379]] == "EQ"
    assert ser[m.sessions[380]] == "BE" and ser[m.sessions[419]] == "BE"
    assert ser[m.sessions[420]] == "EQ"
    assert s.height == 601


def test_s25_rights(m):
    row = _ca(m, "S25")
    assert row.height == 1
    assert row["subject_raw"][0] == "Rights 1:4 @ Premium Rs 40/-"
    assert row["action_type"][0] == "rights" and row["parse_status"][0] == "unparsed"
    assert row["ex_date"][0] == m.sessions[280]


def test_sectors_four_industries(m):
    assert m.sectors["industry"].n_unique() == 4


def _mean_ic(m: SynthMarket) -> float:
    idx = pl.DataFrame({"date": m.sessions, "i": range(len(m.sessions))})
    t = m.truth.join(idx, on="date")
    a = t.select("symbol_key", "i", "alpha")
    r = t.select("symbol_key", (pl.col("i") - 1).alias("i"), pl.col("true_ret").alias("nxt"))
    j = a.join(r, on=["symbol_key", "i"]).drop_nulls()
    ic = (
        j.group_by("i")
        .agg(pl.corr(pl.col("alpha").rank(), pl.col("nxt").rank()).alias("ic"))
        .drop_nulls("ic")["ic"]
    )
    return float(ic.mean())


def test_planted_ic():
    ic = _mean_ic(make_market(planted_ic=0.05))
    assert 0.035 <= ic <= 0.065


def test_no_planted_ic_means_no_ic():
    assert abs(_mean_ic(make_market(planted_ic=0.0))) < 0.02


def test_same_seed_identical_and_different_seed_differs():
    a, b = make_market(seed=3), make_market(seed=3)
    for f in ("eq", "delivery", "corporate_actions", "truth"):
        assert getattr(a, f).equals(getattr(b, f))
    assert not make_market(seed=4).eq.equals(a.eq)


def test_generation_is_fast():
    t0 = time.perf_counter()
    make_market()
    assert time.perf_counter() - t0 < 3.0


def test_tiny_market_exact_numbers():
    t = make_tiny_market()
    # 3 stocks x 12 sessions; AAA closes 101 on session 0 and 103 on session 1 -> prev_close 101.
    assert len(t.sessions) == 12 and t.eq.height == 36
    a = t.eq.filter(pl.col("symbol") == "AAA").sort("date")
    assert a["close"][0] == 101.0 and a["prev_close"][1] == 101.0 and a["close"][1] == 103.0
    assert a["prev_close"][0] == a["open"][0] == 100.0
    # true return session 1 = 103/101 - 1
    tr = t.truth.filter((pl.col("symbol_key") == "AAA") & (pl.col("date") == t.sessions[1]))[
        "true_ret"
    ].item()
    assert tr == pytest.approx(103 / 101 - 1)
    assert dict(t.eq.schema) == dict(CATALOG["bhavcopy_eq"].schema)


def test_write_market_roundtrip(tmp_data_root):
    mk = make_market(n=40)
    write_market(mk, tmp_data_root.paths.data_root)
    back = io.read(tmp_data_root, "bhavcopy_eq").collect()
    assert back.height == mk.eq.height
    assert io.read(tmp_data_root, "bhavcopy_delivery").collect().height == mk.delivery.height
    assert io.read(tmp_data_root, "corporate_actions").collect().height == mk.corporate_actions.height
    assert io.read(tmp_data_root, "symbol_changes").collect().height == 1
    assert io.read(tmp_data_root, "etf_list").collect().height == 1
    first = back.filter(pl.col("symbol") == "S00").sort("date")
    assert first["close"].to_list() == _rows(mk, "S00")["close"].to_list()
