from __future__ import annotations

import polars as pl
import pytest

from stocksense import panel as panel_pkg
from stocksense.core.errors import LockboxError
from stocksense.data import io
from stocksense.labels.forward import (
    LABEL_VERSION,
    build_labels,
    compute_labels,
    load_labels,
    robust_z_and_rank,
)
from tests.fixtures.handmade import weekdays
from tests.fixtures.synth import _settings_for, make_market, make_tiny_market, write_market

N = 30
DATES = weekdays(N)
SESSIONS = pl.DataFrame({"date": DATES, "sx": list(range(N))}, schema_overrides={"sx": pl.Int64})
DAY = {k: DATES[k] for k in range(N)}


def panel_rows(sid: int, sxs, over: dict[int, dict] | None = None) -> pl.DataFrame:
    """Plain rows: open = 100 + k, close = 100.5 + k, adjusted = raw, no dividends or flags."""
    over = over or {}
    rows = []
    for k in sxs:
        r = {
            "sid": sid,
            "date": DAY[k],
            "sx": k,
            "open": 100.0 + k,
            "high": 101.0 + k,
            "close": 100.5 + k,
            "adj_open": 100.0 + k,
            "adj_close": 100.5 + k,
            "div_ret": 0.0,
            "ret_co": 0.001,
            "flag_unexplained_gap": False,
            "flag_locked_up": False,
        }
        r.update(over.get(k, {}))
        rows.append(r)
    return pl.DataFrame(rows, schema_overrides={"sid": pl.Int32, "sx": pl.Int64})


def decide(*pairs: tuple[int, int]) -> pl.DataFrame:
    """Decision rows from (sid, t_index) pairs."""
    return pl.DataFrame(
        {"date": [DAY[t] for _, t in pairs], "sid": [s for s, _ in pairs]},
        schema_overrides={"sid": pl.Int32},
    )


def one(panel: pl.DataFrame, sid: int, t: int) -> dict:
    return compute_labels(panel, decide((sid, t)), SESSIONS).row(0, named=True)


# ---- plain days --------------------------------------------------------------------------------


def test_every_label_on_plain_days():
    r = one(panel_rows(1, range(N)), 1, 2)
    # t = session 2, d1 = 3: open 103, close 103.5
    assert r["y_oc"] == pytest.approx(103.5 / 103 - 1)  # 0.5 / 103 = 0.0048544
    # h=5: x = session 8, adj_open 108 over adj_close(d1) 103.5
    assert r["y_c2o_5"] == pytest.approx(108 / 103.5 - 1)  # 0.0434783
    assert r["y_c2o_10"] == pytest.approx(113 / 103.5 - 1)  # 0.0917874
    assert r["y_c2o_20"] == pytest.approx(123 / 103.5 - 1)  # 0.1884058
    # c2c: close(4) / close(3) = 104.5 / 103.5
    assert r["y_c2c_1"] == pytest.approx(104.5 / 103.5 - 1)  # 0.0096618
    # o2o_5: open(8) / open(3) = 108 / 103
    assert r["y_o2o_5"] == pytest.approx(108 / 103 - 1)  # 0.0485437
    assert (r["end_oc"], r["end_5"], r["end_10"], r["end_20"]) == (DAY[3], DAY[8], DAY[13], DAY[23])
    assert (r["end_c2c_1"], r["end_o2o_5"]) == (DAY[4], DAY[8])
    assert not r["entry_blocked_open"] and not r["entry_blocked_close"] and not r["label_delisted"]


def test_one_row_per_decision_and_key_order():
    p = panel_rows(1, range(N))
    out = compute_labels(p, decide((1, 5), (1, 2), (1, 9)), SESSIONS)
    assert (
        out.height == 3
        and out["date"].to_list() == [DAY[5], DAY[2], DAY[9]]
        or set(out["date"].to_list()) == {DAY[2], DAY[5], DAY[9]}
    )
    assert out.select("date", "sid").is_duplicated().sum() == 0


# ---- the end of data ---------------------------------------------------------------------------


def test_labels_are_null_when_data_ends_first():
    p = panel_rows(1, range(N))
    # t = 25: d1 = 26, d1 + 5 = 31 > last session 29 -> c2o_5 / o2o_5 null; y_oc and c2c_1 exist
    r = one(p, 1, 25)
    assert r["y_oc"] == pytest.approx(126.5 / 126 - 1)
    assert r["y_c2c_1"] == pytest.approx(127.5 / 126.5 - 1)
    assert r["y_c2o_5"] is None and r["y_c2o_10"] is None and r["y_c2o_20"] is None
    assert r["y_o2o_5"] is None and r["end_5"] is None and r["label_delisted"] is False
    # t = 29 (last session): d1 does not exist -> everything null, nothing blocked
    r = one(p, 1, 29)
    assert all(r[c] is None for c in ("y_oc", "y_c2o_5", "y_c2c_1", "y_o2o_5", "end_oc"))
    assert r["entry_blocked_open"] is False and r["entry_blocked_close"] is False
    # t = 28: d1 = 29 exists: y_oc yes, c2c_1 no (d1 + 1 missing)
    r = one(p, 1, 28)
    assert r["y_oc"] is not None and r["y_c2c_1"] is None


def test_no_row_on_d1_nulls_everything():
    p = panel_rows(1, [k for k in range(N) if k != 4])
    r = one(p, 1, 3)  # d1 = 4 missing
    assert all(r[c] is None for c in ("y_oc", "y_c2o_5", "y_c2o_10", "y_c2o_20", "y_c2c_1", "y_o2o_5"))
    assert r["end_oc"] is None and r["end_5"] is None


# ---- splits, dividends, flags ------------------------------------------------------------------


def test_dividend_is_added_once_and_only_inside_the_holding_window():
    p = panel_rows(1, range(N), {5: {"div_ret": 0.01}})
    # t=2: d1=3, window (3, 8] contains the ex-day 5 -> + 0.01 once (also in o2o_5; c2c_1 uses row 4)
    r = one(p, 1, 2)
    assert r["y_c2o_5"] == pytest.approx(108 / 103.5 - 1 + 0.01)
    assert r["y_c2o_20"] == pytest.approx(123 / 103.5 - 1 + 0.01)
    assert r["y_o2o_5"] == pytest.approx(108 / 103 - 1 + 0.01)
    assert r["y_c2c_1"] == pytest.approx(104.5 / 103.5 - 1)  # row 4 has no dividend
    # t=4: d1 = 5 IS the ex-day; the holder bought the close after it -> not included
    r = one(p, 1, 4)
    assert r["y_c2o_5"] == pytest.approx(110 / 105.5 - 1)
    # t=3: d1 = 4, c2c_1 uses row 5 -> + 0.01
    r = one(p, 1, 3)
    assert r["y_c2c_1"] == pytest.approx(105.5 / 104.5 - 1 + 0.01)


def test_split_uses_adjusted_prices_not_raw():
    # raw price falls 5x at session 6 (open 20 instead of 106) but the adjusted series is continuous
    over = {k: {"open": 20.0 + (k - 6), "close": 20.1 + (k - 6), "high": 21.0 + (k - 6)} for k in range(6, N)}
    p = panel_rows(1, range(N), over)
    r = one(p, 1, 2)
    # adjusted: 108 / 103.5 - 1, about +4.3%; a raw calculation would be 22 / 103.5 - 1 = -79%
    assert r["y_c2o_5"] == pytest.approx(108 / 103.5 - 1)
    assert r["y_oc"] == pytest.approx(103.5 / 103 - 1)


def test_unexplained_gap_nulls_labels_whose_window_contains_it():
    p = panel_rows(1, range(N), {6: {"flag_unexplained_gap": True}})
    r = one(p, 1, 2)  # d1 = 3, c2o windows (3, x] with x >= 8 contain row 6; o2o_5 too
    assert r["y_c2o_5"] is None and r["y_c2o_10"] is None and r["y_c2o_20"] is None
    assert r["y_o2o_5"] is None and r["end_5"] is None
    assert r["y_oc"] is not None and r["y_c2c_1"] is not None  # windows {3} and (3, 4]
    # flag exactly on d1 does not touch the later windows
    r = one(p, 1, 5)  # d1 = 6
    assert r["y_c2o_5"] == pytest.approx(111 / 106.5 - 1) and r["y_oc"] is not None


def test_entry_blocked_flags_by_hand():
    over = {
        4: {"open": 105.0, "high": 105.0, "ret_co": 0.0496, "flag_locked_up": True},  # opens at the top
        6: {"open": 106.0, "high": 106.0, "ret_co": 0.03},  # open == high, gap 3% (not near a band)
        8: {"open": 108.0, "high": 109.0, "ret_co": 0.05},  # gap at a band but open < high
        10: {"flag_locked_up": True},
    }
    p = panel_rows(1, range(N), over)
    r = one(p, 1, 3)  # d1 = 4
    assert r["entry_blocked_open"] is True and r["entry_blocked_close"] is True
    # |0.0496 - 0.05| = 0.0004 <= 0.0025 ; other bands are further away
    r = one(p, 1, 5)  # d1 = 6
    assert r["entry_blocked_open"] is False and r["entry_blocked_close"] is False
    r = one(p, 1, 7)  # d1 = 8
    assert r["entry_blocked_open"] is False
    r = one(p, 1, 9)  # d1 = 10: only the close lock
    assert r["entry_blocked_open"] is False and r["entry_blocked_close"] is True


# ---- suspension, delisting ---------------------------------------------------------------------


def test_suspension_inside_the_window_exits_at_the_first_row_back():
    # sid 1 is missing sessions 9..14 (suspended); t = 5: d1 = 6, d1+5 = 11 missing -> x = 15
    p = panel_rows(1, [k for k in range(N) if not 9 <= k <= 14])
    r = one(p, 1, 5)
    # a real holder is stuck until session 15: adj_open(15) = 115 over adj_close(6) = 106.5
    assert r["y_c2o_5"] == pytest.approx(115 / 106.5 - 1)
    assert r["end_5"] == DAY[15]
    # h = 10: d1+10 = 16 present: normal
    assert r["y_c2o_10"] == pytest.approx(116 / 106.5 - 1) and r["end_10"] == DAY[16]
    # o2o_5 needs a row on d1 + 5 exactly: null
    assert r["y_o2o_5"] is None
    assert r["label_delisted"] is False


def test_delisting_haircut_when_the_name_never_trades_again():
    # sid 1 trades sessions 0..5 only; sid 2 keeps the data going to session 29 (24 sessions after 5 > 20)
    p = pl.concat([panel_rows(1, range(6)), panel_rows(2, range(N))])
    r = one(p, 1, 2)  # d1 = 3, d1 + 5 = 8 has no row, never again
    # adj_close(last = 5) / adj_close(3) - 1 - 0.20 = 105.5/103.5 - 1 - 0.2
    assert r["y_c2o_5"] == pytest.approx(105.5 / 103.5 - 1 - 0.20)
    assert r["y_c2o_10"] == pytest.approx(105.5 / 103.5 - 1 - 0.20)
    assert r["y_c2o_20"] == pytest.approx(105.5 / 103.5 - 1 - 0.20)
    assert r["label_delisted"] is True
    # the label is known at the session after the last row
    assert r["end_5"] == DAY[6] and r["end_20"] == DAY[6]
    # y_oc and c2c_1 are ordinary
    assert r["y_oc"] == pytest.approx(103.5 / 103 - 1) and r["y_c2c_1"] == pytest.approx(104.5 / 103.5 - 1)


def test_name_that_stops_within_20_sessions_of_the_end_is_null_not_delisted():
    # sid 1 last row is 20; data ends at 29 (9 sessions later, not more than 20 before the end)
    p = pl.concat([panel_rows(1, range(21)), panel_rows(2, range(N))])
    r = one(p, 1, 14)  # d1 = 15; d1 + 5 = 20 exists; d1 + 10 = 25 does not
    assert r["y_c2o_5"] == pytest.approx(120 / 115.5 - 1)
    assert r["y_c2o_10"] is None and r["y_c2o_20"] is None and r["label_delisted"] is False


# ---- scaling -----------------------------------------------------------------------------------


def test_robust_z_and_rank_by_hand():
    d = DAY[0]
    df = pl.DataFrame(
        {
            "date": [d] * 6 + [DAY[1]] * 3,
            "sid": list(range(6)) + [0, 1, 2],
            "y": [0.0, 1, 2, 3, 4, 100, 1, 1, 5],
        }
    )
    out = robust_z_and_rank(df, ["y"]).sort("date", "sid")
    day1 = out.filter(pl.col("date") == d)
    # median of [0,1,2,3,4,100] = 2.5; |dev| = [2.5,1.5,.5,.5,1.5,97.5] -> MAD 1.5; scale 1.4826 * 1.5
    assert day1["z_y"][0] == pytest.approx(-2.5 / (1.4826 * 1.5))  # -1.1241
    assert day1["z_y"][4] == pytest.approx(1.5 / (1.4826 * 1.5))
    assert day1["z_y"][5] == pytest.approx(5.0)  # 97.5 / 2.2239 = 43.8 clipped to +5
    # ranks 1..6 -> (r - 1)/5 - 0.5
    assert day1["r_y"].to_list() == [pytest.approx(x) for x in (-0.5, -0.3, -0.1, 0.1, 0.3, 0.5)]
    assert day1["z_y"].dtype == pl.Float32 and day1["r_y"].dtype == pl.Float32
    # second date [1, 1, 5]: median 1, |dev| = [0, 0, 4], MAD 0 -> z null; ranks average (1.5, 1.5, 3)
    day2 = out.filter(pl.col("date") == DAY[1])
    assert day2["z_y"].null_count() == 3
    assert day2["r_y"].to_list() == [pytest.approx(-0.25), pytest.approx(-0.25), pytest.approx(0.5)]


def test_nulls_are_ignored_by_scaling():
    d = DAY[0]
    df = pl.DataFrame({"date": [d] * 4, "sid": [0, 1, 2, 3], "y": [1.0, None, 3.0, 2.0]})
    out = robust_z_and_rank(df, ["y"]).sort("sid")
    assert out["r_y"].to_list() == [-0.5, None, 0.5, 0.0]
    # median 2, |dev| [1, 1, 0] -> MAD 1: z = -1/1.4826, null, +1/1.4826, 0
    assert out["z_y"][0] == pytest.approx(-1 / 1.4826) and out["z_y"][1] is None


# ---- end to end on the tiny market -------------------------------------------------------------

_SMALL = {
    "min_history_sessions": 2,
    "min_traded_of_last_60": 1,
    "min_close": 1.0,
    "min_med_turnover_inr": 1.0,
    "u200_min_med_turnover_inr": 1.0,
    "u500_size": 3,
    "u500_exit_rank": 5,
    "u200_size": 2,
    "u200_exit_rank": 3,
}


def _scaled(settings, **kw):
    return settings.model_copy(update={"universe": settings.universe.model_copy(update=kw)})


def test_tiny_market_end_to_end(tmp_path):
    m = make_tiny_market()
    write_market(m, tmp_path)
    s = _scaled(_settings_for(tmp_path), **_SMALL)
    panel_pkg.build_all(s)
    assert build_labels(s) == LABEL_VERSION
    lab = io.read(s, "labels", sub=f"v{LABEL_VERSION}").collect()
    sm = io.read(s, "panel_daily", columns=["sid", "symbol"]).unique().collect()
    lab = lab.join(sm, on="sid")
    t2 = lab.filter(pl.col("date") == m.sessions[2])
    assert t2.height == 3  # members from row index 2
    aaa = t2.filter(pl.col("symbol") == "AAA").row(0, named=True)
    # AAA opens [100,101,103,102,104,105,104,106,107,...] closes [101,103,102,104,105,104,106,107,...]
    # d1 = session 3: open 102, close 104 -> y_oc = 104/102 - 1
    assert aaa["y_oc"] == pytest.approx(104 / 102 - 1)
    # x = session 8: open 107 over close(3) = 104
    assert aaa["y_c2o_5"] == pytest.approx(107 / 104 - 1)
    assert aaa["y_c2c_1"] == pytest.approx(105 / 104 - 1)
    assert aaa["y_o2o_5"] == pytest.approx(107 / 102 - 1)
    assert aaa["end_5"] == m.sessions[8] and aaa["end_oc"] == m.sessions[3]
    # session 13 does not exist and the data does not stop 20 sessions early: null
    assert aaa["y_c2o_10"] is None and aaa["y_c2o_20"] is None
    # z / r of y_c2o_5 across AAA, BBB, CCC at t = 2
    ya = 107 / 104 - 1
    yb = 50 / 52 - 1  # BBB open(8) = 50, close(3) = 52
    yc = 205 / 203 - 1  # CCC open(8) = 205, close(3) = 203
    med = sorted([ya, yb, yc])[1]  # = yc
    mad = sorted([abs(ya - med), abs(yb - med), abs(yc - med)])[1]  # = |ya - yc|
    by = {r["symbol"]: r for r in t2.iter_rows(named=True)}
    assert by["AAA"]["z_y_c2o_5"] == pytest.approx((ya - med) / (1.4826 * mad), rel=1e-5)
    assert by["BBB"]["z_y_c2o_5"] == pytest.approx((yb - med) / (1.4826 * mad), rel=1e-5)
    assert by["CCC"]["z_y_c2o_5"] == pytest.approx(0.0, abs=1e-7)
    # ranks: BBB lowest, CCC middle, AAA highest -> -0.5, 0, 0.5
    assert (by["BBB"]["r_y_c2o_5"], by["CCC"]["r_y_c2o_5"], by["AAA"]["r_y_c2o_5"]) == (-0.5, 0.0, 0.5)
    # the last session has no next session: every label null
    last = lab.filter(pl.col("date") == m.sessions[-1])
    assert last.height > 0 and last["y_oc"].null_count() == last.height
    # one before the last: y_oc exists, c2c_1 does not
    pen = lab.filter(pl.col("date") == m.sessions[-2])
    assert pen["y_oc"].null_count() == 0 and pen["y_c2c_1"].null_count() == pen.height


# ---- the synthetic market's scripted events ----------------------------------------------------


@pytest.fixture(scope="module")
def synth_labels(tmp_path_factory):
    root = tmp_path_factory.mktemp("labels_synth")
    m = make_market()
    write_market(m, root)
    s = _scaled(
        _settings_for(root),
        min_history_sessions=5,
        min_traded_of_last_60=5,
        min_close=1.0,
        min_med_turnover_inr=1.0,
        u200_min_med_turnover_inr=1.0,
    )
    panel_pkg.build_all(s)
    build_labels(s)
    lab = io.read(s, "labels", sub=f"v{LABEL_VERSION}").collect()
    daily = io.read(s, "panel_daily").collect().sort("sid", "date")
    return (
        s,
        m,
        lab.join(daily.select("sid", "symbol").unique("sid", keep="last", maintain_order=True), on="sid"),
        daily,
    )


def _lab(synth, symbol: str, t: int) -> dict:
    _, m, lab, _ = synth
    r = lab.filter((pl.col("symbol") == symbol) & (pl.col("date") == m.sessions[t]))
    assert r.height == 1, (symbol, t)
    return r.row(0, named=True)


def _prod(daily: pl.DataFrame, symbol: str, m, lo: int, hi: int) -> float:
    """prod(1 + ret) over sessions lo..hi of the symbol's panel rows."""
    d = daily.filter((pl.col("symbol") == symbol) & pl.col("date").is_in(m.sessions[lo : hi + 1]))
    assert d.height == hi - lo + 1
    return float((1.0 + d["ret"]).product())


def test_s03_split_label_is_the_true_return(synth_labels):
    _, m, _, daily = synth_labels
    r = _lab(synth_labels, "S03", 196)  # d1 = 197; the 1:5 split is at 200, inside the window
    ret_co = daily.filter((pl.col("symbol") == "S03") & (pl.col("date") == m.sessions[202]))["ret_co"].item()
    # adj_open(202)/adj_close(197) - 1 = prod(1 + ret, 198..201) * (1 + ret_co(202)) - 1
    expected = _prod(daily, "S03", m, 198, 201) * (1.0 + ret_co) - 1.0
    assert r["y_c2o_5"] == pytest.approx(expected, rel=1e-9, abs=1e-9)
    assert abs(r["y_c2o_5"]) < 0.4  # nowhere near the raw -80%


def test_s13_dividend_added_once(synth_labels):
    _, m, _, daily = synth_labels
    div = daily.filter((pl.col("symbol") == "S13") & (pl.col("date") == m.sessions[150]))["div_ret"].item()
    assert div > 0
    r = _lab(synth_labels, "S13", 146)  # d1 = 147, window (147, 152] contains the ex-day 150
    row = lambda k: daily.filter((pl.col("symbol") == "S13") & (pl.col("date") == m.sessions[k])).row(  # noqa: E731
        0, named=True
    )
    base = row(152)["adj_open"] / row(147)["adj_close"] - 1.0
    assert r["y_c2o_5"] == pytest.approx(base + div)
    # the day after the ex-date as decision date: window starts after it -> no dividend
    r2 = _lab(synth_labels, "S13", 150)  # d1 = 151
    base2 = row(156)["adj_open"] / row(151)["adj_close"] - 1.0
    assert r2["y_c2o_5"] == pytest.approx(base2)


def test_s09_suspension_exit_at_first_row_back(synth_labels):
    _, m, _, daily = synth_labels
    r = _lab(synth_labels, "S09", 305)  # d1 = 306; 311 is inside the 310..330 suspension
    assert r["end_5"] == m.sessions[331] and r["end_10"] == m.sessions[331] and r["end_20"] == m.sessions[331]
    row = lambda k: daily.filter((pl.col("symbol") == "S09") & (pl.col("date") == m.sessions[k])).row(  # noqa: E731
        0, named=True
    )
    assert r["y_c2o_5"] == pytest.approx(row(331)["adj_open"] / row(306)["adj_close"] - 1.0)
    # decision on 309: d1 = 310 has no row -> everything null
    assert _has(synth_labels, "S09", 309)
    r = _lab(synth_labels, "S09", 309)
    assert r["y_oc"] is None and r["y_c2o_5"] is None and r["end_5"] is None


def _has(synth, symbol: str, t: int) -> bool:
    _, m, lab, _ = synth
    return lab.filter((pl.col("symbol") == symbol) & (pl.col("date") == m.sessions[t])).height == 1


def test_s11_delisting_haircut(synth_labels):
    _, m, _, daily = synth_labels
    row = lambda k: daily.filter((pl.col("symbol") == "S11") & (pl.col("date") == m.sessions[k])).row(  # noqa: E731
        0, named=True
    )
    r = _lab(synth_labels, "S11", 396)  # d1 = 397; d1 + 5 = 402 never trades; last row is 400
    assert r["label_delisted"] is True
    assert r["y_c2o_5"] == pytest.approx(row(400)["adj_close"] / row(397)["adj_close"] - 1.0 - 0.20)
    assert r["end_5"] == m.sessions[401]
    # earlier decision: h=5 is a normal label, h=20 is delisted
    r = _lab(synth_labels, "S11", 380)  # d1 = 381; d1 + 5 = 386 trades, d1 + 20 = 401 does not
    assert r["label_delisted"] is True
    assert r["y_c2o_5"] == pytest.approx(row(386)["adj_open"] / row(381)["adj_close"] - 1.0)
    assert r["y_c2o_20"] == pytest.approx(row(400)["adj_close"] / row(381)["adj_close"] - 1.0 - 0.20)
    assert r["end_5"] == m.sessions[386] and r["end_20"] == m.sessions[401]
    # a healthy name at the same date is not flagged
    assert _lab(synth_labels, "S00", 380)["label_delisted"] is False


def test_s15_lock_blocks_entry(synth_labels):
    # locked at upper circuit on sessions 450..452 (open = high = low = close = prev * 1.05)
    for t in (449, 450, 451):
        r = _lab(synth_labels, "S15", t)
        assert r["entry_blocked_close"] is True, t
        assert r["entry_blocked_open"] is True, t
    assert _lab(synth_labels, "S15", 448)["entry_blocked_close"] is False
    assert _lab(synth_labels, "S15", 452)["entry_blocked_close"] is False  # d1 = 453, free again


def test_labels_are_only_for_u500_rows_and_unique(synth_labels):
    s, _, lab, _ = synth_labels
    assert lab.select("date", "sid").is_duplicated().sum() == 0
    u = io.read(s, "panel_universe").filter(pl.col("u500")).select("date", "sid").collect()
    assert lab.select("date", "sid").join(u, on=["date", "sid"], how="anti").is_empty()
    assert lab.join(u, on=["date", "sid"], how="semi").height == u.height


# ---- load_labels -------------------------------------------------------------------------------


def test_load_labels_purposes(synth_labels):
    s, m, _, _ = synth_labels
    cut = m.sessions[300]  # lockbox_start
    s2 = s.model_copy(update={"research": s.research.model_copy(update={"lockbox_start": cut})})
    prod = load_labels(s2, "production").collect()
    res = load_labels(s2, "research").collect()
    assert prod["date"].max() > cut
    assert res["date"].max() == m.sessions[299]
    # a research view must not contain outcomes from the lockbox: end dates after research_end are nulled
    late = res.filter(pl.col("date") == m.sessions[298])
    for y, end in (
        ("y_oc", "end_oc"),
        ("y_c2c_1", "end_c2c_1"),
        ("y_c2o_5", "end_5"),
        ("y_c2o_20", "end_20"),
    ):
        assert late[y].null_count() == late.height or late[end].max() <= m.sessions[299], y
    assert res.filter(pl.col("end_20") > m.sessions[299]).height == 0
    assert res.filter(pl.col("end_5") > m.sessions[299]).height == 0
    # the same rows in production keep their values
    p298 = prod.filter(pl.col("date") == m.sessions[298])
    assert p298["y_c2o_20"].null_count() < p298.height
    # an early row is identical in both views
    a = res.filter(pl.col("date") == m.sessions[100]).sort("sid")
    b = prod.filter(pl.col("date") == m.sessions[100]).sort("sid")
    assert a["y_c2o_20"].to_list() == b["y_c2o_20"].to_list()
    with pytest.raises(LockboxError):
        load_labels(s2, "lockbox")
    cols = load_labels(s2, "production", start=m.sessions[10], end=m.sessions[20], columns=["y_oc"]).collect()
    assert set(cols.columns) == {"date", "sid", "y_oc"}
