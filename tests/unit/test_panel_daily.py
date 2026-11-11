from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from stocksense.data import io
from stocksense.panel.adjust import compute_gap_factors, snap
from stocksense.panel.returns import build_daily_panel
from stocksense.panel.security_master import build_security_master
from tests.fixtures.handmade import weekdays
from tests.fixtures.synth import _settings_for, make_market, make_tiny_market, write_market

IDENTITY_TOL = 1e-9


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth_panel")
    m = make_market()
    write_market(m, root)
    s = _settings_for(root)
    sm = build_security_master(s)
    build_daily_panel(s)
    panel = io.read(s, "panel_daily").collect().sort("sid", "date")
    events = io.read(s, "adj_events").collect()
    return s, m, sm, panel, events


def _row(built, symbol: str, idx: int) -> dict:
    _, m, _, panel, _ = built
    r = panel.filter((pl.col("symbol") == symbol) & (pl.col("date") == m.sessions[idx]))
    assert r.height == 1, (symbol, idx)
    return r.row(0, named=True)


def _truth(built, key: str, idx: int) -> float:
    _, m, *_ = built
    return m.truth.filter((pl.col("symbol_key") == key) & (pl.col("date") == m.sessions[idx]))[
        "true_ret"
    ].item()


# ---- global properties --------------------------------------------------------------------------


def test_one_row_per_sid_date_and_row_count(built):
    _, m, sm, panel, _ = built
    assert panel.height == m.eq.height
    assert panel.select("sid", "date").is_duplicated().sum() == 0
    assert panel["series"].is_in(["EQ", "BE", "BZ"]).all()


def test_adj_close_identity_on_all_non_first_rows(built):
    _, _, _, panel, _ = built
    d = panel.sort("sid", "date").with_columns(
        (pl.col("adj_close") / pl.col("adj_close").shift(1).over("sid") - 1.0).alias("_implied"),
        pl.int_range(pl.len()).over("sid").alias("_i"),
    )
    d = d.filter(pl.col("_i") > 0)
    err = (d["_implied"] - d["ret"]).abs().max()
    print(f"max |adj_close identity error| = {err:.3e}")
    assert err < IDENTITY_TOL
    # latest row of every sid is unadjusted
    last = panel.sort("sid", "date").group_by("sid").last()
    assert (last["adj"] - 1.0).abs().max() < 1e-12


def test_first_row_returns_null(built):
    _, _, _, panel, _ = built
    first = panel.sort("sid", "date").group_by("sid", maintain_order=True).first()
    assert first["ret"].null_count() == first.height
    assert first["ret_co"].null_count() == first.height
    assert first["ret_oc"].null_count() == first.height
    assert first["sessions_since_prev"].null_count() == first.height


def test_compounded_tret_matches_truth(built):
    _, m, sm, panel, _ = built
    tr = m.truth.join(sm.select("symbol", "sid"), left_on="symbol_key", right_on="symbol").filter(
        pl.col("true_ret").is_not_null()
    )
    truth = tr.group_by("sid").agg((pl.col("true_ret") + 1.0).log().sum().exp().alias("truth_cum"))
    mine = (
        panel.filter(pl.col("tret").is_not_null())
        .group_by("sid")
        .agg((pl.col("tret") + 1.0).log().sum().exp().alias("my_cum"), pl.col("symbol").first().alias("sym"))
    )
    j = mine.join(truth, on="sid").with_columns((pl.col("my_cum") / pl.col("truth_cum") - 1.0).alias("err"))
    assert j.height == 40
    j = j.join(
        sm.sort("valid_from").group_by("sid").first().select("sid", pl.col("symbol").alias("first_symbol")),
        on="sid",
    )
    worst = j.sort(pl.col("err").abs(), descending=True)
    print("compounded tret mismatch, worst five:", worst.select("first_symbol", "err").head(5).rows())
    # S25: the synthetic price does not drop on the rights date, but the plan's rights rule applies the
    # theoretical ex-rights factor (see DECISIONS), so its compounded return is off by that factor.
    ok = j.filter(pl.col("first_symbol") != "S25")
    assert ok["err"].abs().max() < 0.02
    # S16: the demerger convention drops the overnight gap of that one day; still within 2%
    s25 = j.filter(pl.col("first_symbol") == "S25")["err"].item()
    assert s25 > 0.02  # explained by the rights factor, not a bug: asserted so the exclusion is exact


# ---- scripted events --------------------------------------------------------------------------


def test_s03_split_ca(built):
    r = _row(built, "S03", 200)
    assert r["gap_src"] == "ca"
    assert r["gap_factor"] == pytest.approx(0.2)
    # truth: raw price / 5, so the corrected return equals the true return up to tick rounding
    assert r["ret"] == pytest.approx(_truth(built, "S03", 200), abs=0.003)
    prev = _row(built, "S03", 199)
    assert abs(r["adj_close"] / prev["adj_close"] - 1.0 - r["ret"]) < IDENTITY_TOL
    assert r["adj_volume"] == pytest.approx(r["volume"] / r["adj"])
    # volume x5 after the split: adj_volume is on the old (pre-split) share basis
    assert r["adj"] == pytest.approx(1.0)
    assert prev["adj"] == pytest.approx(0.2, rel=1e-9)


def test_s05_bonus_ca(built):
    r = _row(built, "S05", 250)
    assert r["gap_src"] == "ca" and r["gap_factor"] == pytest.approx(0.5)
    assert r["ret"] == pytest.approx(_truth(built, "S05", 250), abs=0.01)


def test_s06_wrong_table_factor_implied_at_ca(built):
    r = _row(built, "S06", 260)
    assert r["gap_src"] == "implied_at_ca"
    assert r["gap_factor"] == pytest.approx(0.5)
    _, _, _, _, ev = built
    e = ev.filter((pl.col("symbol") == "S06") & (pl.col("gap_src") == "implied_at_ca"))
    assert e.height == 1 and e["f_ca"].item() == pytest.approx(0.667)


def test_s14_suspected_split_without_table_row(built):
    r = _row(built, "S14", 220)
    assert r["gap_src"] == "suspected"
    assert r["gap_factor"] == pytest.approx(0.5)


def test_s16_demerger_implied_event(built):
    r = _row(built, "S16", 330)
    assert r["gap_src"] == "implied_event"
    assert r["ret_co"] == pytest.approx(0.0, abs=1e-12)
    # the overnight move is mechanical, so the day's return is the intraday move
    assert r["ret"] == pytest.approx(r["close"] / r["open"] - 1.0)


def test_s25_rights_factor(built):
    r = _row(built, "S25", 280)
    assert r["gap_src"] == "rights"
    p = r["prev_close"]
    # Rights 1:4 @ Premium Rs 40, face Rs 10 -> issue price 50; f = (4P + 1*50) / (5P)
    assert r["gap_factor"] == pytest.approx((4 * p + 50.0) / (5 * p))


def test_s13_dividend(built):
    r = _row(built, "S13", 150)
    assert r["div_ret"] == pytest.approx(5.0 / r["prev_close"])
    assert r["tret"] - r["ret"] == pytest.approx(r["div_ret"])
    assert r["tret"] == pytest.approx(_truth(built, "S13", 150), abs=0.003)
    assert r["gap_src"] == ""
    assert _row(built, "S13", 151)["div_ret"] == 0.0


def test_s07_rename_is_one_continuous_series(built):
    _, m, _, panel, _ = built
    a, b = _row(built, "S07", 299), _row(built, "S07N", 300)
    assert a["sid"] == b["sid"]
    assert b["gap_src"] == "" and b["sessions_since_prev"] == 1
    assert b["ret"] == pytest.approx(_truth(built, "S07", 300), abs=0.003)
    assert panel.filter(pl.col("sid") == a["sid"]).height == 601
    assert b["industry"] == a["industry"] == "Autos"


def test_s08_rename_continuous(built):
    b = _row(built, "S08N", 310)
    assert b["gap_src"] == "" and b["ret"] == pytest.approx(_truth(built, "S08", 310), abs=0.003)


def test_s09_suspension_flag_stale(built):
    r = _row(built, "S09", 331)
    assert r["sessions_since_prev"] == 22
    assert r["flag_stale"] is True
    assert _row(built, "S09", 332)["flag_stale"] is False


def test_s15_locked_upper_circuit(built):
    for t in (450, 451, 452):
        r = _row(built, "S15", t)
        assert r["flag_locked_up"] is True and r["flag_locked_dn"] is False
        # open = high = low = close in the synthetic lock rows, so the one-price flag is set too
        assert r["flag_one_price"] is True
    assert _row(built, "S15", 449)["flag_locked_up"] is False
    assert _row(built, "S15", 453)["flag_locked_up"] is False


def test_s23_series_switch_one_sid(built):
    assert _row(built, "S23", 379)["series"] == "EQ"
    assert _row(built, "S23", 380)["series"] == "BE"
    assert _row(built, "S23", 420)["series"] == "EQ"
    assert _row(built, "S23", 380)["gap_src"] == ""


def test_s19_first_row_null_and_s11_delisted(built):
    r = _row(built, "S19", 350)
    assert r["ret"] is None and r["adj"] == pytest.approx(r["adj"])
    _, m, _, panel, _ = built
    assert panel.filter(pl.col("symbol") == "S11")["date"].max() == m.sessions[400]


def test_normal_days_have_no_factor(built):
    _, _, _, panel, ev = built
    # only the scripted events carry a gap_src
    srcs = panel.filter(pl.col("gap_src") != "").group_by("symbol").agg(pl.col("gap_src").first())
    assert sorted(srcs["symbol"].to_list()) == ["S03", "S05", "S06", "S14", "S16", "S25"]
    assert panel["flag_bigmove"].sum() == 0
    assert panel["flag_unexplained_gap"].sum() == 0


def test_delivery_and_industry_joined(built):
    _, _, _, panel, _ = built
    assert panel["deliv_pct"].null_count() == 0
    assert set(panel["industry"].unique()) == {"Banks", "Software", "Pharma", "Autos", "UNKNOWN"}
    # NIFTYBEES (S17) has no sectors row
    assert panel.filter(pl.col("symbol") == "NIFTYBEES")["industry"].unique().to_list() == ["UNKNOWN"]


def test_adj_events_audit(built):
    _, _, _, panel, ev = built
    assert set(ev["gap_src"].unique()) == {"ca", "implied_at_ca", "suspected", "implied_event", "rights"}
    assert ev.select("sid", "date", "symbol").is_duplicated().sum() == 0
    assert ev.height == panel.filter(pl.col("gap_src") != "").height


# ---- tiny market: exact arithmetic --------------------------------------------------------------


def test_tiny_market_exact(tmp_path):
    m = make_tiny_market()
    write_market(m, tmp_path)
    s = _settings_for(tmp_path)
    build_security_master(s)
    build_daily_panel(s)
    p = io.read(s, "panel_daily").collect().sort("symbol", "date")
    a = p.filter(pl.col("symbol") == "AAA")
    # AAA day 2: open 103 = prev close 103, close 102 -> ret = ret_oc = 102/103 - 1, ret_co = 0
    r = a.row(2, named=True)
    assert r["ret"] == pytest.approx(102 / 103 - 1)
    assert r["ret_co"] == pytest.approx(0.0)
    assert r["ret_oc"] == pytest.approx(102 / 103 - 1)
    # day 1: open 101 vs prev close 101 (closes: 101,103,...): open[1]=101 -> co = 101/101 - 1 = 0
    assert a.row(1, named=True)["ret_co"] == pytest.approx(0.0)
    assert a.row(1, named=True)["ret"] == pytest.approx(103 / 101 - 1)
    assert (a["adj"] == 1.0).all() and (a["adj_close"] == a["close"]).all()
    assert (a["tret"].drop_nulls() == a["ret"].drop_nulls()).all()
    assert a["industry"].unique().to_list() == ["Banks"]
    assert a["deliv_pct"].unique().to_list() == [50.0]


# ---- compute_gap_factors on hand-made frames ----------------------------------------------------


def _rows(opens, closes, prevs=None, sid=1, n_dates=None):
    n = len(opens)
    d = weekdays(n)
    prevs = prevs or [opens[0]] + list(closes[:-1])
    return pl.DataFrame(
        {
            "sid": [sid] * n,
            "date": d,
            "session_idx": list(range(n)),
            "open": [float(x) for x in opens],
            "close": [float(x) for x in closes],
            "prev_close": [float(x) for x in prevs],
        },
        schema_overrides={"sid": pl.Int32, "session_idx": pl.Int64},
    )


def _ca(rows: list[dict]) -> pl.DataFrame:
    base = {
        "sid": 1,
        "ex_date": None,
        "action_type": "split",
        "ratio_num": None,
        "ratio_den": None,
        "factor_price": None,
        "dividend_amount": None,
        "face_before": None,
        "face_after": None,
        "subject_raw": "",
        "parse_status": "ok",
        "rights_price": None,
    }
    schema = {
        "sid": pl.Int32,
        "ex_date": pl.Date,
        "action_type": pl.Utf8,
        "ratio_num": pl.Float64,
        "ratio_den": pl.Float64,
        "factor_price": pl.Float64,
        "dividend_amount": pl.Float64,
        "face_before": pl.Float64,
        "face_after": pl.Float64,
        "subject_raw": pl.Utf8,
        "parse_status": pl.Utf8,
        "rights_price": pl.Float64,
    }
    return pl.DataFrame([{**base, **r} for r in rows], schema=schema)


def _flat_then_drop(drop_at: int, n: int = 30, factor: float = 0.5):
    opens = [100.0] * drop_at + [100.0 * factor] * (n - drop_at)
    closes = list(opens)
    prevs = [100.0] + closes[:-1]
    return _rows(opens, closes, prevs)


def test_ca_shifted_when_price_drops_two_sessions_after_ex_date():
    rows = _flat_then_drop(12)
    d = rows["date"]
    ca = _ca([{"ex_date": d[10], "factor_price": 0.5}])
    out = compute_gap_factors(rows, ca)
    assert out.height == 1
    r = out.row(0, named=True)
    assert r["date"] == d[12] and r["gap_src"] == "ca_shifted" and r["gap_factor"] == pytest.approx(0.5)


def test_normal_nine_percent_gap_is_not_adjusted():
    opens = [100.0] * 10 + [91.0] * 10
    out = compute_gap_factors(_rows(opens, opens, [100.0] + opens[:-1]), _ca([]))
    assert out.height == 0
    assert out.columns == ["sid", "date", "gap_factor", "gap_src", "r_obs", "f_ca", "ca_types", "note"]


def test_table_factor_without_price_evidence_is_ignored():
    rows = _rows([100.0] * 20, [100.0] * 20)
    ca = _ca([{"ex_date": rows["date"][8], "factor_price": 0.5}])
    out = compute_gap_factors(rows, ca)
    assert out["gap_src"].to_list() == ["ca_ignored"]
    assert out["gap_factor"].to_list() == [1.0]
    assert out["f_ca"].to_list() == [0.5]


def test_table_factor_wrong_direction_is_ignored():
    # table says price halves; the price jumps up 50%: signs disagree -> ignored
    opens = [100.0] * 10 + [150.0] * 10
    out = compute_gap_factors(
        _rows(opens, opens, [100.0] + opens[:-1]), _ca([{"ex_date": weekdays(20)[10], "factor_price": 0.5}])
    )
    assert out["gap_src"].to_list() == ["ca_ignored"]


def test_two_events_same_date_multiply():
    rows = _flat_then_drop(10, factor=0.25)
    d = rows["date"][10]
    ca = _ca(
        [
            {"ex_date": d, "factor_price": 0.5, "action_type": "split"},
            {"ex_date": d, "factor_price": 0.5, "action_type": "bonus", "subject_raw": "b"},
        ]
    )
    out = compute_gap_factors(rows, ca)
    assert out.height == 1 and out["gap_factor"].item() == pytest.approx(0.25)
    assert out["ca_types"].item() == "bonus,split"


def test_rights_not_applied_when_issue_price_above_market():
    rows = _rows([100.0] * 20, [100.0] * 20)
    ca = _ca(
        [
            {
                "ex_date": rows["date"][8],
                "action_type": "rights",
                "ratio_num": 1.0,
                "ratio_den": 4.0,
                "rights_price": 150.0,
            }
        ]
    )
    assert compute_gap_factors(rows, ca).height == 0


def test_rights_numbers_from_table_columns():
    rows = _rows([100.0] * 20, [100.0] * 20)
    ca = _ca(
        [
            {
                "ex_date": rows["date"][8],
                "action_type": "rights",
                "ratio_num": 1.0,
                "ratio_den": 4.0,
                "rights_price": 60.0,
            }
        ]
    )
    out = compute_gap_factors(rows, ca)
    # f = (4*100 + 1*60) / (5*100) = 0.92
    assert out["gap_src"].to_list() == ["rights"] and out["gap_factor"].item() == pytest.approx(0.92)


def test_first_row_never_gets_factor():
    rows = _rows([50.0, 50.0, 50.0, 50.0], [50.0] * 4, [100.0, 50.0, 50.0, 50.0])
    assert compute_gap_factors(rows, _ca([])).height == 0


def test_suspected_needs_price_floor_and_ratio_match():
    # 0.55 matches no suspected ratio within 3% -> untouched; a base under Rs 5 is never adjusted
    out = compute_gap_factors(
        _rows([100.0] * 5 + [55.0] * 5, [100.0] * 5 + [55.0] * 5, [100.0] * 6 + [55.0] * 4), _ca([])
    )
    assert out.height == 0
    low = _rows([4.0] * 5 + [2.0] * 5, [4.0] * 5 + [2.0] * 5, [4.0] * 6 + [2.0] * 4)
    assert compute_gap_factors(low, _ca([])).height == 0


def test_suspected_upward_ratio():
    opens = [100.0] * 6 + [1000.0] * 4
    out = compute_gap_factors(_rows(opens, opens, [100.0] * 7 + [1000.0] * 3), _ca([]))
    assert out["gap_src"].to_list() == ["suspected"] and out["gap_factor"].item() == pytest.approx(10.0)


def test_snap():
    import numpy as np

    got = snap(np.array([0.505, 0.30, 0.6667, 1.0]))
    assert got.tolist() == pytest.approx([0.5, 0.30, 2 / 3, 1.0])
    # 0.30 is 10% from 1/3 (and nothing else) so it is unchanged


def test_dates_are_python_dates():
    assert isinstance(weekdays(1)[0], date)


# ---- event inside the session (ADANIENT 2015-06-03): the open still shows the old price ----------


def _event_day_rows(open_, close_, n=20, at=10):
    opens = [100.0] * at + [open_] + [close_] * (n - at - 1)
    closes = [100.0] * at + [close_] * (n - at)
    return _rows(opens, closes, [100.0] + closes[:-1])


def test_intraday_demerger_gets_close_based_factor():
    rows = _event_day_rows(open_=90.0, close_=20.0)  # open -10%, close -80%
    ca = _ca(
        [{"ex_date": rows["date"][10], "action_type": "scheme_of_arrangement", "parse_status": "unparsed"}]
    )
    out = compute_gap_factors(rows, ca)
    assert out.height == 1
    r = out.row(0, named=True)
    assert r["date"] == rows["date"][10]
    assert r["gap_src"] == "implied_event_close"
    assert r["gap_factor"] == pytest.approx(0.2)


def test_event_with_mild_close_move_and_small_open_gap_is_untouched():
    rows = _event_day_rows(open_=98.0, close_=75.0)  # -25% close: a real move, not mechanical
    ca = _ca(
        [{"ex_date": rows["date"][10], "action_type": "scheme_of_arrangement", "parse_status": "unparsed"}]
    )
    assert compute_gap_factors(rows, ca).height == 0


def test_big_close_crash_without_a_table_event_is_not_adjusted():
    rows = _event_day_rows(open_=98.0, close_=20.0)
    assert compute_gap_factors(rows, _ca([])).height == 0


def test_intraday_event_day_has_zero_ret_and_null_open_split(tmp_data_root):
    from tests.fixtures.handmade import eq_rows, write_raw

    dates = weekdays(20)
    opens = [100.0] * 10 + [90.0] + [20.0] * 9
    closes = [100.0] * 10 + [20.0] * 10
    eq = eq_rows("AAA", dates, closes, opens=opens, prev_closes=[100.0] + closes[:-1])
    ca = _ca([{"ex_date": dates[10], "action_type": "scheme_of_arrangement", "parse_status": "unparsed"}])
    ca = ca.drop("sid").with_columns(pl.lit("AAA").alias("symbol"))
    write_raw(tmp_data_root, bhavcopy_eq=eq, corporate_actions=ca)
    build_security_master(tmp_data_root)
    build_daily_panel(tmp_data_root)
    p = io.read(tmp_data_root, "panel_daily").collect().sort("date")
    r = p.row(10, named=True)
    assert r["gap_src"] == "implied_event_close"
    # whole day mechanical: ret = 20 / (100 * 0.2) - 1 = 0; open/close splits are undefined
    assert r["ret"] == pytest.approx(0.0, abs=1e-12)
    assert r["ret_co"] is None and r["ret_oc"] is None
    assert not r["flag_bigmove"]
    # adjusted history is continuous: adj_close before the event equals the post-event level
    assert p.row(9, named=True)["adj_close"] == pytest.approx(20.0)
