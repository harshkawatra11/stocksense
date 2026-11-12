from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from stocksense.core.clock import IST
from stocksense.data import io
from stocksense.features.base import FeatureContext
from stocksense.features.registry import load_all
from tests.fixtures.feat58 import write_env
from tests.fixtures.handmade import weekdays

REG = load_all()
N = 150
DS = weekdays(N, date(2019, 9, 2))
EVENT = 140  # index of the results day
EVENT_FEATURES = (
    "days_to_exdiv",
    "exdiv_yield",
    "days_to_board_meeting",
    "days_since_result",
    "result_abret",
    "result_abvol",
    "result_gap",
    "ann_count_5",
    "ann_count_21",
    "corp_event_next_5",
)


def ts(i: int, h: int, m: int = 0) -> datetime:
    return datetime.combine(DS[i], datetime.min.time(), tzinfo=IST) + timedelta(hours=h, minutes=m)


def series(s, name, sid=1, as_of=None) -> dict[date, object]:
    df = REG[name].fn(FeatureContext(s, as_of)).collect().filter(pl.col("sid") == sid)
    return dict(zip(df["date"].to_list(), df[name].to_list(), strict=True))


def model_data():
    rng = np.random.default_rng(3)
    m = rng.normal(0.0, 0.01, N)
    m[0] = 0.0
    ra = 0.8 * m + 0.004 * np.sin(np.arange(N))
    ra[0] = 0.0
    close_a = 100.0 * np.cumprod(1 + ra)
    return m, ra, close_a


@pytest.fixture
def world(tmp_data_root):
    s = tmp_data_root
    m, ra, close_a = model_data()
    opn = close_a.copy()
    opn[EVENT + 1] = close_a[EVENT] * 1.03
    turn = [1e7] * N
    turn[EVENT + 1], turn[EVENT + 2] = 3e7, 2e7
    stocks = {
        "AAA": {"sid": 1, "close": list(close_a), "open": list(opn), "turnover": turn},
        "BBB": {"sid": 2, "close": [50.0] * N},
    }
    write_env(s, DS, stocks, market={"date": DS, "ew500_ret": list(m)})
    return s


def ca_row(sym, ex, typ, amount=None, subject="x"):
    cols = ["symbol", "ex_date", "action_type", "dividend_amount", "subject_raw"]
    return pl.DataFrame([(sym, ex, typ, amount, subject)], schema=cols, orient="row")


def put_ca(s, *rows):
    io.upsert(s, "corporate_actions", pl.concat(rows, how="vertical_relaxed"))


def test_exdiv_known_seven_sessions_ahead(world):
    s = world
    put_ca(s, ca_row("AAA", DS[100], "dividend", 5.0, "d1"), ca_row("AAA", DS[120], "dividend", 2.0, "d2"))
    d = series(s, "days_to_exdiv")
    assert d[DS[92]] is None  # first dividend not yet known (known from 7 sessions before the ex-date)
    assert d[DS[93]] == 7 and d[DS[99]] == 1
    assert d[DS[100]] is None  # the ex-date itself is not "next"; the second is not known yet
    assert d[DS[113]] == 7 and d[DS[119]] == 1
    assert series(s, "days_to_exdiv", sid=2)[DS[95]] is None
    y = series(s, "exdiv_yield")
    pan = io.read(s, "panel_daily", columns=["sid", "date", "close"]).collect()
    c95 = pan.filter((pl.col("sid") == 1) & (pl.col("date") == DS[95]))["close"][0]
    assert y[DS[95]] == pytest.approx(5.0 / c95)


def test_exdiv_and_corp_event_ignore_other_types(world):
    s = world
    put_ca(s, ca_row("AAA", DS[100], "split", None, "s1"))
    assert all(v is None for v in series(s, "days_to_exdiv").values())
    f = series(s, "corp_event_next_5")
    assert f[DS[94]] == 0 and f[DS[95]] == 1 and f[DS[99]] == 1 and f[DS[100]] == 0
    assert f[DS[50]] == 0 and series(s, "corp_event_next_5", sid=2)[DS[97]] == 0


def test_corp_event_known_gate(world):
    # a bonus 3 sessions out is flagged; before the "known" date (7 sessions ahead) nothing can be flagged
    s = world
    put_ca(s, ca_row("AAA", DS[100], "bonus", None, "b1"))
    f = series(s, "corp_event_next_5")
    assert [f[DS[i]] for i in (94, 95, 97, 99)] == [0, 1, 1, 1]


def test_board_meetings_gated_by_announcement_time(world):
    s = world
    cols = ["symbol", "meeting_date", "purpose", "description", "announced_ts"]
    rows = [
        ("AAA", DS[110], "Results", "x", ts(90, 10)),
        ("AAA", DS[102], "Dividend", "y", ts(100, 21)),  # announced after 20:00: known the next session
    ]
    io.upsert(s, "board_meetings", pl.DataFrame(rows, schema=cols, orient="row"))
    d = series(s, "days_to_board_meeting")
    assert d[DS[89]] is None  # before the source starts
    assert d[DS[90]] == 20
    assert d[DS[100]] == 10  # the nearer meeting is announced only after 20:00
    assert d[DS[101]] == 1
    assert d[DS[102]] == 0  # the meeting day itself
    assert d[DS[103]] == 7  # first meeting still ahead
    assert d[DS[111]] == 30  # nothing ahead: the cap
    assert series(s, "days_to_board_meeting", sid=2)[DS[95]] == 30


def put_ann(s, rows):
    cols = ["seq_id", "symbol", "ts", "category"]
    io.upsert(s, "announcements", pl.DataFrame(rows, schema=cols, orient="row"))


def test_announcement_counts(world):
    s = world
    rows = [("1", "AAA", ts(60, 10), "Misc"), ("2", "AAA", ts(61, 21), "Misc")]
    rows += [(f"3{k}", "AAA", ts(62, 11, k), "Misc") for k in range(3)]
    put_ann(s, rows)
    c5, c21 = series(s, "ann_count_5"), series(s, "ann_count_21")
    assert c5[DS[59]] is None  # before the source starts
    # per-row counts: day 60 -> 1, day 61 -> 0 (the 21:00 one moves to day 62), day 62 -> 1 + 3
    assert c5[DS[63]] == 5 and c5[DS[64]] == 5
    assert c5[DS[65]] == 4  # day 60 drops out of the 5-row window
    assert c5[DS[66]] == 4 and c5[DS[67]] == 0
    assert c21[DS[68]] == 5 and c21[DS[82]] == 4 and c21[DS[83]] == 0
    assert series(s, "ann_count_5", sid=2)[DS[70]] == 0


def put_results(s):
    put_ann(
        s,
        [
            ("r1", "AAA", ts(EVENT, 16), "Financial Results"),
            ("r2", "AAA", ts(EVENT + 1, 9), "Outcome of Board Meeting"),  # same event, a day later
            ("x1", "AAA", ts(50, 10), "Press Release"),
        ],
    )
    cols = ["seq_number", "symbol", "filing_ts"]
    ri = [("f1", "AAA", ts(EVENT, 17)), ("f2", "AAA", ts(EVENT + 1, 10))]
    io.upsert(s, "results_index", pl.DataFrame(ri, schema=cols, orient="row"))


def test_days_since_result_and_dedup(world):
    s = world
    put_results(s)
    d = series(s, "days_since_result")
    assert d[DS[EVENT - 1]] is None  # before the first result event exists in the data
    assert d[DS[EVENT]] == 0
    assert d[DS[EVENT + 1]] == 1  # the second filing within 3 days is the same event, not a reset
    assert d[DS[EVENT + 9]] == 9
    # BBB has no results: after the source starts its count sits at the cap
    b = series(s, "days_since_result", sid=2)
    assert b[DS[EVENT - 1]] is None and b[DS[EVENT + 3]] == 90


def test_result_gap_vol_and_abnormal_return(world):
    s = world
    put_results(s)
    m, ra, close_a = model_data()
    gap = series(s, "result_gap")
    assert gap[DS[EVENT]] is None
    assert gap[DS[EVENT + 1]] == pytest.approx(0.03)  # reaction session is the one after a 16:00 release
    assert gap[DS[EVENT + 5]] == pytest.approx(0.03)

    lm, la = np.log1p(m), np.log1p(ra)

    def beta(i):  # estimated on the 126 rows BEFORE session i
        a, b = la[i - 126 : i], lm[i - 126 : i]
        return np.cov(a, b)[0, 1] / np.var(b, ddof=1)

    def ab(i):
        return ra[i] - beta(i) * m[i]

    ar = series(s, "result_abret")
    assert ar[DS[EVENT + 1]] is None  # needs both reaction sessions
    assert ar[DS[EVENT + 2]] == pytest.approx(ab(EVENT + 1) + ab(EVENT + 2))
    av = series(s, "result_abvol")
    assert av[DS[EVENT + 1]] is None
    assert av[DS[EVENT + 2]] == pytest.approx((3e7 + 2e7) / 2 / 1e7)


def test_results_need_both_announcement_and_filing(world):
    s = world
    put_ann(s, [("r1", "AAA", ts(EVENT, 16), "Financial Results")])  # no results_index row
    assert all(v is None for v in series(s, "days_since_result").values())
    assert all(v is None for v in series(s, "result_gap").values())


def test_missing_sources_are_all_null(world):
    s = world
    for name in EVENT_FEATURES:
        for sid in (1, 2):
            v = series(s, name, sid=sid)
            assert len(v) == N and all(x is None for x in v.values()), name


def test_truncation_matches_full_for_events(world):
    s = world
    put_ca(s, ca_row("AAA", DS[100], "dividend", 5.0, "d1"), ca_row("AAA", DS[130], "split", None, "s"))
    put_results(s)
    cut = DS[EVENT + 2]
    for name in EVENT_FEATURES:
        full = series(s, name)
        part = series(s, name, as_of=cut)
        for d, v in part.items():
            assert (v is None and full[d] is None) or v == pytest.approx(full[d]), (name, d)
