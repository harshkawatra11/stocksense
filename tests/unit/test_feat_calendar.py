from __future__ import annotations

from datetime import date

import polars as pl

from stocksense.data import io
from stocksense.features.base import FeatureContext
from stocksense.features.calendar_feats import BUDGET_DATES
from stocksense.features.registry import load_all
from tests.fixtures.feat58 import write_env
from tests.fixtures.handmade import weekdays

REG = load_all()


def col(s, name, as_of=None) -> dict[date, object]:
    df = REG[name].fn(FeatureContext(s, as_of)).collect()
    return dict(zip(df["date"].to_list(), df[name].to_list(), strict=True))


def env(s, start=date(2020, 1, 1), n=45, skip=(), extra=()):
    """Weekday sessions from `start` (minus `skip`, plus weekend `extra`)."""
    ds = sorted({d for d in weekdays(n, start) if d not in set(skip)} | set(extra))
    write_env(s, ds, {"A": {"sid": 1, "close": [100.0 + i for i in range(len(ds))]}})
    return ds


def test_dow_including_special_session(tmp_data_root):
    env(tmp_data_root, n=10, extra=[date(2020, 1, 4)])
    d = col(tmp_data_root, "dow")
    assert d[date(2020, 1, 1)] == 2  # Wednesday
    assert d[date(2020, 1, 3)] == 4  # Friday
    assert d[date(2020, 1, 4)] == 5  # Saturday special session


def test_dom_bucket_and_days_to_month_end(tmp_data_root):
    env(tmp_data_root)  # Jan 1 .. Mar 2020
    b = col(tmp_data_root, "dom_bucket")
    e = col(tmp_data_root, "days_to_month_end")
    assert [b[date(2020, 1, k)] for k in (1, 2, 3, 6)] == [0, 0, 0, 1]
    assert [b[date(2020, 1, k)] for k in (28, 29, 30, 31)] == [1, 2, 2, 2]
    assert [e[date(2020, 1, k)] for k in (31, 30, 29, 28)] == [0, 1, 2, 3]


def test_month_end_counts_holidays_out(tmp_data_root):
    # Friday 31 Jan is a holiday: Thursday 30 is the last session
    env(tmp_data_root, skip=[date(2020, 1, 31)])
    e = col(tmp_data_root, "days_to_month_end")
    assert e[date(2020, 1, 30)] == 0 and e[date(2020, 1, 29)] == 1
    assert col(tmp_data_root, "dom_bucket")[date(2020, 1, 28)] == 2


def test_sessions_to_next_holiday_inferred_and_capped(tmp_data_root):
    env(tmp_data_root, skip=[date(2020, 1, 15)])
    v = col(tmp_data_root, "sessions_to_next_holiday")
    assert v[date(2020, 1, 14)] == 0  # the holiday is tomorrow
    assert v[date(2020, 1, 13)] == 1
    assert v[date(2020, 1, 3)] == 7  # sessions Jan 6,7,8,9,10,13,14 before the holiday
    assert v[date(2020, 1, 1)] == 9
    assert v[date(2020, 1, 16)] == 10  # nothing ahead: the cap


def test_sessions_to_next_holiday_uses_published_holidays_when_truncated(tmp_data_root):
    s = tmp_data_root
    env(s, skip=[date(2020, 1, 15)])
    hol = pl.DataFrame({"date": [date(2020, 1, 15)], "description": ["Test Day"]})
    io.upsert(s, "holidays", hol)
    cut = date(2020, 1, 10)
    assert col(s, "sessions_to_next_holiday", as_of=cut)[cut] == 2  # Jan 13, 14 then the holiday
    full = col(s, "sessions_to_next_holiday")
    assert full[cut] == 2


def _fo(rows):
    cols = [
        "symbol",
        "instrument_type",
        "expiry_date",
        "strike_price",
        "option_type",
        "date",
        "close",
        "settle_price",
        "oi",
    ]
    return pl.DataFrame(
        rows, schema=cols, orient="row", schema_overrides={"strike_price": pl.Float64, "option_type": pl.Utf8}
    )


def test_expiry_week_from_fo_data_beats_rule(tmp_data_root):
    s = tmp_data_root
    ds = env(s)
    # January's real expiry is Thursday 23 Jan (not the last Thursday, 30 Jan)
    io.upsert(
        s,
        "fo_bhavcopy",
        _fo([("NIFTY", "IDF", date(2020, 1, 23), None, None, date(2020, 1, 2), 1.0, 1.0, 5)]),
    )
    v = col(s, "is_expiry_week")
    assert v[date(2020, 1, 22)] == 1 and v[date(2020, 1, 20)] == 1
    assert v[date(2020, 1, 29)] == 0 and v[date(2020, 1, 15)] == 0
    # February has no F&O data: the rule gives the last Thursday, 27 Feb -> week of 24 Feb
    assert v[date(2020, 2, 25)] == 1 and v[date(2020, 2, 18)] == 0
    assert len(v) == len(ds)


def test_expiry_rule_moves_to_previous_session_on_holiday(tmp_data_root):
    # last Thursday of Feb 2020 is the 27th; make Thu 27 and Wed 26 holidays -> expiry Tue 25, same week
    env(tmp_data_root, skip=[date(2020, 2, 27)])
    v = col(tmp_data_root, "is_expiry_week")
    assert v[date(2020, 2, 24)] == 1
    # last Thursday of Mar 2020 is the 26th, in the week of 23 Mar; sessions run to ~2 Mar only, so no row


def test_budget_week(tmp_data_root):
    env(tmp_data_root)
    v = col(tmp_data_root, "is_budget_week")
    assert v[date(2020, 1, 28)] == 1  # the week of Saturday 1 Feb 2020 (Mon 27 Jan - Sun 2 Feb)
    assert v[date(2020, 1, 31)] == 1
    assert v[date(2020, 1, 21)] == 0 and v[date(2020, 2, 4)] == 0


def test_budget_table_years_use_actual_dates(tmp_data_root):
    # 2012: budget on 16 March, so 1 February 2012 is NOT a budget week
    s = tmp_data_root
    env(s, start=date(2012, 1, 30), n=50)
    v = col(s, "is_budget_week")
    assert v[date(2012, 1, 31)] == 0
    assert v[date(2012, 3, 14)] == 1 and v[date(2012, 3, 21)] == 0
    assert BUDGET_DATES[2012] == (date(2012, 3, 16),)


def test_all_calendar_features_have_every_session(tmp_data_root):
    ds = env(tmp_data_root, n=12)
    for name in (
        "dow",
        "dom_bucket",
        "days_to_month_end",
        "is_expiry_week",
        "sessions_to_next_holiday",
        "is_budget_week",
    ):
        assert sorted(col(tmp_data_root, name)) == ds
        assert REG[name].label and REG[name].kind == "mkt"
