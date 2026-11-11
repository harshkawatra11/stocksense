from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, FixedClock, SteppingClock
from stocksense.core.state import State
from stocksense.data.ingest_common import IngestContext
from stocksense.data.ingest_corpact import (
    ingest_corpact_range,
    parse_corpact_rows,
    parse_subject,
    reparse_unparsed,
    windows,
)
from stocksense.data.io import read, upsert
from stocksense.data.nse_http import NseClient

FILES = Path(__file__).resolve().parents[1] / "fixtures" / "files"


# ---- parse_subject: the card's exact strings ----------------------------------------------------


def test_bonus_1_1():
    p = parse_subject("Bonus 1:1", None)
    # A:B = 1:1 -> factor B/(A+B) = 1/2
    assert (p.action_type, p.ratio_num, p.ratio_den, p.factor_price, p.parse_status) == (
        "bonus", 1.0, 1.0, 0.5, "ok",
    )  # fmt: skip


def test_bonus_3_2():
    p = parse_subject("Bonus 3:2", None)
    # 3 new for every 2 held: factor 2/(3+2) = 0.4
    assert p.action_type == "bonus" and p.factor_price == pytest.approx(0.4) and p.parse_status == "ok"


def test_split_10_to_1():
    p = parse_subject("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share", None)
    # faces 10 -> 1: factor 1/10; ratio_num = 10, ratio_den = 1
    assert p.action_type == "split" and p.factor_price == pytest.approx(0.1)
    assert (p.face_before, p.face_after, p.ratio_num, p.ratio_den) == (10.0, 1.0, 10.0, 1.0)
    assert p.dividend_amount is None  # the "Per Share" amounts of a split are not dividends
    assert p.parse_status == "ok"


def test_split_10_to_2():
    p = parse_subject("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share", None)
    # faces 10 -> 2: factor 0.2
    assert p.factor_price == pytest.approx(0.2) and (p.face_before, p.face_after) == (10.0, 2.0)


def test_dividend_simple():
    p = parse_subject("Interim Dividend - Rs 2 Per Share", None)
    assert (p.action_type, p.dividend_amount, p.parse_status) == ("dividend", 2.0, "ok")
    assert p.factor_price is None


def test_dividend_with_extra_spaces():
    p = parse_subject("Interim Dividend Rs  - Rs 2.50 Per Share", None)
    assert (p.action_type, p.dividend_amount, p.parse_status) == ("dividend", 2.5, "ok")


def test_two_dividends_are_added():
    p = parse_subject("Interim Dividend - Rs 5 Per Share/ Special Dividend - Rs 10 Per Share", None)
    # 5 + 10 = 15
    assert (p.action_type, p.dividend_amount) == ("dividend", 15.0)


def test_distribution_units():
    p = parse_subject(
        "Third Distribution -Interest Payment - Rs 2.20 Per Unit/ Return Of Capital -Re 0.80 Per Unit", None
    )
    # 2.20 + 0.80 = 3.0, action 'distribution' because the word appears and 'dividend' does not
    assert (p.action_type, p.parse_status) == ("distribution", "ok")
    assert p.dividend_amount == pytest.approx(3.0)


def test_rights_1_7_premium_30_face_10():
    p = parse_subject("Rights 1:7 @ Premium Rs 30/-", 10.0)
    # 1 new share per 7 held at premium 30 on face 10: price 40
    assert (p.action_type, p.ratio_num, p.ratio_den, p.rights_price, p.parse_status) == (
        "rights", 1.0, 7.0, 40.0, "ok",
    )  # fmt: skip
    assert p.factor_price is None and p.dividend_amount is None  # premium is not a dividend


def test_rights_issue_30_37():
    p = parse_subject("Rights Issue 30:37@ Premium Rs 53/-", 10.0)
    assert (p.ratio_num, p.ratio_den, p.rights_price, p.parse_status) == (30.0, 37.0, 63.0, "ok")


def test_rights_fractional_premium():
    p = parse_subject("Rights 4:21 @ Premium Rs 1.80/-", 1.0)
    # premium 1.80 + face 1 = 2.80
    assert (p.ratio_num, p.ratio_den) == (4.0, 21.0) and p.rights_price == pytest.approx(2.80)


def test_rights_without_face_value_is_unparsed_but_keeps_the_ratio():
    p = parse_subject("Rights 1:7 @ Premium Rs 30/-", None)
    # face unknown: 10 is NOT assumed, so no price; the ratio is still recorded
    assert (p.ratio_num, p.ratio_den, p.rights_price, p.parse_status) == (1.0, 7.0, None, "unparsed")


def test_rights_at_par():
    p = parse_subject("Rights 1:5 @ Par", 10.0)
    assert p.rights_price == 10.0 and p.parse_status == "ok"


def test_rights_issue_with_dash_before_ratio():
    p = parse_subject("Rights Issue - 3:5 @ Premium Rs.93/-", 10.0)
    assert (p.ratio_num, p.ratio_den, p.rights_price) == (3.0, 5.0, 103.0)


def test_buyback_unparsed():
    p = parse_subject("Buy Back", None)
    assert (p.action_type, p.parse_status, p.factor_price) == ("buyback", "unparsed", None)


def test_buyback_price_is_not_a_dividend():
    p = parse_subject("Buy Back - Rs 700 Per Share", None)
    assert p.action_type == "buyback" and p.dividend_amount is None


def test_scheme_of_arrangement_unparsed():
    p = parse_subject("Scheme Of Arrangement", None)
    assert (p.action_type, p.parse_status) == ("scheme_of_arrangement", "unparsed")


def test_dividend_without_amount_is_unparsed():
    p = parse_subject("Interim Dividend", None)
    assert (p.action_type, p.dividend_amount, p.parse_status) == ("dividend", None, "unparsed")


def test_bonus_and_split_combined():
    p = parse_subject("Bonus 1:1 And Face Value Split From Rs 10/- To Rs 5/-", None)
    # bonus factor 1/2 times split factor 5/10 = 0.25
    assert p.action_type == "bonus+split" and p.factor_price == pytest.approx(0.25)
    assert (p.face_before, p.face_after) == (10.0, 5.0) and p.parse_status == "ok"


def test_agm_is_noise():
    p = parse_subject("Annual General Meeting", None)
    assert (p.action_type, p.parse_status) == ("noise", "ok")


def test_agm_with_a_dividend_is_a_dividend():
    p = parse_subject("Annual General Meeting/Dividend - Rs 6 Per Share", None)
    assert (p.action_type, p.dividend_amount) == ("dividend", 6.0)


def test_dividend_per_equity_share_wording():
    p = parse_subject("Interim Dividend Of Rs.5/- Per Equity Share", None)
    assert (p.action_type, p.dividend_amount) == ("dividend", 5.0)


def test_consolidation():
    p = parse_subject("Consolidation Of Shares - From Rs 1/- Per Share To Rs 10/- Per Share", None)
    # faces 1 -> 10: factor 10 (> 1)
    assert p.action_type == "consolidation" and p.factor_price == pytest.approx(10.0)


def test_split_without_amounts_is_unparsed():
    p = parse_subject("Face Value Split (Sub-Division)", 10.0)
    assert (p.action_type, p.factor_price, p.parse_status) == ("split", None, "unparsed")


def test_unknown_subject_is_other_unparsed():
    p = parse_subject("Something Entirely Different", None)
    assert (p.action_type, p.parse_status) == ("other", "unparsed")


def test_demerger_and_amalgamation_and_capital_reduction():
    assert parse_subject("Demerger", None).action_type == "demerger"
    assert parse_subject("Amalgamation", None).action_type == "amalgamation"
    assert parse_subject("Scheme Of Merger", None).action_type == "amalgamation"
    assert parse_subject("Capital Reduction", None).action_type == "capital_reduction"


# ---- API rows -----------------------------------------------------------------------------------


def _sample() -> list[dict]:
    return json.loads((FILES / "corpact_api_sample.json").read_text(encoding="utf-8"))


def test_parse_corpact_rows_sample():
    rows = _sample()
    df = parse_corpact_rows(rows)
    # fixture: 12 real rows (3 dividends, 2 splits, 2 bonuses, 2 rights, 3 others)
    assert df.height == len(rows) == 12
    assert df.schema["ex_date"] == pl.Date and df.schema["record_date"] == pl.Date
    first = df.row(0, named=True)
    # CCL, "Dividend - Rs 3 Per Share", ex 01-Sep-2026
    assert (first["symbol"], first["ex_date"], first["action_type"], first["dividend_amount"]) == (
        "CCL", date(2026, 9, 1), "dividend", 3.0,
    )  # fmt: skip
    assert first["source"] == "nse_api" and first["isin"] == "INE421D01014" and first["series"] == "EQ"
    rights = df.filter(pl.col("action_type") == "rights")
    # face value comes from the API field, so the rights rows are fully priced
    assert rights.height == 2 and rights["parse_status"].to_list() == ["ok", "ok"]


def test_parse_corpact_rows_dash_dates_and_face():
    rows = [
        {"symbol": "A", "series": "EQ", "exDate": "-", "recDate": "-", "subject": "Dividend - Rs 1 Per Share",
         "faceVal": "-", "isin": "X"},
        {"symbol": "B", "series": "EQ", "exDate": "02-Nov-2026", "recDate": "-",
         "subject": "Rights 1:7 @ Premium Rs 30/-", "faceVal": "-", "isin": "Y"},
    ]  # fmt: skip
    df = parse_corpact_rows(rows)
    # A has no ex-date: dropped; B has face '-' (null): rights price unknown -> unparsed; record date null
    assert df["symbol"].to_list() == ["B"]
    r = df.row(0, named=True)
    assert r["record_date"] is None and r["rights_price"] is None and r["parse_status"] == "unparsed"


# ---- windows and ingest -------------------------------------------------------------------------


def test_windows_are_31_days_inclusive():
    w = windows(date(2026, 9, 1), date(2026, 11, 5))
    # 01-Sep..01-Oct (31 days), 02-Oct..01-Nov (31 days), 02-Nov..05-Nov
    assert w == [
        (date(2026, 9, 1), date(2026, 10, 1)),
        (date(2026, 10, 2), date(2026, 11, 1)),
        (date(2026, 11, 2), date(2026, 11, 5)),
    ]


class Api:
    """MockTransport handler for /api/corporates-corporateActions; answers `rows` for every window."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, str]] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        q = req.url.params
        self.calls.append((q["from_date"], q["to_date"]))
        a = datetime.strptime(q["from_date"], "%d-%m-%Y").date()
        b = datetime.strptime(q["to_date"], "%d-%m-%Y").date()
        keep = [r for r in self.rows if a <= datetime.strptime(r["exDate"], "%d-%b-%Y").date() <= b]
        return httpx.Response(200, json=keep)


def _ctx(settings, handler) -> IngestContext:
    clock = FixedClock(datetime(2026, 11, 11, 10, 0, tzinfo=IST))
    nse = NseClient(
        SteppingClock(datetime(2026, 11, 11, 10, 0, tzinfo=IST)), transport=httpx.MockTransport(handler)
    )
    state = State(settings, clock)
    state.migrate()
    cal = TradingCalendar.from_dates([date(2026, 11, 6), date(2026, 11, 9)])
    return IngestContext(settings=settings, clock=clock, calendar=cal, nse=nse, state=state)


def test_ingest_range_stores_rows_and_caches_old_windows(tmp_data_root):
    api = Api(_sample())
    ctx = _ctx(tmp_data_root, api)
    res = ingest_corpact_range(ctx, date(2026, 9, 1), date(2026, 11, 5))
    # the sample's ex-dates all fall between 2026-09-01 and 2026-11-04: 12 rows in total
    assert sum(r.rows for r in res) == 12 and all(r.status == "ok" for r in res)
    assert read(tmp_data_root, "corporate_actions").collect().height == 12
    assert len(api.calls) == 3
    cache = tmp_data_root.paths.data_root / "cache" / "corpact" / "2026-09-01_2026-10-01.json"
    assert cache.is_file()
    # second run: the first window ended more than 10 days ago -> served from cache (no call); the
    # two newer windows are mutable (future announcements) and are fetched again
    ingest_corpact_range(ctx, date(2026, 9, 1), date(2026, 11, 5))
    assert len(api.calls) == 3 + 2
    assert read(tmp_data_root, "corporate_actions").collect().height == 12  # idempotent upsert


def test_ingest_range_failure_is_reported(tmp_data_root):
    def boom(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    ctx = _ctx(tmp_data_root, boom)
    res = ingest_corpact_range(ctx, date(2026, 9, 1), date(2026, 9, 10))
    assert [r.status for r in res] == ["failed"]


# ---- reparse ------------------------------------------------------------------------------------


def _stored(subject: str, action: str, status: str, sym: str = "AAA") -> dict:
    return {
        "symbol": sym, "ex_date": date(2024, 1, 2), "action_type": action, "ratio_num": None,
        "ratio_den": None, "factor_price": 1.0, "dividend_amount": None, "face_before": None,
        "face_after": None, "subject_raw": subject, "parse_status": status,
    }  # fmt: skip


def test_reparse_improves_unparsed_rows_and_reports_counts(tmp_data_root):
    rows = [
        _stored("Interim Dividend Of Rs.5/- Per Equity Share", "other", "unparsed", "A"),
        _stored("Rights 1:7 @ Premium Rs 30/-", "rights", "unparsed", "B"),
        _stored("Buy Back", "buyback", "unparsed", "C"),
        _stored("Interim Dividend - Rs 2 Per Share", "dividend", "ok", "D"),  # already ok: untouched
    ]
    schema = {
        "ratio_num": pl.Float64, "ratio_den": pl.Float64, "dividend_amount": pl.Float64,
        "face_before": pl.Float64, "face_after": pl.Float64,
    }  # fmt: skip
    upsert(tmp_data_root, "corporate_actions", pl.DataFrame(rows, schema_overrides=schema))
    ctx = _ctx(tmp_data_root, Api([]))
    before, after = reparse_unparsed(ctx)
    assert before[("other", "unparsed")] == 1 and before[("rights", "unparsed")] == 1
    # A becomes a parsed dividend of 5.0; B keeps 'unparsed' (stored rows have no face value) but gains
    # its ratio; C stays buyback/unparsed; D is untouched
    assert after[("dividend", "ok")] == 1 and ("other", "unparsed") not in after
    assert after[("rights", "unparsed")] == 1 and after[("buyback", "unparsed")] == 1
    df = read(tmp_data_root, "corporate_actions").collect().sort("symbol")
    assert df.height == 4  # keys unchanged: upsert replaced rows, no duplicates
    a, b = df.row(0, named=True), df.row(1, named=True)
    assert (a["action_type"], a["dividend_amount"], a["parse_status"]) == ("dividend", 5.0, "ok")
    assert (b["ratio_num"], b["ratio_den"], b["parse_status"]) == (1.0, 7.0, "unparsed")
    assert b["factor_price"] == 1.0  # stored convention: 1.0 where no price factor is known
    # second run: the dividend is now ok and is not reparsed again; the rest give the same result
    _, after2 = reparse_unparsed(ctx)
    assert after2 == Counter({("rights", "unparsed"): 1, ("buyback", "unparsed"): 1})
    assert read(tmp_data_root, "corporate_actions").collect().height == 4


# ---- wording found in the stored table (beyond the card's strings) --------------------------------


def test_distribution_total_is_not_added_to_its_components():
    p = parse_subject(
        "Distribution Rs 4.80 Per Unit Comprising Of Dividend - Rs 4.37 Per Unit/"
        "Interest Amount - Re 0.43 Per Unit",
        None,
    )
    # the first amount (4.80) is the total; the components after "Comprising Of" must not be added again
    assert p.action_type == "distribution" or p.action_type == "dividend"
    assert p.dividend_amount == pytest.approx(4.80)


def test_rights_abbreviated_premium_and_par_price_are_not_dividends():
    a = parse_subject("Rights 7:13 @Prem Rs.40/-", 10.0)
    assert (a.ratio_num, a.ratio_den, a.rights_price, a.dividend_amount) == (7.0, 13.0, 50.0, None)
    b = parse_subject("Rights - 2:3@ Par Rs 10 Per Share", 10.0)
    # "@ Par Rs 10 Per Share" is the issue price, not a dividend
    assert (b.ratio_num, b.ratio_den, b.rights_price, b.dividend_amount) == (2.0, 3.0, 10.0, None)


def test_demerger_spelling_with_dash():
    assert parse_subject("De-Merger", None).action_type == "demerger"
    assert parse_subject("Scheme Of Demerger", None).action_type == "demerger"


def test_dividend_and_scheme_of_arrangement_keep_both():
    p = parse_subject("Dividend Re.0.50 Per Share And Scheme Of Arrangement", None)
    # the dividend amount is kept, but the scheme means price evidence decides: unparsed
    assert p.action_type == "dividend+scheme_of_arrangement" and p.dividend_amount == 0.5
    assert p.parse_status == "unparsed"
