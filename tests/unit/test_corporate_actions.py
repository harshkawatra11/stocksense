"""Corporate-action parsing tests. NSE's `subject` field is free text
written by ~7,556 different companies over 16 years -- these tests pin
down the grammars actually observed live (Bonus X:Y, face-value splits
in both 'Rs .../- ' and 'Re 1/-' forms, dividends, corporate noise, and
compound bonus+split subjects) plus the properties that matter for
correctness: unparseable text is flagged loudly, never silently
skipped as if it had no effect."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stocksense.data.corporate_actions import parse_action, parse_ca_frame


def test_bonus_ratio_parses_to_correct_dilution_factor() -> None:
    p = parse_action("Bonus 1:1")
    assert p.action_type == "bonus"
    assert p.parse_status == "ok"
    assert p.factor_price == pytest.approx(0.5)  # shares double -> price halves


def test_bonus_2_for_3_parses() -> None:
    p = parse_action("Bonus 2:3")
    assert p.factor_price == pytest.approx(3 / 5)


def test_face_value_split_10_to_2_parses() -> None:
    p = parse_action("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share")
    assert p.action_type == "split"
    assert p.factor_price == pytest.approx(0.2)
    assert p.face_before == 10.0
    assert p.face_after == 2.0


def test_face_value_split_to_re_1_form_parses() -> None:
    """'Re 1/-' (singular rupee) is a real NSE grammar variant, distinct from 'Rs'."""
    p = parse_action("Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share")
    assert p.action_type == "split"
    assert p.factor_price == pytest.approx(0.5)


def test_short_grammar_split_variants_parse() -> None:
    """NSE used at least three grammars for the identical event across
    2010-2026: verbose (tested above), and two short forms -- one with
    dots, one with spaces, neither containing the word 'From' or the
    phrase 'Per Share'."""
    assert parse_action("Fv Split Rs.10 To Rs.5").factor_price == pytest.approx(0.5)
    assert parse_action("Face Value Split Rs.10/- To Re.1/- Per Share").factor_price == pytest.approx(0.1)
    assert parse_action("Face Value Split Rs 10 To Re 1").factor_price == pytest.approx(0.1)


def test_typo_grammar_splt_frm_parses() -> None:
    """Live NSE data contains at least one genuine typo ('Splt'/'Frm' for
    'Split'/'From') that must still be recognized -- confirmed missing
    JSWSTEEL's 2017-01-04 10:1 split during the acceptance check against
    yfinance-implied adjustment jumps before this was added."""
    p = parse_action("Fv Splt Frm Rs 10 To Re 1")
    assert p.action_type == "split"
    assert p.factor_price == pytest.approx(0.1)


def test_compound_bonus_and_split_multiplies_factors() -> None:
    p = parse_action("Bonus 1:1 / Face Value Split From Rs 10/- Per Share To Rs 2/- Per Share")
    assert p.factor_price == pytest.approx(0.5 * 0.2)


def test_dividend_extracts_amount_and_does_not_change_price_factor() -> None:
    p = parse_action("Annual General Meeting/Final Dividend - Rs 6/- Per Share")
    assert p.action_type == "dividend"
    assert p.dividend_amount == pytest.approx(6.0)
    assert p.factor_price == 1.0


def test_dividend_with_period_before_amount_does_not_crash_or_misparse() -> None:
    """Regression: 'Dividend Rs. - 2.80/- Per Share' has a literal '.'
    right after 'Rs' -- an earlier regex let the optional \\.? skip that
    period, leaving it to be captured by the amount group as a lone '.',
    which crashed float() in production during a live backfill."""
    p = parse_action("Annual General Meeting/Dividend Rs. - 2.80/- Per Share")
    assert p.action_type == "dividend"
    assert p.dividend_amount == pytest.approx(2.80)


def test_corporate_noise_is_ignored_not_unparsed() -> None:
    for subject in ["E-Voting", "Annual General Meeting", "Board Meeting"]:
        p = parse_action(subject)
        assert p.action_type == "ignore"
        assert p.parse_status == "ok"
        assert p.factor_price == 1.0


def test_unrecognized_financial_subject_is_flagged_not_silently_dropped() -> None:
    p = parse_action("Something Unusual That Sounds Financial But Matches No Known Grammar")
    assert p.parse_status == "unparsed"
    assert p.action_type == "unparsed"


def test_empty_or_missing_subject_is_unparsed() -> None:
    for subject in ["", None, float("nan")]:
        p = parse_action(subject)
        assert p.parse_status == "unparsed"


def test_parse_ca_frame_produces_write_schema_columns() -> None:
    raw = pd.DataFrame([
        {"symbol": "ECLERX", "exDate": "13-Mar-2026", "subject": "Bonus 1:1"},
        {"symbol": "NOISE", "exDate": "01-Jan-2015", "subject": "E-Voting"},
    ])
    out = parse_ca_frame(raw)
    assert set(out.columns) == {
        "symbol", "ex_date", "action_type", "ratio_num", "ratio_den", "factor_price",
        "dividend_amount", "face_before", "face_after", "subject_raw", "parse_status",
    }
    assert len(out) == 2
    eclerx = out[out["symbol"] == "ECLERX"].iloc[0]
    assert eclerx["ex_date"] == date(2026, 3, 13)
    assert eclerx["action_type"] == "bonus"


def test_parse_ca_frame_drops_rows_with_unparseable_dates() -> None:
    raw = pd.DataFrame([{"symbol": "X", "exDate": "not-a-date", "subject": "Bonus 1:1"}])
    out = parse_ca_frame(raw)
    assert out.empty


def test_parse_ca_frame_empty_input_returns_empty_with_correct_columns() -> None:
    out = parse_ca_frame(pd.DataFrame())
    assert out.empty
    assert "action_type" in out.columns
