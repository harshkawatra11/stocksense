"""Corporate-action parser tests.

Every subject line below is VERBATIM from NSE's own feed. They were found by
surveying 35,074 real subjects across 2010-2026, not invented -- which matters,
because the failure mode here is silent: a missed split leaves a fake -50% or
-90% one-day return in the spine and teaches a model that splits are crashes.
"""

from __future__ import annotations

from datetime import date

import pytest

from stocksense.data.corporate_actions import parse_action, parse_records


# --------------------------------------------------------------------- bonus
@pytest.mark.parametrize(
    ("subject", "factor"),
    [
        ("Bonus 1:1", 0.5),          # 1 new per 1 held -> holder has 2 for every 1
        ("Bonus 3:1", 0.25),         # 3 new per 1 held -> 4 for every 1
        ("Bonus 1:2", 2 / 3),
        ("Bonus 1 : 1", 0.5),        # spaced colon
        ("Bonus 1: 2", 2 / 3),
        ("Bon 2:1", 1 / 3),          # abbreviated
    ],
)
def test_bonus_ratios(subject, factor):
    p = parse_action(subject)
    assert p.action_type == "bonus"
    assert p.factor_price == pytest.approx(factor)


# --------------------------------------------------------------------- splits
@pytest.mark.parametrize(
    ("subject", "factor"),
    [
        ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share", 0.1),
        ("Face Value Split From Rs.10/- To Rs.5/-", 0.5),
        ("Fv Split Rs.10 To Rs.5", 0.5),
        ("Face Value Split From Rs 10 To Re 1", 0.1),
        ("Face Value Split Rs 10 To Rs 2", 0.2),
        ("Sub-Division From Rs 10/- Per Share To Rs 5/- Per Share", 0.5),
    ],
)
def test_split_grammars(subject, factor):
    """NSE has used at least eight phrasings for the identical event."""
    p = parse_action(subject)
    assert p.action_type == "split"
    assert p.factor_price == pytest.approx(factor)


def test_nse_own_typo_still_parses():
    """"Splt"/"Frm" instead of "Split"/"From" -- a genuine typo in NSE's data,
    9 real occurrences. It is a real split and must still adjust the price."""
    p = parse_action("Fv Splt Frm Rs 10 To Rs 5")
    assert p.action_type == "split"
    assert p.factor_price == pytest.approx(0.5)


def test_abbreviated_and_run_together_split_parses():
    """"Rs.5tors.2" is "Rs.5 to Rs.2" with the spaces missing, and "Fv Spl" is
    the abbreviated form. Both appear in real records."""
    p = parse_action("Bon 1:1/Fv Spl Rs.5tors.2")
    assert p.action_type == "bonus+split"
    assert p.factor_price == pytest.approx(0.5 * 0.4)


def test_consolidation_is_a_reverse_split_and_raises_the_price():
    """Face value RISES, share count falls, price rises. The same after/before
    formula gives a factor > 1, so pre-event prices multiply UP. Getting this
    backwards would invent a spectacular fake gain."""
    p = parse_action("Consolidation Of Equity Shares From Re 1 Per Share To Rs 10 Per Share")
    assert p.action_type == "split"
    assert p.factor_price == pytest.approx(10.0)


# ------------------------------------------------------------------ compound
@pytest.mark.parametrize(
    "subject",
    [
        "Bonus 1:1 / Face Value Split From Rs. 10/- To Re. 2/-",
        "Bonus 1:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share",
        "Bonus 1:1 And Face Value Split From Rs.10/- To Rs.2/-",
    ],
)
def test_compound_events_multiply(subject):
    """Both events land on ONE ex-date, so the factors compose: 0.5 * 0.2 = 0.1.
    Applying only one leaves half the discontinuity in the series."""
    p = parse_action(subject)
    assert p.action_type == "bonus+split"
    assert p.factor_price == pytest.approx(0.1)


# ------------------------------------------------------- the non-equity traps
@pytest.mark.parametrize(
    "subject",
    [
        "Scheme Of Arrangement - Bonus Ncrps 4:1",
        "Bonus Ncrps 1:10",
        "Bonus Preference Shares 21:1",
        "Bonus 1 Dvr : 10 Eq Share",
        "Bon 1 Dvr : 20 Eq Shares",
    ],
)
def test_non_equity_bonuses_do_not_adjust_the_equity_price(subject):
    """THE trap. A bonus issue of CRPS / preference / DVR shares is a DIFFERENT
    SECURITY and does not dilute the equity price.

    Reading "Bonus Ncrps 4:1" as a 4:1 equity bonus applies an 80% adjustment to
    a price that never moved -- a fabricated 5x discontinuity. Six of these exist
    in the real 2010-2026 sample.
    """
    p = parse_action(subject)
    assert p.action_type == "preference_bonus"
    assert p.factor_price == 1.0


# ------------------------------------------------------------------ dividends
@pytest.mark.parametrize(
    ("subject", "amount"),
    [
        ("Interim Dividend - Rs 2.50 Per Share", 2.50),
        ("Dividend - Re 0.01 Per Share", 0.01),
        ("Annual General Meeting/Dividend - Rs 6 Per Share", 6.0),
        ("Annual General Meeting And Dividend Rs.3/- Per Share", 3.0),
        ("Agm/Div-Rs.2/- Per Share", 2.0),
        ("Int Div-Rs.1.5 Per Share", 1.5),
        ("Div-Int Re 2.5+Spl Re 2.5", 2.5),
        ("Div-Fin Rs.5+Spl Rs.10   Purpose Revised", 5.0),
    ],
)
def test_dividend_amounts(subject, amount):
    p = parse_action(subject)
    assert p.action_type == "dividend"
    assert p.dividend_amount == pytest.approx(amount)
    assert p.factor_price == 1.0, "a dividend changes no share count"


def test_dividend_with_a_separator_before_the_amount_does_not_crash():
    """Regression: "Rs. - 2.80" put a period and a dash between "Rs" and the
    number. A lax [\\d.]+ captured a lone "." and crashed float() mid-backfill."""
    p = parse_action("Annual General Meeting/Dividend Rs. - 2.80/- Per Share")
    assert p.action_type == "dividend"
    assert p.dividend_amount == pytest.approx(2.80)


def test_special_does_not_get_read_as_split():
    """"Spl" means "Special" in a dividend line and "Split" after "Fv". Reading
    the dividend form as a split would invent a face-value change."""
    p = parse_action("Div - Fin Rs.3 + Spl Rs.5")
    assert p.action_type == "dividend"


# ------------------------------------------------------- honest non-coverage
@pytest.mark.parametrize(
    ("subject", "kind"),
    [
        ("Rights Issue 1:4", "rights"),
        ("Buy Back Of Equity Shares", "buyback"),
        ("Demerger", "demerger"),
        ("Scheme Of Arrangement", "scheme_of_arrangement"),
        ("Amalgamation", "amalgamation"),
        ("Distribution - Rs 5 Per Unit", "distribution"),
    ],
)
def test_unparseable_events_are_recorded_not_dropped(subject, kind):
    """Their ratios are NOT recoverable from the subject text -- a demerger line
    states no exchange ratio at all. Silently dropping them would hide a known
    gap; they are classified, marked unparsed, and counted.

    Rights issues are the most consequential (347 records) because they DO
    dilute price. That limitation is deliberate and documented, not accidental.
    """
    p = parse_action(subject)
    assert p.action_type == kind
    assert p.parse_status == "unparsed"
    assert p.factor_price == 1.0, "never guess a factor we cannot derive"


@pytest.mark.parametrize(
    "subject",
    [
        "Annual General Meeting",
        "Extra Ordinary General Meeting",
        "Board Meeting",
        "E-Voting",
        "Interest Payment",
        "Election Of Directors",
    ],
)
def test_corporate_noise_is_classified_not_left_unparsed(subject):
    """11,000+ of the 35,074 subjects are meetings. Leaving them "unparsed"
    would make the unparsed rate meaningless as a coverage measure."""
    p = parse_action(subject)
    assert p.action_type == "noise"
    assert p.parse_status == "ok"
    assert p.factor_price == 1.0


def test_empty_subject_is_handled():
    assert parse_action("").parse_status == "unparsed"
    assert parse_action(None).parse_status == "unparsed"  # type: ignore[arg-type]


# -------------------------------------------------------------- record parsing
def test_parse_records_builds_the_store_schema():
    from stocksense.data.store import Store

    recs = [
        {"symbol": "RELIANCE", "exDate": "28-Oct-2024", "subject": "Bonus 1:1"},
        {"symbol": "NESTLEIND", "exDate": "05-Jan-2024",
         "subject": "Face Value Split (Sub-Division) - From Rs10/- Per Share To Re 1/- Per Share"},
    ]
    df = parse_records(recs)
    assert list(df.columns) == Store.CA_COLS
    assert df.iloc[0].ex_date == date(2024, 10, 28)
    assert df.iloc[0].factor_price == pytest.approx(0.5)
    assert df.iloc[1].factor_price == pytest.approx(0.1)


def test_parse_records_skips_rows_it_cannot_key():
    """A row with no symbol or an unparseable date cannot be joined to prices, so
    it is dropped rather than written with a null key."""
    recs = [
        {"symbol": "", "exDate": "28-Oct-2024", "subject": "Bonus 1:1"},
        {"symbol": "X", "exDate": "not-a-date", "subject": "Bonus 1:1"},
        {"symbol": "GOOD", "exDate": "28-Oct-2024", "subject": "Bonus 1:1"},
    ]
    df = parse_records(recs)
    assert len(df) == 1
    assert df.iloc[0].symbol == "GOOD"
