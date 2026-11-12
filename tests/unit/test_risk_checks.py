from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from stocksense.portfolio.risk import Check, RiskContext, entries_allowed, pre_trade_checks

IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2026, 11, 12, 8, 50, tzinfo=IST)
LAST = date(2026, 11, 11)

NAMES = [
    "data_fresh", "features_fresh", "model_fresh", "universe_size",
    "score_sane", "capital_read_today", "quotes_alive", "no_missing_session",
]  # fmt: skip


def good() -> RiskContext:
    return RiskContext(
        now=NOW,
        last_completed_session=LAST,
        panel_last_date=LAST,
        features_last_date=LAST,
        null_share_today={"price": 0.02, "delivery": 0.10},
        null_share_mean_60={"price": 0.02, "delivery": 0.09},
        model_fit_times=[NOW - timedelta(days=2), NOW - timedelta(days=20)],
        n_u500=480,
        score_std_today=0.29,
        score_std_mean_60=0.29,
        score_null_frac=0.0,
        capital_read_at=NOW - timedelta(minutes=5),
        last_quote_ts=NOW - timedelta(seconds=40),
        market_open=True,
        missing_sessions_last10=[],
    )


def by_name(checks: list[Check]) -> dict[str, Check]:
    return {c.name: c for c in checks}


def test_all_eight_checks_pass_on_a_good_context():
    checks = pre_trade_checks(good())
    assert [c.name for c in checks] == NAMES
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]
    assert all(isinstance(c.detail, str) and c.detail for c in checks)
    assert entries_allowed(checks) is True


FAILING = {
    "data_fresh": dict(panel_last_date=LAST - timedelta(days=1)),
    "features_fresh": dict(features_last_date=LAST - timedelta(days=3)),
    "model_fresh": dict(model_fit_times=[NOW - timedelta(days=8), NOW - timedelta(days=30)]),
    "universe_size": dict(n_u500=299),
    "score_sane": dict(score_std_today=0.7),
    "capital_read_today": dict(capital_read_at=NOW - timedelta(days=1)),
    "quotes_alive": dict(last_quote_ts=NOW - timedelta(minutes=4)),
    "no_missing_session": dict(missing_sessions_last10=[date(2026, 11, 6)]),
}


@pytest.mark.parametrize("name", NAMES)
def test_each_check_fails_on_its_fixture_and_blocks_entries_only_through_that_check(name):
    checks = pre_trade_checks(replace(good(), **FAILING[name]))
    got = by_name(checks)
    assert not got[name].ok
    assert all(c.ok for c in checks if c.name != name)
    assert entries_allowed(checks) is False


def test_data_missing_entirely_fails():
    assert not by_name(pre_trade_checks(replace(good(), panel_last_date=None)))["data_fresh"].ok


def test_features_fail_on_a_null_share_above_twice_the_60_day_mean():
    ok = replace(good(), null_share_today={"price": 0.039, "delivery": 0.10})
    bad = replace(good(), null_share_today={"price": 0.041, "delivery": 0.10})  # 2 x 0.02 = 0.04
    assert by_name(pre_trade_checks(ok))["features_fresh"].ok
    c = by_name(pre_trade_checks(bad))["features_fresh"]
    assert not c.ok and "price" in c.detail


def test_features_have_an_absolute_floor_for_groups_that_are_never_null():
    ctx = replace(good(), null_share_today={"price": 0.008}, null_share_mean_60={"price": 0.0})
    assert by_name(pre_trade_checks(ctx))["features_fresh"].ok
    ctx = replace(good(), null_share_today={"price": 0.05}, null_share_mean_60={"price": 0.0})
    assert not by_name(pre_trade_checks(ctx))["features_fresh"].ok


def test_model_fresh_needs_one_fit_within_seven_days_and_at_least_one_model():
    assert by_name(pre_trade_checks(replace(good(), model_fit_times=[NOW - timedelta(days=7)])))[
        "model_fresh"
    ].ok
    assert not by_name(pre_trade_checks(replace(good(), model_fit_times=[])))["model_fresh"].ok


def test_universe_size_bounds_are_inclusive():
    for n, ok in [(300, True), (500, True), (299, False), (501, False)]:
        assert by_name(pre_trade_checks(replace(good(), n_u500=n)))["universe_size"].ok is ok


def test_score_sane_std_band_and_null_share():
    for std, ok in [(0.145, True), (0.58, True), (0.14, False), (0.59, False)]:
        assert by_name(pre_trade_checks(replace(good(), score_std_today=std)))["score_sane"].ok is ok
    assert not by_name(pre_trade_checks(replace(good(), score_null_frac=0.02)))["score_sane"].ok
    assert by_name(pre_trade_checks(replace(good(), score_null_frac=0.019)))["score_sane"].ok
    assert not by_name(pre_trade_checks(replace(good(), score_std_today=None)))["score_sane"].ok


def test_capital_must_be_read_on_the_same_day():
    assert by_name(pre_trade_checks(replace(good(), capital_read_at=None)))["capital_read_today"].ok is False


def test_quotes_check_only_applies_in_market_hours():
    stale = replace(good(), last_quote_ts=NOW - timedelta(minutes=30), market_open=False)
    assert by_name(pre_trade_checks(stale))["quotes_alive"].ok
    none_open = replace(good(), last_quote_ts=None)
    assert not by_name(pre_trade_checks(none_open))["quotes_alive"].ok
    edge = replace(good(), last_quote_ts=NOW - timedelta(minutes=3))
    assert by_name(pre_trade_checks(edge))["quotes_alive"].ok


def test_entries_allowed_is_false_for_a_single_failure_and_true_for_empty_pass():
    checks = pre_trade_checks(good())
    assert entries_allowed(checks)
    assert not entries_allowed([*checks[:-1], Check("no_missing_session", False, "x")])
