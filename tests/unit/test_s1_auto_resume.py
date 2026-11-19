"""Backtest-only auto-resume of the S1 book-drawdown stop (DECISIONS: T9.5 auto-resume variant)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from stocksense.core.calendar import TradingCalendar
from stocksense.portfolio.construct import DayScores, NameRow, PanelDay, S1Policy, book_stop_key
from tests.fixtures.engine_panel import settings, zero_slippage
from tests.fixtures.handmade import weekdays

S = settings()
SESS = weekdays(80)
CAL = TradingCalendar.from_dates(SESS)
N = 20


class _Src:
    def scores(self, t):
        return DayScores({i: 0.4 - 0.05 * i for i in range(14)}, {i: 0.05 for i in range(14)})


class _Access:
    def day(self, t, next_date):
        rows = {
            i: NameRow(close=100.0, vol_21=0.005, turnover_rank=10, med_turn_20=1e10, industry=f"I{i}")
            for i in range(14)
        }
        return PanelDay(t, next_date, rows)


class _Kv:
    def __init__(self):
        self.d = {}

    def kv_get(self, key, default=None):
        return self.d.get(key, default)

    def kv_set(self, key, value):
        self.d[key] = value


def _policy(**kw):
    return S1Policy(
        SimpleNamespace(target="y_c2o_5"), S, _Src(), _Access(), S.costs, zero_slippage(), k_max=10, **kw
    )


def _drive(pol, navs):
    """Feed one NAV per session (no positions); returns the gross target the policy planned each day."""
    out = []
    for t, nav in zip(SESS, navs, strict=False):
        book = SimpleNamespace(calendar=CAL, positions={}, cash=nav, reserved=0.0)
        pol.decide(t, book, nav)
        out.append(pol.plans[t].gross_target)
    return out


# 5 sessions at 5e6, a 20% fall (stop fires on session 5), then flat at 4e6
NAVS = [5e6] * 5 + [4e6] * 60


def test_default_stop_is_permanent_in_a_backtest():
    gt = _drive(_policy(), NAVS)
    assert gt[4] > 0 and all(g == 0.0 for g in gt[5:])


def test_auto_resume_reenters_exactly_n_sessions_after_the_stop_fired():
    gt = _drive(_policy(auto_resume_sessions=N), NAVS)
    fired = 5  # the first session whose plan sees the stop (dd 20% at the close of index 5)
    assert all(g == 0.0 for g in gt[fired : fired + N])
    assert gt[fired + N] > 0.0  # cleared N sessions after it fired, peak re-based so dd is 0 again
    assert all(g > 0.0 for g in gt[fired + N :])


def test_auto_resume_can_stop_again_and_resume_again():
    navs = [5e6] * 5 + [4e6] * 30 + [3e6] * 40  # a second 25% fall after the first resume
    gt = _drive(_policy(auto_resume_sessions=N), navs)
    assert gt[5 + N] > 0 and gt[34] > 0
    assert gt[35] == 0.0 and gt[35 + N - 1] == 0.0 and gt[35 + N] > 0.0


def test_auto_resume_waits_for_a_flat_book():
    pol = _policy(auto_resume_sessions=N)
    held = SimpleNamespace(product="delivery", qty=10, entry_date=SESS[0], entry_price=100.0)
    gt = []
    for i, (t, nav) in enumerate(zip(SESS, NAVS, strict=False)):
        pos = {0: held} if i < 40 else {}
        book = SimpleNamespace(calendar=CAL, positions=pos, cash=nav, reserved=0.0)
        pol.decide(t, book, nav)
        gt.append(pol.plans[t].gross_target)
    assert gt[39] == 0.0 and gt[40] > 0.0  # N elapsed long before, but the position held until 40


def test_live_policy_with_a_state_can_never_auto_resume():
    with pytest.raises(ValueError, match="backtest"):
        _policy(auto_resume_sessions=N, state=_Kv())


def test_live_stop_with_a_state_stays_until_resume_clears_it():
    kv = _Kv()
    gt = _drive(_policy(state=kv), NAVS)
    assert kv.d[book_stop_key("s1")] == "1" and all(g == 0.0 for g in gt[5:])
    pol = _policy(state=kv)
    kv.d[book_stop_key("s1")] = "0"  # `stocksense resume --book s1`
    assert _drive(pol, [4e6] * 3)[-1] > 0.0


def test_n_must_be_positive():
    with pytest.raises(ValueError):
        _policy(auto_resume_sessions=0)
