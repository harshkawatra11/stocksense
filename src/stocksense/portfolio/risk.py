"""Pre-trade risk checks (PLAN T9.3).

`pre_trade_checks(ctx)` evaluates eight conditions on a `RiskContext` the daemon (or a test) fills from the
store; any failure means no NEW entries today (exits still run), and the full list is kept in the decision
record. Pure functions: nothing here reads a file or a clock.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

MODEL_MAX_AGE_DAYS = 7
UNIVERSE_MIN, UNIVERSE_MAX = 300, 500
SCORE_STD_BAND = (0.5, 2.0)
SCORE_MAX_NULL_FRAC = 0.02
NULL_SHARE_FACTOR = 2.0
NULL_SHARE_FLOOR = 0.01  # a group that is never null may reach 1% before it counts as a failure
QUOTE_MAX_AGE = timedelta(minutes=3)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class RiskContext:
    """Everything the eight checks need, all as of `now` (a timezone-aware datetime)."""

    now: datetime
    last_completed_session: date
    panel_last_date: date | None
    features_last_date: date | None
    null_share_today: Mapping[str, float]  # per feature group, share of null cells on the last session
    null_share_mean_60: Mapping[str, float]  # the same, mean over the previous 60 sessions
    model_fit_times: Sequence[datetime]  # fit time of each champion/shadow model in use
    n_u500: int
    score_std_today: float | None  # cross-sectional std of today's blended score
    score_std_mean_60: float | None
    score_null_frac: float | None
    capital_read_at: datetime | None  # when the broker's capital was last read
    last_quote_ts: datetime | None
    market_open: bool
    missing_sessions_last10: Sequence[date]  # the validator's missing sessions among the last 10


def _data_fresh(c: RiskContext) -> Check:
    ok = c.panel_last_date is not None and c.panel_last_date >= c.last_completed_session
    return Check(
        "data_fresh",
        ok,
        f"panel last date {c.panel_last_date}, last completed session {c.last_completed_session}",
    )


def _features_fresh(c: RiskContext) -> Check:
    if c.features_last_date is None or c.features_last_date < c.last_completed_session:
        return Check(
            "features_fresh",
            False,
            f"features last date {c.features_last_date} is behind {c.last_completed_session}",
        )
    bad = []
    for g, share in sorted(c.null_share_today.items()):
        mean = c.null_share_mean_60.get(g)
        if mean is None:
            continue  # no history for this group: nothing to compare with
        if share > max(NULL_SHARE_FACTOR * mean, NULL_SHARE_FLOOR) + 1e-12:
            bad.append(f"{g} {share:.3f} > 2 x {mean:.3f}")
    if bad:
        return Check("features_fresh", False, "null share too high: " + "; ".join(bad))
    return Check(
        "features_fresh",
        True,
        f"features through {c.features_last_date}, null shares within 2x their 60-day mean",
    )


def _model_fresh(c: RiskContext) -> Check:
    if not c.model_fit_times:
        return Check("model_fresh", False, "no champion or shadow model")
    newest = max(c.model_fit_times)
    age = c.now - newest
    return Check(
        "model_fresh",
        age <= timedelta(days=MODEL_MAX_AGE_DAYS),
        f"newest fit {age.total_seconds() / 86400:.1f} days ago",
    )


def _universe_size(c: RiskContext) -> Check:
    return Check("universe_size", UNIVERSE_MIN <= c.n_u500 <= UNIVERSE_MAX, f"n_u500 = {c.n_u500}")


def _score_sane(c: RiskContext) -> Check:
    if c.score_std_today is None or c.score_std_mean_60 is None or c.score_null_frac is None:
        return Check("score_sane", False, "score statistics are missing")
    lo, hi = SCORE_STD_BAND
    std_ok = lo * c.score_std_mean_60 <= c.score_std_today <= hi * c.score_std_mean_60
    null_ok = c.score_null_frac < SCORE_MAX_NULL_FRAC
    detail = (
        f"score std {c.score_std_today:.3f} vs 60-day mean {c.score_std_mean_60:.3f}; "
        f"null share {c.score_null_frac:.3f}"
    )
    return Check("score_sane", std_ok and null_ok, detail)


def _capital_read_today(c: RiskContext) -> Check:
    if c.capital_read_at is None:
        return Check("capital_read_today", False, "capital has not been read from the broker")
    ok = c.capital_read_at.astimezone(c.now.tzinfo).date() == c.now.date()
    return Check("capital_read_today", ok, f"capital read at {c.capital_read_at.isoformat()}")


def _quotes_alive(c: RiskContext) -> Check:
    if not c.market_open:
        return Check("quotes_alive", True, "outside market hours: not required")
    if c.last_quote_ts is None:
        return Check("quotes_alive", False, "no quote received")
    age = c.now - c.last_quote_ts
    return Check("quotes_alive", age <= QUOTE_MAX_AGE, f"newest quote {age.total_seconds():.0f} s old")


def _no_missing_session(c: RiskContext) -> Check:
    miss = list(c.missing_sessions_last10)
    return Check(
        "no_missing_session",
        not miss,
        "none missing in the last 10 sessions" if not miss else f"missing: {miss}",
    )


def pre_trade_checks(ctx: RiskContext) -> list[Check]:
    """The eight checks, always in this order."""
    return [
        _data_fresh(ctx), _features_fresh(ctx), _model_fresh(ctx), _universe_size(ctx),
        _score_sane(ctx), _capital_read_today(ctx), _quotes_alive(ctx), _no_missing_session(ctx),
    ]  # fmt: skip


def entries_allowed(checks: Sequence[Check]) -> bool:
    """New entries are allowed only when every check passed (exits are never blocked)."""
    return all(c.ok for c in checks)
