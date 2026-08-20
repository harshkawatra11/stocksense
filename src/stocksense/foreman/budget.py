"""
Budget and kill-switch enforcement (docs/19-foreman.md). On-demand
cadence per the user's choice — this module doesn't schedule anything;
it caps a single `foreman run` invocation so a goal that can't be solved
doesn't retry forever or silently run up cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from stocksense.core.config import get_settings

DEFAULT_MAX_INVOCATIONS_PER_DAY = 200  # fallback only if Settings itself can't be read; see check_budget
DEFAULT_MAX_TOKENS_ESTIMATE_PER_DAY = 2_000_000.0


@dataclass(frozen=True)
class BudgetStatus:
    within_budget: bool
    invocations_used: int
    invocations_limit: int
    reason: str | None = None


def check_budget(store, max_invocations: int | None = None) -> BudgetStatus:
    """Phase F4: `max_invocations` defaults to
    Settings.foreman_max_invocations_per_day (live-editable from the
    desktop app), not a hardcoded constant -- pass an explicit value to
    override for one call (tests do this)."""
    if max_invocations is None:
        max_invocations = get_settings().foreman_max_invocations_per_day
    today = date.today()
    current = store.get_or_create_budget(today)
    used = int(current["invocations"])
    if used >= max_invocations:
        return BudgetStatus(False, used, max_invocations, reason=f"daily invocation cap ({max_invocations}) reached")
    return BudgetStatus(True, used, max_invocations)


class KillSwitch:
    """In-process cooperative stop flag. `stocksense foreman stop`
    (a separate process) can't reach into a running loop directly, so in
    practice this is checked at the top of each goal iteration within a
    single `foreman run` invocation; cross-process stopping works because
    state lives in job_runs/goals (DuckDB), which the runner already
    checks on resume — a genuinely running background process is future
    work (the plan's scheduled-loop phase), not this on-demand core."""

    def __init__(self) -> None:
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    @property
    def stopped(self) -> bool:
        return self._stopped
