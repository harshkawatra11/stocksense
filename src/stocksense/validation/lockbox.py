from __future__ import annotations

from datetime import date, timedelta

from stocksense.core.config import Settings


def research_end(settings: Settings) -> date:
    """Last research date: the day before lockbox_start."""
    return settings.research.lockbox_start - timedelta(days=1)
