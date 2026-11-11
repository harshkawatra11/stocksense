from __future__ import annotations

from stocksense.core.config import Settings


def build_all(settings: Settings) -> None:
    """Run the four panel builds in order: security master, daily panel, universe, market frame."""
    from stocksense.panel.market import build_market
    from stocksense.panel.returns import build_daily_panel
    from stocksense.panel.security_master import build_security_master
    from stocksense.panel.universe import build_universe

    build_security_master(settings)
    build_daily_panel(settings)
    build_universe(settings)
    build_market(settings)
