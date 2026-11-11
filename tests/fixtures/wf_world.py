"""A synthetic market with panel, universe, labels and a test feature store, for walk-forward tests.

The test registry has `alpha_t` (the hidden alpha of the synthetic market, known at t), two seed features
and `mkt_ret_1` (a market-state feature). A `leaky_next` feature (next session's return) is available
separately for the look-ahead self-test. Fold windows are small and sit inside the synthetic period.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from stocksense import panel as panel_pkg
from stocksense.core.calendar import TradingCalendar
from stocksense.core.config import Settings
from stocksense.data import io
from stocksense.features.base import FeatureContext, FeatureSpec, feature
from stocksense.features.build import build_features
from stocksense.features.registry import REGISTRY, load_all
from stocksense.labels.forward import build_labels
from stocksense.validation.splits import Fold
from tests.fixtures.synth import SynthMarket, _settings_for, make_market, write_market

# (name, test_start, test_end): six two-month windows after ~6 months of universe history
WINDOWS = (
    ("f1", date(2019, 5, 1), date(2019, 6, 30)),
    ("f2", date(2019, 7, 1), date(2019, 8, 31)),
    ("f3", date(2019, 9, 1), date(2019, 10, 31)),
    ("f4", date(2019, 11, 1), date(2019, 12, 31)),
    ("f5", date(2020, 1, 1), date(2020, 2, 29)),
    ("f6", date(2020, 3, 1), date(2020, 4, 17)),
)


@dataclass(frozen=True)
class WFWorld:
    settings: Settings
    market: SynthMarket
    calendar: TradingCalendar
    registry: dict[str, FeatureSpec]
    folds: list[Fold]


def make_registry(settings: Settings, market: SynthMarket) -> dict[str, FeatureSpec]:
    """alpha_t + ret_1, vol_21, mkt_ret_1 in a private registry."""
    load_all()
    reg: dict[str, FeatureSpec] = {n: REGISTRY[n] for n in ("ret_1", "vol_21", "mkt_ret_1")}
    sm = io.read(settings, "security_master").collect().select("sid", "symbol")
    alpha = (
        market.truth.select("symbol_key", "date", "alpha")
        .join(sm, left_on="symbol_key", right_on="symbol")
        .select("date", "sid", "alpha")
        .unique(["date", "sid"])
    )

    @feature("alpha_t", group="test", kind="xs", registry=reg, label="hidden alpha known at t")
    def alpha_t(ctx: FeatureContext) -> pl.LazyFrame:
        return (
            ctx.panel()
            .select("date", "sid")
            .join(alpha.lazy(), on=["date", "sid"], how="left")
            .select("date", "sid", pl.col("alpha").alias("alpha_t"))
        )

    return reg


def leaky_spec() -> dict[str, FeatureSpec]:
    """A registry with one feature that reads the NEXT session's return (look-ahead)."""
    reg: dict[str, FeatureSpec] = {}

    @feature("leaky_next", group="test", kind="xs", registry=reg, label="tomorrow's return (a bug)")
    def leaky_next(ctx: FeatureContext) -> pl.LazyFrame:
        return ctx.panel().select("date", "sid", pl.col("ret").shift(-1).over("sid").alias("leaky_next"))

    return reg


def build_world(root: Path, planted_ic: float, n: int = 150) -> WFWorld:
    """Write a market of `n` stocks under `root`; build panel, labels and the test feature store."""
    m = make_market(n=n, planted_ic=planted_ic)
    write_market(m, root)
    s = _settings_for(root)
    panel_pkg.build_all(s)
    build_labels(s)
    reg = make_registry(s, m)
    build_features(s, min_valid=10, registry=reg)
    cal = TradingCalendar.from_dates(m.sessions)
    folds = [Fold(name, cal.prev_session(a), a, b) for name, a, b in WINDOWS]
    return WFWorld(s, m, cal, reg, folds)
