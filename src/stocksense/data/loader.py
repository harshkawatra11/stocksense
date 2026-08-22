"""
The shared candle-loading path (Phase D2 / Phase G). Extracted from
cli/main.py so cli/main.py, harness/loops.py (the reconcile loop), and
research scripts can all share exactly one loading path rather than
each hand-rolling their own — the same discipline that motivated
routing research/bhavcopy_rerun_sweep.py through this path instead of
letting it call store.read_candles() directly (see
research/preregistration_bhavcopy_rerun.md). Living in cli/main.py
would make this unimportable from harness/loops.py without a circular
import (cli/main.py already imports from harness.loops), which is the
concrete reason this lives here instead.
"""

from __future__ import annotations

import structlog
import pandas as pd

from stocksense.data.adjust import quarantine_unexplained_jumps, read_adjusted_candles
from stocksense.data.store import Store
from stocksense.data.universe_pit import filter_to_point_in_time_universe
from stocksense.data.validate import quarantine_symbols
from stocksense.features.engine import build_features, feature_columns
from stocksense.labels.forward_return import add_forward_return_labels, add_relative_forward_return

log = structlog.get_logger(__name__)


def load_candles(settings, store: Store, turnover_rank_band: tuple[float, float] | None = None) -> pd.DataFrame:
    """Source switch (docs/17-data-spine.md, Phase D2): 'candles'
    preserves the exact Phase 0 path (yfinance, fixed 98-symbol
    universe) unchanged, so those numbers stay reproducible. 'bhavcopy'
    pulls point-in-time NSE data through the corporate-action adjustment
    layer (data/adjust.py) -- the widest candidate symbol set is read
    first, then narrowed to the point-in-time-tradeable universe per
    date if use_point_in_time_universe is set, rather than narrowed to
    a hardcoded list up front.

    `turnover_rank_band`: passed straight through to
    universe_pit.filter_to_point_in_time_universe when
    use_point_in_time_universe is set -- a liquidity-rank-proxy cap band
    (see universe_pit.universe_as_of's docstring) for restricting to
    e.g. mid/small cap rather than the full point-in-time-tradeable set.
    Ignored (with no effect) when use_point_in_time_universe is False,
    since 'candles'/no-PIT-filter has no notion of a per-date rank band.
    """
    if settings.price_source == "candles":
        return store.read_candles()

    if settings.price_source != "bhavcopy":
        raise ValueError(f"unknown price_source {settings.price_source!r}, expected 'candles' or 'bhavcopy'")

    bounds = store.con.execute("SELECT MIN(date), MAX(date) FROM bhavcopy_eq").fetchone()
    if bounds[0] is None:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source"])
    min_d, max_d = bounds
    symbols = store.con.execute("SELECT DISTINCT symbol FROM bhavcopy_eq WHERE series = 'EQ'").fetchdf()["symbol"].tolist()
    candles = read_adjusted_candles(store, symbols, min_d, max_d, basis=settings.return_basis)

    if settings.use_point_in_time_universe:
        candles = filter_to_point_in_time_universe(
            store, candles, min_turnover_inr=settings.min_avg_daily_turnover_inr, min_price_inr=settings.min_price_inr,
            turnover_rank_band=turnover_rank_band,
        )
    return candles


def load_features_and_labels(horizon: int, turnover_rank_band: tuple[float, float] | None = None, settings=None, store: Store | None = None):
    """Loads candles via load_candles, applies the source-appropriate
    quarantine, and builds features + horizon labels -- one call for
    the common "give me a clean, feature-built, labeled frame for this
    horizon" need shared by train_candidate, the reconcile loop, and
    research scripts.

    `settings`/`store`: optional overrides so a caller that already has
    a settings object and/or an open Store (e.g. to keep one connection
    for a whole reconcile run) doesn't have to construct fresh ones. If
    `store` is passed in, it is the caller's responsibility to close it
    -- this function only closes a Store it opened itself.
    """
    from stocksense.core.config import get_settings

    settings = settings or get_settings()
    owns_store = store is None
    store = store or Store(settings.duckdb_path)
    candles = load_candles(settings, store, turnover_rank_band=turnover_rank_band)

    # Quarantine symbols with a detected adjustment anomaly before
    # anything downstream touches them. The detector is source-dependent
    # and this branch matters: data/validate.quarantine_symbols compares
    # adj_close to raw close, which is only a bug signal when close is
    # ALREADY split-adjusted at the source (true for yfinance's `candles`
    # — see the ADANIENT 8.6x-jump bug in research/phase0_verdict.md's
    # "Run 3"). For bhavcopy-sourced data close is raw by construction
    # (data/corporate_actions.py), so adj_close/close legitimately steps
    # at every real split/bonus — applying the yfinance check there is
    # itself a bug: it quarantined RELIANCE, TCS, and ~600 other names
    # for their genuine corporate actions, found live during Phase D2.
    # The bhavcopy-appropriate check instead flags adjusted-price jumps
    # with NO matching corporate-action record.
    if settings.price_source == "bhavcopy":
        candles, quarantined = quarantine_unexplained_jumps(store, candles)
    else:
        candles, quarantined = quarantine_symbols(candles)
    if owns_store:
        store.close()
    if quarantined:
        log.warning("quarantined_symbols_with_adjustment_anomalies", symbols=quarantined, price_source=settings.price_source)

    feats = build_features(candles)
    fcols = [c for c in feature_columns(feats) if c != "mkt_ret_1b"]

    labeled = add_forward_return_labels(candles, horizon_bars=horizon)
    labeled = add_relative_forward_return(labeled, horizon_bars=horizon)
    return candles, feats, fcols, labeled
