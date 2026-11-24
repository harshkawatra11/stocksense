"""Delivery flow: how much of each day's volume was taken home, against the stock's own recent past."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from stocksense.core.config import Settings
from stocksense.data import io

WINDOW = 20
MIN_OBS = 15
WARMUP_DAYS = 60  # calendar days read before `start` so the first day has a 20-session baseline


def delivery_daily(panel: pl.DataFrame) -> pl.DataFrame:
    """Per (date, sid): deliv_pct, deliv_z20 vs the previous 20 sessions, deliv_value_cr, deliv_spike."""
    base = pl.col("deliv_pct").shift(1).over("sid")
    p = panel.sort("sid", "date").with_columns(
        base.rolling_mean(WINDOW, min_samples=MIN_OBS).over("sid").alias("_mu"),
        base.rolling_std(WINDOW, min_samples=MIN_OBS).over("sid").alias("_sd"),
    )
    z = pl.when(pl.col("_sd") > 0).then((pl.col("deliv_pct") - pl.col("_mu")) / pl.col("_sd"))
    return p.select(
        "date",
        "sid",
        "deliv_pct",
        z.alias("deliv_z20"),
        (pl.col("deliv_qty") * pl.col("close") / 1e7).alias("deliv_value_cr"),
        ((z > 2) & (pl.col("close") > pl.col("open"))).fill_null(False).alias("deliv_spike"),
    )


def build_delivery_daily(settings: Settings, start: date, end: date) -> int:
    """Compute and upsert flow_delivery_daily for start..end; returns rows written."""
    panel = io.read(
        settings,
        "panel_daily",
        start - timedelta(days=WARMUP_DAYS),
        end,
        ["sid", "date", "open", "close", "deliv_qty", "deliv_pct"],
    ).collect()
    out = delivery_daily(panel).filter(pl.col("date").is_between(start, end))
    if out.height:
        io.upsert(settings, "flow_delivery_daily", out)
    return out.height
