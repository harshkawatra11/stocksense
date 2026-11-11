"""Market frame: equal-weight market returns, breadth and dispersion (PLAN T3.4)."""

from __future__ import annotations

import polars as pl

from stocksense.core.config import Settings
from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.data import io
from stocksense.panel.security_master import session_frame

log = get_logger(__name__)

_BREADTH_WINDOW = 200


def _prev_day_members(universe: pl.DataFrame, flag: str, sess: pl.DataFrame) -> pl.DataFrame:
    """(date, sid): the sids that were `flag` members on the session BEFORE `date`."""
    nxt = (
        sess.sort("date").select(pl.col("date"), pl.col("date").shift(-1).alias("_next")).drop_nulls("_next")
    )
    return (
        universe.filter(pl.col(flag))
        .select("date", "sid")
        .join(nxt, on="date")
        .select(pl.col("_next").alias("date"), "sid")
    )


def _index_returns(settings: Settings, name: str, out_name: str) -> pl.DataFrame:
    """(date, <out_name>): close-to-close return of an index over its own consecutive observations."""
    ix = (
        io.read(settings, "index_daily", columns=["index_name", "date", "close"])
        .filter(pl.col("index_name") == name)
        .select("date", "close")
        .collect()
        .sort("date")
    )
    return ix.select("date", (pl.col("close") / pl.col("close").shift(1) - 1.0).alias(out_name))


def build_market(settings: Settings) -> None:
    """Write panel/market.parquet: one row per session with previous-session u500 members (fractions)."""
    if not io.list_partitions(settings, "panel_universe"):
        raise DataNotAvailable("panel_universe is missing: run build_universe first")
    sess = session_frame(settings)
    uni = io.read(settings, "panel_universe", columns=["date", "sid", "u500", "u200"]).collect()
    daily = (
        io.read(
            settings,
            "panel_daily",
            columns=["sid", "date", "tret", "ret", "ret_co", "ret_oc", "adj_close", "flag_bigmove"],
        )
        .collect()
        .sort("sid", "date")
        .with_columns(
            pl.col("adj_close")
            .rolling_mean(_BREADTH_WINDOW, min_samples=_BREADTH_WINDOW)
            .over("sid")
            .alias("_ma")
        )
        .with_columns((pl.col("adj_close") > pl.col("_ma")).alias("_above"))
    )

    m500 = _prev_day_members(uni, "u500", sess)
    j500 = m500.join(daily, on=["sid", "date"], how="left")
    clean = j500.filter(~pl.col("flag_bigmove").fill_null(False))
    out = m500.group_by("date").agg(pl.len().cast(pl.Int32).alias("n_u500"))
    out = out.join(
        clean.group_by("date").agg(
            pl.col("tret").mean().alias("ew500_ret"),
            pl.col("ret_co").mean().alias("ew500_co"),
            pl.col("ret_oc").mean().alias("ew500_oc"),
            pl.col("ret").std().alias("xs_dispersion"),
        ),
        on="date",
        how="left",
    ).join(
        j500.group_by("date").agg(pl.col("_above").drop_nulls().mean().alias("breadth_200")),
        on="date",
        how="left",
    )
    j200 = _prev_day_members(uni, "u200", sess).join(daily, on=["sid", "date"], how="left")
    ew200 = (
        j200.filter(~pl.col("flag_bigmove").fill_null(False))
        .group_by("date")
        .agg(pl.col("tret").mean().alias("ew200_ret"))
    )
    out = out.join(ew200, on="date", how="left")
    for name, col in (("NIFTY 50", "nifty50_ret"), ("NIFTY 500", "nifty500_ret")):
        out = out.join(_index_returns(settings, name, col), on="date", how="left")
    out = out.with_columns(
        [
            pl.lit(None, dtype=pl.Float64).alias(c)
            for c in ("nifty50_ret", "nifty500_ret")
            if c not in out.columns
        ]
    ).sort("date")
    io.write_partition(settings, "panel_market", "all", out)
    log.info("market_built", rows=out.height)
