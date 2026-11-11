"""Tradable universe u500 / u200 with hysteresis, computed forward in time (PLAN T3.3)."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from stocksense.core.config import Settings
from stocksense.core.log import get_logger
from stocksense.data import io
from stocksense.panel.security_master import session_frame

log = get_logger(__name__)

FNO_START = date(2024, 7, 8)  # F&O bhavcopy coverage starts here
FNO_PROXY_RANK = 150  # before FNO_START: in_fno = turnover_rank <= 150 (documented proxy)
FNO_CARRY_SESSIONS = 5
_TRADED_WINDOW = "60i"
_STALE_WINDOW = "20i"


def _excluded_sids(settings: Settings) -> pl.DataFrame:
    """sids that are ETFs: any symbol in etf_list, or symbol / latest name matching etf_name_regex."""
    cfg = settings.universe
    sm = io.read(settings, "security_master", columns=["sid", "symbol", "name_last"]).collect()
    etf = io.read(settings, "etf_list", columns=["symbol"]).collect()["symbol"].unique()
    bad = (
        pl.col("symbol").is_in(etf.to_list())
        | pl.col("symbol").str.contains(cfg.etf_name_regex)
        | pl.col("name_last").fill_null("").str.contains(cfg.etf_name_regex)
    )
    return sm.filter(bad).select("sid").unique()


def _eligibility(settings: Settings) -> pl.DataFrame:
    """Per (sid, date) rows with point-in-time eligibility columns (rolling windows end at the row)."""
    cfg = settings.universe
    sess = session_frame(settings).rename({"idx": "session_idx"})
    d = (
        io.read(
            settings,
            "panel_daily",
            columns=["sid", "date", "symbol", "series", "close", "turnover_inr", "flag_stale"],
        )
        .collect()
        .join(sess, on="date", how="left")
        .sort("sid", "date")
    )
    d = d.with_columns(pl.lit(1, dtype=pl.Int32).alias("_one")).with_columns(
        pl.int_range(pl.len()).over("sid").alias("_prior_rows"),
        pl.col("_one")
        .rolling_sum_by("session_idx", _TRADED_WINDOW, closed="right")
        .over("sid")
        .alias("_traded60"),
        pl.col("turnover_inr").rolling_median(60, min_samples=1).over("sid").alias("med_turn_60"),
        pl.col("flag_stale")
        .cast(pl.Int8)
        .rolling_max_by("session_idx", _STALE_WINDOW, closed="right")
        .over("sid")
        .alias("_stale20"),
    )
    excluded = _excluded_sids(settings)
    ok = (
        pl.col("series").is_in(list(cfg.series_allowed))
        & (pl.col("_prior_rows") >= cfg.min_history_sessions)
        & (pl.col("_traded60") >= cfg.min_traded_of_last_60)
        & (pl.col("close") >= cfg.min_close)
        & (pl.col("med_turn_60") >= cfg.min_med_turnover_inr)
        & (pl.col("_stale20") == 0)
    )
    d = d.filter(ok.fill_null(False)).join(excluded, on="sid", how="anti")
    return d.select("sid", "date", "session_idx", "med_turn_60").with_columns(
        pl.col("med_turn_60")
        .rank("ordinal", descending=True)
        .over("date")
        .cast(pl.Int32)
        .alias("turnover_rank")
    )


def _hysteresis(elig: pl.DataFrame, settings: Settings) -> pl.DataFrame:
    """Add u500 / u200 by walking the sessions forward, carrying the previous session's members."""
    cfg = settings.universe
    elig = elig.sort("date", "turnover_rank")
    dates = elig["date"].to_numpy()
    sids = elig["sid"].to_numpy()
    rank = elig["turnover_rank"].to_numpy()
    med = elig["med_turn_60"].to_numpy()
    _, starts = np.unique(dates, return_index=True)
    bounds = [*starts.tolist(), len(dates)]
    u500 = np.zeros(len(dates), dtype=bool)
    u200 = np.zeros(len(dates), dtype=bool)
    prev500 = np.empty(0, dtype=sids.dtype)
    prev200 = np.empty(0, dtype=sids.dtype)
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        s, r, m = sids[a:b], rank[a:b], med[a:b]
        floor200 = m >= cfg.u200_min_med_turnover_inr
        in500 = (r <= cfg.u500_size) | (np.isin(s, prev500) & (r <= cfg.u500_exit_rank))
        in200 = ((r <= cfg.u200_size) | (np.isin(s, prev200) & (r <= cfg.u200_exit_rank))) & floor200
        u500[a:b], u200[a:b] = in500, in200
        prev500, prev200 = s[in500], s[in200]
    return elig.with_columns(pl.Series("u500", u500), pl.Series("u200", u200))


def _in_fno(settings: Settings, out: pl.DataFrame) -> pl.DataFrame:
    """STF membership from fo_bhavcopy (carried over <= 5 missing sessions); rank proxy before it."""
    sess = session_frame(settings)
    sm = io.read(settings, "security_master", columns=["sid", "symbol"]).collect()
    fo = (
        io.read(settings, "fo_bhavcopy", start=FNO_START, columns=["symbol", "instrument_type", "date"])
        .filter(pl.col("instrument_type") == "STF")
        .select("symbol", "date")
        .unique()
        .collect()
        .join(sm.unique("symbol"), on="symbol", how="inner")
        .select(pl.col("date").alias("src_date"), "sid")
        .with_columns(pl.lit(True).alias("_fno"))
    )
    fo_dates = fo.select("src_date").unique().sort("src_date")
    if fo_dates.is_empty():
        src = pl.DataFrame(schema={"date": pl.Date, "src_date": pl.Date})
    else:
        src = (
            sess.filter(pl.col("date") >= FNO_START)
            .sort("date")
            .join_asof(
                fo_dates.join(sess.rename({"date": "src_date", "idx": "src_idx"}), on="src_date").sort(
                    "src_date"
                ),
                left_on="date",
                right_on="src_date",
                strategy="backward",
            )
            .filter((pl.col("idx") - pl.col("src_idx")) <= FNO_CARRY_SESSIONS)
            .select("date", "src_date")
        )
    out = out.join(src, on="date", how="left").join(fo, on=["src_date", "sid"], how="left")
    in_fno = (
        pl.when(pl.col("date") >= FNO_START)
        .then(pl.col("_fno").fill_null(False))
        .otherwise(pl.col("turnover_rank") <= FNO_PROXY_RANK)
    )
    return out.with_columns(in_fno.alias("in_fno")).drop("src_date", "_fno")


def build_universe(settings: Settings) -> None:
    """Write panel/universe/part-YYYY.parquet (one row per ELIGIBLE sid-date; others are not members).

    Membership on date t uses only rows <= t: rolling windows end at t, ranks are within t's
    cross-section, and the hysteresis walks forward carrying yesterday's member sets.
    """
    for p in io.list_partitions(settings, "panel_universe"):
        p.unlink()
    elig = _eligibility(settings)
    if elig.is_empty():
        log.warning("universe_empty")
        return
    out = _hysteresis(elig, settings)
    out = _in_fno(settings, out).drop("session_idx")
    out = out.with_columns(pl.col("date").dt.year().alias("_y"))
    for (year,), part in sorted(out.partition_by("_y", as_dict=True).items()):
        io.write_partition(settings, "panel_universe", str(year), part.drop("_y"))
    log.info("universe_built", rows=out.height, u500=int(out["u500"].sum()), u200=int(out["u200"].sum()))
