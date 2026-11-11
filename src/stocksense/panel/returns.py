"""Daily panel: one row per company per session with correct returns and adjusted prices (PLAN T3.2)."""

from __future__ import annotations

from datetime import date

import polars as pl

from stocksense.core.config import Settings
from stocksense.core.log import get_logger
from stocksense.data import io
from stocksense.panel.adjust import compute_gap_factors
from stocksense.panel.security_master import SERIES, session_frame, symbol_sid_map

log = get_logger(__name__)

_BANDS = (0.02, 0.05, 0.10, 0.20)
_LOCK_TOL = 0.0025
_RAW_COLS = [
    "symbol",
    "series",
    "date",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "turnover_inr",
    "n_trades",
]


def _load_rows(settings: Settings, sm: pl.DataFrame, sess: pl.DataFrame) -> pl.DataFrame:
    """EQ/BE/BZ rows joined to sid and session index; one row per (sid, date), EQ > BE > BZ."""
    prio = pl.when(pl.col("series") == "EQ").then(0).when(pl.col("series") == "BE").then(1).otherwise(2)
    return (
        io.read(settings, "bhavcopy_eq", columns=_RAW_COLS)
        .filter(pl.col("series").is_in(SERIES))
        .join(sm.select("symbol", "sid").lazy(), on="symbol", how="inner")
        .join(sess.rename({"idx": "session_idx"}).lazy(), on="date", how="left")
        .with_columns(prio.alias("_prio"))
        .collect()
        .sort("sid", "date", "_prio")
        .unique(subset=["sid", "date"], keep="first", maintain_order=True)
        .drop("_prio")
    )


def _ca_with_sid(settings: Settings, sm: pl.DataFrame) -> pl.DataFrame:
    ca = io.read(settings, "corporate_actions").collect()
    ca = ca.join(sm.select("symbol", "sid").unique("symbol"), on="symbol", how="inner")
    listed = (
        io.read(settings, "listed_equities", columns=["symbol", "face_value", "as_of"])
        .collect()
        .sort("as_of")
        .group_by("symbol", maintain_order=True)
        .last()
        .select("symbol", pl.col("face_value").alias("face_fallback"))
    )
    return ca.join(listed, on="symbol", how="left")


def _industry_by_sid(settings: Settings, sm: pl.DataFrame) -> pl.DataFrame:
    """Latest sectors snapshot mapped to sid (any symbol of the company)."""
    sec = io.read(settings, "sectors", columns=["symbol", "industry", "as_of"]).collect()
    if sec.is_empty():
        return pl.DataFrame(schema={"sid": pl.Int32, "industry": pl.Utf8})
    return (
        sec.join(sm.select("symbol", "sid").unique("symbol"), on="symbol", how="inner")
        .filter(pl.col("industry").is_not_null())
        .sort("as_of", "symbol")
        .group_by("sid", maintain_order=True)
        .last()
        .select("sid", "industry")
    )


def _core(rows: pl.DataFrame, gaps: pl.DataFrame, divs: pl.DataFrame) -> pl.DataFrame:
    """Gap factors, returns, adjusted prices and flags for the whole history (vectorised per sid)."""
    g = gaps.filter(pl.col("gap_src") != "ca_ignored").select("sid", "date", "gap_factor", "gap_src")
    df = (
        rows.join(g, on=["sid", "date"], how="left")
        .join(divs, on=["sid", "date"], how="left")
        .with_columns(
            pl.col("gap_factor").fill_null(1.0),
            pl.col("gap_src").fill_null(""),
            pl.col("div_amt").fill_null(0.0),
        )
        .sort("sid", "date")
    )
    lc = pl.col("close").shift(1).over("sid")
    first = pl.int_range(pl.len()).over("sid") == 0
    base = pl.when(pl.col("prev_close") > 0).then(pl.col("prev_close")).otherwise(lc)
    df = df.with_columns(
        base.alias("_base"),
        lc.alias("_lc"),
        first.alias("_first"),
        (pl.col("session_idx") - pl.col("session_idx").shift(1).over("sid"))
        .cast(pl.Int32)
        .alias("sessions_since_prev"),
    )
    denom = pl.col("_base") * pl.col("gap_factor")
    ok_open = pl.col("open") > 0
    df = df.with_columns(
        pl.when(~pl.col("_first") & (pl.col("close") > 0)).then(pl.col("close") / denom - 1.0).alias("ret"),
        pl.when(~pl.col("_first") & ok_open).then(pl.col("open") / denom - 1.0).alias("ret_co"),
        pl.when(~pl.col("_first") & ok_open & (pl.col("close") > 0))
        .then(pl.col("close") / pl.col("open") - 1.0)
        .alias("ret_oc"),
        (pl.col("div_amt") / pl.col("_base")).alias("div_ret"),
    )
    df = df.with_columns((pl.col("ret") + pl.col("div_ret")).alias("tret"))
    # adjusted prices: step g per row, adj[t] = product of g over LATER rows of the sid
    step = (
        pl.when(pl.col("_first") | pl.col("_base").is_null())
        .then(1.0)
        .otherwise(pl.col("gap_factor") * pl.col("_base") / pl.col("_lc"))
    )
    df = df.with_columns(step.alias("_g"))
    df = df.with_columns(
        pl.col("_g").shift(-1).fill_null(1.0).reverse().cum_prod().reverse().over("sid").alias("adj")
    )
    df = df.with_columns(
        (pl.col("open") * pl.col("adj")).alias("adj_open"),
        (pl.col("high") * pl.col("adj")).alias("adj_high"),
        (pl.col("low") * pl.col("adj")).alias("adj_low"),
        (pl.col("close") * pl.col("adj")).alias("adj_close"),
        (pl.col("volume") / pl.col("adj")).alias("adj_volume"),
    )
    up = pl.min_horizontal(*[(pl.col("ret") - b).abs() for b in _BANDS]) <= _LOCK_TOL
    dn = pl.min_horizontal(*[(pl.col("ret") + b).abs() for b in _BANDS]) <= _LOCK_TOL
    df = df.with_columns(
        (pl.col("ret").abs() > 0.40).fill_null(False).alias("flag_bigmove"),
        (
            (pl.col("sessions_since_prev") == 1)
            & ((pl.col("_base") / pl.col("_lc") - 1.0).abs() > 0.02)
            & (pl.col("gap_src") == "")
        )
        .fill_null(False)
        .alias("flag_unexplained_gap"),
        (pl.col("sessions_since_prev") > 5).fill_null(False).alias("flag_stale"),
        (pl.col("high") == pl.col("low")).alias("flag_one_price"),
        ((pl.col("close") == pl.col("high")) & up).fill_null(False).alias("flag_locked_up"),
        ((pl.col("close") == pl.col("low")) & dn).fill_null(False).alias("flag_locked_dn"),
    )
    return df.drop("_base", "_lc", "_first", "_g", "div_amt", "session_idx")


def _dividends(ca: pl.DataFrame, sess: pl.DataFrame) -> pl.DataFrame:
    """(sid, date, div_amt): dividend per share on the first session on/after each ex-date."""
    d = ca.filter(pl.col("dividend_amount").is_not_null() & (pl.col("dividend_amount") > 0))
    if d.is_empty():
        return pl.DataFrame(schema={"sid": pl.Int32, "date": pl.Date, "div_amt": pl.Float64})
    d = d.group_by(["sid", "ex_date"]).agg(pl.col("dividend_amount").sum().alias("div_amt")).sort("ex_date")
    d = d.join_asof(
        sess.select(pl.col("date").alias("_sd")).sort("_sd"),
        left_on="ex_date",
        right_on="_sd",
        strategy="forward",
    ).drop_nulls("_sd")
    return d.group_by(["sid", pl.col("_sd").alias("date")]).agg(pl.col("div_amt").sum())


def build_daily_panel(settings: Settings) -> None:
    """Write panel/daily/part-YYYY.parquet and panel/adj_events.parquet (returns are fractions)."""
    sm = symbol_sid_map(settings)
    sess = session_frame(settings)
    rows = _load_rows(settings, sm, sess)
    ca = _ca_with_sid(settings, sm)
    gaps = compute_gap_factors(rows, ca)
    divs = _dividends(ca, sess)
    industry = _industry_by_sid(settings, sm)
    core = _core(rows, gaps, divs).with_columns(pl.col("date").dt.year().alias("_y"))
    del rows

    n_rows = 0
    for (year,), part in sorted(core.partition_by("_y", as_dict=True).items()):
        part = part.drop("_y")
        dl = io.read(
            settings,
            "bhavcopy_delivery",
            start=date(year, 1, 1),
            end=date(year, 12, 31),
            columns=["symbol", "series", "date", "deliv_qty", "deliv_pct"],
        ).collect()
        part = (
            part.join(dl, on=["symbol", "series", "date"], how="left")
            .join(industry, on="sid", how="left")
            .with_columns(pl.col("industry").fill_null("UNKNOWN"))
        )
        io.write_partition(settings, "panel_daily", str(year), part)
        n_rows += part.height

    ev = gaps.join(sm.select("sid", "symbol").unique("sid", keep="last"), on="sid", how="left", suffix="_m")
    # symbol valid on the event date (the row's symbol); fall back to the master's latest symbol
    sym = core.select("sid", "date", pl.col("symbol").alias("_row_sym"))
    ev = (
        ev.join(sym, on=["sid", "date"], how="left")
        .with_columns(pl.coalesce("_row_sym", "symbol").alias("symbol"))
        .drop("_row_sym")
    )
    ev = ev.sort(pl.col("gap_src") == "ca_ignored", "sid", "date").unique(
        subset=["sid", "date", "symbol"], keep="first", maintain_order=True
    )
    io.write_partition(settings, "adj_events", "all", ev)
    log.info("daily_panel_built", rows=n_rows, adj_events=ev.height)
