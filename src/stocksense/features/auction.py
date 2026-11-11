"""Auction features for strategy S2 (PLAN T4.4): the opening gap and what is known at 09:07.

For a trade date `d` (the session after decision date `t`) and the sids in `u200` on `t`:
`gap = O(d) / (C(t) * gap_factor(d)) - 1`. In history `O(d)` is the official open (`gap_source =
'official_open'`); live it is the 09:07 indicative price (`gap_source = 'indicative_0907'`), so the
difference between the two can be measured from the first day. Nothing after `O(d)` is used: the
only day-`d` panel columns read are `open` and `gap_factor`.

Stored as `features/auction/part-YYYY.parquet`, keyed by `(trade_date, sid)`.
"""

from __future__ import annotations

from datetime import date, time

import polars as pl

from stocksense.core.config import Settings
from stocksense.core.errors import DataNotAvailable
from stocksense.core.paths import Paths
from stocksense.features.base import FeatureContext, log_ret, roll_std
from stocksense.validation.lockbox import guard

GAP_SOURCE_HISTORY = "official_open"
GAP_SOURCE_LIVE = "indicative_0907"
MIN_INDUSTRY = 5
PREOPEN_END = time(9, 8)  # the pre-open order-entry and matching window ends at 09:08
MACRO_SERIES = ("SPX", "NDX", "USDINR", "BRENT")
OUT_COLUMNS = (
    "trade_date",
    "sid",
    "decision_date",
    "gap",
    "gap_z",
    "gap_rel_mkt",
    "gap_rel_ind",
    "mkt_gap",
    "gap_rank",
    "preopen_imbalance",
    "spx_overnight",
    "ndx_overnight",
    "usdinr_overnight",
    "brent_overnight",
    "gap_source",
)


def gap_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add gap, gap_z, mkt_gap, gap_rel_mkt, gap_rel_ind, gap_rank (fractions) per trade_date.

    Needs columns trade_date, sid, industry, price (O(d) or indicative), close_t, gap_factor, vol_21.
    """
    g = pl.col("gap")
    day = "trade_date"
    ind = [day, "industry"]
    ok_ind = (pl.col("industry") != "UNKNOWN") & (g.count().over(ind) >= MIN_INDUSTRY)
    out = df.with_columns(
        pl.when((pl.col("price") > 0) & (pl.col("close_t") > 0))
        .then(pl.col("price") / (pl.col("close_t") * pl.col("gap_factor").fill_null(1.0)) - 1.0)
        .alias("gap")
    )
    return out.with_columns(
        pl.when(pl.col("vol_21") > 0).then(g / pl.col("vol_21")).alias("gap_z"),
        g.median().over(day).alias("mkt_gap"),
        pl.when(ok_ind).then(g.median().over(ind)).alias("_ind_med"),
        pl.when(g.count().over(day) > 1)
        .then((g.rank("average").over(day) - 1.0) / (g.count().over(day) - 1).cast(pl.Float64) - 0.5)
        .alias("gap_rank"),
    ).with_columns(
        (g - pl.col("mkt_gap")).alias("gap_rel_mkt"),
        (g - pl.col("_ind_med")).alias("gap_rel_ind"),
    )


def _overnight(ctx: FeatureContext, pairs: pl.DataFrame) -> pl.DataFrame:
    """(trade_date, <s>_overnight): ln(newest value at 09:07 on d / newest value at 20:00 on t)."""
    p = (
        pairs.select("trade_date", "decision_date")
        .unique()
        .with_columns(
            (pl.col("decision_date").cast(pl.Datetime("us")) + pl.duration(hours=20))
            .dt.replace_time_zone("Asia/Kolkata")
            .alias("_cut_dec"),
            (pl.col("trade_date").cast(pl.Datetime("us")) + pl.duration(hours=9, minutes=7))
            .dt.replace_time_zone("Asia/Kolkata")
            .alias("_cut_open"),
        )
    )
    out = p.select("trade_date")
    for s in MACRO_SERIES:
        m = (
            ctx.raw("macro_daily", ["series", "close", "available_ts"])
            .filter(pl.col("series").str.to_uppercase() == s)
            .select("available_ts", pl.col("close").alias("_v"))
            .drop_nulls("available_ts")
            .sort("available_ts")
            .collect()
        )
        a = (
            p.sort("_cut_dec")
            .join_asof(m, left_on="_cut_dec", right_on="available_ts")
            .select("trade_date", pl.col("_v").alias("_v_dec"))
        )
        b = (
            p.sort("_cut_open")
            .join_asof(m, left_on="_cut_open", right_on="available_ts")
            .select("trade_date", pl.col("_v").alias("_v_open"))
        )
        col = f"{s.lower()}_overnight"
        j = a.join(b, on="trade_date").select(
            "trade_date", (pl.col("_v_open") / pl.col("_v_dec")).log().alias(col)
        )
        out = out.join(j, on="trade_date", how="left")
    return out


def _imbalance(ctx: FeatureContext) -> pl.DataFrame:
    """(symbol, trade_date, preopen_imbalance): last pre-open snapshot up to 09:08 of each session."""
    return (
        ctx.raw("preopen", ["symbol", "ts", "buy_qty", "sell_qty"])
        .with_columns(pl.col("ts").dt.date().alias("trade_date"), pl.col("ts").dt.time().alias("_t"))
        .filter(pl.col("_t") <= PREOPEN_END)
        .sort("symbol", "ts")
        .group_by("symbol", "trade_date")
        .last()
        .select(
            "symbol",
            "trade_date",
            pl.when(pl.col("buy_qty") + pl.col("sell_qty") > 0)
            .then((pl.col("buy_qty") - pl.col("sell_qty")) / (pl.col("buy_qty") + pl.col("sell_qty")))
            .alias("preopen_imbalance"),
        )
        .collect()
    )


def _with_vol(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.sort("sid", "date")
        .with_columns(log_ret("ret").alias("_la"))
        .with_columns(roll_std("_la", 21).alias("vol_21"))
    )


def _finish(ctx: FeatureContext, rows: pl.DataFrame, imb: pl.DataFrame, source: str) -> pl.DataFrame:
    df = gap_features(rows)
    df = df.join(imb, on=["symbol", "trade_date"], how="left")
    df = df.join(_overnight(ctx, df.select("trade_date", "decision_date")), on="trade_date", how="left")
    return df.with_columns(pl.lit(source).alias("gap_source")).select(OUT_COLUMNS).sort("trade_date", "sid")


def auction_frame(ctx: FeatureContext) -> pl.DataFrame:
    """Historical auction features for every trade date in the context (decision date = previous session)."""
    sess = ctx.sessions().collect()
    nxt = sess.select(
        pl.col("date").alias("decision_date"), pl.col("date").shift(-1).alias("trade_date")
    ).drop_nulls()
    panel = _with_vol(ctx.panel().collect())
    u200 = (
        ctx.universe().filter(pl.col("u200")).select(pl.col("date").alias("decision_date"), "sid").collect()
    )
    on_t = panel.select(
        "sid",
        pl.col("date").alias("decision_date"),
        pl.col("close").alias("close_t"),
        "vol_21",
        "industry",
    )
    on_d = panel.select(
        "sid",
        pl.col("date").alias("trade_date"),
        "symbol",
        pl.col("open").alias("price"),
        "gap_factor",
    )
    rows = (
        u200.join(nxt, on="decision_date", how="inner")
        .join(on_t, on=["decision_date", "sid"], how="inner")
        .join(on_d, on=["trade_date", "sid"], how="inner")
    )
    return _finish(ctx, rows, _imbalance(ctx), GAP_SOURCE_HISTORY)


def auction_features_live(
    ctx: FeatureContext, preopen_frame: pl.DataFrame, panel_tail: pl.DataFrame
) -> pl.DataFrame:
    """The 09:07 decision frame: gap from the indicative price, one row per u200 stock with a pre-open row.

    `preopen_frame`: symbol, ts, indicative_price, buy_qty, sell_qty and optionally gap_factor (a corporate
    action known for today; default 1.0). `panel_tail`: recent `ctx.panel()` rows (needs >= 22 sessions for
    gap_z). The decision date is the last panel_tail session before the pre-open date.
    """
    trade_date = preopen_frame.select(pl.col("ts").dt.date().max())[0, 0]
    dec = panel_tail.filter(pl.col("date") < trade_date)["date"].max()
    if dec is None:
        raise DataNotAvailable("panel_tail has no session before the pre-open date")
    panel = _with_vol(panel_tail)
    u200 = ctx.universe().filter(pl.col("u200") & (pl.col("date") == dec)).select("sid").collect()
    t_rows = panel.filter(pl.col("date") == dec).select(
        "sid", "symbol", pl.col("close").alias("close_t"), "vol_21", "industry"
    )
    gf = pl.col("gap_factor") if "gap_factor" in preopen_frame.columns else pl.lit(1.0)
    pre = (
        preopen_frame.sort("ts")
        .group_by("symbol", maintain_order=True)
        .last()
        .select(
            "symbol", pl.col("indicative_price").alias("price"), gf.alias("gap_factor"), "buy_qty", "sell_qty"
        )
    )
    rows = (
        u200.join(t_rows, on="sid", how="inner")
        .join(pre, on="symbol", how="inner")
        .with_columns(pl.lit(trade_date).alias("trade_date"), pl.lit(dec).alias("decision_date"))
    )
    imb = rows.select(
        "symbol",
        "trade_date",
        pl.when(pl.col("buy_qty") + pl.col("sell_qty") > 0)
        .then((pl.col("buy_qty") - pl.col("sell_qty")) / (pl.col("buy_qty") + pl.col("sell_qty")))
        .alias("preopen_imbalance"),
    )
    return _finish(ctx, rows.drop("buy_qty", "sell_qty"), imb, GAP_SOURCE_LIVE)


# ---- store -------------------------------------------------------------------------------------


def _dir(settings: Settings):
    return Paths(settings).features / "auction"


def build_auction_features(settings: Settings, start: date | None = None, end: date | None = None) -> int:
    """Compute historical auction features for trade dates in [start, end] and merge them into
    features/auction/part-YYYY.parquet by (trade_date, sid); returns the number of rows computed."""
    df = auction_frame(FeatureContext(settings, as_of=end))
    if start is not None:
        df = df.filter(pl.col("trade_date") >= start)
    if end is not None:
        df = df.filter(pl.col("trade_date") <= end)
    out_dir = _dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    for (year,), part in (
        df.with_columns(pl.col("trade_date").dt.year().alias("_y")).partition_by("_y", as_dict=True).items()
    ):
        part = part.drop("_y")
        path = out_dir / f"part-{year}.parquet"
        if path.is_file():
            old = pl.read_parquet(path)
            lo, hi = part["trade_date"].min(), part["trade_date"].max()
            keep = old.filter((pl.col("trade_date") < lo) | (pl.col("trade_date") > hi))
            part = pl.concat([keep, part], how="vertical_relaxed")
        tmp = path.with_name(path.name + ".tmp")
        part.sort("trade_date", "sid").write_parquet(tmp, compression="zstd")
        tmp.replace(path)
    return df.height


def load_auction_features(
    settings: Settings, purpose: str = "research", start: date | None = None, end: date | None = None
) -> pl.LazyFrame:
    """Stored auction features as a lazy frame ('research' stops at lockbox_start - 1 day)."""
    files = sorted(_dir(settings).glob("part-*.parquet"))
    if not files:
        raise DataNotAvailable("no auction features built yet (features/auction is empty)")
    lf = pl.concat([pl.scan_parquet(f) for f in files], how="vertical_relaxed")
    if start is not None:
        lf = lf.filter(pl.col("trade_date") >= start)
    if end is not None:
        lf = lf.filter(pl.col("trade_date") <= end)
    return guard(lf, "trade_date", purpose, settings)
