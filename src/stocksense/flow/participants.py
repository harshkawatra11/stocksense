"""Participant positioning: index-futures long ratio, index option net by FII, DII, PRO and CLIENT."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from stocksense.core.config import Settings
from stocksense.data import io

CLIENTS = ("FII", "DII", "PRO", "CLIENT")
WARMUP_DAYS = 30  # calendar days read before `start` so the 5-session change exists on day one


def participant_columns() -> list[str]:
    """Every output column name besides `date`, in order."""
    cols = []
    for c in CLIENTS:
        k = c.lower()
        cols += [f"{k}_ratio", f"{k}_ratio_chg1", f"{k}_ratio_chg5", f"{k}_call_net", f"{k}_put_net"]
    return cols + ["fii_cash_net", "dii_cash_net"]


def participants_daily(poi: pl.DataFrame, fii_dii: pl.DataFrame) -> pl.DataFrame:
    """Wide per-date frame: index-futures long ratio and its 1/5-session change, option net, cash net."""
    tot = pl.col("fut_idx_long") + pl.col("fut_idx_short")
    per = poi.select(
        "date",
        pl.col("client_type").str.to_uppercase().alias("ct"),
        pl.when(tot > 0).then(pl.col("fut_idx_long") / tot).alias("ratio"),
        (pl.col("opt_idx_call_long") - pl.col("opt_idx_call_short")).alias("call_net"),
        (pl.col("opt_idx_put_long") - pl.col("opt_idx_put_short")).alias("put_net"),
    )
    dates = per.select("date").unique().sort("date")
    out = dates
    for c in CLIENTS:
        k = c.lower()
        one = per.filter(pl.col("ct") == c).select(
            "date",
            pl.col("ratio").alias(f"{k}_ratio"),
            pl.col("call_net").alias(f"{k}_call_net"),
            pl.col("put_net").alias(f"{k}_put_net"),
        )
        out = out.join(one, on="date", how="left")
    out = out.sort("date")
    for c in CLIENTS:
        r = pl.col(f"{c.lower()}_ratio")
        out = out.with_columns(
            (r - r.shift(1)).alias(f"{c.lower()}_ratio_chg1"),
            (r - r.shift(5)).alias(f"{c.lower()}_ratio_chg5"),
        )
    for c in ("FII", "DII"):
        cash = fii_dii.filter(pl.col("category").str.to_uppercase() == c).select(
            "date", pl.col("net_value").alias(f"{c.lower()}_cash_net")
        )
        out = out.join(cash.unique("date", keep="last"), on="date", how="left")
    return out.select("date", *participant_columns()).sort("date")


def build_participants_daily(settings: Settings, start: date, end: date) -> int:
    """Compute and upsert flow_participants_daily for start..end; returns rows written."""
    poi = io.read(settings, "participant_oi", start - timedelta(days=WARMUP_DAYS), end).collect()
    cash = io.read(settings, "fii_dii", start, end).collect()
    out = participants_daily(poi, cash).filter(pl.col("date").is_between(start, end))
    if out.height:
        io.upsert(settings, "flow_participants_daily", out)
    return out.height
