"""Bulk and block deals: per stock and day, money bought and sold and who moved the most."""

from __future__ import annotations

import tomllib
from datetime import date
from pathlib import Path

import polars as pl

from stocksense.core.config import Settings
from stocksense.data import io

_EMPTY = {
    "date": pl.Date,
    "symbol": pl.Utf8,
    "buy_value_cr": pl.Float64,
    "sell_value_cr": pl.Float64,
    "net_value_cr": pl.Float64,
    "n_deals": pl.Int64,
    "top_client": pl.Utf8,
    "flagged": pl.Boolean,
}


def load_clients(path: Path = Path("config/flow_clients.toml")) -> set[str]:
    """Client names to highlight, from config/flow_clients.toml (empty set if the file is missing)."""
    if not path.is_file():
        return set()
    with path.open("rb") as f:
        return set(tomllib.load(f).get("clients", []))


def deals_summary(deals: pl.DataFrame, clients: set[str]) -> pl.DataFrame:
    """Per (date, symbol): buy, sell and net value (cr), deal count, top client by |net value|, flagged."""
    if deals.height == 0:
        return pl.DataFrame(schema=_EMPTY)
    is_buy = pl.col("side").str.to_uppercase().str.starts_with("B")
    d = deals.with_columns(
        (pl.col("qty") * pl.col("price") / 1e7).alias("_v"),
    ).with_columns(pl.when(is_buy).then(pl.col("_v")).otherwise(-pl.col("_v")).alias("_signed"))
    per_client = d.group_by("date", "symbol", "client").agg(pl.col("_signed").sum().alias("_net"))
    top = (
        per_client.with_columns(pl.col("_net").abs().alias("_abs"))
        .sort("date", "symbol", "_abs", "client", descending=[False, False, True, False])
        .group_by("date", "symbol", maintain_order=True)
        .agg(pl.col("client").first().alias("top_client"))
    )
    agg = d.group_by("date", "symbol").agg(
        pl.col("_v").filter(is_buy).sum().alias("buy_value_cr"),
        pl.col("_v").filter(~is_buy).sum().alias("sell_value_cr"),
        pl.len().cast(pl.Int64).alias("n_deals"),
        pl.col("client").is_in(sorted(clients)).any().alias("flagged"),
    )
    return (
        agg.join(top, on=["date", "symbol"])
        .with_columns((pl.col("buy_value_cr") - pl.col("sell_value_cr")).alias("net_value_cr"))
        .select(list(_EMPTY))
        .sort("date", "symbol")
    )


def build_deals_daily(
    settings: Settings, start: date, end: date, clients_path: Path = Path("config/flow_clients.toml")
) -> int:
    """Summarise stored deals for start..end into flow_deals_daily; returns rows written."""
    deals = io.read(settings, "deals", start, end).collect()
    out = deals_summary(deals, load_clients(clients_path))
    if out.height:
        io.upsert(settings, "flow_deals_daily", out)
    return out.height
