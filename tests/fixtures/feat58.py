"""Hand-made panel environment for feature unit tests (writes raw + panel datasets to a temp root)."""

from __future__ import annotations

from datetime import date

import polars as pl

from stocksense.core.config import Settings
from stocksense.data import io


def write_env(
    s: Settings,
    dates: list[date],
    stocks: dict[str, dict],
    market: dict[str, list] | None = None,
) -> None:
    """stocks: symbol -> {sid, close, open?, high?, low?, turnover?, industry?}. Returns are close-to-close
    fractions; panel columns not given are filled with simple defaults. Everything is u500 and u200."""
    eq, pan, uni = [], [], []
    for sym, d in stocks.items():
        close = [float(x) for x in d["close"]]
        opn = [float(x) for x in d.get("open", close)]
        high = [float(x) for x in d.get("high", [max(o, c) for o, c in zip(opn, close, strict=True)])]
        low = [float(x) for x in d.get("low", [min(o, c) for o, c in zip(opn, close, strict=True)])]
        turn = [float(x) for x in d.get("turnover", [1e7] * len(close))]
        for i, dt in enumerate(dates):
            prev = close[i - 1] if i else opn[i]
            eq.append(
                (
                    sym,
                    "EQ",
                    dt,
                    opn[i],
                    high[i],
                    low[i],
                    close[i],
                    prev,
                    close[i],
                    1000,
                    turn[i],
                    10.0,
                    "legacy",
                )
            )
            ret = close[i] / prev - 1.0 if i else None
            pan.append(
                {
                    "sid": d["sid"],
                    "date": dt,
                    "symbol": sym,
                    "series": "EQ",
                    "open": opn[i],
                    "high": high[i],
                    "low": low[i],
                    "close": close[i],
                    "prev_close": prev,
                    "turnover_inr": turn[i],
                    "gap_factor": 1.0,
                    "adj_open": opn[i],
                    "adj_high": high[i],
                    "adj_low": low[i],
                    "adj_close": close[i],
                    "ret": ret,
                    "ret_co": (opn[i] / prev - 1.0) if i else None,
                    "ret_oc": close[i] / opn[i] - 1.0,
                    "tret": ret,
                    "industry": d.get("industry", "UNKNOWN"),
                }
            )
            uni.append(
                {
                    "date": dt,
                    "sid": d["sid"],
                    "turnover_rank": 1,
                    "med_turn_60": turn[i],
                    "u500": True,
                    "u200": True,
                    "in_fno": False,
                }
            )
    cols = [
        "symbol",
        "series",
        "date",
        "open",
        "high",
        "low",
        "close",
        "prev_close",
        "last_price",
        "volume",
        "turnover_inr",
        "n_trades",
        "era",
    ]
    io.upsert(s, "bhavcopy_eq", pl.DataFrame(eq, schema=cols, orient="row"))
    pdf = pl.DataFrame(pan)
    for (y,), part in (
        pdf.with_columns(pl.col("date").dt.year().alias("_y")).partition_by("_y", as_dict=True).items()
    ):
        io.write_partition(s, "panel_daily", str(y), part.drop("_y"))
    udf = pl.DataFrame(uni)
    for (y,), part in (
        udf.with_columns(pl.col("date").dt.year().alias("_y")).partition_by("_y", as_dict=True).items()
    ):
        io.write_partition(s, "panel_universe", str(y), part.drop("_y"))
    if market is None:
        m = pdf.group_by("date").agg(pl.col("ret").mean().alias("ew500_ret")).sort("date")
        market = {"date": m["date"].to_list(), "ew500_ret": m["ew500_ret"].to_list()}
    io.write_partition(s, "panel_market", "all", pl.DataFrame(market, strict=False))
