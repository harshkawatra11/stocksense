"""Which kinds of stocks a strategy's trades worked best in (PLAN-M M5.6).

Ported from market-lab lab/swing_live.py stock_profile (MIT, ba3f771); changes: numpy and polars, no pandas.
Units: `net` is the trade's net result in percent (Market Lab's JSON units); `risk` and `rs63` are percent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

MIN_SECTOR = 25
MIN_OTHER = 40
MIN_BEST_TRADES = 4
BEST_N = 40
SIZE_LABELS = ("Large cap (₹50k cr+)", "Mid cap (₹15–50k cr)", "Small cap (< ₹15k cr)", "Unknown")
LIQ_LABELS = ("Turnover < ₹10 cr/day", "₹10–30 cr/day", "₹30–100 cr/day", "₹100 cr+/day")
VOL_LABELS = ("Tight base (risk < 6%)", "Risk 6–9%", "Risk 9–13%", "Wide base (risk > 13%)")
RS_LABELS = ("Weaker than Nifty", "RS 0–15", "RS 15–30", "RS 30+ (leaders)")
TITLES = (
    ("sector", "Sector"),
    ("size", "Company size"),
    ("liq", "Liquidity"),
    ("vol", "Base width"),
    ("rsb", "Strength vs Nifty (3 months)"),
)


def stock_profile(
    trades: list[dict[str, Any]], classification: dict[str, dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Group a strategy's closed trades by stock traits and rank stocks.

    `trades` rows carry sym, net (%), risk (%), rs63 (points) and val_cr (daily turnover, INR crore);
    `classification[sym]` may carry `sector` and `mcap` (INR). Returns ({title: [{k, n, avg, win}]},
    best_stocks[{sym, n, avg, tot, win, sector}]). Groups need 25 trades (sector) or 40 (others); a stock
    needs 4 trades to be listed, and the 40 best by average are kept.
    """
    if not trades:
        return {title: [] for _, title in TITLES}, []
    df = pl.DataFrame(
        {
            "sym": [t["sym"] for t in trades],
            "net": [float(t["net"]) for t in trades],
            "risk": [float(t["risk"]) for t in trades],
            "rs63": [float(t["rs63"]) if t.get("rs63") is not None else np.nan for t in trades],
            "val_cr": [float(t["val_cr"]) if t.get("val_cr") is not None else np.nan for t in trades],
        }
    )
    cls = {s: classification.get(s) or {} for s in df["sym"].unique().to_list()}
    sector = [cls[s].get("sector") or "Other" for s in df["sym"]]
    mc = np.array([(cls[s].get("mcap") or 0) / 1e7 for s in df["sym"]])  # INR crore
    size = np.where(
        mc >= 50000,
        SIZE_LABELS[0],
        np.where(mc >= 15000, SIZE_LABELS[1], np.where(mc > 0, SIZE_LABELS[2], SIZE_LABELS[3])),
    )
    df = df.with_columns(
        pl.Series("sector", sector),
        pl.Series("size", size.tolist()),
        pl.Series("liq", _bucket(df["val_cr"].to_numpy(), [10, 30, 100], 0.0, LIQ_LABELS)),
        pl.Series("vol", _bucket(df["risk"].to_numpy(), [6, 9, 13], 0.0, VOL_LABELS)),
        pl.Series("rsb", _bucket(df["rs63"].to_numpy(), [0, 15, 30], -999.0, RS_LABELS)),
    )
    out: dict[str, list[dict[str, Any]]] = {}
    for col, title in TITLES:
        g = (
            df.drop_nulls(col)
            .group_by(col)
            .agg(
                pl.len().alias("n"),
                pl.col("net").mean().alias("avg"),
                (pl.col("net") > 0).mean().mul(100).alias("win"),
            )
            .filter(pl.col("n") >= (MIN_SECTOR if col == "sector" else MIN_OTHER))
            .sort("avg", col, descending=[True, False])
        )
        out[title] = [
            {
                "k": str(r[col]),
                "n": int(r["n"]),
                "avg": round(float(r["avg"]), 2),
                "win": round(float(r["win"]), 1),
            }
            for r in g.iter_rows(named=True)
        ]
    s = (
        df.group_by("sym")
        .agg(
            pl.len().alias("n"),
            pl.col("net").mean().alias("avg"),
            pl.col("net").sum().alias("tot"),
            (pl.col("net") > 0).mean().mul(100).alias("win"),
        )
        .filter(pl.col("n") >= MIN_BEST_TRADES)
        .sort("avg", "sym", descending=[True, False])
        .head(BEST_N)
    )
    best = [
        {
            "sym": r["sym"],
            "n": int(r["n"]),
            "avg": round(float(r["avg"]), 2),
            "tot": round(float(r["tot"]), 1),
            "win": round(float(r["win"]), 1),
            "sector": cls[r["sym"]].get("sector", ""),
        }
        for r in s.iter_rows(named=True)
    ]
    return out, best


def _bucket(x: np.ndarray, inner: list[float], lo: float, labels: tuple[str, ...]) -> list[str | None]:
    """Right-closed bins (lo, inner[0]], ..., (inner[-1], inf); NaN and values <= lo get None (pandas.cut)."""
    pos = np.searchsorted(np.asarray(inner, dtype=float), x, side="left")
    ok = np.isfinite(x) & (x > lo)
    return [labels[int(p)] if k else None for p, k in zip(pos, ok, strict=True)]
