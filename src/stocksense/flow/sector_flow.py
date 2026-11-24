"""Sector money flow: which sector is taking turnover, and on which side of its open.

flow_dir = sum(value_i * sign(from_open_i)) / sum(value_i), in -1..+1. Live it is built from the tape rows
(from_open in percent); at the end of the day from `panel_daily` (turnover_inr, ret_oc). Descriptive only.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from stocksense.core.config import Settings
from stocksense.data import io
from stocksense.marketlab.tape import sector_of

RATIO_MIN = 1.2
DIR_MIN = 0.2
HIST_SESSIONS = 20
HIST_MIN = 5
_COLS = ["date", "sector", "flow_dir", "value_cr", "turnover_share", "n", "up", "down", "deliv_spikes"]


def flow_dir(value_cr: list[float | None], from_open: list[float | None]) -> float | None:
    """Value-weighted share of buyers over sellers, -1..+1; rows with a missing value or move are skipped."""
    pairs = [(v, m) for v, m in zip(value_cr, from_open, strict=True) if v is not None and m is not None]
    tot = sum(v for v, _ in pairs)
    if not pairs or tot <= 0:
        return None
    return float(sum(v * float(np.sign(m)) for v, m in pairs) / tot)


def rotation(share_ratio: float | None, fdir: float | None) -> str:
    """'in' or 'out' when turnover share is over 1.2x normal and flow_dir beyond +-0.2, else 'neutral'."""
    if share_ratio is None or fdir is None or share_ratio <= RATIO_MIN:
        return "neutral"
    if fdir > DIR_MIN:
        return "in"
    if fdir < -DIR_MIN:
        return "out"
    return "neutral"


def live_sector_flow(
    rows: list[dict[str, Any]],
    mean_share: dict[str, float | None],
    quad_labels: dict[str, str],
    spikes: set[str],
) -> list[dict[str, Any]]:
    """One entry per tape sector (percent units in `rows`): flow_dir, share, ratio, rotation, breadth."""
    grand = sum(r.get("value_cr") or 0.0 for r in rows)
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["sector"], []).append(r)
    out = []
    for name, rs in by.items():
        fd = flow_dir([r.get("value_cr") for r in rs], [r.get("from_open") for r in rs])
        share = sum(r.get("value_cr") or 0.0 for r in rs) / grand if grand > 0 else None
        base = mean_share.get(name)
        ratio = share / base if share is not None and base else None
        quads: dict[str, int] = {}
        for r in rs:
            q = quad_labels.get(r["sym"])
            if q:
                quads[q] = quads.get(q, 0) + 1
        out.append(
            {
                "name": name,
                "flow_dir": None if fd is None else round(fd, 4),
                "turnover_share": None if share is None else round(share, 4),
                "share_ratio": None if ratio is None else round(ratio, 3),
                "rotation": rotation(ratio, fd),
                "up": sum((r.get("from_open") or 0) > 0 for r in rs),
                "down": sum((r.get("from_open") or 0) < 0 for r in rs),
                "oi_quads": quads,
                "deliv_spikes": sum(r["sym"] in spikes for r in rs),
            }
        )
    return sorted(out, key=lambda s: -(s["flow_dir"] if s["flow_dir"] is not None else -9))


class ValueHistory:
    """Per-session snapshots of each sector's traded value by minute, so today's share can be compared."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def record(self, day: date, minute: int, values: dict[str, float]) -> None:
        """Store each sector's value_cr at `minute` (since 09:15) of `day`, merging with earlier minutes."""
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.root / f"{day}.json"
        cur = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        cur[str(minute)] = {k: float(v) for k, v in values.items()}
        p.write_text(json.dumps(cur), encoding="utf-8")

    def mean_share(self, today: date, minute: int) -> dict[str, float]:
        """Mean turnover share per sector at or before `minute` over up to 20 earlier sessions; {} under 5."""
        if not self.root.exists():
            return {}
        files = sorted(f for f in self.root.glob("*.json") if f.stem < today.isoformat())
        shares: dict[str, list[float]] = {}
        used = 0
        for f in files[-HIST_SESSIONS:]:
            snap = json.loads(f.read_text(encoding="utf-8"))
            ms = [int(m) for m in snap if int(m) <= minute]
            if not ms:
                continue
            vals = snap[str(max(ms))]
            tot = sum(vals.values())
            if tot <= 0:
                continue
            used += 1
            for k, v in vals.items():
                shares.setdefault(k, []).append(v / tot)
        if used < HIST_MIN:
            return {}
        return {k: float(np.mean(v)) for k, v in shares.items()}


def sector_flow_daily(panel: pl.DataFrame, spikes: pl.DataFrame | None) -> pl.DataFrame:
    """Per (date, sector) from daily bars: flow_dir, value_cr, turnover_share, n, up, down, deliv_spikes."""
    pairs = panel.select("symbol", "industry").unique()
    mp = pl.DataFrame(
        {
            "symbol": pairs["symbol"],
            "industry": pairs["industry"],
            "sector": [
                sector_of(s, i or "") for s, i in zip(pairs["symbol"], pairs["industry"], strict=True)
            ],
        }
    )
    p = panel.join(mp, on=["symbol", "industry"], how="left", nulls_equal=True).with_columns(
        (pl.col("turnover_inr") / 1e7).alias("value_cr"),
        pl.col("ret_oc").sign().alias("sgn"),
    )
    if spikes is not None and spikes.height:
        p = p.join(spikes.select("date", "sid", "deliv_spike"), on=["date", "sid"], how="left")
    else:
        p = p.with_columns(pl.lit(False).alias("deliv_spike"))
    g = p.group_by("date", "sector").agg(
        (pl.col("value_cr") * pl.col("sgn").fill_null(0.0)).sum().alias("_num"),
        pl.col("value_cr").sum().alias("value_cr"),
        pl.len().cast(pl.Int64).alias("n"),
        (pl.col("ret_oc") > 0).sum().cast(pl.Int64).alias("up"),
        (pl.col("ret_oc") < 0).sum().cast(pl.Int64).alias("down"),
        pl.col("deliv_spike").fill_null(False).sum().cast(pl.Int64).alias("deliv_spikes"),
    )
    return (
        g.with_columns(
            pl.when(pl.col("value_cr") > 0).then(pl.col("_num") / pl.col("value_cr")).alias("flow_dir"),
            (pl.col("value_cr") / pl.col("value_cr").sum().over("date")).alias("turnover_share"),
        )
        .select(_COLS)
        .sort("date", "sector")
    )


def build_sector_flow_daily(settings: Settings, start: date, end: date) -> int:
    """Compute and upsert flow_sector_daily for start..end (F&O and NIFTY 500 rows); returns rows written."""
    uni = io.read(settings, "panel_universe", start, end).filter(pl.col("u500") | pl.col("in_fno"))
    cols = ["sid", "date", "symbol", "series", "industry", "turnover_inr", "ret_oc"]
    panel = io.read(settings, "panel_daily", start, end, cols).filter(pl.col("series") == "EQ")
    panel = panel.join(uni.select("date", "sid"), on=["date", "sid"], how="inner").collect()
    if not panel.height:
        return 0
    spikes = io.read(settings, "flow_delivery_daily", start, end, ["date", "sid", "deliv_spike"]).collect()
    out = sector_flow_daily(panel.drop("series"), spikes)
    io.upsert(settings, "flow_sector_daily", out)
    return out.height
