"""Out-of-fold reporting for a trial (PLAN T7.1/T7.2 acceptance): IC, t-stat, IC by year, decile means."""

from __future__ import annotations

from typing import Any

import polars as pl

from stocksense.backtest.metrics import ic_stats


def oof_report(preds: pl.DataFrame, labels: pl.DataFrame, target: str) -> dict[str, Any]:
    """Score vs the RAW label `target` (rank IC does not care; the decile means are realised returns).

    preds: date, sid, score; labels: date, sid, <target>. Returns mean_ic, ic_std, ic_tstat, n_dates,
    by_year {year: mean IC}, frac_pos_years, deciles (mean realised label by score decile, 0 = lowest score),
    decile_spread (top minus bottom) and monotonicity.
    """
    j = preds.join(labels.select("date", "sid", target), on=["date", "sid"], how="inner")
    s = ic_stats(j, target)
    by_year = {int(k): float(v) for k, v in s["ic_by_year"].items()}
    dec = [float(x) for x in s["decile_mean_label"]]
    pos = sum(1 for v in by_year.values() if v > 0)
    return {
        "mean_ic": s["mean_ic"], "ic_std": s["ic_std"], "ic_tstat": s["ic_tstat"], "n_dates": s["n_dates"],
        "by_year": by_year, "frac_pos_years": (pos / len(by_year)) if by_year else float("nan"),
        "deciles": dec, "decile_spread": (dec[-1] - dec[0]) if dec else float("nan"),
        "monotonicity": s["monotonicity"],
    }  # fmt: skip


def group_gain_share(gain: dict[str, float], group_of: dict[str, str]) -> dict[str, float]:
    """Share of the total gain per feature group (zeros when nothing was split on)."""
    tot = sum(gain.values())
    out: dict[str, float] = {}
    for name, g in gain.items():
        grp = group_of.get(name, "other")
        out[grp] = out.get(grp, 0.0) + (g / tot if tot > 0 else 0.0)
    return out
