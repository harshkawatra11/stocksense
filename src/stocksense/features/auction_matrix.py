"""The stored auction features (T4.4) as extra model columns keyed by DECISION date (PLAN T7.2, y_oc trial).

`features/auction` is keyed by (trade_date, sid) with the decision date t = the session before; a model row
is (t, sid), so the frame is re-keyed to `date = decision_date`. Columns get the prefix `auc_`. Historical
values use the official open (`gap_source = 'official_open'`); live they come from the 09:07 indicative price.
"""

from __future__ import annotations

import polars as pl

from stocksense.core.config import Settings
from stocksense.features.auction import load_auction_features

AUCTION_PREFIX = "auc_"
_MKT_COLUMNS = ("mkt_gap", "spx_overnight", "ndx_overnight", "usdinr_overnight", "brent_overnight")
_VALUE_COLUMNS = (
    "gap", "gap_z", "gap_rel_mkt", "gap_rel_ind", "mkt_gap", "gap_rank", "preopen_imbalance",
    "spx_overnight", "ndx_overnight", "usdinr_overnight", "brent_overnight",
)  # fmt: skip


def auction_columns(
    settings: Settings, purpose: str = "research"
) -> tuple[pl.LazyFrame, list[str], list[str]]:
    """(frame[date, sid, auc_*], names, kinds); columns that are entirely null are dropped."""
    lf = load_auction_features(settings, purpose)
    df = lf.select("decision_date", "sid", *_VALUE_COLUMNS).collect()
    keep = [c for c in _VALUE_COLUMNS if df[c].null_count() < df.height]
    out = df.select(
        pl.col("decision_date").alias("date"), "sid", *[pl.col(c).alias(AUCTION_PREFIX + c) for c in keep]
    )
    names = [AUCTION_PREFIX + c for c in keep]
    kinds = ["mkt" if c in _MKT_COLUMNS else "xs" for c in keep]
    return out.lazy(), names, kinds
