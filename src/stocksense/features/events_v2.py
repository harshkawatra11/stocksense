"""Event intensity and drift (G2-E7): Hawkes-style announcement intensities, category counts, result drift.

Announcement intensity. Each announcement of a company is assigned to the first session of the company on
or after the session whose 20:00 IST cutoff is at or after its timestamp (the decision rule of
`features.events`), so a feature for day t only counts announcements published by t's cutoff. With one
count `n_s` per (stock, calendar session s) the intensity at session t is the exponential-kernel sum
    ann_intensity_tau(t) = sum over s <= t of n_s * exp(-(t - s) / tau),    tau in {2, 10}
(age in calendar sessions, so a trading gap ages the events). It is computed with the first-order recursion
`lam_t = exp(-1/tau) lam_{t-1} + n_t` (scipy.signal.lfilter over a dense stock x session matrix, in chunks of
stocks), which is the kernel sum exactly. Like the other event features it is null before the source's first
announcement and 0 afterwards when a stock has none.

Category counts. Announcements of each of the eight most frequent categories of `docs/DATA.md` (matched
case-insensitively on the stripped name) counted over the stock's last 21 rows: `ann_cat_<name>_21`.

Result drift. `result_drift_proxy = result_abret * exp(-sessions_since_result / 20)`: the post-announcement
drift literature's decay of the two-day abnormal return around the last results (`features.events`).

The module registers its features when imported but is NOT in `features.registry.FEATURE_MODULES` (E9 adds
it once, for feature version 2), so the nightly build and the live system are unchanged.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.signal import lfilter

from stocksense.features.base import FeatureContext, FeatureSpec, feature, roll_sum
from stocksense.features.events import _known_date, _null_frame, _panel, _results
from stocksense.features.registry import REGISTRY

TAUS = (2.0, 10.0)
DRIFT_TAU = 20.0
CAT_WINDOW = 21
CHUNK = 400  # stocks per dense matrix

# the eight most frequent announcement categories (docs/DATA.md), lower case -> feature column
CATEGORIES: dict[str, str] = {
    "updates": "ann_cat_updates_21",
    "analysts/institutional investor meet/con. call updates": "ann_cat_analyst_21",
    "outcome of board meeting": "ann_cat_outcome_21",
    "financial result updates": "ann_cat_result_21",
    "press release": "ann_cat_press_21",
    "shareholders meeting": "ann_cat_shareholder_21",
    "copy of newspaper publication": "ann_cat_newspaper_21",
    "trading window": "ann_cat_trading_21",
}
INTENSITY_OUTPUTS = ("ann_intensity_2", "ann_intensity_10")
OUTPUTS = (*INTENSITY_OUTPUTS, *CATEGORIES.values(), "result_drift_proxy")


# ---- the kernel ----------------------------------------------------------------------------------------


def exp_intensity(counts: np.ndarray, tau: float) -> np.ndarray:
    """Exponential-kernel intensity per row of `counts` [stocks, sessions]: sum_s n_s e^{-(t-s)/tau}."""
    a = float(np.exp(-1.0 / tau))
    return lfilter([1.0], [1.0, -a], np.asarray(counts, dtype=float), axis=-1)


def row_intensity(events: pl.DataFrame, rows: pl.DataFrame, tau: float, n_sessions: int) -> pl.Series:
    """Intensity at each (sid, sx) of `rows` from `events` (sid, sx, n); 0.0 for a stock without events."""
    out = np.zeros(rows.height)
    ev_sids = events["sid"].unique().sort().to_list()
    if not ev_sids or not rows.height:
        return pl.Series("intensity", out)
    rsid = rows["sid"].to_numpy()
    rsx = rows["sx"].to_numpy().astype(int)
    esid = events["sid"].to_numpy()
    esx = events["sx"].to_numpy().astype(int)
    en = events["n"].cast(pl.Float64).to_numpy()
    for lo in range(0, len(ev_sids), CHUNK):
        chunk = np.array(ev_sids[lo : lo + CHUNK])
        pos = {int(s): k for k, s in enumerate(chunk)}
        mat = np.zeros((len(chunk), n_sessions))
        sel = np.isin(esid, chunk)
        idx = np.array([pos[int(s)] for s in esid[sel]], dtype=int)
        np.add.at(mat, (idx, esx[sel]), en[sel])
        lam = exp_intensity(mat, tau)
        rsel = np.flatnonzero(np.isin(rsid, chunk))
        ridx = np.array([pos[int(s)] for s in rsid[rsel]], dtype=int)
        out[rsel] = lam[ridx, rsx[rsel]]
    return pl.Series("intensity", out)


# ---- categories ----------------------------------------------------------------------------------------


def category_name(category: str | None) -> str | None:
    """The `ann_cat_*_21` column an announcement category counts toward (None: not one of the eight)."""
    if category is None:
        return None
    return CATEGORIES.get(category.strip().lower())


def category_counts(events: pl.DataFrame, panel: pl.DataFrame, window: int = CAT_WINDOW) -> pl.DataFrame:
    """date, sid and one column per category: announcements over the stock's last `window` rows. `events` has
    sid, date (the row the announcement is assigned to) and col (the category column); `panel` has date, sid,
    sx. Null where fewer than `window` rows exist; 0 for a category without events."""
    base = panel.select("date", "sid", "sx").sort("sid", "date")
    cols = []
    for col in CATEGORIES.values():
        cnt = (
            events.filter(pl.col("col") == col)
            .group_by("sid", "date")
            .agg(pl.len().cast(pl.Float64).alias("n"))
        )
        r = (
            base.join(cnt, on=["sid", "date"], how="left")
            .with_columns(pl.col("n").fill_null(0.0))
            .select(roll_sum("n", window).alias(col))
        )
        cols.append(r)
    return pl.concat([base.select("date", "sid"), *cols], how="horizontal")


# ---- result drift --------------------------------------------------------------------------------------


def result_drift_expr() -> pl.Expr:
    """result_abret * exp(-days_since_result / 20) (null when there is no abnormal return)."""
    return (pl.col("result_abret") * (-pl.col("days_since_result") / DRIFT_TAU).exp()).alias(
        "result_drift_proxy"
    )


# ---- features ------------------------------------------------------------------------------------------


def _announcement_rows(ctx: FeatureContext) -> tuple[pl.DataFrame, pl.Timestamp | None] | None:
    """(sid, date, sx, col, adate) per announcement placed on the company's first row on/after its known
    date, and the first known date of the source; None when there are no announcements."""
    cache = ctx.__dict__.setdefault("_events_v2", {})
    if "ann" in cache:
        return cache["ann"]
    panel = _panel(ctx)
    a = ctx.raw("announcements", ["symbol", "ts", "category"]).drop_nulls(["symbol", "ts"]).collect()
    if a.is_empty():
        cache["ann"] = None
        return None
    a = a.with_columns(_known_date(pl.col("ts")).alias("adate"))
    start = a["adate"].min()
    keys = panel.select("symbol", "sid", "date", "sx").sort("date")
    j = (
        a.sort("adate")
        .join_asof(
            keys, left_on="adate", right_on="date", by="symbol", strategy="forward", check_sortedness=False
        )
        .drop_nulls("date")
    )
    cat = pl.col("category").str.strip_chars().str.to_lowercase().replace_strict(CATEGORIES, default=None)
    j = j.select("sid", "date", "sx", cat.alias("col"))
    cache["ann"] = (j, start)
    return cache["ann"]


def _intensity_feature(ctx: FeatureContext) -> pl.LazyFrame:
    panel = _panel(ctx)
    got = _announcement_rows(ctx)
    if got is None:
        return _null_frame(panel, *INTENSITY_OUTPUTS).lazy()
    ev, start = got
    n_ev = ev.group_by("sid", "sx").agg(pl.len().cast(pl.Float64).alias("n"))
    rows = panel.select("date", "sid", "sx")
    n_sessions = int(rows["sx"].max()) + 1
    out = rows.select("date", "sid")
    cols = []
    for tau, name in zip(TAUS, INTENSITY_OUTPUTS, strict=True):
        lam = row_intensity(n_ev, rows, tau, n_sessions)
        cols.append(pl.when(rows["date"] >= start).then(lam).alias(name))
    return out.with_columns(*cols).lazy()


def _categories_feature(ctx: FeatureContext) -> pl.LazyFrame:
    panel = _panel(ctx)
    got = _announcement_rows(ctx)
    names = tuple(CATEGORIES.values())
    if got is None:
        return _null_frame(panel, *names).lazy()
    ev, start = got
    out = category_counts(ev.drop_nulls("col"), panel)
    gate = pl.col("date") >= start
    return out.with_columns(*[pl.when(gate).then(pl.col(n)).alias(n) for n in names]).lazy()


def _drift_feature(ctx: FeatureContext) -> pl.LazyFrame:
    return _results(ctx).select("date", "sid", result_drift_expr()).lazy()


feature(
    "ann_intensity",
    group="events",
    kind="xs",
    window=100,
    version=2,
    label="How intense the company's recent announcement flow is (2- and 10-session memory).",
    outputs=INTENSITY_OUTPUTS,
)(_intensity_feature)
feature(
    "ann_categories",
    group="events",
    kind="xs",
    window=CAT_WINDOW,
    version=2,
    label="Announcements of the eight most common kinds over the last month.",
    outputs=tuple(CATEGORIES.values()),
)(_categories_feature)
feature(
    "result_drift",
    group="events",
    kind="xs",
    window=1,
    version=2,
    label="The beat or miss at the last results, fading with the time since they came out.",
    outputs=("result_drift_proxy",),
)(_drift_feature)

SPECS: tuple[FeatureSpec, ...] = (
    REGISTRY["ann_intensity"],
    REGISTRY["ann_categories"],
    REGISTRY["result_drift"],
)
