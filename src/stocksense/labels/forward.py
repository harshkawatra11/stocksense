"""Forward labels for every (decision date t, sid) in u500 (PLAN T4.5, D.3 timing).

d1 is the session after t; the labels measure what a buyer at the close of d1 ("c2o", the delivery
book: buy near the close, sell at an open) or at the open of d1 ("oc", "o2o") would have earned. All
returns are fractions; split/bonus adjustment comes from the adjusted prices and dividends are added back
through `div_ret`.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from stocksense.core.config import Settings
from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.core.paths import Paths
from stocksense.data import io
from stocksense.features.base import FeatureContext
from stocksense.validation.lockbox import guard, research_end

log = get_logger(__name__)

LABEL_VERSION = 1
HORIZONS = (5, 10, 20)
DELIST_HAIRCUT = 0.20
DELIST_MIN_SESSIONS = 20  # last row more than this many sessions before the end of data -> delisted
BANDS = (0.02, 0.05, 0.10, 0.20)
BAND_TOL = 0.0025
MAD_SCALE = 1.4826
Z_CLIP = 5.0

Y_COLS = ["y_oc", "y_c2o_5", "y_c2o_10", "y_c2o_20", "y_c2c_1", "y_o2o_5"]
# (labels that end on `end` column)
_END_GROUPS = [
    (["y_oc"], "end_oc"),
    (["y_c2c_1"], "end_c2c_1"),
    (["y_c2o_5"], "end_5"),
    (["y_c2o_10"], "end_10"),
    (["y_c2o_20"], "end_20"),
    (["y_o2o_5"], "end_o2o_5"),
]
_PANEL_COLS = [
    "sid",
    "date",
    "sx",
    "open",
    "high",
    "close",
    "adj_open",
    "adj_close",
    "div_ret",
    "ret_co",
    "flag_unexplained_gap",
    "flag_locked_up",
]


def robust_z_and_rank(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Add z_<c> (robust z within the date, clipped to +-5, null if MAD = 0) and r_<c> (average rank
    scaled to [-0.5, 0.5]) as Float32 for each column in `cols`; nulls are ignored."""
    exprs: list[pl.Expr] = []
    for c in cols:
        x = pl.col(c)
        med = x.median().over("date")
        mad = (x - med).abs().median().over("date")
        z = pl.when(mad > 0).then(((x - med) / (MAD_SCALE * mad)).clip(-Z_CLIP, Z_CLIP))
        n = x.count().over("date")
        r = pl.when(n > 1).then((x.rank("average").over("date") - 1.0) / (n - 1).cast(pl.Float64) - 0.5)
        exprs += [z.cast(pl.Float32).alias(f"z_{c}"), r.cast(pl.Float32).alias(f"r_{c}")]
    return df.with_columns(exprs)


def compute_labels(panel: pl.DataFrame, decisions: pl.DataFrame, sessions: pl.DataFrame) -> pl.DataFrame:
    """Labels for the (date, sid) rows in `decisions`.

    `panel` needs the columns in _PANEL_COLS (sx = calendar session index); `sessions` is (date, sx) for
    the whole calendar (its last session is the end of data). The result has one row per decision row.
    """
    data_end = int(sessions["sx"].max())  # type: ignore[arg-type]
    sx_date = dict(zip(sessions["sx"].to_list(), sessions["date"].to_list(), strict=True))

    def date_of_expr(e: pl.Expr) -> pl.Expr:
        return e.replace_strict(sx_date, default=None, return_dtype=pl.Date)

    def date_of(col: str) -> pl.Expr:
        return date_of_expr(pl.col(col))

    p = (
        panel.select(_PANEL_COLS)
        .sort("sid", "sx")
        .with_columns(
            pl.col("flag_unexplained_gap").fill_null(False).cast(pl.Int32).cum_sum().over("sid").alias("_cg"),
            pl.col("div_ret").fill_null(0.0).cum_sum().over("sid").alias("_cd"),
        )
    )
    d = (
        decisions.select("date", "sid")
        .with_row_index("__i")
        .join(sessions.rename({"sx": "tx"}), on="date", how="left")
        .with_columns((pl.col("tx") + 1).alias("k1"))
    )
    d = d.with_columns((pl.col("k1") + 1).alias("k2"), (pl.col("k1") + 5).alias("k5"))

    d = d.join(
        p.select(
            "sid",
            pl.col("sx").alias("k1"),
            pl.col("open").alias("o1"),
            pl.col("high").alias("h1"),
            pl.col("close").alias("c1"),
            pl.col("adj_open").alias("ao1"),
            pl.col("adj_close").alias("ac1"),
            pl.col("ret_co").alias("rc1"),
            pl.col("flag_locked_up").alias("lu1"),
            pl.col("_cg").alias("g1"),
            pl.col("_cd").alias("v1"),
            pl.lit(True).alias("has1"),
        ),
        on=["sid", "k1"],
        how="left",
    )
    d = d.join(
        p.select(
            "sid",
            pl.col("sx").alias("k2"),
            pl.col("adj_close").alias("ac2"),
            pl.col("div_ret").alias("dr2"),
            pl.col("_cg").alias("g2"),
            pl.lit(True).alias("has2"),
        ),
        on=["sid", "k2"],
        how="left",
    )
    d = d.join(
        p.select(
            "sid",
            pl.col("sx").alias("k5"),
            pl.col("adj_open").alias("ao5"),
            pl.col("_cg").alias("g5"),
            pl.col("_cd").alias("v5"),
            pl.lit(True).alias("has5"),
        ),
        on=["sid", "k5"],
        how="left",
    )
    last = p.group_by("sid", maintain_order=True).agg(
        pl.col("sx").last().alias("last_sx"),
        pl.col("adj_close").last().alias("last_ac"),
        pl.col("_cg").last().alias("last_cg"),
    )
    d = d.join(last, on="sid", how="left")

    # first row of the sid at or after d1 + h (a real holder is stuck until it trades again)
    right = p.select(
        "sid",
        pl.col("sx").alias("xs"),
        pl.col("adj_open").alias("xo"),
        pl.col("_cg").alias("xg"),
        pl.col("_cd").alias("xv"),
    ).sort("xs")
    for h in HORIZONS:
        left = d.select("__i", "sid", (pl.col("k1") + h).alias("tgt")).drop_nulls("tgt").sort("tgt")
        j = left.join_asof(
            right, left_on="tgt", right_on="xs", by="sid", strategy="forward", check_sortedness=False
        )
        j = j.select(
            "__i",
            pl.col("xs").alias(f"xs{h}"),
            pl.col("xo").alias(f"xo{h}"),
            pl.col("xg").alias(f"xg{h}"),
            pl.col("xv").alias(f"xv{h}"),
        )
        d = d.join(j, on="__i", how="left")

    has1 = pl.col("has1").fill_null(False)
    ok_open = has1 & (pl.col("o1") > 0) & (pl.col("c1") > 0)
    out: list[pl.Expr] = [
        pl.when(ok_open).then(pl.col("c1") / pl.col("o1") - 1.0).alias("y_oc"),
        pl.when(ok_open).then(date_of("k1")).alias("end_oc"),
    ]
    ok_c2c = (
        has1 & pl.col("has2").fill_null(False) & (pl.col("ac1") > 0) & ((pl.col("g2") - pl.col("g1")) == 0)
    )
    out += [
        pl.when(ok_c2c)
        .then(pl.col("ac2") / pl.col("ac1") - 1.0 + pl.col("dr2").fill_null(0.0))
        .alias("y_c2c_1"),
        pl.when(ok_c2c).then(date_of("k2")).alias("end_c2c_1"),
    ]
    ok_o2o = (
        has1 & pl.col("has5").fill_null(False) & (pl.col("ao1") > 0) & ((pl.col("g5") - pl.col("g1")) == 0)
    )
    out += [
        pl.when(ok_o2o)
        .then(pl.col("ao5") / pl.col("ao1") - 1.0 + (pl.col("v5") - pl.col("v1")))
        .alias("y_o2o_5"),
        pl.when(ok_o2o).then(date_of("k5")).alias("end_o2o_5"),
    ]
    delisted_any: list[pl.Expr] = []
    for h in HORIZONS:
        xs = pl.col(f"xs{h}")
        stuck_ok = has1 & (pl.col("ac1") > 0) & xs.is_not_null() & ((pl.col(f"xg{h}") - pl.col("g1")) == 0)
        gone = (
            has1
            & (pl.col("ac1") > 0)
            & xs.is_null()
            & ((data_end - pl.col("last_sx")) > DELIST_MIN_SESSIONS)
            & ((pl.col("last_cg") - pl.col("g1")) == 0)
        )
        y_norm = pl.col(f"xo{h}") / pl.col("ac1") - 1.0 + (pl.col(f"xv{h}") - pl.col("v1"))
        y_gone = pl.col("last_ac") / pl.col("ac1") - 1.0 - DELIST_HAIRCUT
        out += [
            pl.when(stuck_ok).then(y_norm).when(gone).then(y_gone).alias(f"y_c2o_{h}"),
            pl.when(stuck_ok)
            .then(date_of(f"xs{h}"))
            .when(gone)
            .then(date_of_expr(pl.col("last_sx") + 1))
            .alias(f"end_{h}"),
        ]
        delisted_any.append(gone)
    blocked_open = (
        (pl.col("o1") == pl.col("h1"))
        & (pl.min_horizontal(*[(pl.col("rc1") - b).abs() for b in BANDS]) <= BAND_TOL)
    ).fill_null(False)
    out += [
        blocked_open.alias("entry_blocked_open"),
        pl.col("lu1").fill_null(False).alias("entry_blocked_close"),
        pl.any_horizontal(*delisted_any).alias("label_delisted"),
    ]
    res = d.with_columns(out).sort("__i")
    cols = [
        "date",
        "sid",
        *Y_COLS,
        "end_oc",
        "end_5",
        "end_10",
        "end_20",
        "entry_blocked_open",
        "entry_blocked_close",
        "label_delisted",
        "end_c2c_1",
        "end_o2o_5",
    ]
    return robust_z_and_rank(res.select(cols), Y_COLS)


# ---- build and load ----------------------------------------------------------------------------


def _root(settings: Settings) -> Path:
    return Paths(settings).labels


def current_version(settings: Settings) -> int | None:
    """The version in labels/CURRENT, or None if no labels were built."""
    p = _root(settings) / "CURRENT"
    return int(p.read_text(encoding="utf-8").strip()) if p.is_file() else None


def build_labels(settings: Settings) -> int:
    """Compute labels for every u500 (date, sid) and write labels/v<N>/part-YYYY.parquet; returns N."""
    ctx = FeatureContext(settings)
    decisions = ctx.universe().filter(pl.col("u500")).select("date", "sid").collect()
    if decisions.is_empty():
        raise DataNotAvailable("panel/universe has no u500 rows; run build-panel first")
    panel = ctx.panel().select(_PANEL_COLS).collect()
    sessions = ctx.sessions().collect()
    lab = compute_labels(panel, decisions, sessions)
    del panel
    sub = f"v{LABEL_VERSION}"
    vdir = _root(settings) / sub
    if vdir.is_dir():
        for old in vdir.glob("part-*.parquet"):
            old.unlink()
    for (year,), part in (
        lab.with_columns(pl.col("date").dt.year().alias("_y")).partition_by("_y", as_dict=True).items()
    ):
        io.write_partition(settings, "labels", str(year), part.drop("_y").sort("date", "sid"), sub=sub)
    (_root(settings) / "CURRENT").write_text(f"{LABEL_VERSION}\n", encoding="utf-8", newline="\n")
    log.info("labels_built", version=LABEL_VERSION, rows=lab.height)
    return LABEL_VERSION


def load_labels(
    settings: Settings,
    purpose: str,
    start=None,
    end=None,
    columns: list[str] | None = None,
) -> pl.LazyFrame:
    """The current labels as a lazy frame. 'research' stops at lockbox_start - 1 day AND nulls every
    label whose end date falls after that day (so no outcome from the lockbox period is visible);
    'production' returns everything; 'lockbox' only inside open_lockbox(...)."""
    v = current_version(settings)
    if v is None:
        raise DataNotAvailable("no labels built yet (labels/CURRENT is missing)")
    extra = [c for _, e in _END_GROUPS for c in [e]] + ["label_delisted"] if purpose == "research" else []
    cols = None if columns is None else list(dict.fromkeys(["date", "sid", *columns, *extra]))
    lf = io.read(settings, "labels", start=start, end=end, columns=cols, sub=f"v{v}")
    lf = guard(lf, "date", purpose, settings)
    if purpose != "research":
        return lf
    rd = research_end(settings)
    have = set(lf.collect_schema().names())
    exprs: list[pl.Expr] = []
    for ys, e in _END_GROUPS:
        if e not in have:
            continue
        late = pl.col(e) > rd
        targets = [t for y in ys for t in (y, f"z_{y}", f"r_{y}") if t in have]
        exprs += [pl.when(late).then(None).otherwise(pl.col(t)).alias(t) for t in targets]
        exprs.append(pl.when(late).then(None).otherwise(pl.col(e)).alias(e))
    if "label_delisted" in have:
        ends = [pl.col(e) > rd for e in ("end_5", "end_10", "end_20") if e in have]
        if ends:
            exprs.append(
                pl.when(pl.any_horizontal(*ends))
                .then(False)
                .otherwise(pl.col("label_delisted"))
                .alias("label_delisted")
            )
    lf = lf.with_columns(exprs)
    return lf if columns is None else lf.select(["date", "sid", *columns])
