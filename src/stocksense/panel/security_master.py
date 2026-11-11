"""Security master: one permanent `sid` per company across symbol renames (PLAN T3.1)."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from stocksense.core.config import Settings
from stocksense.core.log import get_logger
from stocksense.core.paths import Paths
from stocksense.data import io

log = get_logger(__name__)

SERIES = ("EQ", "BE", "BZ")
_FAR_FUTURE = date(2099, 12, 31)
_AUTO_PRICE_TOL = 0.001
_AUTO_MAX_SESSIONS = 5
_AUTO_END_MARGIN_DAYS = 30


# ---- nodes ------------------------------------------------------------------------------------


def session_frame(settings: Settings) -> pl.DataFrame:
    """(date, idx): every distinct bhavcopy_eq session date, 0-based in order."""
    d = io.read(settings, "bhavcopy_eq", columns=["date"]).select(pl.col("date").unique().sort()).collect()
    return d.with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int64))


def _nodes(settings: Settings, sessions: pl.DataFrame) -> pl.DataFrame:
    """One row per symbol seen in EQ/BE/BZ with its first/last date and boundary prices."""
    prio = pl.when(pl.col("series") == "EQ").then(0).when(pl.col("series") == "BE").then(1).otherwise(2)
    rows = (
        io.read(
            settings,
            "bhavcopy_eq",
            columns=["symbol", "series", "date", "prev_close", "close", "isin", "name"],
        )
        .filter(pl.col("series").is_in(SERIES))
        .with_columns(prio.alias("_prio"))
    )
    nodes = (
        rows.group_by("symbol")
        .agg(
            pl.col("date").min().alias("first_date"),
            pl.col("date").max().alias("last_date"),
            pl.col("prev_close").sort_by(["date", "_prio"]).first().alias("first_prev_close"),
            pl.col("close").sort_by(["date", "_prio"], descending=[False, True]).last().alias("last_close"),
            pl.col("isin").sort_by("date").drop_nulls().last().alias("isin_last"),
            pl.col("name").sort_by("date").drop_nulls().last().alias("name_last"),
        )
        .collect()
    )
    idx = sessions.rename({"date": "first_date", "idx": "first_idx"})
    nodes = nodes.join(idx, on="first_date", how="left")
    idx = sessions.rename({"date": "last_date", "idx": "last_idx"})
    return nodes.join(idx, on="last_date", how="left").sort("symbol")


# ---- edges ------------------------------------------------------------------------------------


def _official_edges(settings: Settings, nodes: pl.DataFrame) -> pl.DataFrame:
    """Edge A: symbol_changes rows whose dates fit the two symbols' observed life spans."""
    sc = io.read(settings, "symbol_changes").collect()
    if sc.is_empty():
        return pl.DataFrame(schema={"src": pl.Utf8, "dst": pl.Utf8, "gap": pl.Int64})
    old = nodes.select(pl.col("symbol").alias("old_symbol"), pl.col("last_date").alias("old_last"))
    new = nodes.select(pl.col("symbol").alias("new_symbol"), pl.col("first_date").alias("new_first"))
    e = (
        sc.select("old_symbol", "new_symbol", "change_date")
        .unique()
        .join(old, on="old_symbol")
        .join(new, on="new_symbol")
        .filter(pl.col("old_symbol") != pl.col("new_symbol"))
        .filter(
            (pl.col("old_last") >= pl.col("change_date") - timedelta(days=15))
            & (pl.col("old_last") <= pl.col("change_date") + timedelta(days=5))
            & (pl.col("new_first") >= pl.col("change_date") - timedelta(days=5))
            & (pl.col("new_first") <= pl.col("change_date") + timedelta(days=15))
        )
    )
    return e.select(
        pl.col("old_symbol").alias("src"),
        pl.col("new_symbol").alias("dst"),
        (pl.col("new_first") - pl.col("old_last")).dt.total_days().abs().alias("gap"),
    )


def _auto_candidates(
    settings: Settings, nodes: pl.DataFrame, data_end: date, official_src: set[str]
) -> pl.DataFrame:
    """Edge B candidates: A stops, B starts 1-5 sessions later at A's last close."""
    a = nodes.filter(
        (pl.col("last_date") < data_end - timedelta(days=_AUTO_END_MARGIN_DAYS))
        & ~pl.col("symbol").is_in(list(official_src))
    ).select(
        pl.col("symbol").alias("src"),
        pl.col("last_date").alias("d1"),
        pl.col("last_close").alias("a_close"),
        pl.col("last_idx"),
    )
    b = nodes.select(
        pl.col("symbol").alias("dst"),
        pl.col("first_date").alias("d2"),
        pl.col("first_prev_close").alias("b_prev"),
        pl.col("first_idx"),
    )
    listed = io.read(settings, "listed_equities", columns=["symbol", "listing_date"]).collect()
    a = (
        a.with_columns(pl.int_ranges(1, _AUTO_MAX_SESSIONS + 1).alias("_k"))
        .explode("_k")
        .with_columns((pl.col("last_idx") + pl.col("_k")).alias("first_idx"))
    )
    cand = (
        a.join(b, on="first_idx")
        .filter(pl.col("src") != pl.col("dst"))
        .filter((pl.col("b_prev") / pl.col("a_close") - 1.0).abs() < _AUTO_PRICE_TOL)
    )
    if cand.is_empty():
        return cand.select("src", "dst", "d1", "d2", "a_close", "b_prev")
    fresh = (
        cand.select("dst", "d2")
        .unique()
        .join(listed.rename({"symbol": "dst"}), on="dst")
        .filter((pl.col("listing_date") - pl.col("d2")).dt.total_days().abs() <= 3)
        .select("dst")
        .unique()
    )
    cand = cand.join(fresh, on="dst", how="anti")
    cand = cand.with_columns(pl.len().over("src").alias("_na"), pl.len().over("dst").alias("_nb")).filter(
        (pl.col("_na") == 1) & (pl.col("_nb") == 1)
    )
    return cand.select("src", "dst", "d1", "d2", "a_close", "b_prev")


class _UnionFind:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[rb] = ra
        return True


# ---- build ------------------------------------------------------------------------------------


def build_security_master(settings: Settings) -> pl.DataFrame:
    """Build and write panel/security_master.parquet (+ rename_autodetect.csv, sid_merges.csv).

    Returns one row per (sid, symbol); valid_from/valid_to are dates (no unit conversion involved).
    """
    paths = Paths(settings)
    sessions = session_frame(settings)
    data_end: date = sessions["date"].max()
    nodes = _nodes(settings, sessions)
    symbols = nodes["symbol"].to_list()

    # --- edges: official first (closest date first), then auto-detected ---------------------
    official = _official_edges(settings, nodes).sort(["gap", "src", "dst"])
    uf = _UnionFind()
    has_out: set[str] = set()
    has_in: set[str] = set()
    edges: list[tuple[str, str]] = []

    def accept(src: str, dst: str) -> bool:
        if src in has_out or dst in has_in or not uf.union(src, dst):
            return False
        has_out.add(src)
        has_in.add(dst)
        edges.append((src, dst))
        return True

    n_official = sum(accept(r["src"], r["dst"]) for r in official.iter_rows(named=True))
    official_src = set(has_out)
    cand = _auto_candidates(settings, nodes, data_end, official_src)
    auto_rows = []
    for r in cand.sort(["d1", "src"]).iter_rows(named=True):
        if accept(r["src"], r["dst"]):
            auto_rows.append(r)
    auto = pl.DataFrame(
        auto_rows,
        schema={
            "src": pl.Utf8,
            "dst": pl.Utf8,
            "d1": pl.Date,
            "d2": pl.Date,
            "a_close": pl.Float64,
            "b_prev": pl.Float64,
        },
    ).rename(
        {
            "src": "old_symbol",
            "dst": "new_symbol",
            "d1": "last_date",
            "d2": "first_date",
            "a_close": "last_close",
            "b_prev": "first_prev_close",
        }
    )
    paths.panel.mkdir(parents=True, exist_ok=True)
    auto.write_csv(paths.panel / "rename_autodetect.csv")

    # --- sids ---------------------------------------------------------------------------------
    existing: dict[str, int] = {}
    if io.list_partitions(settings, "security_master"):
        old = io.read(settings, "security_master", columns=["sid", "symbol"]).collect()
        existing = dict(zip(old["symbol"].to_list(), old["sid"].to_list(), strict=True))
    max_sid = max(existing.values(), default=0)

    first_date = dict(zip(nodes["symbol"], nodes["first_date"], strict=True))
    comps: dict[str, list[str]] = {}
    for s in symbols:
        comps.setdefault(uf.find(s), []).append(s)
    for members in comps.values():
        members.sort(key=lambda s: (first_date[s], s))

    sid_of: dict[str, int] = {}
    merge_rows: list[dict] = []
    new_chains: list[list[str]] = []
    for members in comps.values():
        have = sorted({existing[s] for s in members if s in existing})
        if not have:
            new_chains.append(members)
            continue
        keep = have[0]
        if len(have) > 1:
            log.warning(
                "sid_merge", symbols=",".join(members), kept=keep, dropped=",".join(map(str, have[1:]))
            )
            for s in members:
                if s in existing and existing[s] != keep:
                    merge_rows.append(
                        {"symbol": s, "old_sid": existing[s], "kept_sid": keep, "chain": ">".join(members)}
                    )
        for s in members:
            sid_of[s] = keep
    new_chains.sort(key=lambda m: (first_date[m[0]], m[0]))
    for i, members in enumerate(new_chains, start=1):
        for s in members:
            sid_of[s] = max_sid + i
    if merge_rows:
        merges = pl.DataFrame(merge_rows)
        path = paths.panel / "sid_merges.csv"
        if path.is_file():
            merges = pl.concat(
                [pl.read_csv(path, schema_overrides={"old_sid": pl.Int64, "kept_sid": pl.Int64}), merges]
            )
        merges.write_csv(path)

    out = nodes.select(
        pl.col("symbol").replace_strict(sid_of, return_dtype=pl.Int32).alias("sid"),
        "symbol",
        pl.col("first_date").alias("valid_from"),
        pl.when(pl.col("symbol").is_in(list(has_out)))
        .then(pl.col("last_date"))
        .otherwise(pl.lit(_FAR_FUTURE))
        .alias("valid_to"),
        "isin_last",
        "name_last",
        "first_date",
        "last_date",
    ).sort("sid", "valid_from")
    io.write_partition(settings, "security_master", "all", out)
    log.info(
        "security_master_built",
        sids=out["sid"].n_unique(),
        symbols=out.height,
        official=n_official,
        auto=len(auto_rows),
    )
    return out


def symbol_sid_map(settings: Settings) -> pl.DataFrame:
    """(symbol, valid_from, valid_to, sid): the symbol -> sid lookup frame (dates are IST sessions)."""
    sm = io.read(settings, "security_master").collect()
    return sm.select("symbol", "valid_from", "valid_to", "sid")
