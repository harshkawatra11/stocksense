"""Build and load the feature store: rank-scaled features per (date, sid) in u500 (PLAN T4.1)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

from stocksense.core.clock import Clock, SystemClock
from stocksense.core.config import Settings
from stocksense.core.errors import DataNotAvailable, DataValidationError
from stocksense.core.log import get_logger
from stocksense.core.paths import Paths
from stocksense.data import io
from stocksense.features.base import FeatureContext, FeatureSpec
from stocksense.features.registry import load_all
from stocksense.validation.lockbox import guard
from stocksense.validation.trials import git_sha

log = get_logger(__name__)

MIN_VALID = 50  # fewer valid values on a date than this -> the ranked column is null for that date
WINSOR = (0.01, 0.99)


# ---- versions ----------------------------------------------------------------------------------


def _root(settings: Settings) -> Path:
    return Paths(settings).features


def current_version(settings: Settings) -> int | None:
    """The version number in features/CURRENT, or None if no features were built yet."""
    p = _root(settings) / "CURRENT"
    if not p.is_file():
        return None
    return int(p.read_text(encoding="utf-8").strip())


def _manifest_path(settings: Settings, version: int) -> Path:
    return _root(settings) / f"v{version}" / "manifest.json"


def _pairs(reg: dict[str, FeatureSpec]) -> list[tuple[str, int]]:
    return sorted((o, s.version) for s in reg.values() for o in s.outputs)


def _choose_version(settings: Settings, reg: dict[str, FeatureSpec], version: int | None) -> int:
    if version is not None:
        return version
    cur = current_version(settings)
    if cur is None:
        return 1
    mp = _manifest_path(settings, cur)
    if not mp.is_file():
        return cur
    man = json.loads(mp.read_text(encoding="utf-8"))
    have = sorted((f["name"], f["version"]) for f in man["features"])
    return cur if have == _pairs(reg) else cur + 1


# ---- cross-sectional transform -----------------------------------------------------------------


def cross_section_transform(df: pl.DataFrame, cols: list[str], min_valid: int = MIN_VALID) -> pl.DataFrame:
    """Per date: winsorise `cols` at the 1st/99th percentile (kept as `<c>__raw`, Float32) and replace
    each by (rank_average - 1)/(n_valid - 1) - 0.5 (Float32); null stays null; all null on dates with
    fewer than `min_valid` valid values. Non-finite values are treated as null."""
    tmp = {c: f"__w_{c}" for c in cols}
    clean = {
        c: pl.when(pl.col(c).cast(pl.Float64).is_finite()).then(pl.col(c).cast(pl.Float64)) for c in cols
    }
    out = df.with_columns(
        *[
            clean[c]
            .clip(
                clean[c].quantile(WINSOR[0], "linear").over("date"),
                clean[c].quantile(WINSOR[1], "linear").over("date"),
            )
            .alias(tmp[c])
            for c in cols
        ]
    )
    exprs: list[pl.Expr] = []
    for c in cols:
        w = pl.col(tmp[c])
        n = w.count().over("date")
        ranked = (w.rank("average").over("date") - 1.0) / (n - 1).cast(pl.Float64) - 0.5
        exprs.append(
            pl.when((n >= min_valid) & (n > 1)).then(ranked).otherwise(None).cast(pl.Float32).alias(c)
        )
        exprs.append(w.cast(pl.Float32).alias(f"{c}__raw"))
    return out.with_columns(*exprs).drop(list(tmp.values()))


# ---- build -------------------------------------------------------------------------------------


def _feature_frame(spec: FeatureSpec, ctx: FeatureContext, base: pl.DataFrame) -> pl.DataFrame:
    """The spec's outputs aligned to `base` (date, sid): xs by (date, sid), mkt broadcast by date."""
    keys = ["date", "sid"] if spec.kind == "xs" else ["date"]
    res = spec.fn(ctx).collect()
    missing = [c for c in (*keys, *spec.outputs) if c not in res.columns]
    if missing:
        raise DataValidationError(f"feature {spec.name}: function did not return column(s) {missing}")
    res = res.select(*keys, *spec.outputs)
    if res.select(keys).is_duplicated().any():
        raise DataValidationError(f"feature {spec.name}: duplicate {keys} rows")
    return base.select(keys).unique(maintain_order=True).join(res, on=keys, how="left")


def _assemble(
    reg: dict[str, FeatureSpec], ctx: FeatureContext, base: pl.DataFrame, start: date | None, min_valid: int
) -> pl.DataFrame:
    frame = base
    xs_cols: list[str] = []
    for name in sorted(reg):
        spec = reg[name]
        part = _feature_frame(spec, ctx, base)
        keys = ["date", "sid"] if spec.kind == "xs" else ["date"]
        frame = frame.join(part, on=keys, how="left")
        if spec.kind == "xs":
            xs_cols.extend(spec.outputs)
        else:
            frame = frame.with_columns([pl.col(o).cast(pl.Float32) for o in spec.outputs])
    if xs_cols:
        frame = cross_section_transform(frame, xs_cols, min_valid)
    if start is not None:
        frame = frame.filter(pl.col("date") >= start)
    order = ["date", "sid"]
    for name in sorted(reg):
        spec = reg[name]
        for o in spec.outputs:
            order.extend([o, f"{o}__raw"] if spec.kind == "xs" else [o])
    return frame.select(order).sort("date", "sid")


def _manifest(settings: Settings, reg: dict[str, FeatureSpec], version: int, clock: Clock) -> dict:
    """Manifest with first_valid_date and coverage (non-null share on/after it) read from the store."""
    lf = io.read(settings, "features", sub=f"v{version}")
    outs = [(o, s) for _, s in sorted(reg.items()) for o in s.outputs]
    first = lf.select(
        [pl.col("date").filter(pl.col(o).is_not_null()).min().alias(o) for o, _ in outs]
    ).collect()
    fvd = {o: first[o][0] for o, _ in outs}
    cov = lf.select(
        [
            pl.col(o).is_not_null().filter(pl.col("date") >= fvd[o]).mean().alias(o)
            if fvd[o] is not None
            else pl.lit(0.0).alias(o)
            for o, _ in outs
        ]
    ).collect()
    return {
        "version": version,
        "created": clock.now().isoformat(),
        "git_sha": git_sha(),
        "features": [
            {
                "name": o,
                "group": s.group,
                "kind": s.kind,
                "version": s.version,
                "first_valid_date": None if fvd[o] is None else fvd[o].isoformat(),
                "coverage": float(cov[o][0]),
                "label": s.label,
            }
            for o, s in outs
        ],
    }


def build_features(
    settings: Settings,
    start: date | None = None,
    end: date | None = None,
    version: int | None = None,
    *,
    min_valid: int = MIN_VALID,
    registry: dict[str, FeatureSpec] | None = None,
    clock: Clock | None = None,
) -> int:
    """Compute, rank-scale and store every registered feature for u500 rows; returns the version number.

    `start` (incremental mode) recomputes from full history but rewrites only dates >= start, merging by
    key; it is ignored when the set of (name, version) pairs changed (a new version is built in full).
    `end` cuts the data at that date (as_of). Rank-scaled xs values and `<name>__raw` are Float32.
    """
    clk = clock if clock is not None else SystemClock()
    reg = load_all() if registry is None else registry
    if not reg:
        raise DataValidationError("no features registered")
    cur = current_version(settings)
    v = _choose_version(settings, reg, version)
    incremental = start is not None and v == cur and _manifest_path(settings, v).is_file()
    ctx = FeatureContext(settings, as_of=end)
    base = ctx.universe().filter(pl.col("u500")).select("date", "sid").collect()
    if base.is_empty():
        raise DataNotAvailable("panel/universe has no u500 rows; run build-panel first")
    frame = _assemble(reg, ctx, base, start if incremental else None, min_valid)
    sub = f"v{v}"
    if incremental:
        io.upsert(settings, "features", frame, sub=sub, clock=clk)
    else:
        vdir = _root(settings) / sub
        for old in vdir.glob("part-*.parquet") if vdir.is_dir() else []:
            old.unlink()
        for (year,), part in (
            frame.with_columns(pl.col("date").dt.year().alias("_y")).partition_by("_y", as_dict=True).items()
        ):
            io.write_partition(settings, "features", str(year), part.drop("_y"), sub=sub, clock=clk)
    man = _manifest(settings, reg, v, clk)
    mp = _manifest_path(settings, v)
    mp.write_text(json.dumps(man, indent=1), encoding="utf-8", newline="\n")
    (_root(settings) / "CURRENT").write_text(f"{v}\n", encoding="utf-8", newline="\n")
    log.info(
        "features_built", version=v, rows=frame.height, features=len(man["features"]), incremental=incremental
    )
    return v


def load_features(
    settings: Settings,
    purpose: str,
    start: date | None = None,
    end: date | None = None,
    columns: list[str] | None = None,
) -> pl.LazyFrame:
    """The current feature version as a lazy frame; 'research' stops at lockbox_start - 1 day,
    'production' does not, 'lockbox' only inside open_lockbox(...)."""
    v = current_version(settings)
    if v is None:
        raise DataNotAvailable("no features built yet (features/CURRENT is missing)")
    cols = None if columns is None else list(dict.fromkeys(["date", "sid", *columns]))
    lf = io.read(settings, "features", start=start, end=end, columns=cols, sub=f"v{v}")
    return guard(lf, "date", purpose, settings)
