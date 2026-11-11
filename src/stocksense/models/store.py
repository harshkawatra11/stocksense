"""Model store (PLAN T7.4): models/<model_id>/ on disk plus the `models` and `model_events` tables.

Directory: spec.json, weights.* (written by the model), feature_manifest.json (ordered feature names and the
feature version), fit_info.json, train_range.json, git_sha.txt (validation.json is added later by the gate).
A status change appends a `model_events` row. A strategy family prefix (the part of `family` before the first
'-': s1, s2, s3) has at most one champion: promoting one demotes the previous champion to 'shadow' in the same
transaction.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from stocksense.core.clock import Clock, SystemClock
from stocksense.core.config import Settings
from stocksense.core.ids import model_id as make_model_id
from stocksense.core.ids import spec_hash8
from stocksense.core.paths import Paths
from stocksense.core.state import State
from stocksense.data import io
from stocksense.models.base import Model, ModelSpec, build_model
from stocksense.validation.trials import git_sha

STATUSES = ("candidate", "shadow", "champion", "retired")
PRED_MODES = ("walkforward", "live")


def family_prefix(family: str) -> str:
    """'s1' for 's1-ridge' (the whole family when it has no dash)."""
    return family.split("-", 1)[0]


def model_dir(settings: Settings, model_id: str) -> Path:
    return Paths(settings).models / model_id


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n"
    )


def save_model(
    settings: Settings,
    state: State,
    model: Model,
    spec: ModelSpec,
    train_end: date,
    meta: dict[str, Any],
) -> str:
    """Write the model directory and register a 'candidate' row; returns the model_id.

    meta keys used: feature_names, feature_kinds (optional), feature_version, fit_info (dict), train_start
    (optional), note (optional)."""
    spec_dict = spec.to_dict()
    mid = make_model_id(spec.family, spec.target, spec_dict, train_end)
    d = model_dir(settings, mid)
    d.mkdir(parents=True, exist_ok=True)
    _write_json(d / "spec.json", spec_dict)
    model.save(d)
    _write_json(
        d / "feature_manifest.json",
        {
            "features": list(meta.get("feature_names", [])),
            "kinds": list(meta.get("feature_kinds", [])),
            "version": meta.get("feature_version"),
        },
    )
    _write_json(d / "fit_info.json", meta.get("fit_info", {}))
    _write_json(d / "train_range.json", {"train_start": meta.get("train_start"), "train_end": train_end})
    (d / "git_sha.txt").write_text(f"{git_sha() or 'unknown'}\n", encoding="utf-8", newline="\n")
    now = state._now_iso()
    con = state.connect()
    try:
        con.execute("BEGIN IMMEDIATE")
        con.execute(
            "INSERT INTO models(model_id, family, target, spec_hash, train_end, created_at, status, "
            "validated, "
            "path, note) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (mid, spec.family, spec.target, spec_hash8(spec_dict), train_end.isoformat(), now, "candidate", 0,
             str(d), meta.get("note")),
        )  # fmt: skip
        con.execute(
            "INSERT INTO model_events(ts, model_id, event, detail) VALUES(?,?,?,?)",
            (now, mid, "saved", f"train_end={train_end.isoformat()}"),
        )
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()
    return mid


def load_model(settings: Settings, model_id: str) -> Model:
    """Rebuild the model from spec.json and restore its weights."""
    d = model_dir(settings, model_id)
    spec = ModelSpec.from_dict(json.loads((d / "spec.json").read_text(encoding="utf-8")))
    model = build_model(spec)
    model.load(d)
    return model


def set_status(state: State, model_id: str, status: str, note: str = "") -> None:
    """Move a model to `status`, logging an event; a new champion demotes its prefix's champion to shadow."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    now = state._now_iso()
    con = state.connect()
    try:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT family FROM models WHERE model_id=?", (model_id,)).fetchone()
        if row is None:
            raise KeyError(model_id)
        if status == "champion":
            prefix = family_prefix(row[0])
            rivals = [
                r[0]
                for r in con.execute(
                    "SELECT model_id, family FROM models WHERE status='champion' AND model_id<>?", (model_id,)
                ).fetchall()
                if family_prefix(r[1]) == prefix
            ]
            for other in rivals:
                con.execute("UPDATE models SET status='shadow' WHERE model_id=?", (other,))
                con.execute(
                    "INSERT INTO model_events(ts, model_id, event, detail) VALUES(?,?,?,?)",
                    (now, other, "status:shadow", f"demoted: {model_id} became champion of {prefix}"),
                )
        con.execute("UPDATE models SET status=? WHERE model_id=?", (status, model_id))
        con.execute(
            "INSERT INTO model_events(ts, model_id, event, detail) VALUES(?,?,?,?)",
            (now, model_id, f"status:{status}", note),
        )
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def current(state: State, prefix: str, status: str) -> str | None:
    """The newest model of a strategy family prefix (s1, s2, ...) with this status, or None."""
    con = state.connect()
    try:
        rows = con.execute(
            "SELECT model_id, family FROM models WHERE status=? ORDER BY created_at DESC, rowid DESC",
            (status,),
        ).fetchall()
    finally:
        con.close()
    for mid, fam in rows:
        if family_prefix(fam) == prefix:
            return str(mid)
    return None


def append_preds(
    settings: Settings, model_id: str, frame: pl.DataFrame, mode: str, clock: Clock | None = None
) -> None:
    """Upsert scores (date, sid, score) into preds/<model_id>/ on (date, sid), stamped made_at and mode."""
    if mode not in PRED_MODES:
        raise ValueError(f"mode must be one of {PRED_MODES}, got {mode!r}")
    clk = clock if clock is not None else SystemClock()
    out = frame.select("date", "sid", "score").with_columns(
        pl.lit(clk.now()).alias("made_at"), pl.lit(mode).alias("mode")
    )
    io.upsert(settings, "preds", out, sub=model_id, clock=clk)
