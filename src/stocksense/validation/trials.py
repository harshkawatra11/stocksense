from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from stocksense.core.ids import canonical_json

if TYPE_CHECKING:
    from stocksense.core.state import State

PLANNING_FAMILY = "planning-exploration"
PLANNING_NOTE = "signals examined on development data while writing the plan, 2026-11-11"
N_PLANNING_TRIALS = 30


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _hash(family: str, target: str, spec: Any) -> str:
    text = canonical_json({"family": family, "target": target, "spec": spec})
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def register_trial(state: State, family: str, target: str, spec: Any) -> str:
    """Record a spec before any test-fold evaluation; returns its trial_id (existing one if repeated)."""
    h = _hash(family, target, spec)
    con = state.connect()
    try:
        row = con.execute("SELECT trial_id FROM trials WHERE spec_hash=?", (h,)).fetchone()
        if row is not None:
            return str(row[0])
        trial_id = f"{family}-{target}-{h[:10]}"
        con.execute(
            "INSERT INTO trials(trial_id, created_at, family, target, spec_json, spec_hash, status, git_sha) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (trial_id, state._now_iso(), family, target, canonical_json(spec), h, "registered", git_sha()),
        )
        return trial_id
    finally:
        con.close()


def finish_trial(state: State, trial_id: str, metrics: dict[str, Any], returns_path: str | Path) -> None:
    """Attach metrics (per-period Sharpe under `sharpe_daily`) and the daily-returns file; once only."""
    con = state.connect()
    try:
        row = con.execute("SELECT status FROM trials WHERE trial_id=?", (trial_id,)).fetchone()
        if row is None:
            raise KeyError(trial_id)
        if row[0] == "done":
            raise ValueError(f"trial {trial_id} is already finished")
        con.execute(
            "UPDATE trials SET status='done', metrics_json=?, returns_path=? WHERE trial_id=?",
            (canonical_json(metrics), str(returns_path), trial_id),
        )
    finally:
        con.close()


def n_trials(state: State, family: str | None = None) -> int:
    """Number of registered trials (seeds included), optionally for one family."""
    con = state.connect()
    try:
        if family is None:
            return int(con.execute("SELECT count(*) FROM trials").fetchone()[0])
        return int(con.execute("SELECT count(*) FROM trials WHERE family=?", (family,)).fetchone()[0])
    finally:
        con.close()


def sharpe_variance_across_trials(state: State, family: str | None = None) -> float:
    """Sample variance (ddof=1) of per-period Sharpes of finished trials; nan with fewer than two."""
    import json

    con = state.connect()
    try:
        sql = "SELECT metrics_json FROM trials WHERE status='done' AND metrics_json IS NOT NULL"
        args: tuple[str, ...] = ()
        if family is not None:
            sql += " AND family=?"
            args = (family,)
        rows = con.execute(sql, args).fetchall()
    finally:
        con.close()
    srs = []
    for (m,) in rows:
        v = json.loads(m).get("sharpe_daily")
        if v is not None and math.isfinite(v):
            srs.append(float(v))
    return float(np.var(srs, ddof=1)) if len(srs) >= 2 else float("nan")


def seed_planning_trials(state: State) -> None:
    """Insert the 30 planning-exploration rows (idempotent by trial_id)."""
    con = state.connect()
    try:
        for i in range(1, N_PLANNING_TRIALS + 1):
            tid = f"{PLANNING_FAMILY}-{i:02d}"
            con.execute(
                "INSERT OR IGNORE INTO trials(trial_id, created_at, family, target, spec_json, spec_hash, "
                "status, metrics_json, returns_path, git_sha, note) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    tid,
                    state._now_iso(),
                    PLANNING_FAMILY,
                    None,
                    None,
                    hashlib.sha1(tid.encode("utf-8")).hexdigest(),
                    "done",
                    None,
                    None,
                    None,
                    PLANNING_NOTE,
                ),
            )
    finally:
        con.close()
