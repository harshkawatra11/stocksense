"""Append-only attempt registry. EVERY evaluated configuration registers here.

That count IS the n_trials fed to robustness.deflated_sharpe_ratio, which is
what makes a wider sweep raise its own bar instead of gaming itself.
Registering only survivors would understate n_trials and silently inflate
every DSR the system reports.

PROTECTED. Do not edit after it lands.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

_VALID_VERDICTS = {"screened_out", "gate_fail", "gate_pass", "vault_fail", "promoted"}
_VALID_FAIL_REASONS = {"low_ic", "unstable_ic", "fast_decay", "cost_drag", "capacity", None}


def config_hash(family: str, params: dict) -> str:
    """sha256 over (family, sorted params) -- deterministic, order-independent,
    the idempotency key used everywhere a config needs a stable identity."""
    payload = json.dumps({"family": family, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def register_attempt(
    store,
    *,
    hypothesis_id: str,
    config_hash: str,  # the caller computes this via the module-level config_hash() above
    config_json: str,
    family: str,
) -> str:
    """Insert one attempt, return its attempt_id.

    Idempotent on (hypothesis_id, config_hash): a resumed sweep that re-offers
    a config it already registered gets the SAME attempt_id back rather than a
    duplicate row, via ON CONFLICT DO UPDATE ... RETURNING. This is what makes
    an interrupted-and-resumed sweep count each config exactly once.
    """
    existing = store.con.execute(
        "SELECT attempt_id FROM evaluation_attempts WHERE hypothesis_id = ? AND config_hash = ?",
        [hypothesis_id, config_hash],
    ).fetchone()
    if existing:
        return existing[0]

    attempt_id = uuid.uuid4().hex
    store.con.execute(
        "INSERT INTO evaluation_attempts "
        "(attempt_id, hypothesis_id, family, config_hash, config_json, registered_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [attempt_id, hypothesis_id, family, config_hash, config_json, datetime.now()],
    )
    return attempt_id


def record_result(
    store,
    attempt_id: str,
    *,
    verdict: str,
    metrics_json: str | None = None,
    fail_reason: str | None = None,
) -> None:
    """Attach the outcome to an already-registered attempt.

    verdict: screened_out | gate_fail | gate_pass | vault_fail | promoted
    fail_reason: low_ic | unstable_ic | fast_decay | cost_drag | capacity | None
    """
    if verdict not in _VALID_VERDICTS:
        raise ValueError(f"verdict must be one of {sorted(_VALID_VERDICTS)}, got {verdict!r}")
    if fail_reason not in _VALID_FAIL_REASONS:
        raise ValueError(
            f"fail_reason must be one of {sorted(r for r in _VALID_FAIL_REASONS if r)} or None, "
            f"got {fail_reason!r}"
        )
    store.con.execute(
        "UPDATE evaluation_attempts SET verdict = ?, fail_reason = ?, metrics_json = ? "
        "WHERE attempt_id = ?",
        [verdict, fail_reason, metrics_json, attempt_id],
    )


def count_attempts(reader, hypothesis_id: str | None = None) -> int:
    """n_trials for the Deflated Sharpe. Counts EVERY attempt, not just
    survivors -- registering only survivors would understate n_trials and
    silently inflate the DSR reported for a real result."""
    if not reader.exists("evaluation_attempts"):
        return 0
    sql = "SELECT count(*) AS n FROM {evaluation_attempts}"
    params = []
    if hypothesis_id:
        sql += " WHERE hypothesis_id = ?"
        params.append(hypothesis_id)
    df = reader.sql(sql, params)
    return int(df.iloc[0]["n"]) if not df.empty else 0


def trial_sharpe_std(reader, hypothesis_id: str) -> float:
    """Cross-sectional std of Sharpe across this hypothesis's attempts -- the
    other DSR input. MEASURED from the sweep's recorded metrics, never
    assumed to be 1.0. Sharpe is read from each attempt's metrics_json.

    Returns nan if fewer than 2 attempts carry a parseable sharpe -- there is
    no dispersion to measure from 0 or 1 points.
    """
    if not reader.exists("evaluation_attempts"):
        return float("nan")
    df = reader.sql(
        "SELECT metrics_json FROM {evaluation_attempts} WHERE hypothesis_id = ? "
        "AND metrics_json IS NOT NULL",
        [hypothesis_id],
    )
    if df.empty:
        return float("nan")

    sharpes = []
    for raw in df["metrics_json"]:
        try:
            metrics = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(metrics, dict) and "sharpe" in metrics and metrics["sharpe"] is not None:
            sharpes.append(float(metrics["sharpe"]))

    if len(sharpes) < 2:
        return float("nan")
    return float(np.std(sharpes, ddof=1))


def read_attempts(reader, hypothesis_id: str | None = None) -> pd.DataFrame:
    """Full attempt rows, for the search leaderboard / audit."""
    if not reader.exists("evaluation_attempts"):
        return pd.DataFrame(
            columns=[
                "attempt_id", "hypothesis_id", "family", "config_hash", "config_json",
                "registered_at", "verdict", "fail_reason", "metrics_json",
            ]
        )
    sql = "SELECT * FROM {evaluation_attempts}"
    params = []
    if hypothesis_id:
        sql += " WHERE hypothesis_id = ?"
        params.append(hypothesis_id)
    return reader.sql(sql + " ORDER BY registered_at", params)
