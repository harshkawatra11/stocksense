"""
Phase K1: the sealed vault -- the "out-of-sample gate on data never seen".

THE PROBLEM THIS SOLVES. `evaluation/walkforward.py`'s folds are genuinely
purged and embargoed, but every sweep reuses the SAME test windows, every run,
forever -- and each fold's test window is folded into the NEXT fold's training
set (walkforward.py:91). So there is no terminal data this project has not
already looked at, repeatedly. `evaluation/attempts.py` acknowledges this by
counting attempts and tightening the significance threshold; it does not
prevent it. A search loop makes it acute: hundreds of evaluations against data
the loop can see is how you find prettier noise faster.

WHAT THIS IS. A hard date ceiling enforced at `data/loader.py:load_candles` --
the single choke point every research script, the reconcile loop, and
train_candidate already funnel through. Research sees 2010-01-01 .. 2024-12-31.
The 406 trading days from 2025-01-01 onward are unreachable without an explicit,
audited, once-per-hypothesis unseal.

Measured cost of the seal, at the time it was set:
    research 2010-2024 : 5,622,601 rows / 3,704 trading days  (86%)
    SEALED   2025-01+  :   914,160 rows /   406 trading days  (14%)

WHY A MECHANISM AND NOT A CONVENTION. A convention is a comment asking future
sessions to be careful. This raises. The whole value of a holdout is that it was
untouched, and that property cannot be maintained by good intentions across many
sessions and an automated loop.

ONE UNSEAL PER HYPOTHESIS, EVER. A second unseal for the same hypothesis_id is
refused outright. Looking twice and keeping the better answer is exactly the
selection bias the holdout exists to measure.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

VAULT_SEAL_DATE = date(2025, 1, 1)
"""Rows dated on or after this are sealed. Research code never sees them."""


class VaultSealed(Exception):
    """Raised when sealed data is requested without a valid unseal, or when a
    hypothesis tries to unseal a second time."""


@dataclass(frozen=True)
class UnsealToken:
    """Proof that a specific, pre-registered hypothesis has been granted its one
    look at the holdout. Passing this to `load_candles` lifts the ceiling."""

    unseal_id: str
    attempt_id: str
    hypothesis_id: str
    preregistration_path: str
    preregistration_hash: str


def unseal(
    store,
    *,
    attempt_id: str,
    hypothesis_id: str,
    preregistration_path: str | Path,
    reason: str,
    requested_by: str = "user",
) -> UnsealToken:
    """Grants one -- and only ever one -- look at the sealed period.

    Refuses unless ALL of these hold:
      1. The pre-registration file exists AND is committed to git. An
         uncommitted file could still be edited after seeing the result, which
         would make the pre-registration worthless.
      2. `attempt_id` exists in `evaluation_attempts` -- the unseal must be tied
         to a counted trial, so it lands in the DSR's N.
      3. No prior `vault_unseals` row exists for this `hypothesis_id`.

    Reuses `evaluation.attempts._is_committed` rather than reimplementing the
    git check -- one definition of "committed", not two that can drift.
    """
    import hashlib

    from stocksense.evaluation.attempts import _is_committed

    path = Path(preregistration_path)
    if not path.exists():
        raise FileNotFoundError(f"preregistration file does not exist: {path}")
    if not _is_committed(path):
        raise VaultSealed(
            f"preregistration file is not committed to git: {path} -- commit it BEFORE "
            "unsealing, or the pre-registration provides no protection at all"
        )

    attempt_row = store.con.execute(
        "SELECT attempt_id FROM evaluation_attempts WHERE attempt_id = ?", [attempt_id]
    ).fetchone()
    if attempt_row is None:
        raise VaultSealed(
            f"attempt_id {attempt_id!r} is not registered -- register the attempt first "
            "(evaluation.attempts.register_attempt) so this unseal is counted in N"
        )

    prior = store.read_vault_unseals(hypothesis_id=hypothesis_id)
    if not prior.empty:
        raise VaultSealed(
            f"hypothesis {hypothesis_id!r} has already used its one unseal "
            f"(unseal_id={prior.iloc[0]['unseal_id']}, at {prior.iloc[0]['requested_at']}). "
            "A holdout looked at twice is not a holdout."
        )

    preregistration_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    unseal_id = str(uuid.uuid4())[:12]
    store.insert_vault_unseal({
        "unseal_id": unseal_id,
        "attempt_id": attempt_id,
        "hypothesis_id": hypothesis_id,
        "preregistration_path": str(path),
        "preregistration_hash": preregistration_hash,
        "requested_at": datetime.now(timezone.utc),
        "requested_by": requested_by,
        "reason": reason,
    })
    return UnsealToken(
        unseal_id=unseal_id,
        attempt_id=attempt_id,
        hypothesis_id=hypothesis_id,
        preregistration_path=str(path),
        preregistration_hash=preregistration_hash,
    )


def apply_seal(candles, date_col: str = "date"):
    """Drops sealed-period rows. Called by `load_candles` when no token is
    supplied. Separate function so it is directly testable and so any future
    loader path can apply the identical rule rather than reimplementing it."""
    import pandas as pd

    if candles.empty or date_col not in candles.columns:
        return candles
    keep = pd.to_datetime(candles[date_col]).dt.date < VAULT_SEAL_DATE
    return candles[keep]
