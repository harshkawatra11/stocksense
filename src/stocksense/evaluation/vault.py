"""The sealed holdout. One unseal per hypothesis, ever.

Enforced at the single choke point every research path is meant to funnel
through (`apply_seal`, called from a project-wide `load_candles` once that
exists): without an `UnsealToken`, every row dated on or after
`VAULT_SEAL_DATE` is dropped before a strategy, a search, or a walk-forward
fold ever sees it.

`VAULT_SEAL_DATE = 2025-07-01` is the user's own decision: it withholds
roughly the most recent 14 months / ~290 trading days (~8% of the full
2010-> history) while leaving 2010 -> 2025-H1 for research, which still spans
the 2020 crash and the 2021-24 bull run. Enough holdout that a single DSR/PBO
test on it means something; not so much that the search is blind to the
modern regime.

PROTECTED. Do not edit after it lands.
"""

from __future__ import annotations

import hashlib
import subprocess
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

VAULT_SEAL_DATE = date(2025, 7, 1)


class VaultSealed(RuntimeError):
    """Raised when an unseal is refused: a missing/uncommitted/modified
    pre-registration, an unknown attempt, or a hypothesis that has already
    used its one unseal."""


@dataclass(frozen=True)
class UnsealToken:
    unseal_id: str
    attempt_id: str
    hypothesis_id: str
    preregistration_path: str
    preregistration_sha256: str
    issued_at: datetime


def apply_seal(
    df: pd.DataFrame,
    date_col: str = "date",
    token: UnsealToken | None = None,
) -> pd.DataFrame:
    """Drop rows on/after VAULT_SEAL_DATE unless a token is presented.

    Logs the withheld row count at INFO (or WARNING with the unseal_id when a
    token lifts the ceiling) -- an unseal must be visible in the log, not a
    silent state change.
    """
    if df.empty:
        return df

    dates = pd.to_datetime(df[date_col])
    sealed_mask = dates.dt.date >= VAULT_SEAL_DATE

    if token is not None:
        log.warning(
            "vault_unsealed",
            unseal_id=token.unseal_id,
            hypothesis_id=token.hypothesis_id,
            rows_unsealed=int(sealed_mask.sum()),
        )
        return df

    n_dropped = int(sealed_mask.sum())
    if n_dropped:
        log.info("vault_ceiling_applied", rows_dropped=n_dropped, seal_date=str(VAULT_SEAL_DATE))
    return df.loc[~sealed_mask].reset_index(drop=True)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )


def _is_committed_and_unmodified(path: Path, repo_root: Path) -> bool:
    """The file must exist, be tracked by git, and have no uncommitted diff
    against HEAD -- committed-then-edited must refuse exactly like never
    committed, or a pre-registration could be quietly loosened after landing.
    """
    if not path.exists():
        return False
    rel = path.resolve().relative_to(repo_root.resolve())
    tracked = _git("ls-files", "--error-unmatch", str(rel), cwd=repo_root)
    if tracked.returncode != 0:
        return False
    diff = _git("diff", "--quiet", "HEAD", "--", str(rel), cwd=repo_root)
    return diff.returncode == 0


def unseal(
    store,
    *,
    attempt_id: str,
    hypothesis_id: str,
    preregistration_path: str | Path,
    reason: str,
    repo_root: str | Path,
    requested_by: str = "user",
) -> UnsealToken:
    """Refuse unless ALL hold:

      1. the pre-registration file EXISTS, is COMMITTED, and is UNMODIFIED
         relative to HEAD.
      2. `attempt_id` exists in `evaluation_attempts`.
      3. NO prior `vault_unseals` row exists for this `hypothesis_id`.

    Then write the vault_unseals row and return the token. Uses the writer's
    OWN DuckDB connection for both checks -- not a possibly-stale Parquet
    snapshot from the last publish() -- because this must see attempts and
    unseals from the same session that has not yet been published.
    """
    prereg_path = Path(preregistration_path)
    repo_root = Path(repo_root)

    if not _is_committed_and_unmodified(prereg_path, repo_root):
        raise VaultSealed(
            f"pre-registration {prereg_path} must be committed to git and unmodified "
            "before a hypothesis may unseal the vault"
        )

    attempt_exists = store.con.execute(
        "SELECT count(*) FROM evaluation_attempts WHERE attempt_id = ?", [attempt_id]
    ).fetchone()[0]
    if not attempt_exists:
        raise VaultSealed(f"attempt_id {attempt_id!r} is not registered in evaluation_attempts")

    if store.vault_unseals_for(hypothesis_id) > 0:
        raise VaultSealed(f"hypothesis {hypothesis_id!r} has already used its one unseal")

    token = UnsealToken(
        unseal_id=uuid.uuid4().hex,
        attempt_id=attempt_id,
        hypothesis_id=hypothesis_id,
        preregistration_path=str(prereg_path),
        preregistration_sha256=_file_sha256(prereg_path),
        issued_at=datetime.now(),
    )
    store.record_vault_unseal(
        dict(
            unseal_id=token.unseal_id,
            attempt_id=token.attempt_id,
            hypothesis_id=token.hypothesis_id,
            preregistration_path=token.preregistration_path,
            preregistration_sha256=token.preregistration_sha256,
            issued_at=token.issued_at,
            requested_by=requested_by,
            reason=reason,
        )
    )
    return token
