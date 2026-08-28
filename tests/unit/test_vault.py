"""Phase K1: the sealed vault.

The load-bearing test is `test_default_load_candles_never_returns_sealed_rows`,
which runs against the REAL database rather than a fixture -- a seal that holds
on synthetic data but leaks on the actual store would be worse than no seal,
because it would be believed.
"""

from __future__ import annotations

import subprocess
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from stocksense.data.store import Store
from stocksense.data.vault import (
    VAULT_SEAL_DATE,
    UnsealToken,
    VaultSealed,
    apply_seal,
    unseal,
)


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


@pytest.fixture()
def committed_prereg(tmp_path):
    """A real git repo with one committed file -- `unseal` refuses anything
    else, so the happy path needs a genuinely committed file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    f = repo / "preregistration_vault_test.md"
    f.write_text("# fixed before any result\n", encoding="utf-8")
    subprocess.run(["git", "add", f.name], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "prereg"], cwd=repo, check=True)
    return f


def _register_attempt_row(store, attempt_id: str = "att-1") -> str:
    store.con.execute(
        """
        INSERT INTO evaluation_attempts
            (attempt_id, hypothesis_id, preregistration_path, preregistration_hash,
             holdout_id, holdout_spec_json, attempt_index, registered_at, registered_by,
             status, base_alpha)
        VALUES (?, 'h1', 'p.md', 'abc', 'hold-1', '{}', 1, ?, 'user', 'registered', 0.10)
        """,
        [attempt_id, datetime.now(timezone.utc)],
    )
    return attempt_id


# ---- apply_seal ----


def test_apply_seal_drops_rows_on_and_after_the_seal_date() -> None:
    df = pd.DataFrame({
        "symbol": ["A"] * 4,
        "date": pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-01", "2025-06-01"]),
    })
    kept = apply_seal(df)
    assert len(kept) == 2
    assert kept["date"].max() == pd.Timestamp("2024-12-31")


def test_apply_seal_is_inclusive_of_the_boundary_day() -> None:
    """2025-01-01 itself is SEALED, not the last research day."""
    df = pd.DataFrame({"symbol": ["A"], "date": pd.to_datetime([VAULT_SEAL_DATE])})
    assert apply_seal(df).empty


def test_apply_seal_handles_empty_and_missing_column() -> None:
    assert apply_seal(pd.DataFrame()).empty
    odd = pd.DataFrame({"symbol": ["A"]})
    assert len(apply_seal(odd)) == 1  # no date column -> nothing to seal, returned untouched


# ---- the real-database guarantee ----


def test_default_load_candles_never_returns_sealed_rows() -> None:
    """Against the REAL store. If this ever fails, every research result
    produced afterwards is contaminated and the holdout is gone."""
    from stocksense.core.config import get_settings
    from stocksense.data.loader import load_candles

    settings = get_settings()
    settings.price_source = "bhavcopy"
    settings.use_point_in_time_universe = True

    store = Store(settings.duckdb_path, read_only=True)
    try:
        raw_max = store.con.execute("SELECT MAX(date) FROM bhavcopy_eq WHERE series='EQ'").fetchone()[0]
        if raw_max is None or raw_max < VAULT_SEAL_DATE:
            pytest.skip("database holds no sealed-period data, so there is nothing to seal")
        candles = load_candles(settings, store)
    finally:
        store.close()

    assert not candles.empty
    assert candles["date"].max().date() < VAULT_SEAL_DATE


# ---- unseal ----


def test_unseal_requires_the_preregistration_to_exist(tmp_store, tmp_path) -> None:
    _register_attempt_row(tmp_store)
    with pytest.raises(FileNotFoundError):
        unseal(tmp_store, attempt_id="att-1", hypothesis_id="h1",
               preregistration_path=tmp_path / "nope.md", reason="r")


def test_unseal_requires_a_committed_preregistration(tmp_store, tmp_path) -> None:
    """An uncommitted file could still be edited after seeing the result."""
    _register_attempt_row(tmp_store)
    repo = tmp_path / "repo2"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    loose = repo / "preregistration_loose.md"
    loose.write_text("not committed", encoding="utf-8")
    with pytest.raises(VaultSealed, match="not committed"):
        unseal(tmp_store, attempt_id="att-1", hypothesis_id="h1",
               preregistration_path=loose, reason="r")


def test_unseal_requires_a_registered_attempt(tmp_store, committed_prereg) -> None:
    """The unseal must be tied to a counted trial, so it lands in the DSR's N."""
    with pytest.raises(VaultSealed, match="not registered"):
        unseal(tmp_store, attempt_id="never-registered", hypothesis_id="h1",
               preregistration_path=committed_prereg, reason="r")


def test_unseal_succeeds_and_is_recorded(tmp_store, committed_prereg) -> None:
    _register_attempt_row(tmp_store)
    token = unseal(tmp_store, attempt_id="att-1", hypothesis_id="h1",
                   preregistration_path=committed_prereg, reason="final gate")
    assert isinstance(token, UnsealToken)
    rows = tmp_store.read_vault_unseals("h1")
    assert len(rows) == 1
    assert rows.iloc[0]["unseal_id"] == token.unseal_id
    assert rows.iloc[0]["reason"] == "final gate"


def test_second_unseal_for_same_hypothesis_is_refused(tmp_store, committed_prereg) -> None:
    """A holdout looked at twice is not a holdout."""
    _register_attempt_row(tmp_store)
    unseal(tmp_store, attempt_id="att-1", hypothesis_id="h1",
           preregistration_path=committed_prereg, reason="first")
    with pytest.raises(VaultSealed, match="already used its one unseal"):
        unseal(tmp_store, attempt_id="att-1", hypothesis_id="h1",
               preregistration_path=committed_prereg, reason="second")


def test_a_different_hypothesis_gets_its_own_unseal(tmp_store, committed_prereg) -> None:
    _register_attempt_row(tmp_store, "att-1")
    tmp_store.con.execute(
        """
        INSERT INTO evaluation_attempts
            (attempt_id, hypothesis_id, preregistration_path, preregistration_hash,
             holdout_id, holdout_spec_json, attempt_index, registered_at, registered_by,
             status, base_alpha)
        VALUES ('att-2', 'h2', 'p.md', 'abc', 'hold-1', '{}', 2, ?, 'user', 'registered', 0.10)
        """,
        [datetime.now(timezone.utc)],
    )
    unseal(tmp_store, attempt_id="att-1", hypothesis_id="h1",
           preregistration_path=committed_prereg, reason="a")
    unseal(tmp_store, attempt_id="att-2", hypothesis_id="h2",
           preregistration_path=committed_prereg, reason="b")
    assert len(tmp_store.read_vault_unseals()) == 2


def test_token_lifts_the_ceiling() -> None:
    """With a token the sealed rows come back -- otherwise the final gate could
    never be run at all."""
    from stocksense.data.loader import _apply_vault_ceiling

    df = pd.DataFrame({
        "symbol": ["A"] * 3,
        "date": pd.to_datetime(["2024-12-31", "2025-06-01", "2026-01-01"]),
    })
    token = UnsealToken(unseal_id="u1", attempt_id="a1", hypothesis_id="h1",
                        preregistration_path="p.md", preregistration_hash="abc")
    assert len(_apply_vault_ceiling(df, token)) == 3
    assert len(_apply_vault_ceiling(df, None)) == 1
