"""Tests for the sealed vault -- the untouchable final holdout.

Every research path must funnel through apply_seal before it can see rows
on/after VAULT_SEAL_DATE, and a hypothesis gets exactly one unseal, ever.
"""

from __future__ import annotations

import subprocess
from datetime import date, datetime

import pandas as pd
import pytest

from stocksense.data.store import Store
from stocksense.evaluation import attempts
from stocksense.evaluation.vault import (
    VAULT_SEAL_DATE,
    UnsealToken,
    VaultSealed,
    apply_seal,
    unseal,
)


def _panel(dates: list[date]) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "symbol": ["X"] * len(dates), "v": range(len(dates))})


# ---------------------------------------------------------------- apply_seal
def test_seal_drops_rows_on_or_after_the_seal_date():
    dates = [date(2025, 6, 29), date(2025, 6, 30), date(2025, 7, 1), date(2025, 7, 2)]
    out = apply_seal(_panel(dates))
    assert list(out["date"]) == [date(2025, 6, 29), date(2025, 6, 30)]


def test_seal_keeps_the_boundary_date_out_it_is_sealed_not_the_day_before():
    """VAULT_SEAL_DATE itself must be excluded -- >= not >."""
    out = apply_seal(_panel([VAULT_SEAL_DATE]))
    assert out.empty


def test_seal_is_a_no_op_on_data_entirely_before_the_seal():
    dates = [date(2020, 1, 1), date(2021, 1, 1)]
    out = apply_seal(_panel(dates))
    assert len(out) == 2


def test_seal_on_empty_frame_returns_it_unchanged():
    empty = pd.DataFrame(columns=["date", "symbol", "v"])
    assert apply_seal(empty).empty


def test_a_valid_token_lifts_the_ceiling():
    dates = [date(2025, 6, 30), date(2025, 7, 1), date(2026, 1, 1)]
    token = UnsealToken(
        unseal_id="u1", attempt_id="a1", hypothesis_id="h1",
        preregistration_path="x.md", preregistration_sha256="abc", issued_at=datetime.now(),
    )
    out = apply_seal(_panel(dates), token=token)
    assert len(out) == 3


# --------------------------------------------------------------------- unseal
@pytest.fixture()
def repo(tmp_path):
    """A throwaway git repo with committed and uncommitted markdown files,
    standing in for the real project repo's pre-registration docs."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)

    committed = tmp_path / "committed.md"
    committed.write_text("pre-registration, frozen\n")
    subprocess.run(["git", "add", "committed.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=tmp_path, check=True)

    uncommitted = tmp_path / "uncommitted.md"
    uncommitted.write_text("never committed\n")

    return tmp_path


@pytest.fixture()
def store_with_attempt(tmp_path):
    db, pq = tmp_path / "hot.duckdb", tmp_path / "parquet"
    s = Store(db, pq)
    aid = attempts.register_attempt(
        s, hypothesis_id="h1", config_hash="c1", config_json="{}", family="test"
    )
    yield s, aid
    s.close()


def test_unseal_requires_a_committed_preregistration(repo, store_with_attempt):
    store, attempt_id = store_with_attempt
    with pytest.raises(VaultSealed, match="committed"):
        unseal(
            store, attempt_id=attempt_id, hypothesis_id="h1",
            preregistration_path=repo / "uncommitted.md", reason="test", repo_root=repo,
        )


def test_unseal_requires_an_unmodified_preregistration(repo, store_with_attempt):
    """Committed then edited must refuse exactly like never committed -- a
    pre-registration could otherwise be quietly loosened after landing."""
    store, attempt_id = store_with_attempt
    path = repo / "committed.md"
    path.write_text("quietly edited after committing\n")

    with pytest.raises(VaultSealed, match="committed"):
        unseal(
            store, attempt_id=attempt_id, hypothesis_id="h1",
            preregistration_path=path, reason="test", repo_root=repo,
        )


def test_unseal_requires_a_registered_attempt(repo, store_with_attempt):
    store, _ = store_with_attempt
    with pytest.raises(VaultSealed, match="not registered"):
        unseal(
            store, attempt_id="does-not-exist", hypothesis_id="h1",
            preregistration_path=repo / "committed.md", reason="test", repo_root=repo,
        )


def test_a_valid_unseal_succeeds_and_is_recorded(repo, store_with_attempt):
    store, attempt_id = store_with_attempt
    token = unseal(
        store, attempt_id=attempt_id, hypothesis_id="h1",
        preregistration_path=repo / "committed.md", reason="final gate test", repo_root=repo,
    )
    assert isinstance(token, UnsealToken)
    assert token.hypothesis_id == "h1"
    assert store.vault_unseals_for("h1") == 1


def test_second_unseal_for_same_hypothesis_raises(repo, store_with_attempt):
    store, attempt_id = store_with_attempt
    unseal(
        store, attempt_id=attempt_id, hypothesis_id="h1",
        preregistration_path=repo / "committed.md", reason="first", repo_root=repo,
    )
    with pytest.raises(VaultSealed, match="already used"):
        unseal(
            store, attempt_id=attempt_id, hypothesis_id="h1",
            preregistration_path=repo / "committed.md", reason="second attempt", repo_root=repo,
        )


def test_a_different_hypothesis_can_still_unseal(repo, tmp_path):
    db, pq = tmp_path / "hot.duckdb", tmp_path / "parquet"
    store = Store(db, pq)
    a1 = attempts.register_attempt(store, hypothesis_id="h1", config_hash="c1", config_json="{}", family="f")
    a2 = attempts.register_attempt(store, hypothesis_id="h2", config_hash="c2", config_json="{}", family="f")

    unseal(store, attempt_id=a1, hypothesis_id="h1", preregistration_path=repo / "committed.md",
           reason="r", repo_root=repo)
    token2 = unseal(store, attempt_id=a2, hypothesis_id="h2", preregistration_path=repo / "committed.md",
                     reason="r", repo_root=repo)
    assert token2.hypothesis_id == "h2"
    store.close()
