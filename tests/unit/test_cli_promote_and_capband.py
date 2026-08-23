"""Phase H1: `promote-model` and `train-candidate --cap-band` CLI tests.

promote-model is the first code path that has EVER set a model to
'live' -- apply_gate_decision only produces 'shadow'/'archived',
apply_forward_record_decision only demotes 'live'->'shadow'. These
tests check the state-machine boundary directly: only 'shadow' models
may be promoted, and promotion is otherwise refused with a clear reason.

--cap-band tests check the wiring (which turnover_rank_band and
settings reach train_candidate_core) without running a real walk-
forward train, which needs real bhavcopy data this test db doesn't
have -- that end-to-end path is already covered by
research/bhavcopy_rerun_sweep.py's real run."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from stocksense.data.store import Store


@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKSENSE_DUCKDB_PATH", str(tmp_path / "test.duckdb"))
    from typer.testing import CliRunner
    return CliRunner(), tmp_path / "test.duckdb"


def _insert_model(db_path, model_id: str, lifecycle_state: str) -> None:
    store = Store(db_path)
    store.con.execute(
        """
        INSERT INTO model_registry (
            model_id, model_type, horizon_bars, top_n, feature_schema_version,
            created_at, lifecycle_state, artifact_path
        ) VALUES (?, 'cross_sectional_ranker', 10, 10, 'v1', ?, ?, 'unused.joblib')
        """,
        [model_id, datetime.now(timezone.utc), lifecycle_state],
    )
    store.close()


def test_promote_model_shadow_to_live(cli_env) -> None:
    runner, db_path = cli_env
    _insert_model(db_path, "m1", "shadow")
    from stocksense.cli.main import app

    result = runner.invoke(app, ["promote-model", "m1"])
    assert result.exit_code == 0
    assert "shadow -> live" in result.output

    store = Store(db_path)
    row = store.con.execute("SELECT lifecycle_state, promoted_at FROM model_registry WHERE model_id='m1'").fetchdf().iloc[0]
    store.close()
    assert row["lifecycle_state"] == "live"
    assert row["promoted_at"] is not None


@pytest.mark.parametrize("state", ["candidate", "archived", "live"])
def test_promote_model_refuses_non_shadow(cli_env, state) -> None:
    runner, db_path = cli_env
    _insert_model(db_path, "m1", state)
    from stocksense.cli.main import app

    result = runner.invoke(app, ["promote-model", "m1"])
    assert result.exit_code == 1
    assert "not 'shadow'" in result.output or "refusing" in result.output

    store = Store(db_path)
    row = store.con.execute("SELECT lifecycle_state FROM model_registry WHERE model_id='m1'").fetchdf().iloc[0]
    store.close()
    assert row["lifecycle_state"] == state  # unchanged


def test_promote_model_unknown_id(cli_env) -> None:
    runner, db_path = cli_env
    from stocksense.cli.main import app

    result = runner.invoke(app, ["promote-model", "does-not-exist"])
    assert result.exit_code == 1
    assert "No model with id" in result.output


def test_train_candidate_cap_band_rejects_unknown_value(cli_env) -> None:
    runner, db_path = cli_env
    from stocksense.cli.main import app

    result = runner.invoke(app, ["train-candidate", "--cap-band", "mega"])
    assert result.exit_code == 1
    assert "Unknown --cap-band" in result.output


def test_train_candidate_cap_band_forces_bhavcopy_and_passes_band(cli_env, monkeypatch) -> None:
    """Checks the wiring: --cap-band mid must (a) force price_source=
    bhavcopy and use_point_in_time_universe=True even though env vars
    left them at defaults, and (b) pass the resolved (0.5, 0.8) tuple
    through as turnover_rank_band -- without running a real train
    (mocked, since this test db has no bhavcopy data)."""
    captured = {}

    def fake_train_candidate_core(horizon, top_n, cost_bps, store, settings=None, turnover_rank_band=None):
        captured["settings"] = settings
        captured["turnover_rank_band"] = turnover_rank_band
        from stocksense.models.train_candidate import TrainCandidateResult
        return TrainCandidateResult(model_id=None, verdict=None, lifecycle_state=None, n_fold_results=0)

    # train_candidate imports train_candidate_core locally inside the
    # function body (looked up fresh on every call), so patching it at
    # its real source is what actually takes effect.
    import stocksense.models.train_candidate as tc_mod
    monkeypatch.setattr(tc_mod, "train_candidate_core", fake_train_candidate_core)

    runner, db_path = cli_env
    from stocksense.cli.main import app

    result = runner.invoke(app, ["train-candidate", "--cap-band", "mid"])

    assert captured["turnover_rank_band"] == (0.5, 0.8)
    assert captured["settings"].price_source == "bhavcopy"
    assert captured["settings"].use_point_in_time_universe is True
    assert "No fold results produced" in result.output  # the mocked None-state result's message
    assert result.exit_code == 1


def test_train_candidate_full_pit_cap_band_passes_none_band(cli_env, monkeypatch) -> None:
    captured = {}

    def fake_train_candidate_core(horizon, top_n, cost_bps, store, settings=None, turnover_rank_band=None):
        captured["turnover_rank_band"] = turnover_rank_band
        from stocksense.models.train_candidate import TrainCandidateResult
        return TrainCandidateResult(model_id=None, verdict=None, lifecycle_state=None, n_fold_results=0)

    import stocksense.models.train_candidate as tc_mod
    monkeypatch.setattr(tc_mod, "train_candidate_core", fake_train_candidate_core)

    runner, db_path = cli_env
    from stocksense.cli.main import app
    runner.invoke(app, ["train-candidate", "--cap-band", "full_pit"])

    assert captured["turnover_rank_band"] is None


def test_train_candidate_no_cap_band_leaves_settings_untouched(cli_env, monkeypatch) -> None:
    """Regression: omitting --cap-band entirely must be byte-for-byte
    the pre-Phase-H behavior -- no forced price_source, no band."""
    captured = {}

    def fake_train_candidate_core(horizon, top_n, cost_bps, store, settings=None, turnover_rank_band=None):
        captured["settings"] = settings
        captured["turnover_rank_band"] = turnover_rank_band
        from stocksense.models.train_candidate import TrainCandidateResult
        return TrainCandidateResult(model_id=None, verdict=None, lifecycle_state=None, n_fold_results=0)

    import stocksense.models.train_candidate as tc_mod
    monkeypatch.setattr(tc_mod, "train_candidate_core", fake_train_candidate_core)

    runner, db_path = cli_env
    from stocksense.cli.main import app
    runner.invoke(app, ["train-candidate"])

    assert captured["turnover_rank_band"] is None
    assert captured["settings"].price_source == "candles"  # untouched default
