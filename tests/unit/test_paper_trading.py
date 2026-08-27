"""Phase J2: paper trading engine tests. The engine mirrors evaluation.
backtest.simulate_portfolio's own loop shape (target weights -> no-trade-
band -> turnover cost -> drifted weights forward by realized return) so
paper NAV stays directly comparable to the number that actually cleared
the gate -- these tests pin that arithmetic down with a hand-computable
single-symbol scenario before trusting anything more complex."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from stocksense.data.store import Store
from stocksense.paper.account import close_account, get_account, list_accounts, open_paper_account
from stocksense.paper.engine import run_pending_rebalances
from stocksense.paper.scorecard import paper_scorecard, real_capital_readiness


@pytest.fixture()
def tmp_store(tmp_path):
    store = Store(tmp_path / "test.duckdb")
    yield store
    store.close()


def _register_model(store, model_id: str, horizon_bars: int = 2, metrics_json: str | None = None) -> None:
    store.con.execute(
        """
        INSERT INTO model_registry
            (model_id, model_type, horizon_bars, top_n, feature_schema_version,
             created_at, lifecycle_state, artifact_path)
        VALUES (?, 'cross_sectional_ranker', ?, 1, 'phase0-v1', ?, 'live', 'unused')
        """,
        [model_id, horizon_bars, datetime.now(timezone.utc)],
    )
    if metrics_json is not None:
        store.con.execute("UPDATE model_registry SET metrics_json = ? WHERE model_id = ?", [metrics_json, model_id])


def _write_bhavcopy(store, rows: list[tuple[str, date, float]]) -> None:
    df = pd.DataFrame([
        {"symbol": s, "series": "EQ", "date": d, "open": c, "high": c, "low": c, "close": c,
         "prev_close": c, "volume": 1000.0, "turnover_inr": c * 1000.0, "era": "udiff"}
        for s, d, c in rows
    ])
    store.write_bhavcopy_eq(df)


def _write_predictions(store, model_id: str, rows: list[tuple[str, str, float]], horizon_bars: int = 2) -> None:
    """rows: (as_of_date, symbol, score)"""
    df = pd.DataFrame([
        {
            "run_id": f"r-{d}-{s}", "symbol": s, "as_of_date": d, "horizon_bars": horizon_bars,
            "score": score, "rank": 1, "model_version": model_id, "horizon_type": "monthly",
            "predicted_return": score, "confidence": None, "feature_snapshot_hash": "x",
        }
        for d, s, score in rows
    ])
    store.write_predictions(df)


def test_open_paper_account_rejects_unknown_model(tmp_store) -> None:
    with pytest.raises(ValueError):
        open_paper_account(tmp_store, name="acc", model_id="nope", model_type="cross_sectional_ranker", horizon_bars=10, top_n=10)


def test_open_and_list_accounts(tmp_store) -> None:
    _register_model(tmp_store, "m1")
    account = open_paper_account(tmp_store, name="Acc A", model_id="m1", model_type="cross_sectional_ranker", horizon_bars=2, top_n=1)
    assert account.status == "active"
    df = list_accounts(tmp_store)
    assert len(df) == 1
    assert df.iloc[0]["account_id"] == account.account_id

    close_account(tmp_store, account.account_id)
    closed = get_account(tmp_store, account.account_id)
    assert closed.status == "closed"


def test_single_symbol_nav_compounds_correctly_across_rebalances(tmp_store) -> None:
    """AAA held continuously across 3 rebalance points (d0 -> d2 -> d4),
    +10% each period. Expected NAV = 1.10 * 1.10 = 1.21 (only the FIRST
    rebalance pays entry cost; subsequent 'hold' actions cost nothing,
    matching optimizer.rebalance's own no-trade-band classification)."""
    _register_model(tmp_store, "m1", horizon_bars=2)
    dates = [date(2026, 1, d) for d in (5, 6, 7, 8, 9)]  # d0..d4
    _write_bhavcopy(tmp_store, [
        ("AAA", dates[0], 100.0), ("AAA", dates[2], 110.0), ("AAA", dates[4], 121.0),
        ("BBB", dates[0], 50.0), ("BBB", dates[2], 50.0), ("BBB", dates[4], 50.0),
    ])
    _write_predictions(tmp_store, "m1", [
        (str(dates[0]), "AAA", 0.9), (str(dates[0]), "BBB", 0.1),
        (str(dates[1]), "AAA", 0.9), (str(dates[1]), "BBB", 0.1),
        (str(dates[2]), "AAA", 0.9), (str(dates[2]), "BBB", 0.1),
        (str(dates[3]), "AAA", 0.9), (str(dates[3]), "BBB", 0.1),
        (str(dates[4]), "AAA", 0.9), (str(dates[4]), "BBB", 0.1),
    ], horizon_bars=2)

    account = open_paper_account(tmp_store, name="Acc", model_id="m1", model_type="cross_sectional_ranker", horizon_bars=2, top_n=1)
    runs = run_pending_rebalances(tmp_store, account)

    assert len(runs) == 3  # rebalance points at index 0, 2, 4
    nav = tmp_store.read_paper_daily_nav(account.account_id)
    assert len(nav) == 3
    final_nav = float(nav["nav_units"].iloc[-1])
    # first rebalance pays entry cost (small, equity_delivery), so allow
    # a small tolerance below the frictionless 1.21 rather than an exact match
    assert 1.15 < final_nav <= 1.21
    assert nav["cum_return"].iloc[-1] == pytest.approx(final_nav - 1.0)


def test_run_pending_rebalances_is_idempotent(tmp_store) -> None:
    _register_model(tmp_store, "m1", horizon_bars=2)
    dates = [date(2026, 1, d) for d in (5, 6, 7)]
    _write_bhavcopy(tmp_store, [("AAA", dates[0], 100.0), ("AAA", dates[2], 110.0)])
    _write_predictions(tmp_store, "m1", [
        (str(dates[0]), "AAA", 0.9), (str(dates[1]), "AAA", 0.9), (str(dates[2]), "AAA", 0.9),
    ], horizon_bars=2)

    account = open_paper_account(tmp_store, name="Acc", model_id="m1", model_type="cross_sectional_ranker", horizon_bars=2, top_n=1)
    first = run_pending_rebalances(tmp_store, account)
    second = run_pending_rebalances(tmp_store, account)

    assert len(first) == 2
    assert len(second) == 0  # nothing new to process
    nav = tmp_store.read_paper_daily_nav(account.account_id)
    assert len(nav) == 2  # no duplicate rows


def test_order_missing_price_is_rejected_not_silently_dropped(tmp_store) -> None:
    _register_model(tmp_store, "m1", horizon_bars=2)
    dates = [date(2026, 1, d) for d in (5, 6, 7)]
    # AAA has NO bhavcopy row on the rebalance date -- price genuinely missing
    _write_predictions(tmp_store, "m1", [
        (str(dates[0]), "AAA", 0.9),
    ], horizon_bars=2)

    account = open_paper_account(tmp_store, name="Acc", model_id="m1", model_type="cross_sectional_ranker", horizon_bars=2, top_n=1)
    runs = run_pending_rebalances(tmp_store, account)

    orders = tmp_store.read_paper_orders(account.account_id)
    non_hold = orders[orders["action"] != "hold"]
    assert len(non_hold) == 1
    assert non_hold.iloc[0]["fill_status"] == "rejected"
    assert non_hold.iloc[0]["rejection_reason"] == "no_price_on_rebalance_date"


def test_scorecard_empty_account_reports_zero_not_crash(tmp_store) -> None:
    _register_model(tmp_store, "m1", horizon_bars=2)
    account = open_paper_account(tmp_store, name="Acc", model_id="m1", model_type="cross_sectional_ranker", horizon_bars=2, top_n=1)
    card = paper_scorecard(tmp_store, account.account_id)
    assert card["n_rebalances"] == 0


def test_paper_run_all_steps_every_active_account(tmp_path, monkeypatch) -> None:
    """The scheduled-task entry point: paper-run-all must pick up every
    ACTIVE account without the caller naming any account_id, and must
    skip closed ones."""
    db_path = tmp_path / "test.duckdb"
    monkeypatch.setenv("STOCKSENSE_DUCKDB_PATH", str(db_path))
    from typer.testing import CliRunner

    from stocksense.cli.main import app

    store = Store(db_path)
    _register_model(store, "m1", horizon_bars=2)
    dates = [date(2026, 1, d) for d in (5, 6, 7)]
    _write_bhavcopy(store, [("AAA", dates[0], 100.0), ("AAA", dates[2], 110.0)])
    _write_predictions(store, "m1", [
        (str(dates[0]), "AAA", 0.9), (str(dates[1]), "AAA", 0.9), (str(dates[2]), "AAA", 0.9),
    ], horizon_bars=2)
    active = open_paper_account(store, name="Active", model_id="m1", model_type="cross_sectional_ranker", horizon_bars=2, top_n=1)
    closed = open_paper_account(store, name="Closed", model_id="m1", model_type="cross_sectional_ranker", horizon_bars=2, top_n=1)
    close_account(store, closed.account_id)
    store.close()

    runner = CliRunner()
    result = runner.invoke(app, ["paper-run-all"])

    assert result.exit_code == 0, result.output
    assert active.account_id in result.output
    assert closed.account_id not in result.output

    store2 = Store(db_path)
    nav_active = store2.read_paper_daily_nav(active.account_id)
    nav_closed = store2.read_paper_daily_nav(closed.account_id)
    store2.close()
    assert len(nav_active) == 2
    assert len(nav_closed) == 0


def test_paper_run_all_reports_no_active_accounts(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "test.duckdb"
    monkeypatch.setenv("STOCKSENSE_DUCKDB_PATH", str(db_path))
    Store(db_path).close()
    from typer.testing import CliRunner

    from stocksense.cli.main import app

    result = CliRunner().invoke(app, ["paper-run-all"])
    assert result.exit_code == 0
    assert "No active paper accounts" in result.output


def test_readiness_reports_not_ready_with_reasons_when_insufficient(tmp_store) -> None:
    """A handful of rebalances must never yield ready=True -- the module
    docstring's promise: falling short means the record continues."""
    _register_model(tmp_store, "m1", horizon_bars=2, metrics_json='{"fold_alphas": [0.01, 0.02, -0.01, 0.015, 0.005]}')
    dates = [date(2026, 1, d) for d in (5, 6, 7)]
    _write_bhavcopy(tmp_store, [("AAA", dates[0], 100.0), ("AAA", dates[2], 110.0)])
    _write_predictions(tmp_store, "m1", [
        (str(dates[0]), "AAA", 0.9), (str(dates[1]), "AAA", 0.9), (str(dates[2]), "AAA", 0.9),
    ], horizon_bars=2)

    account = open_paper_account(tmp_store, name="Acc", model_id="m1", model_type="cross_sectional_ranker", horizon_bars=2, top_n=1)
    run_pending_rebalances(tmp_store, account)

    verdict = real_capital_readiness(tmp_store, account.account_id, "m1")
    assert verdict.ready is False
    assert "min_rebalances" in verdict.reasons_not_ready
