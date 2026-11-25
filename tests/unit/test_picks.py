"""M5.9: Daily Picks from gate-passing strategies (strategies/picks.py, /api/picks)."""

# ruff: noqa: E501
from __future__ import annotations

import json
import re
import sqlite3
from datetime import date, datetime

import polars as pl
import pytest
from fastapi.testclient import TestClient

from stocksense.api.app import create_app
from stocksense.core.clock import IST, FixedClock
from stocksense.core.paths import Paths
from stocksense.core.state import State
from stocksense.strategies import picks, scan
from stocksense.validation.trials import n_trials, register_trial

ASOF = date(2026, 11, 20)
NAMES = ["mean_rank_ic", "ic_tstat", "net_sharpe", "net_sharpe_2x_slip", "excess_tstat", "frac_positive_years", "dsr", "pbo", "max_drawdown"]  # fmt: skip


@pytest.fixture
def clock():
    return FixedClock(datetime(2026, 11, 20, 18, 0, tzinfo=IST))


@pytest.fixture
def state(tmp_data_root, clock):
    st = State(tmp_data_root, clock)
    st.migrate()
    return st


def eval_json(
    settings, key, trial, family="m-swing", *, passed=True, dsr=0.97, pbo=0.2, sharpe=1.5, trades=200
):
    lines = [
        {"number": i + 1, "name": n, "passed": passed or n != "dsr", "value": dsr if n == "dsr" else pbo if n == "pbo" else 1.0}
        for i, n in enumerate(NAMES)
    ]  # fmt: skip
    ev = {
        "strategy": key, "strat": key, "trial_id": trial, "family": family, "trades": trades,
        "net_sharpe": sharpe, "net_sharpe_by_capital": {"500000": sharpe},
        "verdict": "PASS" if passed else "FAIL", "gate": {"kind": "research-v2", "passed": passed, "lines": lines},
    }  # fmt: skip
    d = settings.paths.data_root / "reports" / "m" / "strategies" / trial
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval.json").write_text(json.dumps(ev), encoding="utf-8")


def write_scan(settings, rows):
    """rows: (sym, strat, p_in, stop, target, rs63)."""
    data = [
        {"sym": s, "strat": k, "state": "SIGNAL", "t_in": int(datetime(2026, 11, 21).timestamp()), "p_in": p,
         "stop": st, "target": tg, "rs63": rs, "dist_pct": 0.0, "risk_pct": 5.0, "trigger": p, "close": p}
        for s, k, p, st, tg, rs in rows
    ]  # fmt: skip
    df = pl.DataFrame(data, schema=scan.ROW_SCHEMA)
    out = scan.scan_path(settings, ASOF)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)


def build(settings, state, clock, capital=100_000.0, **kw):
    return picks.build_picks(settings, state, ASOF, capital, clock=clock, **kw)


def journal_rows(state, sql="SELECT trade_date, strat, sym, side, ref_price, stop, status FROM lab_journal"):
    con = state.connect()
    try:
        return [tuple(r) for r in con.execute(sql).fetchall()]
    finally:
        con.close()


def test_no_passing_strategy_gives_an_empty_gated_list_and_names_the_trial_count(tmp_data_root, state, clock):
    register_trial(state, "m-swing", "y_c2o_10", {"k": 1})
    eval_json(tmp_data_root, "ml_swing_vcp", "t-fail", passed=False)
    write_scan(tmp_data_root, [("AAA", "ml_swing_vcp", 100.0, 95.0, 115.0, 12.0)])
    out = build(tmp_data_root, state, clock)
    assert out["gated"] == [] and out["n_strategies_passing"] == 0
    assert (
        out["n_trials_total"] == n_trials(state) >= 31
        and f"n_trials_total = {out['n_trials_total']}" in out["note"]
    )
    assert out["note"].startswith("No strategy has passed the gate yet") and "not evidence" in out["note"]
    (ex,) = out["exploratory"]
    assert ex["sym"] == "AAA" and ex["label"] == picks.LABEL and ex["exploratory"] is True


def test_two_passing_strategies_on_one_stock_add_their_weights(tmp_data_root, state, clock):
    eval_json(tmp_data_root, "ml_swing_vcp", "t-a", dsr=0.96, trades=200, sharpe=1.2)
    eval_json(
        tmp_data_root, "ml_swing_breakout20", "t-b", dsr=0.98, trades=50, sharpe=1.8
    )  # weight 0.98 x 0.5
    write_scan(
        tmp_data_root,
        [
            ("AAA", "ml_swing_vcp", 100.0, 95.0, 115.0, 12.0),
            ("AAA", "ml_swing_breakout20", 100.0, 94.0, None, 12.0),
        ],
    )
    out = build(tmp_data_root, state, clock)
    (p,) = out["gated"]
    assert p["evidence"] == pytest.approx(0.96 * 1.0 + 0.98 * 0.5)
    assert set(p["strategies"]) == {"ml_swing_vcp", "ml_swing_breakout20"} and p["best_trial"] == "t-a"
    assert "2 gated strategies" in p["reason"] and p["stop"] == 95.0 and p["target"] == 115.0
    assert out["n_strategies_passing"] == 2 and out["exploratory"] == []


def test_a_failing_strategy_never_reaches_the_gated_list_and_an_exploratory_one_stays_apart(
    tmp_data_root, state, clock
):
    eval_json(tmp_data_root, "ml_swing_vcp", "t-ok")
    eval_json(tmp_data_root, "ml_swing_breakout20", "t-bad", passed=False)
    write_scan(
        tmp_data_root,
        [
            ("GOOD", "ml_swing_vcp", 100.0, 95.0, 110.0, 5.0),
            ("BAD", "ml_swing_breakout20", 50.0, 48.0, 56.0, 30.0),
        ],
    )
    out = build(tmp_data_root, state, clock)
    assert [p["sym"] for p in out["gated"]] == ["GOOD"]
    assert [e["sym"] for e in out["exploratory"]] == ["BAD"] and out["exploratory"][0]["label"] == picks.LABEL
    assert "label" not in out["gated"][0]


def test_only_all_nine_v2_lines_count(tmp_data_root, state, clock):
    eval_json(tmp_data_root, "ml_swing_vcp", "t-eight", passed=False)  # the DSR line fails
    p = tmp_data_root.paths.data_root / "reports/m/strategies/t-eight/eval.json"
    ev = json.loads(p.read_text("utf-8"))
    ev["verdict"] = "PASS"  # a stale verdict does not matter: the nine lines are read
    p.write_text(json.dumps(ev), "utf-8")
    assert picks.passing_strategies(tmp_data_root) == {}
    for ln in ev["gate"]["lines"]:
        ln["passed"] = True
    ev["gate"]["kind"] = "research-v1"
    p.write_text(json.dumps(ev), "utf-8")
    assert picks.passing_strategies(tmp_data_root) == {}  # a v1 report is not a v2 pass
    ev["gate"]["kind"] = "research-v2"
    p.write_text(json.dumps(ev), "utf-8")
    assert set(picks.passing_strategies(tmp_data_root)) == {"ml_swing_vcp"}


def test_sizing_takes_the_smaller_of_risk_and_slice():
    assert picks.size_position(100_000.0, 0.01, 10, 100.0, 95.0) == (100, 10_000.0, 500.0)  # min(200, 100)
    assert (
        picks.size_position(1_000_000.0, 0.01, 10, 100.0, 95.0)[0] == 1000
    )  # slice 100,000/100 vs risk 2,000
    assert (
        picks.size_position(100_000.0, 0.01, 10, 100.0, 50.0)[0] == 20
    )  # risk binds: 1,000 / 50 vs slice 100
    assert picks.size_position(100_000.0, 0.01, 10, 100.0, None)[0] == 100  # no stop: the slice only
    assert picks.size_position(None, 0.01, 10, 100.0, 95.0) == (0, 0.0, 0.0)
    assert picks.size_position(0.0, 0.01, 10, 100.0, 95.0) == (0, 0.0, 0.0)


def test_capital_in_the_signature_sizes_the_pick_and_none_says_so(tmp_data_root, state, clock):
    eval_json(tmp_data_root, "ml_swing_vcp", "t-ok")
    write_scan(tmp_data_root, [("AAA", "ml_swing_vcp", 100.0, 95.0, 110.0, 5.0)])
    sized = build(tmp_data_root, state, clock, 100_000.0, persist=False)
    assert sized["gated"][0]["qty"] == 100 and sized["gated"][0]["notional_inr"] == 10_000.0
    none = build(tmp_data_root, state, clock, None, persist=False)
    assert none["gated"][0]["qty"] == 0 and "Capital is unavailable" in none["note"]


def test_the_file_is_written_once_per_date_unless_forced(tmp_data_root, state, clock):
    eval_json(tmp_data_root, "ml_swing_vcp", "t-ok")
    write_scan(tmp_data_root, [("AAA", "ml_swing_vcp", 100.0, 95.0, 110.0, 5.0)])
    first = build(tmp_data_root, state, clock, 100_000.0)
    path = picks.picks_path(tmp_data_root, ASOF)
    assert path.is_file() and json.loads(path.read_text("utf-8")) == first
    mtime = path.stat().st_mtime_ns
    again = build(tmp_data_root, state, clock, 400_000.0)  # a different capital does not rewrite the day
    assert again == first and path.stat().st_mtime_ns == mtime
    forced = build(tmp_data_root, state, clock, 400_000.0, force=True)
    assert forced["gated"][0]["qty"] == 400 and forced != first  # slice 40,000 / 100; risk would allow 800


def test_every_gated_pick_is_journalled_once(tmp_data_root, state, clock):
    eval_json(tmp_data_root, "ml_swing_vcp", "t-ok")
    write_scan(tmp_data_root, [("AAA", "ml_swing_vcp", 100.0, 95.0, 110.0, 5.0)])
    build(tmp_data_root, state, clock)
    build(tmp_data_root, state, clock, force=True)
    assert journal_rows(state) == [("2026-11-20", "picks:t-ok", "AAA", 1, 100.0, 95.0, "planned")]


def test_a_cross_sectional_strategy_fires_on_its_top_names_without_a_stop(tmp_data_root, state, clock):
    eval_json(tmp_data_root, "k151_mom_cum", "t-xs", family="m-xs", dsr=0.99, trades=500)
    write_scan(tmp_data_root, [])
    out = build(
        tmp_data_root, state, clock, 100_000.0,
        xs_scores=lambda k, d: ["XXX", "YYY"], closes=lambda d, s: {"XXX": 200.0, "YYY": 50.0},
    )  # fmt: skip
    got = {p["sym"]: p for p in out["gated"]}
    assert (
        set(got) == {"XXX", "YYY"}
        and got["XXX"]["stop"] is None
        and got["XXX"]["strategies"] == ["k151_mom_cum"]
    )
    assert got["XXX"]["qty"] == 50 and got["YYY"]["qty"] == 200  # equal slices of 10,000
    assert journal_rows(state, "SELECT count(*) FROM lab_journal") == [(0,)]  # no stop, no journal row


def test_picks_step_is_idempotent_by_date_and_skips_a_missing_scan(tmp_data_root, state, clock):
    eval_json(tmp_data_root, "ml_swing_vcp", "t-ok")
    assert (
        picks.picks_step(tmp_data_root, state, clock, ASOF, 100_000.0)["status"] == "skipped"
    )  # no scan yet
    write_scan(tmp_data_root, [("AAA", "ml_swing_vcp", 100.0, 95.0, 110.0, 5.0)])
    res = picks.picks_step(tmp_data_root, state, clock, ASOF, 100_000.0)
    assert res == {"status": "done", "asof": "2026-11-20", "n_gated": 1}
    assert state.job_status("picks", ASOF) == "done"
    assert picks.picks_step(tmp_data_root, state, clock, ASOF, 100_000.0)["status"] == "skipped"


@pytest.fixture
def client(tmp_data_root, tmp_path, clock, state):
    app = create_app(tmp_data_root, start_feeds=False, clock=clock, web_out=tmp_path / "none")
    with TestClient(app) as c:
        yield c


def _keys(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from _keys(v)
    elif isinstance(x, list):
        for v in x:
            yield from _keys(v)


BANNED = re.compile(r"(capital|funds|client|token)", re.I)


def test_api_has_no_capital_funds_client_or_token_keys(client, tmp_data_root, state, clock):
    eval_json(tmp_data_root, "ml_swing_vcp", "t-ok")
    write_scan(tmp_data_root, [("AAA", "ml_swing_vcp", 100.0, 95.0, 110.0, 5.0)])
    build(tmp_data_root, state, clock, 250_000.0)
    r = client.get("/api/picks")
    assert r.status_code == 200
    j = r.json()
    assert (
        j["asof"] == "2026-11-20"
        and j["gated"][0]["qty"] == 250
        and j["gated"][0]["notional_inr"] == 25_000.0
    )
    assert not [k for k in _keys(j) if BANNED.search(k)]
    h = client.get("/api/picks/history?days=30").json()
    assert h["paper"] is True and h["rows"][0]["sym"] == "AAA" and h["rows"][0]["status"] == "planned"
    assert not [k for k in _keys(h) if BANNED.search(k)]


def test_api_previews_without_storing_and_reports_errors(client, tmp_data_root):
    assert client.get("/api/picks").json() == {"error": "run: stocksense lab scan"}
    assert client.get("/api/picks?asof=garbage").status_code == 400
    write_scan(tmp_data_root, [("AAA", "ml_swing_vcp", 100.0, 95.0, 110.0, 5.0)])
    j = client.get("/api/picks?asof=2026-11-20").json()
    assert j["gated"] == [] and j["exploratory"][0]["label"] == picks.LABEL
    assert not picks.picks_path(tmp_data_root, ASOF).exists()  # a preview is never stored
    assert client.post("/api/picks").status_code == 405


def test_history_reports_the_realised_outcome_of_a_closed_pick(tmp_data_root, state, clock):
    eval_json(tmp_data_root, "ml_swing_vcp", "t-ok")
    write_scan(tmp_data_root, [("AAA", "ml_swing_vcp", 100.0, 95.0, 110.0, 5.0)])
    build(tmp_data_root, state, clock)
    con = sqlite3.connect(Paths(tmp_data_root).sqlite)
    con.execute("UPDATE lab_journal SET status='closed', closed_ts='2026-11-27', close_price=110.0")
    con.commit()
    con.close()
    (row,) = picks.history(tmp_data_root, clock, 30)
    assert row["status"] == "closed" and row["close_price"] == 110.0
    assert 9.0 < row["net_pct"] < 10.0 and row["r_mult"] == pytest.approx(row["net_pct"] / 5.0, rel=1e-6)
    assert picks.history(tmp_data_root, FixedClock(datetime(2027, 3, 1, tzinfo=IST)), 30) == []
