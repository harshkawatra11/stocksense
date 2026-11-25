"""M5.0: the pre-registered strategy library (config/strategy_library.toml), frozen by a SHA pin."""

from __future__ import annotations

import hashlib
import tomllib
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from stocksense.strategies import base, exits
from stocksense.strategies import library as lib

REPO = Path(__file__).resolve().parents[2]
LIBRARY = REPO / "config" / "strategy_library.toml"

# SHA-256 of config/strategy_library.toml with CRLF normalized to LF. Pre-registered on 2026-11-25, before any
# gated strategy result existed (PLAN-M Part 0 rule 13). If this fails, someone edited it: revert and put
# a new strategy in config/strategy_library_<YYYYMMDD>.toml with its own pinned SHA instead.
LIBRARY_SHA = "e09fdc982cabe72b9ed7ec187637891b78b13e1e4989cdfea3c80a6b488e6db7"

VALID_EXITS = {*exits.EXITS_1D, *exits.EXITS_INTRA, *exits.EXITS_EXTRA, "expiry"}


@pytest.fixture(scope="module")
def doc() -> dict:
    with LIBRARY.open("rb") as f:
        return tomllib.load(f)


def library_hash(path: Path) -> str:
    """SHA-256 hex of the file after normalizing \\r\\n to \\n."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_library_file_is_frozen():
    assert library_hash(LIBRARY) == LIBRARY_SHA


def test_hash_checker_ignores_line_endings_and_catches_edits(tmp_path):
    f = tmp_path / "lib.toml"
    f.write_bytes(LIBRARY.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert library_hash(f) == LIBRARY_SHA
    f.write_bytes(f.read_bytes().replace(b"slots = 10", b"slots = 12"))
    assert library_hash(f) != LIBRARY_SHA


def test_protocol_matches_the_card(doc):
    p = doc["protocol"]
    assert p["research_end"] == date(2025, 9, 30)
    assert p["capital_grid"] == [100000, 200000, 500000, 1000000] and p["reference_capital"] == 500000
    assert p["slots"] == 10 and p["gate_version"] == 2 and p["priority"] == "rs63"
    assert (p["ic_label_daily"], p["ic_label_intraday"]) == ("y_c2o_10", "y_oc")
    assert p["intraday_trends"] == ["D", "I", "DI"]


def test_every_registered_strategy_is_listed_and_nothing_else(doc):
    reg = base.load_library()
    assert set(doc["strategy"]) == set(reg)
    for key, s in doc["strategy"].items():
        spec = reg[key]
        assert (s["family"], s["timeframe"], s["kind"], s["source"]) == (
            spec.family,
            spec.timeframe,
            spec.kind,
            spec.source,
        )
        assert s["status"] == spec.status and tuple(s["exits"]) == spec.exits
        assert s["default_exit"] == spec.default_exit and s["params"] == {}
        assert s["eval"] == lib.eval_kind(spec) and s["target"] == lib.target_of(spec)


def test_every_exit_exists(doc):
    bad = {(k, e) for k, s in doc["strategy"].items() for e in s["exits"] if e not in VALID_EXITS}
    assert not bad
    for k, s in doc["strategy"].items():
        if s["exits"]:
            assert s["default_exit"] in s["exits"], k


def test_market_lab_rules_for_exits(doc):
    for k, s in doc["strategy"].items():
        if s["source"] == "marketlab_swing":
            assert ("mr5" in s["exits"]) == (s["kind"] == "mr"), k
        if s["source"] == "marketlab_intraday":
            assert s["exits"] == list(lib.INTRA_EXITS) and s["default_exit"] == "rr2", k


def test_each_family_has_ten_trials_for_pbo_and_counts_are_printed(doc, capsys):
    trials: Counter[str] = Counter()
    for s in doc["strategy"].values():
        if s["status"] != "ready" or s["eval"] == lib.KIND_INFO:
            continue
        trials[s["family"]] += len(s.get("trends", [None]))
    with capsys.disabled():
        print("\ntrials per family:", dict(sorted(trials.items())), "total", sum(trials.values()))
    assert set(trials) == {"m-swing", "m-intra", "m-xs", "m-index", "m-opt"}
    assert all(n >= 10 for n in trials.values()), trials  # gate line 8 fails below 10 finished trials


def test_await_data_and_informational_entries_are_listed_but_not_run(doc):
    waiting = {k for k, s in doc["strategy"].items() if s["status"] == "await_data"}
    assert len(waiting) == 7
    assert {"k151_nb_news", "tier_b_smt", "flow_gex_regime", "flow_preopen_imb"} <= waiting
    info = {k for k, s in doc["strategy"].items() if s["eval"] == lib.KIND_INFO}
    assert info == {"k151_cash_carry", "k151_gold_diversify"}


def test_option_trials_are_returns_on_notional_against_cash(doc):
    opt = {k: s for k, s in doc["strategy"].items() if s["family"] == "m-opt"}
    assert len(opt) == 118 and "k151_dispersion" in opt
    assert all(s["benchmark"] == "cash" and s["eval"] == lib.KIND_OPTIONS for s in opt.values())
    assert all(s["benchmark"] != "cash" for k, s in doc["strategy"].items() if s["family"] != "m-opt")


def test_intraday_per_stock_strategies_get_one_trial_per_trend_filter(doc):
    intra = {k: s for k, s in doc["strategy"].items() if s["eval"] == lib.KIND_INTRA}
    assert intra and all(s["trends"] == ["D", "I", "DI"] for s in intra.values())
    assert all(s["target"] == "y_oc" for s in intra.values())
    assert not [k for k, s in doc["strategy"].items() if "trends" in s and s["eval"] != lib.KIND_INTRA]


def test_every_151_section_names_its_paper_section_and_equation(doc):
    rows = {k: s for k, s in doc["strategy"].items() if s["source"] == "k151"}
    assert len(rows) > 100
    assert all("paper_section" in s and "paper_eq" in s for s in rows.values())
    assert all(s["paper_section"] for s in rows.values())
    assert (
        rows["k151_vrp_straddle"]["paper_section"] == "7.4" and rows["k151_vrp_straddle"]["paper_eq"] == "96"
    )


def test_library_section_reads_one_strategy():
    s = lib.library_section("ml_swing_vcp")
    assert s["default_exit"] == "rr3_40" and s["eval"] == "swing" and s["target"] == "y_c2o_10"
    assert lib.research_end() == date(2025, 9, 30)
    with pytest.raises(KeyError):
        lib.library_section("no_such_strategy")
