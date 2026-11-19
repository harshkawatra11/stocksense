"""The Gate G1 research run (PLAN T9.5): finish the model trials, run the portfolio grid, pick one candidate
per family, run the research gate on each, fit the candidates in production mode and keep a summary.

Everything is idempotent: trials are registered by spec hash, a finished trial's result files are reused and
a stored production fit is returned as it is. No threshold or grid point is ever changed here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stocksense.backtest.metrics import perf_stats
from stocksense.core.calendar import TradingCalendar
from stocksense.core.config import Settings
from stocksense.core.log import get_logger
from stocksense.core.state import State
from stocksense.models import store
from stocksense.models.base import ModelSpec
from stocksense.validation import g1
from stocksense.validation.g1 import (
    FAMILY_REF,
    G1Config,
    G1Runner,
    TrialPlan,
    TrialResult,
)
from stocksense.validation.gate import (
    CandidateResult,
    GateReport,
    save_candidate_result,
    validate_candidate,
)
from stocksense.validation.trials import n_trials, register_trial
from stocksense.validation.walkforward import backtest_walk_forward, run_walk_forward

log = get_logger(__name__)

N_FOLDS = 12


@dataclass
class G1Outcome:
    """What a run produced: the portfolio trials, the references, the candidates and their gate reports."""

    plans: list[TrialPlan] = field(default_factory=list)
    results: dict[str, TrialResult] = field(default_factory=dict)
    candidates: dict[str, str] = field(default_factory=dict)  # "s1"/"s2" -> candidate (trial) id
    reports: dict[str, GateReport] = field(default_factory=dict)
    production: dict[str, list[str]] = field(default_factory=dict)  # "s1"/"s2" -> production model ids
    notes: list[str] = field(default_factory=list)


# ---- step 0: finish the model trials ---------------------------------------------------------------


def folds_done(settings: Settings, trial_id: str) -> int:
    from stocksense.core.paths import Paths

    p = Paths(settings).preds / trial_id / "folds_done.json"
    return len(json.loads(p.read_text(encoding="utf-8"))) if p.is_file() else 0


def finish_model_trials(
    settings: Settings,
    state: State,
    calendar: TradingCalendar,
    specs: dict[str, ModelSpec],
    *,
    n_folds: int = N_FOLDS,
    k: int = 10,
) -> dict[str, str]:
    """Give every model trial whose walk-forward is complete (all folds stored) its top-k backtest, which
    records its metrics and returns and marks it done. A trial with missing folds is left alone (it is run
    by `stocksense train`). Returns {spec name: trial id or 'incomplete'}."""
    out: dict[str, str] = {}
    for name, spec in sorted(specs.items()):
        tid = register_trial(state, spec.family, spec.target, spec.to_dict())
        if folds_done(settings, tid) < n_folds:
            out[name] = "incomplete"
            continue
        con = state.connect()
        try:
            status = con.execute("SELECT status FROM trials WHERE trial_id=?", (tid,)).fetchone()[0]
        finally:
            con.close()
        if status != "done":
            res = run_walk_forward(spec, settings, state, calendar=calendar)  # every fold is stored: no fit
            backtest_walk_forward(res, spec, settings, state, calendar=calendar, k=k)
            log.info("model_trial_finished", trial=tid)
        out[name] = tid
    return out


# ---- step 1-3: the portfolio trials and references ----------------------------------------------------


def _best(plans: Sequence[TrialPlan], results: dict[str, TrialResult]) -> TrialPlan | None:
    ok = [p for p in plans if p.trial_id in results and math.isfinite(results[p.trial_id].ref_sharpe)]
    return max(ok, key=lambda p: (results[p.trial_id].ref_sharpe, p.trial_id)) if ok else None


def reference_plans(
    best: TrialPlan, factors: Sequence[tuple[str, float]] = g1.SINGLE_FACTORS
) -> list[TrialPlan]:
    """The same-table references of T9.5 step 3 built on the best S1 trial: the same S1 policy with entry at
    the open and the two single-factor portfolios (ew500 and NIFTY 500 come from the benchmark series)."""
    sp = best.spec
    plans = [TrialPlan("ref-open", FAMILY_REF, best.target, {**sp, "entry": "open"}, "S1 entry at the open")]
    for feat, sign in factors:
        plans.append(
            TrialPlan(
                "ref-factor",
                FAMILY_REF,
                best.target,
                {
                    "g1": g1.G1_VERSION,
                    "strategy": "s1-ref",
                    "factor": feat,
                    "sign": sign,
                    "target": best.target,
                    "k_max": sp["k_max"],
                    "overlay": sp["overlay"],
                    "start": sp["start"],
                    "end": sp["end"],
                    "capitals": sp["capitals"],
                    "reference_capital": sp["reference_capital"],
                    "stress": sp["stress"],
                    **(
                        {"auto_resume_sessions": sp["auto_resume_sessions"]}
                        if "auto_resume_sessions" in sp
                        else {}
                    ),
                },
                f"single factor {'-' if sign < 0 else ''}{feat}",
            )  # fmt: skip
        )
    return plans


def benchmark_rows(runner: G1Runner) -> list[dict[str, Any]]:
    """ew500 and NIFTY 500 over the run window (no cost, no slippage: they are index series)."""
    bm = runner.world.benchmark
    out = []
    for col, label in (
        ("ew500_ret", "ew500 (equal-weight u500, no cost)"),
        ("nifty500_ret", "NIFTY 500 (index)"),
    ):
        if col not in bm.columns:
            continue
        d = bm.select("date", col).drop_nulls(col).sort("date")
        if d.height < 30:
            out.append({"label": label, "available": False})
            continue
        s = perf_stats(d[col].to_numpy(), None, dates=d["date"].to_list())
        out.append({"label": label, "available": True, "sharpe": s["sharpe"], "cagr": s["cagr"],
                    "max_drawdown": s["max_drawdown"], "n_days": d.height})  # fmt: skip
    return out


def run_portfolio_trials(
    runner: G1Runner,
    state: State,
    out: G1Outcome,
    factors: Sequence[tuple[str, float]] = g1.SINGLE_FACTORS,
) -> None:
    s1 = {t: g1.surviving_models(state, "s1", t) for t in g1.S1_TARGETS}
    s2 = g1.surviving_models(state, "s2", g1.S2_TARGET)
    for t, ms in s1.items():
        if not ms:
            out.notes.append(f"no surviving S1 model for {t}: no portfolio trials")
    if not s2:
        out.notes.append("no surviving S2 model: no S2 trials")
    plans = runner.register(g1.plan_grid(runner.cfg, s1, s2))
    for p in plans:
        out.results[p.trial_id] = runner.run_trial(p)  # type: ignore[index]
        log.info("g1_trial_done", trial=p.trial_id, label=p.label)
    out.plans = list(plans)
    best_s1 = _best([p for p in plans if p.kind == "s1"], out.results)
    if best_s1 is None:
        return
    extra = runner.register(
        [
            TrialPlan(
                "s1-meta",
                best_s1.family,
                best_s1.target,
                {**best_s1.spec, "meta": "walk-forward", "base_trial": best_s1.trial_id},
                f"{best_s1.label} + meta-label",
            ),
            *reference_plans(best_s1, factors),
        ]
    )
    for p in extra:
        out.results[p.trial_id] = runner.run_trial(p)  # type: ignore[index]
        log.info("g1_trial_done", trial=p.trial_id, label=p.label)
    out.plans += extra


# ---- step 4: candidates and the research gate ---------------------------------------------------------


def pick_candidates(out: G1Outcome) -> dict[str, TrialPlan]:
    """Per family the trial with the highest net Sharpe at the reference capital. The meta-label variant is
    eligible only through that same rule (the card keeps it only if it improves the Sharpe)."""
    pick: dict[str, TrialPlan] = {}
    s1 = _best([p for p in out.plans if p.kind in ("s1", "s1-meta")], out.results)
    s2 = _best([p for p in out.plans if p.kind == "s2"], out.results)
    if s1 is not None:
        pick["s1"] = s1
    if s2 is not None:
        pick["s2"] = s2
    return pick


def make_candidate(runner: G1Runner, plan: TrialPlan, result: TrialResult) -> CandidateResult:
    sp = plan.spec
    models = sp["models"]
    blend = runner.blended(plan.target, models)
    daily = result.daily().select("date", "net", "net_2x", "benchmark", "gross")
    return CandidateResult(
        candidate_id=str(plan.trial_id), family=plan.family, target=plan.target, daily=daily,
        ic={"mean_ic": float(blend.ic["mean_ic"]), "ic_tstat": float(blend.ic["ic_tstat"])},
        trial_ids=tuple(models),
    )  # fmt: skip


def run_gate(runner: G1Runner, settings: Settings, state: State, out: G1Outcome) -> None:
    for fam, plan in pick_candidates(out).items():
        cand = make_candidate(runner, plan, out.results[str(plan.trial_id)])
        save_candidate_result(cand, settings)
        out.candidates[fam] = cand.candidate_id
        out.reports[fam] = validate_candidate(cand.candidate_id, settings, state)


def _trial_spec(state: State, trial_id: str) -> dict[str, Any]:
    con = state.connect()
    try:
        row = con.execute("SELECT spec_json FROM trials WHERE trial_id=?", (trial_id,)).fetchone()
    finally:
        con.close()
    return dict(json.loads(row[0]))


# ---- step 6: production fits ---------------------------------------------------------------------------


def fit_candidates(
    settings: Settings,
    state: State,
    calendar: TradingCalendar,
    out: G1Outcome,
    plans: dict[str, TrialPlan],
    fit: Callable[..., str] | None = None,
) -> None:
    """Fit each candidate's models on all research data (as of the last research session; the lockbox is not
    read) and store them as `shadow`: the research gate decides `champion` (S2 is at most `shadow`)."""
    from stocksense.learn.nightly import fit_production, last_research_session

    do_fit = fit or fit_production
    as_of = last_research_session(settings, calendar)
    for fam, plan in plans.items():
        ids = []
        for mid in plan.spec["models"]:
            spec = ModelSpec.from_dict(_trial_spec(state, mid))
            note = f"G1 candidate {plan.trial_id}: gate {'PASS' if out.reports[fam].passed else 'FAIL'}"
            model_id = do_fit(spec, settings, state, as_of, calendar=calendar, note=note)
            store.set_status(state, model_id, "shadow", note)
            ids.append(model_id)
        out.production[fam] = ids


# ---- the run -----------------------------------------------------------------------------------------


def run_g1(
    settings: Settings,
    state: State,
    calendar: TradingCalendar,
    cfg: G1Config | None = None,
    *,
    specs: dict[str, ModelSpec] | None = None,
    production: bool = True,
    runner: G1Runner | None = None,
    fit: Callable[..., str] | None = None,
    factors: Sequence[tuple[str, float]] = g1.SINGLE_FACTORS,
) -> G1Outcome:
    """The whole research run on whatever model trials are finished. Returns the outcome (also written to
    data_store/reports/g1/outcome.json by `write_outcome`)."""
    out = G1Outcome()
    if specs:
        status = finish_model_trials(settings, state, calendar, specs)
        out.notes += [f"model trial {n}: {t}" for n, t in status.items() if t == "incomplete"]
    runner = runner or G1Runner(settings, state, calendar, cfg)
    run_portfolio_trials(runner, state, out, factors)
    g1._write_json(g1.g1_dir(settings) / "benchmarks.json", benchmark_rows(runner))
    run_gate(runner, settings, state, out)
    if production:
        fit_candidates(settings, state, calendar, out, pick_candidates(out), fit)
    write_outcome(settings, state, out, runner.cfg.s1_auto_resume)
    return out


def write_outcome(settings: Settings, state: State, out: G1Outcome, auto_resume: int | None = None) -> Path:
    """outcome.json; the auto-resume variant writes its own file and never touches the original run's."""
    d = g1.g1_dir(settings)
    d.mkdir(parents=True, exist_ok=True)
    doc = {
        "trials_total": n_trials(state),
        "candidates": out.candidates,
        "reports": {k: r.to_dict() for k, r in out.reports.items()},
        "production": out.production,
        "notes": out.notes,
        "plans": [
            {"trial_id": p.trial_id, "kind": p.kind, "label": p.label, "target": p.target} for p in out.plans
        ],
    }
    path = d / ("outcome.json" if auto_resume is None else "outcome_s1_auto_resume.json")
    g1._write_json(path, doc)
    return path
