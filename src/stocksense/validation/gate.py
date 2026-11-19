from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from stocksense.backtest.metrics import ic_stats, perf_stats
from stocksense.core.clock import IST, Clock, SystemClock
from stocksense.core.config import Settings
from stocksense.core.paths import Paths
from stocksense.core.state import State
from stocksense.validation.dsr import deflated_sharpe
from stocksense.validation.lockbox import open_lockbox as _open_lockbox
from stocksense.validation.pbo import pbo_cscv
from stocksense.validation.trials import n_trials, sharpe_variance_across_trials

MIN_PBO_TRIALS = 10

LockboxRunner = Callable[[str, Settings], np.ndarray]


@dataclass(frozen=True)
class CandidateResult:
    """Everything the research gate needs about one candidate (all returns are daily fractions).

    daily: columns date (Date), net (net return at the reference capital, all costs and slippage), net_2x
    (the same with every slippage component doubled) and optionally benchmark (ew500_ret; absent or null means
    a zero benchmark, as for S2). The IC statistics are either precomputed (`ic` with mean_ic and ic_tstat) or
    computed from out-of-fold `preds` (date, score and `label_col`)."""

    candidate_id: str
    family: str
    target: str
    daily: pl.DataFrame
    ic: dict[str, float] | None = None
    preds: pl.DataFrame | None = None
    label_col: str | None = None
    trial_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateLine:
    """One line of a gate: its number, name, outcome, the measured value and the bar it was held to."""

    number: int
    name: str
    passed: bool
    value: float | None
    threshold: float
    op: str
    text: str


@dataclass(frozen=True)
class GateReport:
    """A gate result: `kind` is 'research-v1', 'research-v2' or 'lockbox'; `lockbox` holds the lockbox outcome
    if requested."""

    candidate_id: str
    family: str
    kind: str
    lines: tuple[GateLine, ...]
    lockbox: dict[str, Any] | None = None
    generated_at: str | None = None

    @property
    def passed(self) -> bool:
        return all(ln.passed for ln in self.lines)

    def table(self) -> str:
        """A printable table, one row per line: number, PASS/FAIL, name and the number shown."""
        rows = [f"{self.kind} gate for {self.candidate_id}: {'PASS' if self.passed else 'FAIL'}"]
        rows += [f"  {ln.number}. {'PASS' if ln.passed else 'FAIL'}  {ln.text}" for ln in self.lines]
        if self.lockbox is not None:
            rows.append(f"  lockbox: {self.lockbox.get('status')} {self.lockbox.get('reason', '')}".rstrip())
        return "\n".join(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "kind": self.kind,
            "passed": self.passed,
            "generated_at": self.generated_at,
            "lines": [
                {
                    "number": ln.number, "name": ln.name, "passed": ln.passed, "value": _num(ln.value),
                    "threshold": ln.threshold, "op": ln.op, "text": ln.text,
                }
                for ln in self.lines
            ],
            "lockbox": self.lockbox,
        }  # fmt: skip


def _num(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def _line(
    number: int, name: str, value: float | None, threshold: float, op: str, extra: str = ""
) -> GateLine:
    ok = (
        value is not None
        and math.isfinite(value)
        and (value >= threshold if op == ">=" else value <= threshold)
    )
    shown = "nan" if value is None or not math.isfinite(value) else f"{value:.4f}"
    return GateLine(number, name, ok, value, threshold, op, f"{name} {shown} {op} {threshold:g}{extra}")


# ---- research gate -------------------------------------------------------------------------------


def _family_returns(state: State, family: str) -> np.ndarray | None:
    """Daily returns of the family's finished trials with a returns file, on common dates (T x n)."""
    con = state.connect()
    try:
        rows = con.execute(
            "SELECT trial_id, returns_path FROM trials WHERE family=? AND status='done' "
            "AND returns_path IS NOT NULL ORDER BY trial_id",
            (family,),
        ).fetchall()
    finally:
        con.close()
    merged: pl.DataFrame | None = None
    for k, (_tid, path) in enumerate(rows):
        p = Path(path)
        if not p.is_file():
            continue
        df = pl.read_parquet(p).select(pl.col("date").cast(pl.Date), pl.col("ret").alias(f"r{k}"))
        merged = df if merged is None else merged.join(df, on="date", how="inner")
    if merged is None or merged.width < 2:
        return None
    return merged.sort("date").drop("date").to_numpy()


def _ic_values(c: CandidateResult) -> tuple[float, float]:
    if c.ic is not None:
        return float(c.ic["mean_ic"]), float(c.ic["ic_tstat"])
    if c.preds is None or c.label_col is None:
        raise ValueError("CandidateResult needs `ic` or `preds` with `label_col`")
    s = ic_stats(c.preds, c.label_col)
    return float(s["mean_ic"]), float(s["ic_tstat"])


# Families whose benchmark is zero by construction (the auction book is not a long-only equity exposure).
ZERO_BENCHMARK_FAMILIES = ("g1-s2", "g2-s2strict")


def exposure_matched_benchmark(gross: np.ndarray, benchmark: np.ndarray) -> np.ndarray:
    """Gate v2 benchmark: b_t = gross_{t-1} * benchmark_t, the previous day's gross exposure of the book
    times the day's equal-weight market return; the first day uses exposure 0."""
    g = np.asarray(gross, dtype=float)
    prev = np.concatenate([[0.0], g[:-1]]) if len(g) else g
    return prev * np.asarray(benchmark, dtype=float)


def _as_ist(ts: datetime) -> datetime:
    return ts.replace(tzinfo=IST) if ts.tzinfo is None else ts


def _all_trials_since(state: State, trial_ids: tuple[str, ...], effective_from: datetime) -> bool:
    """True if there is at least one trial id and every one was created at or after `effective_from`."""
    if not trial_ids:
        return False
    con = state.connect()
    try:
        marks = ",".join("?" for _ in trial_ids)
        rows = con.execute(
            f"SELECT trial_id, created_at FROM trials WHERE trial_id IN ({marks})", tuple(trial_ids)
        ).fetchall()
    finally:
        con.close()
    if len(rows) != len(set(trial_ids)):
        return False
    return all(
        r[1] is not None and _as_ist(datetime.fromisoformat(r[1])) >= _as_ist(effective_from) for r in rows
    )


def select_gate_version(candidate: CandidateResult, settings: Settings, state: State) -> int:
    """2 when gate v2 is configured and every trial of the candidate was registered at or after its
    `effective_from`; otherwise 1 (earlier trials stay judged by v1)."""
    g2 = settings.gate_v2
    if g2 is None:
        return 1
    return 2 if _all_trials_since(state, candidate.trial_ids, g2.effective_from) else 1


def evaluate_research_gate(
    candidate: CandidateResult, settings: Settings, state: State, version: int | None = None
) -> GateReport:
    """The nine research lines of T6.6. Version 1 reads config/gate.toml (never altered); version 2 reads
    config/gate_v2.toml: DSR variance within the candidate's family and the excess t against an
    exposure-matched benchmark (G2-B1). `version=None` selects by the candidates' trial creation times."""
    ver = select_gate_version(candidate, settings, state) if version is None else version
    if ver not in (1, 2):
        raise ValueError(f"unknown gate version {ver}")
    if ver == 2 and settings.gate_v2 is None:
        raise ValueError("gate v2 is not configured (config/gate_v2.toml missing)")
    g = settings.gate.research if ver == 1 else settings.gate_v2.research  # type: ignore[union-attr]
    daily = candidate.daily.sort("date")
    dates = daily["date"].to_list()
    net = daily["net"].to_numpy().astype(float)
    net2 = daily["net_2x"].to_numpy().astype(float)
    bench = (
        daily["benchmark"].fill_null(0.0).to_numpy().astype(float) if "benchmark" in daily.columns else None
    )
    if ver == 2 and g.excess_benchmark == "exposure_matched":  # type: ignore[union-attr]
        if candidate.family.startswith(ZERO_BENCHMARK_FAMILIES):
            bench = None
        else:
            if "gross" not in daily.columns:
                raise ValueError("gate v2 needs gross")
            gross = daily["gross"].fill_null(0.0).to_numpy().astype(float)
            bench = exposure_matched_benchmark(gross, np.zeros(len(net)) if bench is None else bench)
    mean_ic, ic_t = _ic_values(candidate)
    st = perf_stats(net, bench, dates=dates)
    st2 = perf_stats(net2, None, dates=dates)
    t = len(net)
    sd = float(np.std(net, ddof=1)) if t > 1 else 0.0
    sr_daily = float(net.mean()) / sd if sd > 0 else 0.0

    n_all = n_trials(state)
    if ver == 2 and g.dsr_variance_scope == "family":  # type: ignore[union-attr]
        var_sr = sharpe_variance_across_trials(state, family=candidate.family)
        if not math.isfinite(var_sr):
            var_sr = 1.0 / t if t else float("nan")
    else:
        var_sr = sharpe_variance_across_trials(state)
    dsr = deflated_sharpe(sr_daily, t, st["skew"], st["kurtosis"], n_all, var_sr)

    mat = _family_returns(state, candidate.family)
    pbo_line: GateLine
    if mat is None or mat.shape[1] < MIN_PBO_TRIALS:
        n = 0 if mat is None else mat.shape[1]
        pbo_line = GateLine(
            8, "pbo", False, None, g.max_pbo, "<=",
            f"pbo FAIL (too few trials to estimate): {n} finished trials of family "
            f"{candidate.family!r}, need >= {MIN_PBO_TRIALS}",
        )  # fmt: skip
    else:
        pbo = float(pbo_cscv(mat)["pbo"])
        pbo_line = _line(8, "pbo", pbo, g.max_pbo, "<=", f" ({mat.shape[1]} trials, {mat.shape[0]} days)")

    lines = (
        _line(1, "mean_rank_ic", mean_ic, g.min_mean_rank_ic, ">="),
        _line(2, "ic_tstat", ic_t, g.min_ic_tstat, ">="),
        _line(3, "net_sharpe", st["sharpe"], g.min_net_sharpe, ">="),
        _line(4, "net_sharpe_2x_slip", st2["sharpe"], g.min_net_sharpe_2x_slip, ">="),
        _line(5, "excess_tstat", st["excess_tstat"], g.min_excess_tstat, ">="),
        _line(6, "frac_positive_years", st.get("frac_positive_years"), g.min_frac_positive_years, ">="),
        _line(7, "dsr", dsr, g.min_dsr, ">=", f" (N={n_all} trials)"),
        pbo_line,
        _line(9, "max_drawdown", st["max_drawdown"], g.max_drawdown, "<="),
    )
    return GateReport(candidate.candidate_id, candidate.family, f"research-v{ver}", lines)


# ---- lockbox gate --------------------------------------------------------------------------------


def evaluate_lockbox_gate(candidate_id: str, net_returns: np.ndarray, settings: Settings) -> GateReport:
    """The two lockbox lines on the daily net returns of [lockbox_start, last date]."""
    g = settings.gate.lockbox
    r = np.asarray(net_returns, dtype=float)
    sharpe = perf_stats(r, None)["sharpe"] if len(r) else float("nan")
    total = float(np.prod(1.0 + r) - 1.0) if len(r) else float("nan")
    lines = (
        _line(1, "net_sharpe", sharpe, g.min_net_sharpe, ">="),
        _line(2, "net_return", total, g.min_net_return, ">="),
    )
    return GateReport(candidate_id, "", "lockbox", lines)


# ---- persistence ---------------------------------------------------------------------------------


def _model_dir(settings: Settings, candidate_id: str) -> Path:
    return Paths(settings).models / candidate_id


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_report(report: GateReport, settings: Settings) -> Path:
    """Write models/<candidate>/validation.json and return its path."""
    path = _model_dir(settings, report.candidate_id) / "validation.json"
    _atomic_write_text(path, json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def save_candidate_result(candidate: CandidateResult, settings: Settings) -> Path:
    """Persist models/<candidate>/candidate_result.json + daily.parquet (preds are not persisted: store the
    precomputed `ic` instead)."""
    d = _model_dir(settings, candidate.candidate_id)
    d.mkdir(parents=True, exist_ok=True)
    ic = candidate.ic
    if ic is None:
        m, t = _ic_values(candidate)
        ic = {"mean_ic": m, "ic_tstat": t}
    meta = {
        "candidate_id": candidate.candidate_id, "family": candidate.family, "target": candidate.target,
        "ic": ic, "trial_ids": list(candidate.trial_ids),
    }  # fmt: skip
    candidate.daily.write_parquet(d / "daily.parquet")
    path = d / "candidate_result.json"
    _atomic_write_text(path, json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return path


def load_candidate_result(candidate_id: str, settings: Settings) -> CandidateResult:
    """Read what save_candidate_result wrote."""
    d = _model_dir(settings, candidate_id)
    meta = json.loads((d / "candidate_result.json").read_text(encoding="utf-8"))
    daily = pl.read_parquet(d / "daily.parquet")
    return CandidateResult(
        meta["candidate_id"], meta["family"], meta["target"], daily, ic=meta["ic"],
        trial_ids=tuple(meta.get("trial_ids", ())),
    )  # fmt: skip


def _with_trial_gross(cand: CandidateResult, settings: Settings) -> CandidateResult:
    """Candidates stored before G2-B1 have no `gross` column; when the candidate id is a G1 trial id, take it
    from that trial's daily.parquet (same dates, same book). Otherwise return the candidate unchanged."""
    path = Paths(settings).reports / "g1" / "trials" / cand.candidate_id / "daily.parquet"
    if not path.is_file():
        return cand
    gross = pl.read_parquet(path).select("date", "gross")
    return replace(cand, daily=cand.daily.join(gross, on="date", how="left"))


# ---- the callable the CLI will wrap --------------------------------------------------------------


def validate_candidate(
    candidate_id: str,
    settings: Settings,
    state: State,
    open_lockbox: bool = False,
    *,
    clock: Clock | None = None,
    lockbox_runner: LockboxRunner | None = None,
    gate_version: int | None = None,
) -> GateReport:
    """Run the research gate for a stored candidate, write its validation.json and, with `open_lockbox`,
    run the lockbox only if all nine lines pass (otherwise the request is recorded as refused and nothing is
    opened). `lockbox_runner(candidate_id, settings)` must return the daily net returns of the frozen spec's
    production-style walk-forward on [lockbox_start, last date]; it is called inside open_lockbox(...)."""
    clk = clock if clock is not None else SystemClock()
    cand = load_candidate_result(candidate_id, settings)
    if "gross" not in cand.daily.columns:
        cand = _with_trial_gross(cand, settings)
    report = replace(
        evaluate_research_gate(cand, settings, state, version=gate_version),
        generated_at=clk.now().isoformat(),
    )
    if open_lockbox:
        if not report.passed:
            failed = [ln.number for ln in report.lines if not ln.passed]
            report = replace(
                report,
                lockbox={"status": "refused", "reason": f"research gate failed on lines {failed}"},
            )
        else:
            if lockbox_runner is None:
                raise NotImplementedError("no lockbox runner supplied (the walk-forward runner is T6.7)")
            with _open_lockbox(settings, clk, candidate_id, "validate --open-lockbox"):
                returns = lockbox_runner(candidate_id, settings)
            lb = evaluate_lockbox_gate(candidate_id, returns, settings)
            report = replace(
                report,
                lockbox={
                    "status": "passed" if lb.passed else "failed",
                    "n_sessions": int(len(returns)),
                    "lines": [ln for ln in lb.to_dict()["lines"]],
                },
            )
    write_report(report, settings)
    return report


__all__ = [
    "CandidateResult",
    "GateLine",
    "GateReport",
    "evaluate_lockbox_gate",
    "evaluate_research_gate",
    "exposure_matched_benchmark",
    "select_gate_version",
    "load_candidate_result",
    "save_candidate_result",
    "validate_candidate",
    "write_report",
]
