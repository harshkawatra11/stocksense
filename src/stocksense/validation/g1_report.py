"""docs/RESULTS-G1.md sections 1, 2 and 5, rendered from the trial registry and the files the G1 run left.

Nothing here computes a new result except the IC tables (out-of-fold scores against the stored labels);
every other number is read back from `result.json`, `validation.json` and the registry."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from stocksense.core.config import Settings
from stocksense.core.paths import Paths
from stocksense.core.state import State
from stocksense.data import io
from stocksense.labels.forward import load_labels
from stocksense.validation import g1
from stocksense.validation.ic_report import oof_report
from stocksense.validation.trials import n_trials

ANN = math.sqrt(252.0)
PLACEHOLDER = "_Not run in this task; filled in by the Kronos and sequence-model tasks._"


def _f(x: Any, nd: int = 2, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x * 100:.{nd}f}%" if pct else f"{x:.{nd}f}"


def _table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _registry(state: State) -> list[dict[str, Any]]:
    con = state.connect()
    try:
        rows = con.execute(
            "SELECT trial_id, family, target, status, spec_json, metrics_json FROM trials ORDER BY created_at, trial_id"  # noqa: E501
        ).fetchall()
    finally:
        con.close()
    out = []
    for tid, fam, tgt, st, spec, met in rows:
        out.append(
            {
                "id": tid,
                "family": fam,
                "target": tgt,
                "status": st,
                "spec": json.loads(spec) if spec else {},
                "metrics": json.loads(met) if met else {},
            }  # fmt: skip
        )
    return out


def _oof(
    settings: Settings, trial_id: str, target: str, cache: dict[str, pl.DataFrame]
) -> dict[str, Any] | None:
    if not (Paths(settings).preds / trial_id).is_dir():
        return None
    preds = io.read(settings, "preds", sub=trial_id).collect().select("date", "sid", "score")
    if target not in cache:
        cache[target] = load_labels(settings, "research", columns=[target]).collect()
    return oof_report(preds, cache[target], target)


def _model_label(r: dict[str, Any]) -> str:
    prm = r["spec"].get("params", {}) or {}
    if "feature" in prm:
        return f"{'-' if prm.get('sign', 1) < 0 else ''}{prm['feature']}"
    return r["family"]


def ic_section(
    settings: Settings, rows: list[dict[str, Any]], title: str, cache: dict[str, pl.DataFrame]
) -> str:
    res = []
    for r in rows:
        rep = _oof(settings, r["id"], r["target"], cache)
        if rep is not None:
            res.append((r, rep))
    if not res:
        return f"{title}\n\nNo trials with predictions."
    years = sorted({y for _, rep in res for y in rep["by_year"]})
    head = ["trial", "target", "IC x100", "t", "yrs+", "top-10 net Sharpe", *[str(y) for y in years]]
    body = []
    for r, rep in res:
        by = rep["by_year"]
        sh = r["metrics"].get("sharpe_daily")
        body.append(
            [
                f"{_model_label(r)} [{r['id']}]",
                r["target"],
                f"{rep['mean_ic'] * 100:+.2f}",
                f"{rep['ic_tstat']:+.1f}",
                f"{sum(1 for v in by.values() if v > 0)}/{len(by)}",
                _f(sh * ANN) if sh is not None else "n/a",
                *[(f"{by[y] * 100:+.1f}" if y in by else "") for y in years],
            ]  # fmt: skip
        )
    return f"{title}\n\n" + _table(head, body)


def _gate_table(rep: dict[str, Any]) -> str:
    rows = [
        [str(ln["number"]), ln["name"], "PASS" if ln["passed"] else "FAIL", ln["text"]] for ln in rep["lines"]
    ]
    return _table(["#", "line", "result", "measured vs bar"], rows)


def _yearly(daily: pl.DataFrame) -> str:
    d = (
        daily.with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("year")
        .agg(pl.col("net").alias("net"), pl.col("benchmark").alias("bm"))
        .sort("year")
    )
    rows = []
    for y, net, bm in d.iter_rows():
        n = np.asarray(net, dtype=float)
        sd = n.std(ddof=1) if len(n) > 1 else 0.0
        b = np.asarray([v for v in bm if v is not None], dtype=float)
        rows.append(
            [
                str(y),
                str(len(n)),
                _f(float(np.prod(1 + n) - 1), 1, True),
                _f(float(n.mean() / sd * ANN) if sd > 0 else None),
                _f(float(np.prod(1 + b) - 1), 1, True) if len(b) else "n/a",
            ]
        )
    return _table(["year", "days", "net return", "net Sharpe", "ew500 return"], rows)


def _portfolio_row(settings: Settings, rid: str, r: dict[str, Any], ref: float) -> list[str] | None:
    res = g1.load_result(settings, rid)
    if res is None:
        return None
    d = res.data
    b = d["by_capital"][str(int(ref))]
    return [
        d["label"], f"[{rid}]", _f(b["sharpe"]), _f(d["stress_2x"]["sharpe"]), _f(b["cagr"], 1, True),
        _f(b["max_drawdown"], 1, True), _f(b["frac_positive_years"], 0, True), _f(b["mean_turnover"], 1, True),  # noqa: E501
        _f(b["cost_drag_annual"], 1, True), _f(d.get("excess_tstat")), str(b["n_trades"]),
    ]  # fmt: skip


def section5(settings: Settings, state: State, outcome: dict[str, Any] | None) -> str:
    reg = _registry(state)
    ref = float(settings.gate.research.reference_capital_inr)
    parts: list[str] = ["## 5. Gate G1 research run and decision (T9.5)"]
    total = n_trials(state)
    parts.append(
        f"Total registered trials (the N used by the deflated Sharpe ratio): **{total}**. "
        f"Reference capital: {ref:,.0f} INR. Fills for S2 are at the official open (flag `auction-assumption`)."  # noqa: E501
    )

    # 5.1 every trial
    rows = []
    for r in reg:
        m = r["metrics"]
        sh = m.get("sharpe_daily")
        ic = m.get("mean_ic")
        rows.append(
            [
                r["id"],
                r["family"],
                r["target"] or "",
                r["status"],
                f"{ic * 100:+.2f}" if ic is not None else "",
                _f(sh * ANN) if sh is not None else "",
            ]  # fmt: skip
        )
    parts.append(
        "### 5.1 Every trial in the registry\n\n"
        "Columns: out-of-fold mean IC x100 and annualised net Sharpe of the top-10 (model trials) or of the "
        "portfolio run (G1 trials). The 30 `planning-exploration` rows are the signals examined while the plan "  # noqa: E501
        "was written; they count toward N.\n\n"
        + _table(["trial", "family", "target", "status", "IC x100", "net Sharpe"], rows)
    )

    # 5.2 portfolio trials
    head = ["trial", "id", "net Sharpe", "2x-slip Sharpe", "CAGR", "max DD", "yrs+", "turnover/day",
            "cost drag/yr", "excess t", "trades"]  # fmt: skip
    for kind_title, fams in (("S1 (delivery book)", ("g1-s1",)), ("S2 (auction book)", ("g1-s2",)),
                             ("References", ("g1-ref",))):  # fmt: skip
        body = []
        for r in reg:
            if r["family"] in fams:
                row = _portfolio_row(settings, r["id"], r, ref)
                if row is not None:
                    body.append(row)
        parts.append(
            f"### 5.2 Portfolio trials: {kind_title}\n\n"
            + (_table(head, body) if body else "No trials of this kind ran.")
        )

    # benchmarks
    bm = _benchmark(settings)
    if bm:
        parts.append("### 5.3 Index references (no cost, no slippage)\n\n" + bm)

    # capital grid
    cands = (outcome or {}).get("candidates", {})
    for fam, cid in cands.items():
        res = g1.load_result(settings, cid)
        if res is None:
            continue
        rows = [
            [f"{int(c):,}", _f(b["sharpe"]), _f(b["cagr"], 1, True), _f(b["max_drawdown"], 1, True),
             _f(b["cost_drag_annual"], 1, True), _f(b["mean_turnover"], 1, True), str(b["n_trades"])]
            for c, b in sorted(res.data["by_capital"].items(), key=lambda kv: float(kv[0]))
        ]  # fmt: skip
        parts.append(
            f"### 5.4 Capital grid of the {fam.upper()} candidate [{cid}]\n\n"
            + _table(
                ["capital INR", "net Sharpe", "CAGR", "max DD", "cost drag/yr", "turnover/day", "trades"],
                rows,
            )
        )

    # gates
    for fam, rep in (outcome or {}).get("reports", {}).items():
        verdict = "PASS" if rep["passed"] else "FAIL"
        parts.append(
            f"### 5.5 Research gate, {fam.upper()} candidate [{rep['candidate_id']}]: {verdict}\n\n{_gate_table(rep)}"  # noqa: E501
        )
        res = g1.load_result(settings, rep["candidate_id"])
        if res is not None:
            parts.append(f"Per-year net result of {fam.upper()} candidate:\n\n{_yearly(res.daily())}")

    # strict capture
    for r in reg:
        if r["family"] == "g1-s2":
            res = g1.load_result(settings, r["id"])
            if res is not None:
                sc = res.data.get("strict_capture") or {}
                parts.append(f"strict_capture of {r['id']}: `{json.dumps(sc, sort_keys=True, default=str)}`")
    return "\n\n".join(parts)


def _benchmark(settings: Settings) -> str:
    p = g1.g1_dir(settings) / "benchmarks.json"
    if not p.is_file():
        return ""
    rows = json.loads(p.read_text(encoding="utf-8"))
    out = [
        [r["label"], _f(r.get("sharpe")), _f(r.get("cagr"), 1, True), _f(r.get("max_drawdown"), 1, True)]
        if r.get("available")
        else [r["label"], "no data", "", ""]
        for r in rows
    ]
    return _table(["series", "Sharpe", "CAGR", "max DD"], out)


def render_results(settings: Settings, state: State, outcome: dict[str, Any] | None = None) -> str:
    if outcome is None:
        p = g1.g1_dir(settings) / "outcome.json"
        outcome = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    reg = _registry(state)
    cache: dict[str, pl.DataFrame] = {}
    floors = [r for r in reg if r["family"].startswith("floor-")]
    gbms = [r for r in reg if r["family"].endswith("-gbm")]
    s1 = ic_section(settings, floors, "### 1.1 Floor trials (ridge and single-factor)", cache)
    s2 = ic_section(settings, gbms, "### 2.1 Gradient-boosted models", cache)
    return (
        "# Gate G1 results\n\n"
        "Generated from the trial registry by `stocksense.validation.g1_report`. No threshold was changed and no "  # noqa: E501
        "grid point added after a result was seen (config/gate.toml is frozen).\n\n"
        f"## 1. Floors\n\n{s1}\n\n## 2. Gradient boosting\n\n{s2}\n\n"
        f"## 3. Sequence model\n\n{PLACEHOLDER}\n\n## 4. Kronos\n\n{PLACEHOLDER}\n\n"
        + section5(settings, state, outcome)
        + "\n"
    )


def write_results(
    settings: Settings, state: State, path: Path, outcome: dict[str, Any] | None = None
) -> Path:
    path.write_text(render_results(settings, state, outcome), encoding="utf-8", newline="\n")
    return path
