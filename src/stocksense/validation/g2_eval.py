"""Evaluation of a G2 model trial against the baseline GBM (Part 4.2 of PLAN-G2, card D0.5).

Metrics, all on the dates where the model's and the baseline's out-of-fold scores both exist, from
`EVAL_START` (2016-01-01) to the research end, dates with at least 50 names, label `r_<target>`:

1. `mean_ic`, `ic_tstat`, `ic_by_year` (daily Spearman of the score with the label);
2. `corr_with_base`: mean daily Spearman between the model's and the baseline's scores;
3. `orth_ic`: per date, OLS (with intercept) of the model's rank on the baseline's rank, Spearman of the
   residuals with the label; mean and Newey-West t with lags = the target's horizon;
4. `blend_gain`: daily IC of `0.5*rank(model)+0.5*rank(base)` minus the baseline's IC; mean and NW t;
5. `net_sharpe_s1`: the model's scores alone through the S1 portfolio at the reference capital
   (G1Runner, auto-resume, registered as a `g2-s1` trial with spec key `model_eval_of`).

Admission (pre-registered): (`orth_ic >= 0.005` and its t >= 2) or (`blend_gain >= 0.003` and its t >= 2),
and `mean_ic >= 0.04`.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from stocksense.backtest.metrics import newey_west_tstat
from stocksense.core.calendar import TradingCalendar
from stocksense.core.config import Settings
from stocksense.core.paths import Paths
from stocksense.core.state import State
from stocksense.data import io
from stocksense.labels.forward import load_labels
from stocksense.validation.lockbox import research_end

EVAL_START = date(2016, 1, 1)
MIN_NAMES = 50
ADMIT_MIN_IC = 0.04
ADMIT_ORTH = 0.005
ADMIT_BLEND = 0.003
ADMIT_T = 2.0
S1_AUTO_RESUME = 20
FAMILY_S1_EVAL = "g2-s1"
_DEGENERATE = 1e-9


def eval_dir(settings: Settings, trial_id: str) -> Path:
    return Paths(settings).reports / "g2" / "models" / trial_id


def horizon(target: str) -> int:
    """Sessions a label looks ahead (the Newey-West lag count): y_c2o_10 -> 10, y_o2o_5 -> 5, y_oc -> 1."""
    m = re.search(r"_(\d+)$", target)
    return int(m.group(1)) if m else 1


# ---- daily series ------------------------------------------------------------------------------------


def daily_series(
    model: pl.DataFrame, base: pl.DataFrame, labels: pl.DataFrame, label_col: str, min_names: int = MIN_NAMES
) -> pl.DataFrame:
    """Per date: n, ic, ic_base, corr_base, orth_ic, ic_blend (rows with a null label dropped; dates with
    fewer than `min_names` joint rows dropped). `model` and `base` carry date, sid, score."""
    df = (
        model.select("date", "sid", pl.col("score").cast(pl.Float64).alias("m"))
        .join(base.select("date", "sid", pl.col("score").cast(pl.Float64).alias("b")), on=["date", "sid"])
        .join(labels.select("date", "sid", pl.col(label_col).cast(pl.Float64).alias("y")), on=["date", "sid"])
        .drop_nulls()
        .filter(pl.col("m").is_finite() & pl.col("b").is_finite() & pl.col("y").is_finite())
        .with_columns(pl.len().over("date").alias("n"))
        .filter(pl.col("n") >= min_names)
    )
    if df.is_empty():
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "n": pl.UInt32,
                "ic": pl.Float64,
                "ic_base": pl.Float64,
                "corr_base": pl.Float64,
                "orth_ic": pl.Float64,
                "ic_blend": pl.Float64,
            }  # fmt: skip
        )
    rank = lambda c: pl.col(c).rank("average").over("date")  # noqa: E731
    df = df.with_columns(rank("m").alias("rm"), rank("b").alias("rb"), rank("y").alias("ry"))
    slope = (
        (pl.col("rm") - pl.col("rm").mean().over("date")) * (pl.col("rb") - pl.col("rb").mean().over("date"))
    ).sum().over("date") / ((pl.col("rb") - pl.col("rb").mean().over("date")) ** 2).sum().over("date")
    df = df.with_columns(
        (
            pl.col("rm")
            - pl.col("rm").mean().over("date")
            - slope * (pl.col("rb") - pl.col("rb").mean().over("date"))
        ).alias("res"),
        (0.5 * pl.col("rm") + 0.5 * pl.col("rb")).alias("bl"),
    )
    df = df.with_columns(
        rank("res").alias("rres"),
        rank("bl").alias("rbl"),
        pl.col("res").std().over("date").alias("res_sd"),
        pl.col("rm").std().over("date").alias("rm_sd"),
    )
    out = (
        df.group_by("date")
        .agg(
            pl.len().alias("n"),
            pl.corr("rm", "ry").alias("ic"),
            pl.corr("rb", "ry").alias("ic_base"),
            pl.corr("rm", "rb").alias("corr_base"),
            pl.corr("rres", "ry").alias("orth_raw"),
            pl.corr("rbl", "ry").alias("ic_blend"),
            pl.col("res_sd").first().alias("res_sd"),
            pl.col("rm_sd").first().alias("rm_sd"),
        )
        .with_columns(
            pl.when(pl.col("res_sd") <= _DEGENERATE * pl.col("rm_sd").clip(lower_bound=1e-300))
            .then(0.0)  # the model's rank is (numerically) a function of the base's: nothing left over
            .otherwise(pl.col("orth_raw"))
            .alias("orth_ic")
        )
        .sort("date")
        .select("date", "n", "ic", "ic_base", "corr_base", "orth_ic", "ic_blend")
    )
    return out


def _mean_t(x: np.ndarray, lags: int) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    return float(x.mean()), float(newey_west_tstat(x, lags=lags))


def summarise(daily: pl.DataFrame, target: str) -> dict[str, Any]:
    """The EP metrics 1-4 from the per-date series."""
    h = horizon(target)
    if daily.is_empty():
        nan = float("nan")
        return {
            "n_dates": 0, "mean_ic": nan, "ic_tstat": nan, "ic_by_year": {}, "base_mean_ic": nan,
            "corr_with_base": nan, "orth_ic": nan, "orth_ic_t": nan, "blend_gain": nan, "blend_gain_t": nan,
            "horizon": h,
        }  # fmt: skip
    ic = daily["ic"].drop_nulls().to_numpy()
    sd = float(np.std(ic, ddof=1)) if len(ic) > 1 else float("nan")
    tstat = float(ic.mean() / sd * math.sqrt(len(ic))) if sd and sd > 0 else float("nan")
    by_year = (
        daily.drop_nulls("ic")
        .with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("year")
        .agg(pl.col("ic").mean())
        .sort("year")
    )
    orth, orth_t = _mean_t(daily["orth_ic"].to_numpy(), h)
    gain = (daily["ic_blend"] - daily["ic_base"]).to_numpy()
    blend, blend_t = _mean_t(gain, h)
    return {
        "n_dates": daily.height,
        "mean_ic": float(ic.mean()) if len(ic) else float("nan"),
        "ic_tstat": tstat,
        "ic_tstat_nw": float(newey_west_tstat(ic, lags=h)) if len(ic) > 1 else float("nan"),
        "ic_by_year": {str(y): float(v) for y, v in zip(by_year["year"], by_year["ic"], strict=True)},
        "base_mean_ic": float(daily["ic_base"].drop_nulls().mean()),  # type: ignore[arg-type]
        "corr_with_base": float(daily["corr_base"].drop_nulls().mean()),  # type: ignore[arg-type]
        "orth_ic": orth, "orth_ic_t": orth_t, "blend_gain": blend, "blend_gain_t": blend_t, "horizon": h,
    }  # fmt: skip


def admission(m: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered rule to the metrics `summarise` produced."""

    def ge(x: float | None, thr: float) -> bool:
        return x is not None and math.isfinite(x) and x >= thr

    own = ge(m.get("mean_ic"), ADMIT_MIN_IC)
    orth = ge(m.get("orth_ic"), ADMIT_ORTH) and ge(m.get("orth_ic_t"), ADMIT_T)
    blend = ge(m.get("blend_gain"), ADMIT_BLEND) and ge(m.get("blend_gain_t"), ADMIT_T)
    ok = own and (orth or blend)
    if ok:
        why = "admitted: " + " and ".join(n for n, f in (("orthogonal IC", orth), ("blend gain", blend)) if f)
    elif not own:
        why = f"no incremental value established: own mean IC below {ADMIT_MIN_IC}"
    else:
        why = "no incremental value established"
    return {"admitted": bool(ok), "reason": why, "own_ic_ok": own, "orth_ok": orth, "blend_ok": blend}


# ---- trials and inputs ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialRow:
    trial_id: str
    family: str
    target: str
    status: str
    spec: dict[str, Any]


def trial_row(state: State, trial_id: str) -> TrialRow:
    con = state.connect()
    try:
        r = con.execute(
            "SELECT trial_id, family, target, status, spec_json FROM trials WHERE trial_id=?", (trial_id,)
        ).fetchone()
    finally:
        con.close()
    if r is None:
        raise KeyError(f"unknown trial {trial_id!r}")
    return TrialRow(r[0], r[1], r[2], r[3], json.loads(r[4]) if r[4] else {})


def find_base_trial(state: State, target: str) -> str:
    """The most recently registered finished GBM walk-forward trial for `target` (s1-gbm; s2-gbm for y_oc)."""
    fam = "s2-gbm" if target == "y_oc" else "s1-gbm"
    con = state.connect()
    try:
        rows = con.execute(
            "SELECT trial_id FROM trials WHERE family=? AND target=? AND status='done' "
            "ORDER BY created_at DESC, trial_id",
            (fam, target),
        ).fetchall()
    finally:
        con.close()
    if not rows:
        raise LookupError(f"no finished {fam} trial for {target}; pass --base")
    return str(rows[0][0])


def load_scores(settings: Settings, trial_id: str, end: date | None = None) -> pl.DataFrame:
    return (
        io.read(settings, "preds", end=end, columns=["date", "sid", "score"], sub=trial_id)
        .collect()
        .sort("date", "sid")
    )


# ---- the S1 backtest of one model's scores -------------------------------------------------------------


@dataclass(frozen=True)
class S1Policy:
    """The S1 portfolio setting used to value a model alone. Card C5 chooses the final policy; until it
    has, this is the best G1 setting (k 10, no overlay) with the backtest-only auto-resume."""

    k_max: int = 10
    overlay: bool = False
    auto_resume: int = S1_AUTO_RESUME


def s1_backtest(
    trial_id: str,
    settings: Settings,
    state: State,
    calendar: TradingCalendar,
    *,
    policy: S1Policy | None = None,
    runner: Any = None,
    capital_inr: float | None = None,
) -> dict[str, Any] | None:
    """Run (or load) the `g2-s1` trial that backtests `trial_id`'s scores alone; None for a non-S1 target."""
    from stocksense.validation import g1

    row = trial_row(state, trial_id)
    if row.target not in g1.S1_TARGETS:
        return None
    pol = policy or S1Policy()
    if runner is None:
        runner = g1.G1Runner(
            settings, state, calendar,
            g1.G1Config(
                capitals=(capital_inr,) if capital_inr is not None else None, reference_capital=capital_inr
            ),
        )  # fmt: skip
    run = runner
    mt = g1.ModelTrial(trial_id, row.family, row.target, row.spec, 0.0, 0.0)
    cfg = replace(run.cfg, s1_auto_resume=pol.auto_resume)
    spec = g1._spec_s1(cfg, row.target, pol.k_max, pol.overlay, [mt], model_eval_of=trial_id)
    plan = g1.TrialPlan("s1", FAMILY_S1_EVAL, row.target, spec, f"S1 of {trial_id} k={pol.k_max}")
    (plan,) = run.register([plan])
    res = run.run_trial(plan)
    ref = res.data["by_capital"][str(int(run.cfg.reference_capital))]
    return {
        "s1_trial": plan.trial_id, "sharpe": ref["sharpe"], "net_2x": res.data["stress_2x"]["sharpe"],
        "mean_turnover": ref.get("mean_turnover"), "mean_gross": ref.get("mean_gross"),
        "excess_tstat": res.data.get("excess_tstat"), "k_max": pol.k_max, "overlay": pol.overlay,
        "auto_resume": pol.auto_resume,
    }  # fmt: skip


# ---- the command -----------------------------------------------------------------------------------------


def evaluate_model(
    trial_id: str,
    settings: Settings,
    state: State,
    calendar: TradingCalendar,
    *,
    base_id: str | None = None,
    eval_start: date = EVAL_START,
    policy: S1Policy | None = None,
    backtest: bool = True,
    runner: Any = None,
    min_names: int | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """All EP metrics of one model trial; writes reports/g2/models/<trial>/eval.json and returns the dict."""
    row = trial_row(state, trial_id)
    base_id = base_id or find_base_trial(state, row.target)
    label_col = f"r_{row.target}"
    end = research_end(settings)
    labels = load_labels(settings, "research", columns=[label_col]).collect()
    model = load_scores(settings, trial_id, end).filter(pl.col("date") >= eval_start)
    base = load_scores(settings, base_id, end).filter(pl.col("date") >= eval_start)
    daily = daily_series(model, base, labels, label_col, MIN_NAMES if min_names is None else min_names)
    m = summarise(daily, row.target)
    out: dict[str, Any] = {
        "trial_id": trial_id, "family": row.family, "target": row.target, "base_trial": base_id,
        "eval_start": eval_start.isoformat(), **m, **admission(m),
    }  # fmt: skip
    if base_id == trial_id:
        out["note"] = "model equals base"
    out["net_sharpe_s1"] = (
        s1_backtest(trial_id, settings, state, calendar, policy=policy, runner=runner) if backtest else None
    )
    if write:
        d = eval_dir(settings, trial_id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "eval.json.tmp"
        tmp.write_text(json.dumps(_clean(out), indent=2), encoding="utf-8", newline="\n")
        tmp.replace(d / "eval.json")
    return out


def _clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, np.generic):
        return _clean(x.item())
    return x


def load_evals(settings: Settings) -> list[dict[str, Any]]:
    root = Paths(settings).reports / "g2" / "models"
    out = []
    for p in sorted(root.glob("*/eval.json")) if root.is_dir() else []:
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def _f(x: Any, nd: int = 4, signed: bool = True) -> str:
    if x is None:
        return "-"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def table(rows: list[dict[str, Any]]) -> str:
    """One line per evaluated trial, grouped by family, for `research g2-table`."""
    head = (
        f"{'trial':<44} {'IC':>8} {'t':>6} {'corrB':>6} {'orth':>8} {'t':>5} {'blend':>8} {'t':>5} "
        f"{'S1 shp':>6} {'2x':>6} {'turn':>6}  admit"
    )
    lines = [head, "-" * len(head)]
    for r in sorted(rows, key=lambda r: (r["family"], r["trial_id"])):
        s = r.get("net_sharpe_s1") or {}
        lines.append(
            f"{r['trial_id']:<44} {_f(r.get('mean_ic'))} {_f(r.get('ic_tstat'), 1)} "
            f"{_f(r.get('corr_with_base'), 2)} {_f(r.get('orth_ic'))} {_f(r.get('orth_ic_t'), 1)} "
            f"{_f(r.get('blend_gain'))} {_f(r.get('blend_gain_t'), 1)} "
            f"{_f(s.get('sharpe'), 2)} {_f(s.get('net_2x'), 2)} {_f(s.get('mean_turnover'), 3, False)}  "
            f"{'YES' if r.get('admitted') else 'no'}"
        )
    return "\n".join(lines)
