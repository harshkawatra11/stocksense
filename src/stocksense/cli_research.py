"""Research commands (PLAN T7.1/T7.2): `train --spec`, `validate`, `research ic-table|wf-status|select`.

Kept apart from cli.py; `register(app)` is the single hook. It replaces the placeholder `train` and
`validate` commands there. Everything here reads with purpose='research' (the lockbox is never opened).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import SystemClock
from stocksense.core.config import Settings, load_settings
from stocksense.core.paths import Paths
from stocksense.core.state import State
from stocksense.data import io
from stocksense.labels.forward import load_labels
from stocksense.models.base import ModelSpec
from stocksense.models.linear import FLOOR_TARGETS, S1_TARGETS, floor_specs
from stocksense.validation.ic_report import oof_report
from stocksense.validation.splits import Fold, walk_forward_folds
from stocksense.validation.trials import register_trial

research_app = typer.Typer(help="Research runs and reports (research period only).", no_args_is_help=True)

GBM_OBJECTIVES = ("regression", "lambdarank")


# ---- named specs ---------------------------------------------------------------------------------------


def selection_path(settings: Settings, target: str, objective: str) -> Path:
    return Paths(settings).models / "_selection" / f"{target}_{objective}.json"


def _floor_name(spec: ModelSpec) -> str:
    if spec.family == "floor-single":
        return f"floor-single-{spec.params['feature']}-{spec.target}"
    return f"floor-ridge-{spec.target}"


def gbm_spec(target: str, objective: str, best: dict) -> ModelSpec:
    """The T7.2 spec for (target, objective) with the selected grid configuration `best`."""
    oc = target == "y_oc"
    params = {
        "objective": objective,
        **{k: best[k] for k in ("num_leaves", "min_child_samples", "colsample_bytree")},
    }
    if oc:
        params["auction"] = True  # the auction features of T4.4, built from the official open
    return ModelSpec(
        family="s2-gbm" if oc else "s1-gbm", target=target, universe="u200" if oc else "u500",
        features="all", params=params, entry="open" if oc else "close",
    )  # fmt: skip


def named_specs(settings: Settings) -> dict[str, ModelSpec]:
    """name -> spec: the 32 floor trials, plus a `gbm-<target>-<objective>` spec for every (target,
    objective) whose hyper-parameter selection has been written (`research select`)."""
    out = {_floor_name(s): s for s in floor_specs()}
    for target in FLOOR_TARGETS:
        for objective in GBM_OBJECTIVES:
            p = selection_path(settings, target, objective)
            if p.is_file():
                out[f"gbm-{target}-{objective}"] = gbm_spec(
                    target, objective, json.loads(p.read_text(encoding="utf-8"))["best"]
                )
    return out


# ---- helpers ------------------------------------------------------------------------------------------


def _settings(ctx: typer.Context) -> Settings:
    if isinstance(ctx.obj, Settings):
        return ctx.obj
    return load_settings(ctx.obj["config_dir"])


def parse_folds(text: str, calendar: TradingCalendar) -> list[Fold]:
    """'name:YYYY-MM-DD:YYYY-MM-DD,...' -> folds whose train_end is the session before test_start."""
    folds = []
    for part in text.split(","):
        name, a, b = part.strip().split(":")
        start = date.fromisoformat(a)
        folds.append(Fold(name, calendar.prev_session(start), start, date.fromisoformat(b)))
    return folds


def _trials(state: State, family: str | None) -> list[tuple[str, str, str, str, str]]:
    con = state.connect()
    try:
        rows = con.execute(
            "SELECT trial_id, family, target, status, spec_json FROM trials "
            "WHERE family != 'planning-exploration' ORDER BY created_at, trial_id"
        ).fetchall()
    finally:
        con.close()
    return [r for r in rows if family is None or r[1].startswith(family)]


def _folds_done(settings: Settings, trial_id: str) -> list[str]:
    p = Paths(settings).preds / trial_id / "folds_done.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else []


# ---- commands -----------------------------------------------------------------------------------------


def train(
    ctx: typer.Context,
    spec: Annotated[
        str | None, typer.Option("--spec", help="Name of a registered spec (see --list).")
    ] = None,
    list_specs: Annotated[bool, typer.Option("--list", help="List the spec names and exit.")] = False,
    seeds: Annotated[str, typer.Option(help="Comma separated seeds.")] = "1,2,3",
    folds: Annotated[str | None, typer.Option(help="Custom folds 'name:start:end,...' (tests).")] = None,
    inner_months: Annotated[int, typer.Option(help="Inner-validation months.")] = 12,
) -> None:
    """Run the walk-forward for a named spec (research purpose; resumable per fold)."""
    from stocksense.models.gbm_diag import DiagGbm
    from stocksense.validation.walkforward import run_walk_forward

    settings = _settings(ctx)
    specs = named_specs(settings)
    if list_specs:
        for name in sorted(specs):
            typer.echo(name)
        return
    if spec is None or spec not in specs:
        typer.echo(f"unknown spec {spec!r}; use --list", err=True)
        raise typer.Exit(2)
    ms = specs[spec]
    state = State(settings)
    state.migrate()
    clock = SystemClock()
    cal = TradingCalendar.from_store(settings, clock)
    fold_list = parse_folds(folds, cal) if folds else walk_forward_folds(settings, cal)
    factory = None
    if ms.family.rsplit("-", 1)[-1] == "gbm":
        tid = register_trial(state, ms.family, ms.target, ms.to_dict())  # idempotent: the id the runner gets
        out_dir = Paths(settings).models / "_wf" / tid
        keys = ("objective", "num_leaves", "min_child_samples", "colsample_bytree", "n_estimators", "n_jobs")

        def factory(s: ModelSpec):  # type: ignore[no-redef]
            return DiagGbm(out_dir, **{k: s.params[k] for k in keys if k in s.params})

    res = run_walk_forward(
        ms, settings, state, purpose="research", seeds=[int(x) for x in seeds.split(",")], folds=fold_list,
        calendar=cal, clock=clock, inner_months=inner_months, model_factory=factory,
    )  # fmt: skip
    typer.echo(f"trial {res.trial_id}")
    for m in res.fold_metrics:
        ic = m.get("mean_ic")
        sec = m.get("seconds")
        typer.echo(
            f"  fold {m['fold']}: " + (f"IC {ic:+.4f} t {m['ic_tstat']:+.1f}" if ic is not None else "-")
            + (f"  {sec:.0f}s" if sec is not None else "")
        )  # fmt: skip
    typer.echo(f"mean IC {res.ic['mean_ic']:+.4f}  t {res.ic['ic_tstat']:+.2f}  dates {res.ic['n_dates']}")


def validate(
    ctx: typer.Context,
    candidate: Annotated[
        str, typer.Argument(help="Candidate id (models/<candidate>/candidate_result.json).")
    ],
    open_lockbox: Annotated[bool, typer.Option("--open-lockbox")] = False,
) -> None:
    """Run the research gate for a stored candidate (T6.6)."""
    from stocksense.validation.gate import validate_candidate

    settings = _settings(ctx)
    state = State(settings)
    state.migrate()
    try:
        report = validate_candidate(candidate, settings, state, open_lockbox=open_lockbox)
    except (FileNotFoundError, NotImplementedError) as e:
        typer.echo(f"cannot validate {candidate!r}: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(report.table())
    raise typer.Exit(0 if report.passed else 1)


@research_app.command("wf-status")
def wf_status(
    ctx: typer.Context,
    family: Annotated[str | None, typer.Option(help="Family prefix filter.")] = None,
    expected: Annotated[int, typer.Option(help="Folds a finished trial has.")] = 12,
) -> None:
    """List trials with their fold progress."""
    settings = _settings(ctx)
    state = State(settings)
    state.migrate()
    rows = _trials(state, family)
    if not rows:
        typer.echo("no trials")
        return
    for tid, fam, target, status, _spec in rows:
        done = len(_folds_done(settings, tid))
        typer.echo(f"{tid:<52} {status:<10} folds {done}/{expected}  ({fam}, {target})")


@research_app.command("ic-table")
def ic_table(
    ctx: typer.Context,
    family: Annotated[str | None, typer.Option(help="Family prefix filter.")] = None,
    out: Annotated[Path | None, typer.Option(help="Also write the table as markdown to this file.")] = None,
) -> None:
    """Out-of-fold IC (mean x100, t-stat, by year) of every trial that has predictions."""
    settings = _settings(ctx)
    state = State(settings)
    state.migrate()
    lab_cache: dict[str, pl.DataFrame] = {}
    results = []
    for tid, fam, target, _status, spec_json in _trials(state, family):
        if not _folds_done(settings, tid):
            continue
        preds = io.read(settings, "preds", sub=tid).collect().select("date", "sid", "score")
        if target not in lab_cache:
            lab_cache[target] = load_labels(settings, "research", columns=[target]).collect()
        label = f"{json.loads(spec_json).get('params', {}).get('feature', '')}"
        results.append((tid, fam, target, label, oof_report(preds, lab_cache[target], target)))
    if not results:
        typer.echo("no trials with predictions")
        return
    years = sorted({y for r in results for y in r[4]["by_year"]})
    head = ["trial", "target", "IC x100", "t", "yrs+", *[str(y) for y in years]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for tid, _fam, target, _label, r in results:
        by = r["by_year"]
        pos = sum(1 for v in by.values() if v > 0)
        cells = [
            tid, target, f"{r['mean_ic'] * 100:+.2f}", f"{r['ic_tstat']:+.1f}", f"{pos}/{len(by)}",
            *[(f"{by[y] * 100:+.1f}" if y in by else "") for y in years],
        ]  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    typer.echo(text)
    if out is not None:
        out.write_text(text + "\n", encoding="utf-8", newline="\n")


@research_app.command("select")
def select(
    ctx: typer.Context,
    target: Annotated[str, typer.Option(help="y_c2o_5, y_c2o_10, y_c2o_20 or y_oc.")],
    objective: Annotated[str, typer.Option(help="regression or lambdarank.")] = "regression",
) -> None:
    """Hyper-parameter selection by inner validation at the anchor folds (registers no trial)."""
    from stocksense.models.gbm import select_hyperparams

    if target not in (*S1_TARGETS, "y_oc") or objective not in GBM_OBJECTIVES:
        typer.echo("bad target or objective", err=True)
        raise typer.Exit(2)
    settings = _settings(ctx)
    state = State(settings)
    state.migrate()
    oc = target == "y_oc"
    res = select_hyperparams(
        target, objective, settings, state, universe="u200" if oc else "u500",
        entry="open" if oc else "close",
        auction=oc,
    )  # fmt: skip
    p = selection_path(settings, target, objective)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    for row in sorted(res["table"], key=lambda r: -r["mean_ic"]):
        typer.echo(f"{row['mean_ic']:+.4f} {row['ic_by_anchor']} {row['config']}")
    typer.echo(f"selected {res['best']} -> {p}")


def register(app: typer.Typer) -> None:
    """Hook for cli.py: replace the placeholder train/validate and add the `research` group."""
    app.registered_commands = [
        c
        for c in app.registered_commands
        if (c.name or getattr(c.callback, "__name__", "")) not in ("train", "validate")
    ]
    app.command("train")(train)
    app.command("validate")(validate)
    app.add_typer(research_app, name="research")
