"""Dashboard commands (PLAN-M): `serve` starts the local read-only server; `lab` holds library tools.

Registered from cli.py.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, Any

import typer

lab_app = typer.Typer(help="Dashboard and strategy-library tools.", no_args_is_help=True)
REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "marketlab"
DEFAULT_CLONE = Path("C:/Users/harsh/Downloads/market-lab")


def serve(
    ctx: typer.Context,
    port: Annotated[int, typer.Option(help="Port on 127.0.0.1 (Market Lab uses 8810).")] = 8765,
    no_feeds: Annotated[bool, typer.Option(help="Do not start the live feed threads.")] = False,
    no_catchup: Annotated[
        bool, typer.Option(help="Do not run the startup catch-up (ingest, panel, scan).")
    ] = False,
) -> None:
    """Start the local dashboard server on 127.0.0.1 (manual start only; Ctrl+C stops it)."""
    import uvicorn

    from stocksense.api.app import create_app
    from stocksense.cli import _settings

    app = create_app(_settings(ctx), start_feeds=not no_feeds)
    app.state.catchup_enabled = not no_catchup
    uvicorn.run(app, host="127.0.0.1", port=port, workers=1, log_level="info")


def _diff(a: Any, b: Any, path: str, bad: list[str]) -> None:
    """Collect differences between two JSON values (floats compared to 1e-9, NaN equal NaN)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            bad.append(f"{path}: keys differ {sorted(set(a) ^ set(b))}")
            return
        for k in a:
            _diff(a[k], b[k], f"{path}.{k}", bad)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            bad.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _diff(x, y, f"{path}[{i}]", bad)
    elif isinstance(a, float) or isinstance(b, float):
        nums = isinstance(a, (int, float)) and isinstance(b, (int, float))
        both_nan = nums and math.isnan(a) and math.isnan(b)
        if not both_nan and not (nums and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)):
            bad.append(f"{path}: {a} vs {b}")
    elif a != b:
        bad.append(f"{path}: {a!r} vs {b!r}")


@lab_app.command("golden-check")
def golden_check(
    clone: Annotated[Path, typer.Option(help="Local market-lab clone.")] = DEFAULT_CLONE,
) -> None:
    """Re-run Market Lab's own functions in its own venv and diff against the committed golden files."""
    py = clone / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        typer.echo(f"clone interpreter not found: {py} (nothing is installed into it)")
        raise typer.Exit(2)
    with tempfile.TemporaryDirectory() as tmp:
        for f in FIXTURES.glob("inputs_*.json"):
            shutil.copy(f, tmp)
        script = REPO / "scripts" / "golden" / "run_marketlab.py"
        run = subprocess.run(
            [str(py), "-I", str(script), str(clone), tmp], capture_output=True, text=True, check=False
        )
        if run.returncode != 0:
            typer.echo(run.stderr[-800:])
            raise typer.Exit(1)
        bad: list[str] = []
        for new in sorted(Path(tmp).glob("golden_*.json")):
            old = FIXTURES / new.name
            _diff(json.loads(new.read_text("utf-8")), json.loads(old.read_text("utf-8")), new.stem, bad)
    for line in bad[:30]:
        typer.echo(line)
    typer.echo("golden files match" if not bad else f"{len(bad)} differences")
    raise typer.Exit(1 if bad else 0)


def register(app: typer.Typer) -> None:
    """Hook for cli.py: add `serve` and the `lab` group."""
    app.command("serve")(serve)
    app.add_typer(lab_app, name="lab")
