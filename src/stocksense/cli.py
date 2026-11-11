from __future__ import annotations

import platform
import re
import shutil
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import polars as pl
import pyarrow.parquet as pq
import typer

from stocksense import __version__
from stocksense.core.calendar import TradingCalendar
from stocksense.core.clock import IST, SystemClock
from stocksense.core.config import Settings, load_settings
from stocksense.core.errors import DataNotAvailable
from stocksense.core.paths import Paths
from stocksense.core.state import State
from stocksense.data.catalog import CATALOG, DatasetSpec
from stocksense.data.io import list_partitions, read
from stocksense.data.status import status_table
from stocksense.data.validate import validate_dataset

app = typer.Typer(help="StockSense command line.", no_args_is_help=True)

_RE_DAY_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.parquet$")


@app.callback()
def main(
    ctx: typer.Context,
    config_dir: Annotated[Path, typer.Option(help="Config folder.")] = Path("config"),
) -> None:
    """StockSense command line."""
    if ctx.obj is None:  # tests inject a Settings through `obj`
        ctx.obj = {"config_dir": config_dir}


def _settings(ctx: typer.Context) -> Settings:
    if isinstance(ctx.obj, Settings):
        return ctx.obj
    return load_settings(ctx.obj["config_dir"])


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


def _not_implemented(phase: int) -> None:
    typer.echo(f"not implemented yet (phase {phase})")
    raise typer.Exit(2)


@app.command("data-status")
def data_status(
    ctx: typer.Context,
    issues: Annotated[bool, typer.Option("--issues", help="Also list every validation issue.")] = False,
) -> None:
    """Dataset freshness and validation report (exit 1 if there is no data)."""
    settings = _settings(ctx)
    clock = SystemClock()
    try:
        calendar = TradingCalendar.from_store(settings, clock)
    except DataNotAvailable:
        typer.echo("no data: bhavcopy_eq is empty or missing")
        raise typer.Exit(1) from None
    table = status_table(settings, calendar, clock)
    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=200,
        tbl_formatting="ASCII_MARKDOWN",
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
        fmt_str_lengths=60,
    ):
        typer.echo(str(table))
    if issues:
        for name in table["dataset"]:
            for i in validate_dataset(settings, name):
                typer.echo(f"{i.dataset}\t{i.code}\t{i.detail}")


@app.command()
def ingest() -> None:
    """Download and store new raw data."""
    _not_implemented(2)


@app.command("build-panel")
def build_panel() -> None:
    """Build the adjusted daily panel."""
    _not_implemented(3)


@app.command("build-features")
def build_features() -> None:
    """Build the feature store."""
    _not_implemented(4)


@app.command("build-labels")
def build_labels() -> None:
    """Build forward-return labels."""
    _not_implemented(4)


@app.command()
def train() -> None:
    """Train a model."""
    _not_implemented(7)


@app.command()
def validate() -> None:
    """Run the validation harness."""
    _not_implemented(6)


@app.command()
def backtest() -> None:
    """Run a backtest."""
    _not_implemented(5)


@app.command()
def run() -> None:
    """Run the daemon."""
    _not_implemented(10)


@app.command()
def job() -> None:
    """Run one scheduled job."""
    _not_implemented(10)


@app.command()
def report() -> None:
    """Write the daily report."""
    _not_implemented(12)


@app.command()
def resume() -> None:
    """Resume after an interruption."""
    _not_implemented(10)


# ---- doctor -----------------------------------------------------------------------------------


def _dataset_files(settings: Settings, spec: DatasetSpec) -> list[Path]:
    """All partition files of a dataset (every sub-folder for nested ones)."""
    if not spec.nested:
        return list_partitions(settings, spec.name)
    base = Path(settings.paths.data_root) / spec.path
    return sorted(base.glob("*/*.parquet")) if base.is_dir() else []


def _last_date(settings: Settings, spec: DatasetSpec, files: list[Path]) -> date | None:
    """Newest session date: nested day files by name; others from the date column of the newest file."""
    if spec.nested:
        days = [m[1] for f in files if (m := _RE_DAY_FILE.match(f.name))]
        return date.fromisoformat(max(days)) if days else None
    if spec.date_col is None:
        return None
    newest = max(files, key=lambda f: f.name)
    col = spec.date_col
    expr = pl.col(col).dt.date() if isinstance(spec.schema[col], pl.Datetime) else pl.col(col)
    # data.io.read conforms the dtype; the lower bound skips every older partition file.
    start = _partition_start(newest.name)
    top = read(settings, spec.name, start=start, columns=[col]).select(expr.max()).collect().item()
    return top


def _partition_start(name: str) -> date | None:
    m = re.match(r"^part-(\d{4})-(\d{2})\.parquet$", name)
    if m:
        return date(int(m[1]), int(m[2]), 1)
    m = re.match(r"^part-(\d{4})\.parquet$", name)
    return date(int(m[1]), 1, 1) if m else None


def _dataset_lines(settings: Settings) -> list[str]:
    lines = []
    for spec in CATALOG.values():
        files = _dataset_files(settings, spec)
        if not files:
            continue
        rows = sum(pq.read_metadata(f).num_rows for f in files)
        last = _last_date(settings, spec, files)
        lines.append(f"  {spec.name:<22}{rows:>12,} rows   last {last if last else '-'}")
    return lines


def _torch_line() -> str:
    try:
        import torch

        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
        return f"torch {torch.__version__}, {dev}"
    except Exception as e:  # noqa: BLE001 - a broken torch install must not stop doctor
        return f"torch unavailable ({type(e).__name__})"


@app.command()
def doctor(ctx: typer.Context) -> None:
    """Environment, data and state health check (exit 1 if the data root is missing)."""
    settings = _settings(ctx)
    paths = Paths(settings)
    out = typer.echo
    out(f"python        {sys.version.split()[0]} ({platform.system()})")
    out(f"torch         {_torch_line()}")
    root = paths.root
    out(f"data root     {root}")
    if not root.is_dir():
        out("ERROR: data root is missing")
        raise typer.Exit(1)
    out(f"free disk     {shutil.disk_usage(root).free / 1e9:.1f} GB")
    out("datasets")
    lines = _dataset_lines(settings)
    for ln in lines or ["  (none)"]:
        out(ln)
    state = State(settings)
    state.migrate()
    con = state.connect()
    try:
        ver = con.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    finally:
        con.close()
    out(f"schema version {ver}")
    hb = paths.heartbeat
    if hb.is_file():
        age = SystemClock().now() - datetime.fromtimestamp(hb.stat().st_mtime, IST)
        out(f"heartbeat     {age.total_seconds():.0f} s ago")
    else:
        out("heartbeat     none")
    out("secrets")
    for name, value in settings.secrets.model_dump().items():
        out(f"  {name:<24}{'set' if value else 'unset'}")
