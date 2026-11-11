from __future__ import annotations

import platform
import re
import shutil
import sys
from datetime import date, datetime, timedelta
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


ingest_app = typer.Typer(help="Download and store new raw data.")
app.add_typer(ingest_app, name="ingest")


@ingest_app.callback(invoke_without_command=True)
def ingest(ctx: typer.Context) -> None:
    """Download and store new raw data."""
    if ctx.invoked_subcommand is None:
        _not_implemented(2)


def _parse_day(text: str, clock: SystemClock) -> date:
    if text.strip().lower() == "yesterday":
        return clock.today() - timedelta(days=1)
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise typer.BadParameter(f"{text!r} is not YYYY-MM-DD or 'yesterday'") from None


@ingest_app.command("eq")
def ingest_eq(
    ctx: typer.Context,
    from_: Annotated[str, typer.Option("--from", help="First day, YYYY-MM-DD.")],
    to: Annotated[str, typer.Option("--to", help="Last day, YYYY-MM-DD or 'yesterday'.")],
    repair: Annotated[
        bool, typer.Option("--repair", help="Run the T2.3 repair order (ignores --from/--to start).")
    ] = False,
    mcp_check: Annotated[
        bool,
        typer.Option(
            "--mcp-check/--no-mcp-check", help="After --repair, cross-check 20 rows with NSE's MCP server."
        ),
    ] = True,
) -> None:
    """Equity bhavcopy and delivery for every calendar day in [from, to]."""
    from collections import Counter

    from stocksense.data.ingest_eq import (
        IngestContext,
        cross_check_mcp,
        ingest_eq_range,
        repair_days,
        sample_pairs,
    )
    from stocksense.data.nse_http import NseClient

    settings = _settings(ctx)
    clock = SystemClock()
    a, b = _parse_day(from_, clock), _parse_day(to, clock)
    state = State(settings, clock)
    state.migrate()
    calendar = TradingCalendar.from_store(settings, clock)
    before = {d for d in calendar.sessions(date(2010, 1, 1), calendar.last_completed_session(clock.now()))}
    nse = NseClient(clock)
    ictx = IngestContext(settings=settings, clock=clock, calendar=calendar, nse=nse, state=state)

    def show(r) -> None:
        if r.status != "skipped":
            typer.echo(f"{r.date} {r.dataset:<18} {r.status:<14} rows={r.rows}", color=False)
            sys.stdout.flush()

    try:
        days = repair_days(ictx, b) if repair else None
        results = ingest_eq_range(ictx, a, b, days=days, on_result=show)
    finally:
        nse.close()
    counts = Counter((r.dataset, r.status) for r in results)
    typer.echo("summary (dataset, status): count")
    for (ds, st), n in sorted(counts.items()):
        typer.echo(f"  {ds:<18} {st:<14} {n}")
    ok_eq = sorted(r.date for r in results if r.dataset == "bhavcopy_eq" and r.status == "ok")
    typer.echo(f"eq sessions added: {len(ok_eq)}")
    for d in ok_eq:
        tag = "" if d in before else "  (new date)"
        typer.echo(f"  {d} {d:%a}{tag}")
    if repair and mcp_check:
        pairs = sample_pairs(settings, ok_eq, 20, seed=11)
        res = cross_check_mcp(settings, pairs)
        typer.echo(f"mcp cross-check: {res.status} checked={res.checked} mismatches={res.mismatches}")
        if res.status == "mismatch":
            raise typer.Exit(1)


def _ingest_context(ctx: typer.Context):
    """(IngestContext, NseClient, SystemClock) wired to the real store and the live NSE client."""
    from stocksense.data.ingest_common import IngestContext
    from stocksense.data.nse_http import NseClient

    settings = _settings(ctx)
    clock = SystemClock()
    state = State(settings, clock)
    state.migrate()
    calendar = TradingCalendar.from_store(settings, clock)
    nse = NseClient(clock)
    return IngestContext(settings=settings, clock=clock, calendar=calendar, nse=nse, state=state), nse, clock


@ingest_app.command("reference")
def ingest_reference_cmd(ctx: typer.Context) -> None:
    """Holidays, symbol changes, listings, ETFs, sectors, price bands and the F&O list."""
    from stocksense.data.ingest_reference import ingest_reference

    ictx, nse, _ = _ingest_context(ctx)
    try:
        results = ingest_reference(ictx)
    finally:
        nse.close()
    for r in results:
        typer.echo(f"{r.dataset:<18} {r.status:<10} rows={r.rows}")
    if any(r.status == "failed" for r in results):
        raise typer.Exit(1)


def _backup_dataset(settings: Settings, dataset: str, tag: str) -> Path:
    """Copy a dataset's folder to data_store/_backup_<tag>/<dataset>/ once (never overwrites)."""
    src = Path(settings.paths.data_root) / "parquet" / dataset
    dst = Path(settings.paths.data_root) / f"_backup_{tag}" / dataset
    if src.is_dir() and not dst.exists():
        shutil.copytree(src, dst)
    return dst


def _echo_result(r) -> None:
    if r.status != "skipped":
        typer.echo(f"{r.date} {r.dataset:<18} {r.status:<14} rows={r.rows}", color=False)
        sys.stdout.flush()


def _echo_summary(results) -> None:
    from collections import Counter

    counts = Counter((r.dataset, r.status) for r in results)
    typer.echo("summary (dataset, status): count")
    for (ds, st), n in sorted(counts.items()):
        typer.echo(f"  {ds:<18} {st:<14} {n}")


@ingest_app.command("fo")
def ingest_fo_cmd(
    ctx: typer.Context,
    from_: Annotated[str, typer.Option("--from", help="First day, YYYY-MM-DD.")],
    to: Annotated[str, typer.Option("--to", help="Last day, YYYY-MM-DD or 'yesterday'.")],
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild", help="First re-parse every cached file in [from, to] over the stored rows."
        ),
    ] = False,
) -> None:
    """F&O bhavcopy for every calendar day in [from, to] (never before 2024-07-08)."""
    from stocksense.data.ingest_fo import ingest_fo_range, rebuild_fo_from_cache

    ictx, nse, clock = _ingest_context(ctx)
    a, b = _parse_day(from_, clock), _parse_day(to, clock)
    try:
        _backup_dataset(ictx.settings, "fo_bhavcopy", "t2x")
        if rebuild:
            per_day = rebuild_fo_from_cache(ictx, a, b)
            typer.echo(f"rebuilt {len(per_day)} cached days, {sum(per_day.values())} rows")
        results = ingest_fo_range(ictx, a, b, on_result=_echo_result)
    finally:
        nse.close()
    _echo_summary(results)


@ingest_app.command("index")
def ingest_index_cmd(
    ctx: typer.Context,
    to: Annotated[str, typer.Option("--to", help="Last day, YYYY-MM-DD or 'yesterday'.")],
    from_: Annotated[str | None, typer.Option("--from", help="First day, YYYY-MM-DD.")] = None,
    backfill: Annotated[
        bool,
        typer.Option("--backfill", help="Walk backward from --to until 60 consecutive days have no file."),
    ] = False,
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Re-parse every cached file with the current name map first.")
    ] = False,
) -> None:
    """Index values and India VIX for every calendar day in [from, to], or backfill backward from --to."""
    from stocksense.data.ingest_index import backfill_index, ingest_index_range, rebuild_index_from_cache

    ictx, nse, clock = _ingest_context(ctx)
    b = _parse_day(to, clock)
    if not backfill and from_ is None and not rebuild:
        raise typer.BadParameter("give --from, or --backfill, or --rebuild")
    try:
        _backup_dataset(ictx.settings, "india_vix", "t2x")
        if rebuild:
            per_day = rebuild_index_from_cache(ictx)
            typer.echo(f"rebuilt {len(per_day)} cached days")
        results = []
        if backfill:
            results = backfill_index(ictx, b, on_result=_echo_result)
        elif from_ is not None:
            results = ingest_index_range(ictx, _parse_day(from_, clock), b, on_result=_echo_result)
    finally:
        nse.close()
    _echo_summary(results)
    ok = sorted(r.date for r in results if r.dataset == "index_daily" and r.status == "ok")
    if ok:
        typer.echo(f"index days added: {len(ok)}, {ok[0]} .. {ok[-1]}")


@ingest_app.command("corpact")
def ingest_corpact_cmd(
    ctx: typer.Context,
    from_: Annotated[str | None, typer.Option("--from", help="First ex-date (default: today - 10).")] = None,
    to: Annotated[str | None, typer.Option("--to", help="Last ex-date (default: today + 45).")] = None,
    reparse: Annotated[
        bool, typer.Option("--reparse", help="Re-run the parser over stored unparsed/rights/other rows.")
    ] = False,
) -> None:
    """Corporate actions in 31-day windows, or --reparse the stored unparsed rows."""
    from stocksense.data.ingest_corpact import (
        FUTURE_DAYS,
        ingest_corpact_range,
        reparse_unparsed,
        table_counts,
    )

    ictx, nse, clock = _ingest_context(ctx)
    _backup_dataset(ictx.settings, "corporate_actions", "t2x")
    today = clock.today()
    try:
        if reparse:
            before = table_counts(ictx)
            reparse_unparsed(ictx)
            after = table_counts(ictx)
            typer.echo(f"{'action_type':<32}{'status':<10}{'before':>8}{'after':>8}")
            for key in sorted(set(before) | set(after)):
                typer.echo(f"{key[0]:<32}{key[1]:<10}{before[key]:>8}{after[key]:>8}")
            typer.echo(f"{'TOTAL':<42}{sum(before.values()):>8}{sum(after.values()):>8}")
            return
        a = _parse_day(from_, clock) if from_ else today - timedelta(days=10)
        b = _parse_day(to, clock) if to else today + timedelta(days=FUTURE_DAYS)
        results = ingest_corpact_range(ictx, a, b, on_result=_echo_result)
    finally:
        nse.close()
    _echo_summary(results)
    typer.echo(f"rows: {sum(r.rows for r in results)}")


@ingest_app.command("announcements")
def ingest_announcements_cmd(
    ctx: typer.Context,
    to: Annotated[str, typer.Option("--to", help="Newest day, YYYY-MM-DD or 'yesterday'.")] = "yesterday",
    from_: Annotated[
        str, typer.Option("--from", help="Oldest day (the walk stops earlier after 12 empty windows).")
    ] = "2010-01-01",
    poll: Annotated[bool, typer.Option("--poll", help="Live poll of today instead of a backfill.")] = False,
) -> None:
    """Corporate announcements: 7-day windows backward from --to (resumable), or --poll today."""
    from stocksense.data.ingest_announcements import backfill_announcements, poll_announcements

    ictx, nse, clock = _ingest_context(ctx)
    try:
        if poll:
            typer.echo(f"new announcements: {poll_announcements(ictx)}")
            return
        results = backfill_announcements(ictx, _parse_day(from_, clock), _parse_day(to, clock), _echo_result)
    finally:
        nse.close()
    _echo_summary(results)
    typer.echo(f"rows: {sum(r.rows for r in results)}")


@ingest_app.command("board-meetings")
def ingest_board_meetings_cmd(
    ctx: typer.Context,
    from_: Annotated[str, typer.Option("--from", help="Oldest meeting date.")] = "2010-01-01",
    to: Annotated[str | None, typer.Option("--to", help="Newest meeting date (default today + 60).")] = None,
) -> None:
    """Board meetings in 30-day windows backward from --to."""
    from stocksense.data.ingest_announcements import ingest_board_meetings

    ictx, nse, clock = _ingest_context(ctx)
    b = _parse_day(to, clock) if to else clock.today() + timedelta(days=60)
    try:
        results = ingest_board_meetings(ictx, _parse_day(from_, clock), b, _echo_result)
    finally:
        nse.close()
    _echo_summary(results)
    typer.echo(f"rows: {sum(r.rows for r in results)}")


@ingest_app.command("results")
def ingest_results_cmd(
    ctx: typer.Context,
    from_: Annotated[str, typer.Option("--from", help="Oldest filing date.")] = "2010-01-01",
    to: Annotated[str, typer.Option("--to", help="Newest filing date.")] = "yesterday",
) -> None:
    """Financial-results filings index (Quarterly and Annual) in 31-day windows backward from --to."""
    from stocksense.data.ingest_announcements import ingest_results_index

    ictx, nse, clock = _ingest_context(ctx)
    try:
        results = ingest_results_index(ictx, _parse_day(from_, clock), _parse_day(to, clock), _echo_result)
    finally:
        nse.close()
    _echo_summary(results)
    typer.echo(f"rows: {sum(r.rows for r in results)}")


@app.command("build-panel")
def build_panel(ctx: typer.Context) -> None:
    """Build the security master, adjusted daily panel, universe and market frame."""
    import time

    from stocksense.panel import build_all

    settings = _settings(ctx)
    t0 = time.monotonic()
    build_all(settings)
    elapsed = time.monotonic() - t0
    daily = read(settings, "panel_daily")
    typer.echo(f"panel_daily rows: {daily.select(pl.len()).collect().item()}")
    by_src = daily.group_by("gap_src").agg(pl.len().alias("n")).sort("n", descending=True).collect()
    typer.echo("gap_src counts:")
    for src, n in by_src.iter_rows():
        typer.echo(f"  {src or '(none)':<16}{n:>10}")
    for name in ("security_master", "panel_universe", "panel_market", "adj_events"):
        n = read(settings, name).select(pl.len()).collect().item()
        typer.echo(f"{name} rows: {n}")
    typer.echo(f"build-panel took {elapsed:.1f} s")


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


def _broker_lines(settings: Settings) -> list[str]:
    """Live read-only broker checks; amounts are never printed, only that they were read."""
    from stocksense.brokers.angel import AngelClient
    from stocksense.brokers.upstox import UpstoxClient
    from stocksense.core.errors import BrokerError

    clock = SystemClock()
    lines: list[str] = []
    sec = settings.secrets
    if not (sec.angel_api_key and sec.angel_client_id and sec.angel_pin and sec.angel_totp_secret):
        lines.append("angel: not configured")
    else:
        angel = AngelClient(settings, clock)
        try:
            angel.login()
            angel.funds()
            lines.append("angel: login ok, available cash read")
        except (BrokerError, OSError) as e:
            lines.append(f"angel: FAILED ({type(e).__name__}: {str(e)[:80]})")
        finally:
            angel.close()
    if not sec.upstox_analytics_token:
        lines.append("upstox: not configured")
    else:
        up = UpstoxClient(settings, clock)
        try:
            today = clock.today()
            df = up.candles("NSE_EQ|INE002A01018", "days", 1, today - timedelta(days=10), today)
            lines.append(f"upstox: token ok, {df.height} daily candles read")
        except (BrokerError, OSError) as e:
            lines.append(f"upstox: FAILED ({type(e).__name__}: {str(e)[:80]})")
        finally:
            up.close()
    return lines


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

    out("brokers")
    for line in _broker_lines(settings):
        out(f"  {line}")
