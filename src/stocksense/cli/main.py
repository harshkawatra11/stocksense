"""StockSense CLI -- the only way anything in this system starts.

There are deliberately NO Windows scheduled tasks. Nothing runs because the
machine booted; everything runs because a command was issued (by the user, or by
Claude Code on the user's behalf). That is a design decision, not an omission:
a background job that dies silently on a stale token or a locked database is
worse than no job at all, and this project has been bitten by exactly that.
"""

from __future__ import annotations

import typer

from stocksense import __version__

app = typer.Typer(
    name="stocksense",
    help="A local quant stack for NSE cash equities.",
    no_args_is_help=True,
    add_completion=False,
)

probe_app = typer.Typer(help="Q0 environment probes -- run these before trusting anything else.")
app.add_typer(probe_app, name="probe")


@app.command()
def version() -> None:
    """Print the version and exit."""
    typer.echo(f"stocksense {__version__}")


@app.command()
def doctor() -> None:
    """One-shot environment report: interpreter, deps, GPU, feeds, credentials."""
    import platform
    import sys

    from stocksense.core.config import get_settings

    typer.echo(f"stocksense {__version__}")
    typer.echo(f"python     {sys.version.split()[0]}  ({platform.platform()})")

    for mod in ("duckdb", "pandas", "numpy", "polars", "scipy", "sklearn", "lightgbm", "numba"):
        try:
            m = __import__(mod)
            typer.echo(f"  {mod:<12} {getattr(m, '__version__', 'ok')}")
        except Exception as exc:  # noqa: BLE001
            typer.secho(f"  {mod:<12} MISSING/BROKEN: {type(exc).__name__}", fg=typer.colors.RED)

    s = get_settings()
    typer.echo("credentials:")
    for k, v in s.redacted().items():
        if v in ("<set>", "<unset>"):
            colour = typer.colors.GREEN if v == "<set>" else typer.colors.YELLOW
            typer.secho(f"  {k:<22} {v}", fg=colour)


@app.command("backfill-daily")
def backfill_daily_cmd(
    start: str = typer.Option("2010-01-01", help="Start date YYYY-MM-DD"),
    end: str = typer.Option(None, help="End date YYYY-MM-DD (default: today)"),
    no_delivery: bool = typer.Option(False, "--no-delivery", help="Skip the delivery-% feed"),
    no_resume: bool = typer.Option(False, "--no-resume", help="Re-fetch days already recorded ok"),
) -> None:
    """Backfill the daily NSE bhavcopy spine (and delivery-%) into the store.

    Genuinely resumable: every day is fetched, written and recorded individually,
    so interrupting this loses at most the day in flight. Re-running skips days
    already recorded ok/empty without issuing a request. Fetches are cached to
    disk, so a replay is fast rather than another few hours of network.
    """
    from datetime import date as _date
    from datetime import datetime as _dt

    from stocksense.core.config import get_settings
    from stocksense.data.nse_bhavcopy import backfill
    from stocksense.data.store import Store, StoreLocked

    s = get_settings()
    start_d = _dt.strptime(start, "%Y-%m-%d").date()
    end_d = _dt.strptime(end, "%Y-%m-%d").date() if end else _date.today()

    typer.echo(f"backfilling {start_d} -> {end_d}")
    try:
        store_cm = Store(s.duckdb_path, s.parquet_root)
    except StoreLocked as exc:
        typer.secho(str(exc), fg=typer.colors.YELLOW)
        raise typer.Exit(code=1) from None
    with store_cm as store:
        stats = backfill(
            store,
            s.data_store / "cache",
            start_d,
            end_d,
            with_delivery=not no_delivery,
            resume=not no_resume,
        )
    typer.echo(
        f"done: {stats['days_ok']} day(s) ingested, {stats['days_empty']} holiday/empty, "
        f"{stats['days_failed']} failed, {stats['skipped']} skipped, "
        f"{stats['rows']:,} rows, {stats['delivery_rows']:,} delivery rows"
    )
    if stats["days_failed"]:
        typer.secho(
            f"{stats['days_failed']} day(s) failed -- re-run this command to retry just those.",
            fg=typer.colors.YELLOW,
        )


@app.command("backfill-corporate-actions")
def backfill_ca_cmd(
    start: str = typer.Option("2010-01-01", help="Start date YYYY-MM-DD"),
    end: str = typer.Option(None, help="End date YYYY-MM-DD (default: today)"),
    no_resume: bool = typer.Option(False, "--no-resume", help="Re-fetch windows already recorded ok"),
) -> None:
    """Backfill NSE corporate actions (splits, bonuses, dividends).

    Nothing downstream is valid without these: bhavcopy carries RAW prices, so a
    1:10 split reads as a 90% crash unless it is adjusted away.
    """
    from datetime import date as _date
    from datetime import datetime as _dt

    from stocksense.core.config import get_settings
    from stocksense.data.corporate_actions import backfill
    from stocksense.data.store import Store, StoreLocked

    s = get_settings()
    start_d = _dt.strptime(start, "%Y-%m-%d").date()
    end_d = _dt.strptime(end, "%Y-%m-%d").date() if end else _date.today()

    typer.echo(f"backfilling corporate actions {start_d} -> {end_d}")
    try:
        store_cm = Store(s.duckdb_path, s.parquet_root)
    except StoreLocked as exc:
        typer.secho(str(exc), fg=typer.colors.YELLOW)
        raise typer.Exit(code=1) from None
    with store_cm as store:
        stats = backfill(store, s.data_store / "cache" / "corpact", start_d, end_d,
                         resume=not no_resume)
    typer.echo(
        f"done: {stats['actions']:,} action(s) across {stats['windows_ok']} window(s), "
        f"{stats['unparsed']:,} unparsed, {stats['windows_failed']} failed, "
        f"{stats['skipped']} skipped"
    )


@app.command("data-status")
def data_status_cmd() -> None:
    """What is actually ingested. Read-only: safe while a backfill is running."""
    from stocksense.core.config import get_settings
    from stocksense.data.store import Reader

    s = get_settings()
    with Reader(s.parquet_root) as r:
        lo, hi = r.bhavcopy_bounds()
        if lo is None:
            typer.echo("bhavcopy: EMPTY -- run 'stocksense backfill-daily'")
            return
        n = r.sql("SELECT count(*) c FROM {bhavcopy_eq}").c.iloc[0]
        days = r.sql("SELECT count(DISTINCT date) c FROM {bhavcopy_eq}").c.iloc[0]
        syms = r.sql("SELECT count(DISTINCT symbol) c FROM {bhavcopy_eq}").c.iloc[0]
        typer.echo(f"bhavcopy : {n:,} rows | {days:,} trading days | {syms:,} symbols | {lo} -> {hi}")
        # ingest_runs is an APPEND-ONLY attempt log -- one row per attempt, so a
        # unit that failed and was later repaired has BOTH rows. Counting rows
        # naively reports long-fixed failures forever, which is a monitor that
        # cries wolf. Report the LATEST status per unit instead, and show the
        # retry history separately.
        runs = r.ingest_runs(limit=1_000_000)
        if not runs.empty:
            latest = (
                runs.sort_values("started_at")
                .groupby(["source", "unit"], as_index=False)
                .last()
            )
            typer.echo(f"units    : {latest.status.value_counts().to_dict()}")
            retried = len(runs) - len(latest)
            if retried:
                typer.echo(f"retries  : {retried} superseded attempt(s) in the log")
            still_failing = latest[latest.status == "failed"]
            if not still_failing.empty:
                typer.secho(
                    f"FAILING  : {len(still_failing)} unit(s) still unresolved, "
                    f"e.g. {still_failing.unit.head(5).tolist()} "
                    f"-- re-run backfill-daily to retry just those.",
                    fg=typer.colors.YELLOW,
                )
            else:
                typer.secho("failing  : none", fg=typer.colors.GREEN)


@probe_app.command("network")
def probe_network() -> None:
    """Q0.1a/Q0.6 -- public IP stability and which data feeds this ISP can reach."""
    from stocksense.probes.base import run_probe
    from stocksense.probes.q0_network import probe_feed_reachability, probe_public_ip

    run_probe(
        "q0_1_public_ip",
        "Is this machine's public IP stable enough to register as Angel One's static IP?",
        probe_public_ip,
    )
    run_probe(
        "q0_6_feed_reachability",
        "Which market-data and news feeds are reachable from this ISP?",
        probe_feed_reachability,
    )


@probe_app.command("compute")
def probe_compute() -> None:
    """Q0.4/Q0.5 -- GPU headroom for Monte Carlo, and whether torch has CUDA here."""
    from stocksense.probes.base import run_probe
    from stocksense.probes.q0_compute import probe_gpu, probe_torch_cuda

    run_probe("q0_4_gpu", "How many Monte Carlo paths fit in the 4GB VRAM budget?", probe_gpu)
    run_probe("q0_5_torch_cuda", "Is there a CUDA-enabled torch wheel for this Python?", probe_torch_cuda)


@probe_app.command("upstox")
def probe_upstox_cmd() -> None:
    """Q0.2 -- does the stored Upstox token still authenticate, and how deep is 1-min history?"""
    from stocksense.probes.base import run_probe
    from stocksense.probes.q0_brokers import probe_upstox

    run_probe("q0_2_upstox", "Does the Upstox market-data feed work from here today?", probe_upstox)


@probe_app.command("angel")
def probe_angel_cmd() -> None:
    """Q0.3 -- read-only Angel One login and account reads. Places no orders."""
    from stocksense.probes.base import run_probe
    from stocksense.probes.q0_brokers import probe_angel_readonly

    run_probe(
        "q0_3_angel_readonly",
        "Can we log in to Angel One and read holdings/positions/tradebook from this ISP?",
        probe_angel_readonly,
    )


@probe_app.command("all")
def probe_all() -> None:
    """Run every Q0 probe in order and print a summary table."""
    from stocksense.probes.base import run_probe
    from stocksense.probes.q0_brokers import probe_angel_readonly, probe_upstox
    from stocksense.probes.q0_compute import probe_gpu, probe_torch_cuda
    from stocksense.probes.q0_network import probe_feed_reachability, probe_public_ip

    specs = [
        ("q0_1_public_ip", "Public IP stable enough to register as a static IP?", probe_public_ip),
        ("q0_6_feed_reachability", "Which feeds are reachable from this ISP?", probe_feed_reachability),
        ("q0_2_upstox", "Does the Upstox feed authenticate today?", probe_upstox),
        ("q0_3_angel_readonly", "Can we read the Angel One account from here?", probe_angel_readonly),
        ("q0_4_gpu", "How many Monte Carlo paths fit in VRAM?", probe_gpu),
        ("q0_5_torch_cuda", "Is there a CUDA torch wheel for this Python?", probe_torch_cuda),
    ]
    results = [run_probe(n, q, f) for n, q, f in specs]

    typer.echo("\n=== Q0 SUMMARY ===")
    colours = {
        "PASS": typer.colors.GREEN,
        "FAIL": typer.colors.RED,
        "BLOCKED": typer.colors.YELLOW,
        "UNKNOWN": typer.colors.WHITE,
    }
    for r in results:
        typer.secho(f"  {r.verdict:<8} {r.name}", fg=colours.get(r.verdict, typer.colors.WHITE))


if __name__ == "__main__":
    app()
