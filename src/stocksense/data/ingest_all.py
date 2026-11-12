from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

import polars as pl

from stocksense.core.log import get_logger
from stocksense.data.ingest_common import IngestContext, IngestResult
from stocksense.data.io import list_partitions, read

log = get_logger(__name__)

#: datasets whose staleness makes `ingest --all --catch-up` exit 1
REQUIRED = ("bhavcopy_eq", "bhavcopy_delivery", "corporate_actions")

Step = tuple[str, Callable[[IngestContext], list[IngestResult]]]


def last_date(ctx: IngestContext, dataset: str) -> date | None:
    """Newest stored date of a dataset (None if it has no files)."""
    from stocksense.data.catalog import get_spec

    if not list_partitions(ctx.settings, dataset):
        return None
    col = get_spec(dataset).date_col
    if col is None:
        return None
    # only the newest partition file is scanned: its bounds come from the name via the start bound
    files = list_partitions(ctx.settings, dataset)
    import re

    m = re.match(r"part-(\d{4})(?:-(\d{2}))?\.parquet", files[-1].name)
    start = date(int(m[1]), int(m[2] or 1), 1) if m else None
    df = read(ctx.settings, dataset, start=start, columns=[col]).collect()
    if df.height == 0:
        return None
    top = df.select(pl.col(col).max()).item()
    return top.date() if hasattr(top, "date") else top


def _since(ctx: IngestContext, dataset: str, fallback_days: int, overlap: int = 0) -> date:
    last = last_date(ctx, dataset)
    today = ctx.clock.today()
    return (last - timedelta(days=overlap - 1)) if last else today - timedelta(days=fallback_days)


def _one_minute_running(ctx: IngestContext) -> bool:
    """True if a 1-minute backfill is mid-run (a `running` bars_1m job started in the last 6 hours)."""
    con = ctx.state.connect()
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM job_runs WHERE job LIKE 'bars_1m:%' AND status='running' "
            "AND started_at > ?",
            ((ctx.clock.now() - timedelta(hours=6)).isoformat(timespec="seconds"),),
        ).fetchone()
    finally:
        con.close()
    return bool(row and row[0])


def default_steps() -> list[Step]:
    """The catch-up order: reference, eq + delivery, fo, index/vix, corporate actions, announcements,
    board meetings and results, macro and misc, intraday 5m and 1m."""

    def reference(ctx):
        from stocksense.data.ingest_reference import ingest_reference

        return ingest_reference(ctx)

    def eq(ctx):
        from stocksense.data.ingest_eq import ingest_eq_range

        a = _since(ctx, "bhavcopy_eq", 10, overlap=3)
        return ingest_eq_range(ctx, a, ctx.clock.today())

    def fo(ctx):
        from stocksense.data.ingest_fo import ingest_fo_range

        a = _since(ctx, "fo_bhavcopy", 10, overlap=3)
        return ingest_fo_range(ctx, a, ctx.clock.today())

    def index(ctx):
        from stocksense.data.ingest_index import ingest_index_range

        a = _since(ctx, "index_daily", 10, overlap=3)
        return ingest_index_range(ctx, a, ctx.clock.today())

    def corpact(ctx):
        from stocksense.data.ingest_corpact import FUTURE_DAYS, ingest_corpact_range

        t = ctx.clock.today()
        return ingest_corpact_range(ctx, t - timedelta(days=10), t + timedelta(days=FUTURE_DAYS))

    def announcements(ctx):
        from stocksense.data.ingest_announcements import backfill_announcements

        t = ctx.clock.today()
        a = _since(ctx, "announcements", 3, overlap=2)
        return backfill_announcements(ctx, min(a, t - timedelta(days=1)), t)

    def board_and_results(ctx):
        from stocksense.data.ingest_announcements import ingest_board_meetings, ingest_results_index

        t = ctx.clock.today()
        return [
            *ingest_board_meetings(ctx, t - timedelta(days=60), t + timedelta(days=60)),
            *ingest_results_index(ctx, t - timedelta(days=62), t),
        ]

    def macro_misc(ctx):
        from stocksense.data.ingest_macro import ingest_macro
        from stocksense.data.ingest_misc import (
            backfill_participant_oi,
            ingest_deals_today,
            ingest_fii_dii,
            ingest_participant_oi_range,
        )

        t = ctx.clock.today()
        out = ingest_macro(ctx)
        last = last_date(ctx, "participant_oi")
        if last is None:
            out += backfill_participant_oi(ctx, t - timedelta(days=1))
        else:
            out += ingest_participant_oi_range(ctx, last + timedelta(days=1), t)
        out.append(ingest_fii_dii(ctx))
        out += ingest_deals_today(ctx)
        return out

    def intraday(ctx):
        from stocksense.data.ingest_intraday import backfill_1m, extend_5m

        sec = ctx.settings.secrets
        out: list[IngestResult] = []
        if sec.angel_api_key and sec.angel_client_id and sec.angel_pin and sec.angel_totp_secret:
            out += extend_5m(ctx)
        if sec.upstox_analytics_token and list_partitions(ctx.settings, "bars_1m"):
            if _one_minute_running(ctx):
                log.info("catch_up_skip_1m", reason="a 1-minute backfill is running")
            else:
                t = ctx.clock.today()
                syms = (
                    read(ctx.settings, "bars_1m", start=t - timedelta(days=62), columns=["symbol"])
                    .select(pl.col("symbol").unique())
                    .collect()["symbol"]
                    .to_list()
                )
                out += backfill_1m(ctx, sorted(syms), (t - timedelta(days=45)).replace(day=1))
        return out

    return [
        ("reference", reference),
        ("eq", eq),
        ("fo", fo),
        ("index", index),
        ("corporate_actions", corpact),
        ("announcements", announcements),
        ("board_results", board_and_results),
        ("macro_misc", macro_misc),
        ("intraday", intraday),
    ]


def catch_up(ctx: IngestContext, steps: list[Step] | None = None) -> list[IngestResult]:
    """Run every ingester in order, each from its own last stored date to the last completed session.

    A failure in one step does not stop the others: it is logged and reported as a `failed` result.
    Returns every result.
    """
    results: list[IngestResult] = []
    for name, fn in steps if steps is not None else default_steps():
        try:
            res = fn(ctx)
        except Exception as e:  # noqa: BLE001 - job boundary: log with traceback, keep going
            log.exception("catch_up_step_failed", step=name, error=str(e))
            res = [IngestResult(name, ctx.clock.today(), 0, "failed")]
        results.extend(res)
        log.info("catch_up_step", step=name, results=len(res), failed=sum(r.status == "failed" for r in res))
    return results


def exit_code(table: pl.DataFrame) -> int:
    """1 if any required dataset (eq, delivery, corporate actions) is not fresh, else 0."""
    req = table.filter(pl.col("dataset").is_in(list(REQUIRED)))
    if req.height < len(REQUIRED):
        return 1
    return 0 if req["fresh"].fill_null(False).all() else 1
