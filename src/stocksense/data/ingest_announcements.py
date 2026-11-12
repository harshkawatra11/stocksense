from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from stocksense.core.clock import IST
from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.data.ingest_common import IngestContext, IngestResult
from stocksense.data.io import read, upsert

log = get_logger(__name__)

DATASET = "announcements"
BOARD = "board_meetings"
RESULTS = "results_index"
EARLIEST = date(2010, 1, 1)
MAX_EMPTY_WINDOWS = 12
FLUSH_EVERY = 4  # windows per upsert of the month partitions (a month file is rewritten on each upsert)
#: a window whose row count reaches this is re-read day by day (the API might silently cap a response)
SPLIT_ABOVE = 9000
#: the truncation check of 2026-11-11 (docs/DATA.md): a 7-day call equals the sum of seven 1-day calls
WINDOW_DAYS = 7
CACHE_FINAL_AGE_DAYS = 2  # a window that ended longer ago than this is final and cached forever

_REF_ANN = "companies-listing/corporate-filings-announcements"
_REF_BM = "companies-listing/corporate-filings-board-meetings"
_REF_RES = "companies-listing/corporate-filings-financial-results"

_ANN_SCHEMA: dict[str, Any] = {
    "seq_id": pl.Utf8,
    "symbol": pl.Utf8,
    "ts": pl.Datetime("us", "Asia/Kolkata"),
    "exch_ts": pl.Datetime("us", "Asia/Kolkata"),
    "category": pl.Utf8,
    "summary": pl.Utf8,
    "attachment_url": pl.Utf8,
    "company": pl.Utf8,
    "isin": pl.Utf8,
    "has_xbrl": pl.Boolean,
    "first_seen_ts": pl.Datetime("us", "Asia/Kolkata"),
}
_BM_SCHEMA: dict[str, Any] = {
    "symbol": pl.Utf8,
    "meeting_date": pl.Date,
    "purpose": pl.Utf8,
    "description": pl.Utf8,
    "announced_ts": pl.Datetime("us", "Asia/Kolkata"),
}
_RES_SCHEMA: dict[str, Any] = {
    "seq_number": pl.Utf8,
    "symbol": pl.Utf8,
    "period_from": pl.Date,
    "period_to": pl.Date,
    "filing_ts": pl.Datetime("us", "Asia/Kolkata"),
    "exch_ts": pl.Datetime("us", "Asia/Kolkata"),
    "audited": pl.Utf8,
    "consolidated": pl.Utf8,
    "relating_to": pl.Utf8,
    "xbrl_url": pl.Utf8,
}

_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
    )
}


# ---- parsing ------------------------------------------------------------------------------------


def _clean(v: Any) -> str | None:
    """Stripped string, or None for null, '' and the API's '-' placeholder."""
    if v is None:
        return None
    s = str(v).strip()
    return None if s in ("", "-") else s


def _parse_ts(v: Any, with_seconds: bool = True) -> datetime | None:
    """`09-Nov-2026 23:45:59` (or `... 17:17` without seconds) as an IST datetime; None if unparsable."""
    s = _clean(v)
    if s is None:
        return None
    try:
        day, mon, rest = s.split("-", 2)
        year, _, clock = rest.partition(" ")
        hh, mm, *ss = (clock or "00:00:00").split(":")
        return datetime(
            int(year), _MONTHS[mon.lower()], int(day), int(hh), int(mm), int(ss[0]) if ss else 0, tzinfo=IST
        )
    except (ValueError, KeyError):
        return None


def _parse_day(v: Any) -> date | None:
    ts = _parse_ts(v)
    return ts.date() if ts else None


def _bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    s = _clean(v)
    if s is None:
        return None
    return s.lower() in ("true", "1", "y", "yes")


def announcement_id(symbol: str | None, an_dt: str | None, text: str | None) -> str:
    """Fallback key for a row without seq_id: sha1(symbol|an_dt|attchmntText)[:16]."""
    return hashlib.sha1(f"{symbol or ''}|{an_dt or ''}|{text or ''}".encode()).hexdigest()[:16]


def parse_announcements(rows: list[dict[str, Any]], first_seen: datetime | None = None) -> pl.DataFrame:
    """announcements rows from the API's JSON list; rows without symbol or parsable `an_dt` are dropped."""
    out: list[dict[str, Any]] = []
    for r in rows:
        sym = _clean(r.get("symbol"))
        ts = _parse_ts(r.get("an_dt"))
        if sym is None or ts is None:
            continue
        text = r.get("attchmntText")
        seq = _clean(r.get("seq_id")) or announcement_id(sym, r.get("an_dt"), text)
        out.append(
            {
                "seq_id": seq,
                "symbol": sym,
                "ts": ts,
                "exch_ts": _parse_ts(r.get("exchdisstime")),
                "category": _clean(r.get("desc")),
                "summary": _clean(text),
                "attachment_url": _clean(r.get("attchmntFile")),
                "company": _clean(r.get("sm_name")),
                "isin": _clean(r.get("sm_isin")),
                "has_xbrl": _bool(r.get("hasXbrl")),
                "first_seen_ts": first_seen,
            }
        )
    df = pl.DataFrame(out, schema=_ANN_SCHEMA) if out else pl.DataFrame(schema=_ANN_SCHEMA)
    return df.unique(subset=["seq_id"], keep="last", maintain_order=True)


def parse_board_meetings(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """board_meetings rows (bm_symbol, bm_date, bm_purpose, bm_desc, bm_timestamp); undated rows dropped."""
    out: list[dict[str, Any]] = []
    for r in rows:
        sym, md, at = (
            _clean(r.get("bm_symbol")),
            _parse_day(r.get("bm_date")),
            _parse_ts(r.get("bm_timestamp")),
        )
        if sym is None or md is None or at is None:
            continue
        out.append(
            {
                "symbol": sym,
                "meeting_date": md,
                "purpose": _clean(r.get("bm_purpose")),
                "description": _clean(r.get("bm_desc")),
                "announced_ts": at,
            }
        )
    df = pl.DataFrame(out, schema=_BM_SCHEMA) if out else pl.DataFrame(schema=_BM_SCHEMA)
    return df.unique(subset=["symbol", "meeting_date", "announced_ts"], keep="last", maintain_order=True)


def parse_results(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """results_index rows (seqNumber, filingDate `%d-%b-%Y %H:%M`, exchdisstime, fromDate, toDate, ...)."""
    out: list[dict[str, Any]] = []
    for r in rows:
        seq, sym, ft = _clean(r.get("seqNumber")), _clean(r.get("symbol")), _parse_ts(r.get("filingDate"))
        if seq is None or sym is None or ft is None:
            continue
        out.append(
            {
                "seq_number": seq,
                "symbol": sym,
                "period_from": _parse_day(r.get("fromDate")),
                "period_to": _parse_day(r.get("toDate")),
                "filing_ts": ft,
                "exch_ts": _parse_ts(r.get("exchdisstime")),
                "audited": _clean(r.get("audited")),
                "consolidated": _clean(r.get("consolidated")),
                "relating_to": _clean(r.get("relatingTo")),
                "xbrl_url": _clean(r.get("xbrl")),
            }
        )
    df = pl.DataFrame(out, schema=_RES_SCHEMA) if out else pl.DataFrame(schema=_RES_SCHEMA)
    return df.unique(subset=["seq_number"], keep="last", maintain_order=True)


# ---- windows and the raw cache -----------------------------------------------------------------


def backward_windows(start: date, end: date, days: int) -> list[tuple[date, date]]:
    """Consecutive windows of `days` days from `end` back to `start`, newest first (inclusive ends)."""
    out: list[tuple[date, date]] = []
    hi = end
    while hi >= start:
        lo = max(hi - timedelta(days=days - 1), start)
        out.append((lo, hi))
        hi = lo - timedelta(days=1)
    return out


def _cache_file(ctx: IngestContext, kind: str, a: date, b: date) -> Path:
    return ctx.settings.paths.data_root / "cache" / kind / f"{a:%Y-%m-%d}_{b:%Y-%m-%d}.json.gz"


def _read_cache(path: Path) -> list[dict[str, Any]]:
    return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))


def _write_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(rows).encode("utf-8")))
    tmp.replace(path)


def _fetch(
    ctx: IngestContext, kind: str, api_path: str, referer: str, a: date, b: date, extra: dict[str, str]
) -> list[dict[str, Any]]:
    """Rows of one window: served from `cache/<kind>/<from>_<to>.json.gz` when final, else the API."""
    path = _cache_file(ctx, kind, a, b)
    final = (ctx.clock.today() - b).days > CACHE_FINAL_AGE_DAYS
    if final and path.is_file():
        return _read_cache(path)
    payload = ctx.nse.get_api(
        api_path,
        {"index": "equities", "from_date": f"{a:%d-%m-%Y}", "to_date": f"{b:%d-%m-%Y}", **extra},
        referer,
    )
    if not isinstance(payload, list):
        raise DataNotAvailable(f"{api_path} {a}..{b}: unexpected payload {type(payload).__name__}")
    if final:
        _write_cache(path, payload)
    return payload


def fetch_announcement_window(ctx: IngestContext, a: date, b: date) -> list[dict[str, Any]]:
    """Announcement rows for [a, b]; an oversized multi-day window is re-read one day at a time."""
    rows = _fetch(ctx, "announcements", "/api/corporate-announcements", _REF_ANN, a, b, {})
    if len(rows) >= SPLIT_ABOVE and a < b:
        log.warning("announcement_window_split", start=str(a), end=str(b), rows=len(rows))
        merged: dict[str, dict[str, Any]] = {}
        d = a
        while d <= b:
            for r in _fetch(ctx, "announcements", "/api/corporate-announcements", _REF_ANN, d, d, {}):
                merged[str(r.get("seq_id") or id(r))] = r
            d += timedelta(days=1)
        rows = list(merged.values())
    return rows


# ---- announcements ------------------------------------------------------------------------------


def backfill_announcements(
    ctx: IngestContext,
    start: date,
    end: date,
    on_result: Callable[[IngestResult], None] | None = None,
) -> list[IngestResult]:
    """Walk backward from `end` to `start` in 7-day windows; stop after 12 consecutive empty windows.

    Raw JSON of every final window is cached gzipped, so a rerun makes no HTTP call for finished
    windows (resumable). Rows are stored in batches of four windows; `first_seen_ts` stays null.
    """
    results: list[IngestResult] = []
    pending: list[pl.DataFrame] = []
    empties = 0

    def flush() -> None:
        if pending:
            upsert(ctx.settings, DATASET, pl.concat(pending), clock=ctx.clock)
            pending.clear()

    for n, (a, b) in enumerate(backward_windows(start, end, WINDOW_DAYS), 1):
        try:
            df = parse_announcements(fetch_announcement_window(ctx, a, b))
        except (DataNotAvailable, httpx.HTTPError, ValueError) as e:
            log.warning("announcement_window_failed", start=str(a), end=str(b), error=str(e))
            res = IngestResult(DATASET, a, 0, "failed")
        else:
            empties = empties + 1 if df.height == 0 else 0
            if df.height:
                pending.append(df)
            res = IngestResult(DATASET, a, df.height, "ok")
        results.append(res)
        if on_result is not None:
            on_result(res)
        if n % FLUSH_EVERY == 0:
            flush()
        if empties >= MAX_EMPTY_WINDOWS:
            log.info("announcement_backfill_stop", reason="empty_windows", at=str(a))
            break
    flush()
    return results


def poll_announcements(ctx: IngestContext) -> int:
    """Live poll: announcements of today; rows with a new seq_id get first_seen_ts = now. Returns new rows."""
    today = ctx.clock.today()
    rows = _fetch(ctx, "announcements", "/api/corporate-announcements", _REF_ANN, today, today, {})
    now = ctx.clock.now()
    df = parse_announcements(rows, first_seen=now)
    if df.height == 0:
        return 0
    known = read(ctx.settings, DATASET, start=today - timedelta(days=1), columns=["seq_id"]).collect()
    new = df.filter(~pl.col("seq_id").is_in(known["seq_id"].implode()))
    if new.height:
        upsert(ctx.settings, DATASET, new, clock=ctx.clock)
    return new.height


# ---- board meetings -----------------------------------------------------------------------------


def ingest_board_meetings(
    ctx: IngestContext,
    start: date,
    end: date,
    on_result: Callable[[IngestResult], None] | None = None,
) -> list[IngestResult]:
    """Board meetings with a meeting date in [start, end], 30-day windows from `end` backward;
    stops after 12 consecutive empty windows."""
    results: list[IngestResult] = []
    empties = 0
    for a, b in backward_windows(start, end, 30):
        try:
            df = parse_board_meetings(
                _fetch(ctx, "board_meetings", "/api/corporate-board-meetings", _REF_BM, a, b, {})
            )
        except (DataNotAvailable, httpx.HTTPError, ValueError) as e:
            log.warning("board_meetings_window_failed", start=str(a), end=str(b), error=str(e))
            res = IngestResult(BOARD, a, 0, "failed")
        else:
            empties = empties + 1 if df.height == 0 else 0
            if df.height:
                upsert(ctx.settings, BOARD, df, clock=ctx.clock)
            res = IngestResult(BOARD, a, df.height, "ok")
        results.append(res)
        if on_result is not None:
            on_result(res)
        if empties >= MAX_EMPTY_WINDOWS:
            break
    return results


# ---- results index ------------------------------------------------------------------------------

#: probe 2026-11-11: the API honours from_date/to_date (filter on filing date) and period=Quarterly|Annual
#: (Half-Yearly returns nothing); without dates it returns only recent filings.
RESULT_PERIODS = ("Quarterly", "Annual")


def ingest_results_index(
    ctx: IngestContext,
    start: date,
    end: date,
    on_result: Callable[[IngestResult], None] | None = None,
) -> list[IngestResult]:
    """Result filings with a filing date in [start, end]: 31-day windows from `end` backward, both
    periods; stops after 12 consecutive windows with no filings."""
    results: list[IngestResult] = []
    empties = 0
    for a, b in backward_windows(start, end, 31):
        total = 0
        try:
            for period in RESULT_PERIODS:
                df = parse_results(
                    _fetch(
                        ctx,
                        f"results_{period.lower()}",
                        "/api/corporates-financial-results",
                        _REF_RES,
                        a,
                        b,
                        {"period": period},
                    )
                )
                if df.height:
                    upsert(ctx.settings, RESULTS, df, clock=ctx.clock)
                total += df.height
        except (DataNotAvailable, httpx.HTTPError, ValueError) as e:
            log.warning("results_window_failed", start=str(a), end=str(b), error=str(e))
            res = IngestResult(RESULTS, a, 0, "failed")
        else:
            empties = empties + 1 if total == 0 else 0
            res = IngestResult(RESULTS, a, total, "ok")
        results.append(res)
        if on_result is not None:
            on_result(res)
        if empties >= MAX_EMPTY_WINDOWS:
            break
    return results
