from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import httpx
import polars as pl

from stocksense.core.errors import DataNotAvailable
from stocksense.core.log import get_logger
from stocksense.data.ingest_common import IngestContext, IngestResult
from stocksense.data.io import list_partitions, read, upsert

log = get_logger(__name__)

DATASET = "corporate_actions"
WINDOW_DAYS = 31
FUTURE_DAYS = 45  # future ex-dates are inputs to features and risk
CACHE_FINAL_AGE_DAYS = 10  # a window that ended longer ago than this is final: cache is never refetched
_RS = r"\b(?:rs\.?|re\.?)"
_NUM = r"(\d+(?:\.\d+)?)"

_RE_SPLIT_KW = re.compile(r"(?:face value )?split|sub-?division")
_RE_CONSOL_KW = re.compile(r"consolidation")
_RE_FROM_TO = re.compile(rf"from\s+{_RS}\s*{_NUM}\s*/?-?\s*(?:per share\s*)?to\s+{_RS}\s*{_NUM}")
_RE_BONUS = re.compile(r"bonus\s*(?:issue\s*)?(\d+)\s*:\s*(\d+)")
_RE_BONUS_KW = re.compile(r"bonus")
_RE_RIGHTS = re.compile(r"rights?\s*(?:issue)?\s*[-:]?\s*(\d+)\s*:\s*(\d+)")
_RE_RIGHTS_KW = re.compile(r"\brights?\b")
_RE_PREMIUM = re.compile(rf"@?\s*(?:a\s*)?prem(?:ium)?\s*(?:of\s*)?{_RS}\s*{_NUM}\s*(?:/-?)?")
_RE_AT_PAR = re.compile(rf"(?:at|@)\s*par\b(?:\s*{_RS}\s*{_NUM}\s*(?:/-?)?(?:\s*per\s+share)?)?")
_RE_AMOUNT = re.compile(rf"{_RS}\s*{_NUM}\s*(?:/-?)?\s*per\s+(?:equity\s+)?(?:share|unit|sh)\b")
_RE_DIVIDEND_KW = re.compile(r"dividend|\bdiv\b|divided")
_RE_DISTRIBUTION_KW = re.compile(r"distribution")
_RE_EVENT_KW = (
    ("buyback", re.compile(r"buy ?back")),
    ("demerger", re.compile(r"de-? ?merger")),
    ("scheme_of_arrangement", re.compile(r"scheme of arrangement")),
    ("amalgamation", re.compile(r"amalgamation|(?<!de)(?<!de-)(?<!de )merger")),
    ("capital_reduction", re.compile(r"capital reduction")),
)
_RE_NOISE = re.compile(
    r"annual general meeting|extra-?\s*ordinary general meeting|general meeting|\bagm\b|\begm\b"
    r"|interest payment|redemption|e-voting|annual closing"
)


@dataclass(frozen=True)
class ParsedAction:
    """A parsed corporate-action subject.

    factor_price is the price multiplier on the ex-date (new/old, 0.5 = halved; None where no factor is
    known); ratio_num/ratio_den are the issue ratio (bonus/rights A:B, split before:after); amounts are in
    rupees; parse_status is 'ok' or 'unparsed' ("price evidence decides").
    """

    action_type: str
    ratio_num: float | None = None
    ratio_den: float | None = None
    factor_price: float | None = None
    dividend_amount: float | None = None
    face_before: float | None = None
    face_after: float | None = None
    rights_price: float | None = None
    parse_status: str = "unparsed"


def _blank(text: str, m: re.Match[str]) -> str:
    return text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]


def parse_subject(subject: str, face_val: float | None) -> ParsedAction:
    """Parse an NSE corporate-action subject (rules of PLAN T2.6, applied in order and combined).

    Factors multiply, dividends add; action_type joins the matched kinds with '+' (bonus, split,
    consolidation, rights, dividend/distribution). `face_val` (rupees, from the API) is needed only for
    the rights price; it is never assumed.
    """
    text = re.sub(r"\s+", " ", subject.strip().lower())
    kinds: list[str] = []
    factors: list[float] = []
    ok = True
    ratio: tuple[float, float] | None = None
    faces: tuple[float | None, float | None] = (None, None)
    rights_price: float | None = None
    scan = text  # text with the spans already consumed by earlier rules blanked out

    # 1 and 2: split / sub-division and consolidation (both read "from Rs X to Rs Y")
    m_ft = _RE_FROM_TO.search(scan)
    is_split, is_consol = _RE_SPLIT_KW.search(scan), _RE_CONSOL_KW.search(scan)
    # rule 3 first in the type order (bonus before split) but the spans are consumed in card order
    bonus = _RE_BONUS.search(scan)
    if is_split or is_consol:
        kind = "split" if is_split else "consolidation"
        if m_ft and float(m_ft[1]) > 0 and float(m_ft[2]) > 0:
            before, after = float(m_ft[1]), float(m_ft[2])
            faces = (before, after)
            factors.append(after / before)
            if kind == "split":
                ratio = ratio or (before / after, 1.0)
            scan = _blank(scan, m_ft)
        else:
            ok = False  # amounts missing: price evidence decides
        kinds.append(kind)
    # 3: bonus A:B means A new shares for every B held
    if bonus:
        a, b = float(bonus[1]), float(bonus[2])
        if a + b > 0:
            factors.append(b / (a + b))
            ratio = (a, b)
            kinds.insert(0, "bonus")
            scan = _blank(scan, bonus)
        else:
            ok = False
    elif _RE_BONUS_KW.search(scan) and not kinds and not _RE_EVENT_KW[2][1].search(scan):
        kinds.append("bonus")
        ok = False
    # 4: rights
    rights = _RE_RIGHTS.search(scan)
    prem = _RE_PREMIUM.search(scan)
    if prem:
        scan = _blank(scan, prem)  # a premium is not a dividend
    par = _RE_AT_PAR.search(scan)
    if par:
        scan = _blank(scan, par)  # nor is a par issue price
    if rights:
        a, b = float(rights[1]), float(rights[2])
        ratio = ratio or (a, b)
        if prem and face_val is not None:
            rights_price = float(prem[1]) + face_val
        elif par and face_val is not None:
            rights_price = face_val
        ok = ok and rights_price is not None
        kinds.append("rights")
        scan = _blank(scan, rights)
    elif _RE_RIGHTS_KW.search(scan) and "rights" not in kinds and not factors:
        kinds.append("rights")
        ok = False

    # 5: dividend / distribution amounts (not for buy-backs, mergers etc. unless 'dividend' is said)
    event = next((name for name, rx in _RE_EVENT_KW if rx.search(text)), None)
    div_kw = _RE_DIVIDEND_KW.search(text) is not None
    dist_kw = _RE_DISTRIBUTION_KW.search(text) is not None
    head = re.split(r"comprising|consist", scan, maxsplit=1)[0]  # components after a total are not added
    pool = head if _RE_AMOUNT.search(head) else scan
    amounts = [float(x) for x in _RE_AMOUNT.findall(pool)] if (event is None or div_kw) else []
    dividend_amount = sum(amounts) if amounts else None
    if amounts or div_kw or dist_kw:
        kinds.append("distribution" if dist_kw and not div_kw else "dividend")
        ok = ok and dividend_amount is not None

    # 6: events with no factor (price evidence decides)
    if event is not None and not factors:
        kinds.append(event)
        ok = False
    # 7: noise, only when nothing else matched
    if not kinds:
        if _RE_NOISE.search(text):
            return ParsedAction("noise", parse_status="ok")
        return ParsedAction("other", parse_status="unparsed")

    factor = None
    if factors:
        factor = 1.0
        for f in factors:
            factor *= f
    return ParsedAction(
        action_type="+".join(dict.fromkeys(kinds)),
        ratio_num=ratio[0] if ratio else None,
        ratio_den=ratio[1] if ratio else None,
        factor_price=factor,
        dividend_amount=dividend_amount,
        face_before=faces[0],
        face_after=faces[1],
        rights_price=rights_price,
        parse_status="ok" if ok else "unparsed",
    )


# ---- API rows -----------------------------------------------------------------------------------


def _dash_to_none(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return None if s in ("", "-") else s


def _parse_day(v: Any) -> date | None:
    s = _dash_to_none(v)
    if s is None:
        return None
    try:
        return datetime.strptime(s.title(), "%d-%b-%Y").date()
    except ValueError:
        return None


def _float_or_none(v: Any) -> float | None:
    s = _dash_to_none(v)
    try:
        return None if s is None else float(s)
    except ValueError:
        return None


_OUT_SCHEMA: dict[str, Any] = {
    "symbol": pl.Utf8,
    "ex_date": pl.Date,
    "action_type": pl.Utf8,
    "ratio_num": pl.Float64,
    "ratio_den": pl.Float64,
    "factor_price": pl.Float64,
    "dividend_amount": pl.Float64,
    "face_before": pl.Float64,
    "face_after": pl.Float64,
    "subject_raw": pl.Utf8,
    "parse_status": pl.Utf8,
    "record_date": pl.Date,
    "series": pl.Utf8,
    "isin": pl.Utf8,
    "source": pl.Utf8,
    "rights_price": pl.Float64,
}


def _row_from(p: ParsedAction) -> dict[str, Any]:
    return {
        "action_type": p.action_type,
        "ratio_num": p.ratio_num,
        "ratio_den": p.ratio_den,
        # stored convention (all 35k adopted rows): 1.0 where no price factor is known; parse_status
        # tells 'price-neutral' (ok) from 'unknown' (unparsed)
        "factor_price": 1.0 if p.factor_price is None else p.factor_price,
        "dividend_amount": p.dividend_amount,
        "face_before": p.face_before,
        "face_after": p.face_after,
        "parse_status": p.parse_status,
        "rights_price": p.rights_price,
    }


def parse_corpact_rows(rows: list[dict[str, Any]], source: str = "nse_api") -> pl.DataFrame:
    """corporate_actions rows from the API's JSON list; rows without a valid ex-date are dropped."""
    out: list[dict[str, Any]] = []
    for r in rows:
        ex = _parse_day(r.get("exDate"))
        subject = (r.get("subject") or "").strip()
        sym = (r.get("symbol") or "").strip()
        if ex is None or not sym or not subject:
            continue
        p = parse_subject(subject, _float_or_none(r.get("faceVal")))
        out.append(
            {
                "symbol": sym,
                "ex_date": ex,
                "subject_raw": subject,
                "record_date": _parse_day(r.get("recDate")),
                "series": _dash_to_none(r.get("series")),
                "isin": _dash_to_none(r.get("isin")),
                "source": source,
                **_row_from(p),
            }
        )
    df = pl.DataFrame(out, schema=_OUT_SCHEMA) if out else pl.DataFrame(schema=_OUT_SCHEMA)
    return df.unique(subset=["symbol", "ex_date", "subject_raw"], keep="last", maintain_order=True)


# ---- ingest -------------------------------------------------------------------------------------


def windows(a: date, b: date) -> list[tuple[date, date]]:
    """Consecutive 31-day windows (inclusive both ends) covering [a, b]."""
    out: list[tuple[date, date]] = []
    cur = a
    while cur <= b:
        end = min(cur + timedelta(days=WINDOW_DAYS - 1), b)
        out.append((cur, end))
        cur = end + timedelta(days=1)
    return out


def _cache_path(ctx: IngestContext, a: date, b: date):
    return ctx.settings.paths.data_root / "cache" / "corpact" / f"{a:%Y-%m-%d}_{b:%Y-%m-%d}.json"


def _fetch_window(ctx: IngestContext, a: date, b: date) -> list[dict[str, Any]]:
    path = _cache_path(ctx, a, b)
    final = (ctx.clock.today() - b).days > CACHE_FINAL_AGE_DAYS
    if final and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = ctx.nse.get_api(
        "/api/corporates-corporateActions",
        {"index": "equities", "from_date": f"{a:%d-%m-%Y}", "to_date": f"{b:%d-%m-%Y}"},
        "companies-listing/corporate-filings-actions",
    )
    if not isinstance(payload, list):
        raise DataNotAvailable(f"corporate actions {a}..{b}: unexpected payload {type(payload).__name__}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def ingest_corpact_range(
    ctx: IngestContext,
    a: date,
    b: date,
    on_result: Callable[[IngestResult], None] | None = None,
) -> list[IngestResult]:
    """Fetch corporate actions with ex-dates in [a, b] in 31-day windows; one result per window."""
    out: list[IngestResult] = []
    for wa, wb in windows(a, b):
        try:
            rows = _fetch_window(ctx, wa, wb)
            df = parse_corpact_rows(rows)
        except (DataNotAvailable, httpx.HTTPError, ValueError) as e:
            log.warning("corpact_window_failed", start=str(wa), end=str(wb), error=str(e))
            res = IngestResult(DATASET, wa, 0, "failed")
        else:
            if df.height:
                upsert(ctx.settings, DATASET, df, clock=ctx.clock)
            log.info("ingest_done", dataset=DATASET, start=str(wa), end=str(wb), rows=df.height)
            res = IngestResult(DATASET, wa, df.height, "ok")
        out.append(res)
        if on_result is not None:
            on_result(res)
    return out


# ---- reparse ------------------------------------------------------------------------------------


def _counts(df: pl.DataFrame) -> Counter[tuple[str, str]]:
    return Counter(zip(df["action_type"].to_list(), df["parse_status"].to_list(), strict=True))


def reparse_unparsed(ctx: IngestContext) -> tuple[Counter[tuple[str, str]], Counter[tuple[str, str]]]:
    """Re-run the parser over stored rows that are 'unparsed' or typed rights/other; upsert the result.

    Returns (before, after): row counts by (action_type, parse_status) over the rows reparsed. Stored rows
    carry no face value, so a rights row keeps 'unparsed' (its ratio is filled in, its price stays null).
    """
    if not list_partitions(ctx.settings, DATASET):
        return Counter(), Counter()
    df = read(ctx.settings, DATASET).collect()
    todo = df.filter(
        (pl.col("parse_status") == "unparsed") | pl.col("action_type").is_in(["rights", "other"])
    )
    if todo.height == 0:
        return Counter(), Counter()
    new_rows: list[dict[str, Any]] = []
    for r in todo.iter_rows(named=True):
        p = parse_subject(r["subject_raw"] or "", None)
        new_rows.append({**r, **_row_from(p)})
    new = pl.DataFrame(new_rows, schema=todo.schema)
    upsert(ctx.settings, DATASET, new, clock=ctx.clock)
    return _counts(todo), _counts(new)


def table_counts(ctx: IngestContext) -> Counter[tuple[str, str]]:
    """Row counts of the whole stored table by (action_type, parse_status)."""
    if not list_partitions(ctx.settings, DATASET):
        return Counter()
    return _counts(read(ctx.settings, DATASET, columns=["action_type", "parse_status"]).collect())
