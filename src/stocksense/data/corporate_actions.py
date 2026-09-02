"""NSE corporate actions: fetch, and parse the free-text `subject` field.

Nothing downstream is valid without this. NSE's bhavcopy carries **raw** prices
with no corporate-action adjustment, so a 1:10 split reads as a 90% crash. Feed
that to a feature engine and it learns that splits are catastrophes.

Every grammar below was derived from a survey of **35,074 real subject lines,
2010-2026**, not from imagination. The observed distribution:

    dividend            20,498      meeting              11,061
    interest payment       996      bonus                   616
    split                  539      OTHER                   458
    rights                 341      buyback                 347
    demerger               118      scheme of arrangement    82

## The adjustment arithmetic

    Bonus X:Y   X new shares for every Y held, so a holder of Y ends with X+Y.
                factor_price = Y / (X + Y).      Bonus 1:1 -> 0.5
    Split A->B  face value falls from Rs A to Rs B, so share count multiplies
                by A/B.  factor_price = B / A.   Rs 10 -> Rs 2 gives 0.2
    Dividend D  no share-count change. factor_price = 1.0; only the
                total-return basis is affected.

`factor_price` multiplies PRE-ex-date prices to put them on the post-ex scale.

## Traps found in the real data, each covered by a test

1. **`Bonus Ncrps 1:10` / `Bonus Preference Shares 21:1`.** A bonus issue of
   *preference* shares. It does NOT dilute the equity price, and treating it as
   an equity bonus applies a large wrong adjustment. Six of these exist in the
   sample. Excluded explicitly.
2. **`Fv Splt Frm Rs 10 To Rs 5`.** A genuine typo in NSE's own data (9
   occurrences) -- "Splt"/"Frm" rather than "Split"/"From". It is a real split
   and must still parse.
3. **`Dividend Rs. - 2.80/- Per Share`.** The period after "Rs" followed by a
   separator crashed a previous build's `float()` when a lax regex captured a
   lone ".".
4. **`Distribution - Rs 5 Per Unit`.** REIT/InvIT distributions, not equity
   dividends. Different instrument, classified separately.

## What is deliberately NOT parsed

Rights issues, buybacks, demergers, schemes of arrangement and amalgamations are
recorded with `parse_status="unparsed"` and an `action_type`, never silently
dropped. Their ratios are not recoverable from the subject text -- a demerger
subject states no share-exchange ratio at all -- so pretending otherwise would
be worse than admitting the gap. They are a known, counted limitation:
~890 records, of which rights issues (341) are the most consequential because
they DO dilute price.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from stocksense.data.store import Store

_CA_URL = "https://www.nseindia.com/api/corporates-corporateActions"
_WARMUP_URL = "https://www.nseindia.com"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

POLITE_DELAY_S = 0.35


# ------------------------------------------------------------------- grammars
# Anchored on the ratio itself rather than on any one phrasing, because NSE has
# used at least eight phrasings for the identical event across 16 years.

# "Bonus 3:1", "Bonus 1 : 1", "Bonus 1: 2", and the abbreviation "Bon 2:1".
# \bBon\b does NOT match inside "Bonus" (there is no word boundary after "Bon"),
# so the optional (?:us)? is what covers both spellings.
_BONUS_RE = re.compile(r"\bBon(?:us)?\b[^0-9:]{0,40}?(\d+)\s*:\s*(\d+)", re.IGNORECASE)

# The word alone, with no ratio. Used only to recognise non-equity bonuses whose
# ratio does not parse, so they are classified rather than left in the tail.
_BONUS_WORD_RE = re.compile(r"\bBon(?:us)?\b", re.IGNORECASE)

# Bonuses of NON-EQUITY instruments do not dilute the equity price. Checked
# BEFORE the bonus rule, or "Scheme Of Arrangement - Bonus Ncrps 4:1" reads as a
# 4:1 equity bonus and mis-adjusts by 80%. DVR (differential voting rights)
# shares are a separate security too: "Bonus 1 Dvr : 10 Eq Share".
_PREF_BONUS_RE = re.compile(
    r"\b(?:n?crps|ncds?|preference\s+shares?|debentures?|dvr)\b", re.IGNORECASE
)

# Every observed face-value-change phrasing, in one expression:
#   "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share"
#   "Face Value Split From Rs.10/- To Rs.5/-"
#   "Fv Split Rs.10 To Rs.5"
#   "Fv Splt Frm Rs 10 To Rs 5"          <- NSE's own typo, 9 real occurrences
#   "Bon 1:1/Fv Spl Rs.5tors.2"          <- abbreviated AND run together
#   "Sub-Division From Rs 10/- Per Share To Rs 5/- Per Share"   <- no "Split"
#   "Consolidation Of Equity Shares From Re 1 Per Share To Rs 10 Per Share"
#
# The real discriminator is the "Rs A ... To Rs B" structure, which does not
# otherwise occur; the leading word only disambiguates. "Spl" is allowed ONLY
# after "Fv"/"Face Value", because on its own it means "Special" in dividend
# lines like "Div-Int Re 2.5+Spl Re 2.5".
#
# CONSOLIDATION / CAPITAL REDUCTION is a REVERSE split: face value rises, share
# count falls, price rises. The same after/before formula handles it --
# Re 1 -> Rs 10 gives factor 10.0, so pre-event prices multiply by 10. Correct
# by construction, and pinned by a test.
_SPLIT_RE = re.compile(
    r"(?:Spli?t|Splt|Sub[\s-]?Division|Consolidation|Capital\s+Reduction"
    r"|(?:Fv|Face\s+Valu[eu]s?)\s*[-\s]*Spl)"
    r".*?"
    r"(?:Rs|Re)\.?\s*(\d+(?:\.\d+)?)\s*/?-?\s*(?:Per\s*(?:Share)?\s*)?"
    r"To\s*(?:Rs|Re)\.?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE | re.DOTALL,
)

# "Dividend - Rs 2.50 Per Share", "Div-Rs.2/-", "Dividend Rs. - 2.80/- Per Share",
# and qualifier forms: "Div-Int Re 2.5+Spl Re 2.5", "Div-Fin Rs.5+Spl Rs.10".
# The amount group REQUIRES a leading digit: a lax [\d.]+ matched a bare "."
# when the separator sat between "Rs." and the number, and crashed float().
_DIVIDEND_RE = re.compile(
    r"\b(?:Dividend|Div)\b[\s\-]*"
    r"(?:(?:Int|Fin|Spl|Interim|Final|Special|Annual)\b[\s\-.]*)*"
    r"(?:Rs|Re)?\.?[\s\-]*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Instruments that are not equity, or events whose ratio is not in the text.
_UNPARSEABLE = {
    "rights": re.compile(r"\brights?\b", re.IGNORECASE),
    "buyback": re.compile(r"\bbuy[\s-]?back\b", re.IGNORECASE),
    "demerger": re.compile(r"\bde[\s-]?merger\b", re.IGNORECASE),
    "scheme_of_arrangement": re.compile(r"\bscheme\s+of\s+arrangement\b", re.IGNORECASE),
    "amalgamation": re.compile(r"\bamalgamation\b", re.IGNORECASE),
    "distribution": re.compile(r"\bdistribution\b", re.IGNORECASE),  # REIT/InvIT
}

# Corporate noise with no price effect whatsoever. Classified, not "unparsed":
# an unparsed count that is 90% AGMs tells you nothing about real coverage.
_NOISE_RE = re.compile(
    r"\b(?:annual\s+general\s+meeting|extra\s*[\s-]?ordinary\s+general\s+meeting|agm|egm|"
    r"board\s+meeting|e-?voting|book\s+closure|interest\s+payment|redemption|"
    r"name\s+change|open\s+offer|postal\s+ballot|election\s+of\s+directors)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedAction:
    action_type: str
    factor_price: float = 1.0
    ratio_num: float | None = None
    ratio_den: float | None = None
    dividend_amount: float | None = None
    face_before: float | None = None
    face_after: float | None = None
    parse_status: str = "ok"


def parse_action(subject: str) -> ParsedAction:
    """Classify one NSE subject line and compute its price-adjustment factor.

    Compound subjects multiply: "Bonus 1:1 / Face Value Split From Rs 10 To Rs 2"
    is genuinely both events on one ex-date, so the factors compose to
    0.5 * 0.2 = 0.1.
    """
    text = (subject or "").strip()
    if not text:
        return ParsedAction("unknown", parse_status="unparsed")

    factor = 1.0
    kinds: list[str] = []
    num = den = face_b = face_a = None

    # Non-equity bonus FIRST -- it must not fall through to the bonus rule.
    # Deliberately keyed off the WORD, not the ratio regex: "Bonus 1 Dvr : 10 Eq
    # Share" puts a qualifier between the number and the colon so the ratio does
    # not match, but it is still unmistakably a DVR bonus and must be recognised
    # as one rather than silently landing in the unparsed tail.
    bonus_match = _BONUS_RE.search(text)
    pref_bonus = bool(_PREF_BONUS_RE.search(text)) and bool(_BONUS_WORD_RE.search(text))

    m = bonus_match
    if m and not pref_bonus:
        x, y = float(m.group(1)), float(m.group(2))
        if x > 0 and y > 0:
            factor *= y / (x + y)
            num, den = x, y
            kinds.append("bonus")

    m = _SPLIT_RE.search(text)
    if m:
        before, after = float(m.group(1)), float(m.group(2))
        if before > 0 and after > 0:
            factor *= after / before
            face_b, face_a = before, after
            kinds.append("split")

    if kinds:
        return ParsedAction(
            action_type="+".join(kinds),
            factor_price=factor,
            ratio_num=num,
            ratio_den=den,
            face_before=face_b,
            face_after=face_a,
        )

    if pref_bonus:
        # Real event, correctly recognised as having NO equity price effect.
        return ParsedAction("preference_bonus", factor_price=1.0)

    for name, rx in _UNPARSEABLE.items():
        if rx.search(text):
            return ParsedAction(name, factor_price=1.0, parse_status="unparsed")

    m = _DIVIDEND_RE.search(text)
    if m:
        return ParsedAction("dividend", factor_price=1.0, dividend_amount=float(m.group(1)))

    if _NOISE_RE.search(text):
        return ParsedAction("noise", factor_price=1.0)

    return ParsedAction("other", factor_price=1.0, parse_status="unparsed")


# -------------------------------------------------------------------- fetching
def _session() -> requests.Session:
    s = requests.Session()
    try:
        # This host is session-gated. The warm-up often returns 403 and the API
        # still works, so its status is deliberately not checked.
        s.get(_WARMUP_URL, headers=_HEADERS, timeout=20)
    except requests.RequestException:
        pass
    return s


def fetch_window(session: requests.Session, start: date, end: date) -> list[dict]:
    """One date window of corporate actions. NSE caps the range, so callers
    should page in quarters or half-years."""
    resp = session.get(
        _CA_URL,
        params={
            "index": "equities",
            "from_date": start.strftime("%d-%m-%Y"),
            "to_date": end.strftime("%d-%m-%Y"),
        },
        headers=_HEADERS,
        timeout=45,
    )
    time.sleep(POLITE_DELAY_S)
    resp.raise_for_status()
    payload = resp.json()
    return payload if isinstance(payload, list) else []


def parse_records(records: list[dict]) -> pd.DataFrame:
    """Turn raw API records into the `corporate_actions` schema."""
    rows = []
    for rec in records:
        symbol = (rec.get("symbol") or "").strip()
        subject = (rec.get("subject") or "").strip()
        ex_raw = (rec.get("exDate") or "").strip()
        if not symbol or not ex_raw:
            continue
        try:
            ex_date = datetime.strptime(ex_raw, "%d-%b-%Y").date()
        except ValueError:
            continue

        p = parse_action(subject)
        rows.append(
            {
                "symbol": symbol,
                "ex_date": ex_date,
                "action_type": p.action_type,
                "ratio_num": p.ratio_num,
                "ratio_den": p.ratio_den,
                "factor_price": p.factor_price,
                "dividend_amount": p.dividend_amount,
                "face_before": p.face_before,
                "face_after": p.face_after,
                "subject_raw": subject,
                "parse_status": p.parse_status,
            }
        )
    return pd.DataFrame(rows, columns=Store.CA_COLS)


def _windows(start: date, end: date, months: int = 6):
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=months * 31), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def backfill(
    store: Store,
    cache_root: Path,
    start: date,
    end: date,
    resume: bool = True,
) -> dict[str, int]:
    """Ingest corporate actions across a date range, window by window.

    Resumable in the same shape as the bhavcopy backfill: each window is
    fetched, written and recorded individually, and a failure is recorded rather
    than raised so one bad window cannot abort the run.
    """
    done = store.completed_units("nse_corp_actions") if resume else set()
    stats = {"windows_ok": 0, "windows_failed": 0, "actions": 0, "unparsed": 0, "skipped": 0}
    session = _session()
    cache_root.mkdir(parents=True, exist_ok=True)

    for w_start, w_end in _windows(start, end):
        unit = f"{w_start.isoformat()}..{w_end.isoformat()}"
        if unit in done:
            stats["skipped"] += 1
            continue

        started = datetime.now()
        run = {
            "run_id": uuid.uuid4().hex[:16],
            "source": "nse_corp_actions",
            "unit": unit,
            "started_at": started,
            "attempt": 1,
        }
        try:
            df = parse_records(fetch_window(session, w_start, w_end))
            if df.empty:
                run |= {"status": "empty", "rows_written": 0, "error": None}
            else:
                store.write_corporate_actions(df)
                stats["actions"] += len(df)
                stats["unparsed"] += int((df.parse_status == "unparsed").sum())
                stats["windows_ok"] += 1
                run |= {"status": "ok", "rows_written": len(df), "error": None}
        except Exception as exc:
            stats["windows_failed"] += 1
            run |= {"status": "failed", "rows_written": 0, "error": f"{type(exc).__name__}: {exc}"}

        run["finished_at"] = datetime.now()
        store.record_ingest_run(run)
        store.publish()

    return stats
