"""
Corporate actions (splits/bonuses/dividends) fetched from NSE's
corporateActions API and parsed into a price-return adjustment factor
per action (docs/17-data-spine.md, Phase D1).

bhavcopy_eq carries NO corporate-action adjustment -- verified directly
against confirmed splits (ECLERX 1:2, PASHUPATI 1:10): `prev_close`
equals the prior RAW close in every case, ratio 1.0. Feeding raw
bhavcopy closes into features without this correction would inject
fake -50% to -90% one-day "returns" on every split/bonus date and teach
a model that a stock split is a crash.

This endpoint (nseindia.com, distinct from the bhavcopy archive host
nsearchives.nseindia.com) is session-gated: it 401s/403s without
cookies obtained by first hitting the plain nseindia.com pages. That
warm-up is done once per `fetch_ca_range` call, since cookies are
reusable across the whole date range -- not once per request.

Dividends are recorded (`dividend_amount`) but NOT converted into a
factor here, because the total-return adjustment factor for a dividend
needs the ex-date price, which this module doesn't have -- that
computation lives in data/adjust.py, applied against the adjusted price
series being built.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
import structlog

from stocksense.core.config import DATA_STORE

log = structlog.get_logger(__name__)

CACHE_DIR = DATA_STORE / "nse_archive" / "corpact"
_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-actions",
    "Accept": "*/*",
}
_POLITE_DELAY_S = 1.0
_WINDOW_DAYS = 89  # ~quarterly; keeps each response small and cache granularity fine


class FetchError(Exception):
    """A genuine fetch failure -- not raised for an empty-but-200 response."""


@dataclass
class ParsedAction:
    action_type: str          # split | bonus | dividend | ignore | unparsed
    ratio_num: float | None
    ratio_den: float | None
    factor_price: float       # 1.0 for anything that doesn't change share count
    dividend_amount: float | None
    face_before: float | None
    face_after: float | None
    parse_status: str         # ok | unparsed


def _cache_path(start: date, end: date) -> Path:
    return CACHE_DIR / f"{start.isoformat()}_{end.isoformat()}.csv"


def _warm_session() -> requests.Session:
    s = requests.Session()
    s.get("https://www.nseindia.com/", headers=_HEADERS, timeout=15)
    s.get("https://www.nseindia.com/companies-listing/corporate-filings-actions", headers=_HEADERS, timeout=15)
    return s


def fetch_ca_window(start: date, end: date, session: requests.Session | None = None) -> pd.DataFrame:
    """Raw corporate-actions rows for [start, end]. Content-hash cached
    to disk (by date-range filename) so a re-run never re-hits the
    session-gated endpoint for a window already fetched."""
    path = _cache_path(start, end)
    if path.exists():
        return pd.read_csv(path)

    sess = session or _warm_session()
    url = (
        "https://www.nseindia.com/api/corporates-corporateActions"
        f"?index=equities&from_date={start.strftime('%d-%m-%Y')}&to_date={end.strftime('%d-%m-%Y')}"
    )
    try:
        resp = sess.get(url, headers=_HEADERS, timeout=25)
    except requests.exceptions.RequestException as e:
        raise FetchError(f"network error fetching CA window {start}..{end}: {e}") from e
    if resp.status_code != 200:
        raise FetchError(f"unexpected status {resp.status_code} fetching CA window {start}..{end}")

    records = resp.json()
    df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    time.sleep(_POLITE_DELAY_S)
    return df


def fetch_ca_range(start: date, end: date) -> Iterator[pd.DataFrame]:
    """Quarterly windows across [start, end], yielded incrementally --
    same resumability shape as nse_archive.fetch_range: a caller that
    writes each window's parsed rows to the database as they arrive
    loses nothing to an interruption partway through the range."""
    session = _warm_session()
    window_start = start
    while window_start <= end:
        window_end = min(window_start + timedelta(days=_WINDOW_DAYS), end)
        try:
            df = fetch_ca_window(window_start, window_end, session=session)
        except FetchError as e:
            log.warning("ca_fetch_failed", start=str(window_start), end=str(window_end), error=str(e))
            df = pd.DataFrame()
        yield df
        window_start = window_end + timedelta(days=1)


# ---- parsing ----

_BONUS_RE = re.compile(r"Bonus\s+(\d+)\s*:\s*(\d+)", re.IGNORECASE)
_FACE_SPLIT_RE = re.compile(
    # NSE has used at least three grammars across 2010-2026 for the same
    # event: verbose ("From Rs 10/- Per Share To Rs 2/- Per Share"),
    # short with dots ("Fv Split Rs.10 To Rs.5"), and short with spaces
    # ("Face Value Split Rs 10 To Re 1"). Anchored on "Split" + the two
    # Rs/Re amounts either side of "To", tolerant of "/-", ".", and an
    # optional "Per Share" -- not on the full verbose phrase.
    r"Spli?t.*?Rs\.?\s*(\d+(?:\.\d+)?)\s*/?-?\s*(?:Per\s*Share\s*)?To\s*(?:Rs|Re)\.?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_DIVIDEND_RE = re.compile(r"Dividend[\s\-]*(?:Rs|Re)\.?[\s\-]*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_action(subject: str) -> ParsedAction:
    """Classifies one CA `subject` string into a structured action.
    Compound subjects (bonus AND a face-value split in the same
    announcement) multiply their price factors. Anything recognized as
    corporate noise with no share-count or cash effect (AGM notices,
    e-voting, board meetings) is `ignore` with factor 1.0. Anything that
    looks financially relevant but doesn't match a known grammar is
    `unparsed` -- reported, never silently dropped."""
    if not isinstance(subject, str) or not subject.strip():
        return ParsedAction("unparsed", None, None, 1.0, None, None, None, "unparsed")

    factor = 1.0
    action_type = None
    ratio_num = ratio_den = None
    face_before = face_after = None
    dividend_amount = None

    bonus_m = _BONUS_RE.search(subject)
    if bonus_m:
        x, y = float(bonus_m.group(1)), float(bonus_m.group(2))
        # Bonus X:Y -- X new shares issued per Y held.
        factor *= y / (x + y)
        ratio_num, ratio_den = x, y
        action_type = "bonus"

    split_m = _FACE_SPLIT_RE.search(subject)
    if split_m:
        before, after = float(split_m.group(1)), float(split_m.group(2))
        if before > 0:
            factor *= after / before
            face_before, face_after = before, after
            action_type = "split" if action_type is None else action_type

    div_m = _DIVIDEND_RE.search(subject)
    is_dividend_only = div_m and action_type is None and "face value" not in subject.lower()
    if is_dividend_only:
        dividend_amount = float(div_m.group(1))
        action_type = "dividend"

    if action_type is not None:
        return ParsedAction(action_type, ratio_num, ratio_den, factor, dividend_amount, face_before, face_after, "ok")

    noise_markers = (
        "agm", "annual general meeting", "board meeting", "e-voting", "evoting",
        "postal ballot", "egm", "extraordinary general meeting", "record date",
        "interest payment", "redemption",
    )
    if any(m in subject.lower() for m in noise_markers):
        return ParsedAction("ignore", None, None, 1.0, None, None, None, "ok")

    return ParsedAction("unparsed", None, None, 1.0, None, None, None, "unparsed")


def parse_ca_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Parses a raw fetch_ca_window frame into the corporate_actions
    write schema. `exDate` arrives as 'DD-Mon-YYYY' (e.g. '05-Jan-2015')."""
    if raw.empty:
        return pd.DataFrame(columns=[
            "symbol", "ex_date", "action_type", "ratio_num", "ratio_den", "factor_price",
            "dividend_amount", "face_before", "face_after", "subject_raw", "parse_status",
        ])

    rows = []
    for _, r in raw.iterrows():
        parsed = parse_action(str(r.get("subject", "")))
        rows.append({
            "symbol": str(r["symbol"]).strip(),
            "ex_date": pd.to_datetime(r["exDate"], format="%d-%b-%Y", errors="coerce").date(),
            "action_type": parsed.action_type,
            "ratio_num": parsed.ratio_num,
            "ratio_den": parsed.ratio_den,
            "factor_price": parsed.factor_price,
            "dividend_amount": parsed.dividend_amount,
            "face_before": parsed.face_before,
            "face_after": parsed.face_after,
            "subject_raw": str(r.get("subject", "")),
            "parse_status": parsed.parse_status,
        })
    df = pd.DataFrame(rows)
    return df[df["ex_date"].notna()]
