"""
Quantify the survivorship bias named in research/phase0_verdict.md item 1:
the Phase 0 universe (data/universe.py) is 98 symbols that are liquid and
listed *today*. This pulls real NSE historical bhavcopy (full-market daily
records, confirmed reachable directly from archives.nseindia.com) at
sparse historical checkpoints, extracts the top-N most-traded symbols at
each checkpoint by turnover, and reports how many of those historically
significant names are absent from the Phase 0 universe — i.e., names that
were prominent once and are not liquid/listed/existing today, which the
current universe silently excludes.

This does not build a full point-in-time universe (that requires the
complete daily archive, ~6,500 files over 26 years — a much larger
ingestion this script deliberately does not attempt). It answers a
narrower, immediately actionable question: how big is the gap, roughly?
"""

from __future__ import annotations

import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from stocksense.data.universe import PHASE0_UNIVERSE

CHECKPOINTS = [
    ("02JAN2001",), ("02JAN2004",), ("03JAN2007",), ("04JAN2010",),
    ("02JAN2013",), ("02JAN2016",), ("02JAN2019",), ("02JAN2022",), ("02JAN2024",),
]

BASE_URLS = [
    "https://archives.nseindia.com/content/historical/EQUITIES/{year}/{mon}/cm{ddmonyyyy}bhav.csv.zip",
    "https://nsearchives.nseindia.com/content/historical/EQUITIES/{year}/{mon}/cm{ddmonyyyy}bhav.csv.zip",
]

MONTH_MAP = {
    "JAN": "JAN", "FEB": "FEB", "MAR": "MAR", "APR": "APR", "MAY": "MAY", "JUN": "JUN",
    "JUL": "JUL", "AUG": "AUG", "SEP": "SEP", "OCT": "OCT", "NOV": "NOV", "DEC": "DEC",
}


def fetch_bhavcopy(ddmonyyyy: str) -> pd.DataFrame | None:
    year = ddmonyyyy[-4:]
    mon = ddmonyyyy[2:5]
    for base in BASE_URLS:
        url = base.format(year=year, mon=mon, ddmonyyyy=ddmonyyyy)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = r.read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                name = zf.namelist()[0]
                with zf.open(name) as f:
                    df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    print(f"  FAILED {ddmonyyyy}: {last_err}")
    return None


def main() -> None:
    phase0_set = set(PHASE0_UNIVERSE)
    all_top_names: dict[str, set[str]] = {}

    for (date_str,) in CHECKPOINTS:
        df = fetch_bhavcopy(date_str)
        if df is None:
            continue
        # Standard NSE bhavcopy columns: SYMBOL, SERIES, ..., TOTTRDQTY / TTL_TRD_QNTY, TOTTRDVAL
        symbol_col = "SYMBOL" if "SYMBOL" in df.columns else df.columns[0]
        series_col = "SERIES" if "SERIES" in df.columns else None
        val_col = next((c for c in df.columns if "TRDVAL" in c.upper() or "TTL_TRD_VAL" in c.upper()), None)

        if series_col:
            df = df[df[series_col].astype(str).str.strip() == "EQ"]

        if val_col:
            top = df.sort_values(val_col, ascending=False).head(150)[symbol_col].astype(str).str.strip()
        else:
            top = df[symbol_col].astype(str).str.strip().head(150)

        top_set = set(top)
        all_top_names[date_str] = top_set
        missing = top_set - phase0_set
        print(f"{date_str}: {len(df)} EQ rows, top150 -> {len(missing)} not in Phase0 universe")
        time.sleep(1.0)

    union_historical = set().union(*all_top_names.values()) if all_top_names else set()
    missing_overall = union_historical - phase0_set

    print(f"\n=== SUMMARY ===")
    print(f"checkpoints fetched: {len(all_top_names)} / {len(CHECKPOINTS)}")
    print(f"union of historically-top150 symbols across checkpoints: {len(union_historical)}")
    print(f"of those, NOT in Phase0's 98-symbol universe: {len(missing_overall)}")
    print(f"\nsample of missing (historically significant, excluded today):")
    print(sorted(missing_overall)[:40])

    out = Path(__file__).parent / "survivorship_gap.txt"
    out.write_text(
        f"checkpoints={len(all_top_names)}/{len(CHECKPOINTS)}\n"
        f"union_top150_across_checkpoints={len(union_historical)}\n"
        f"missing_from_phase0_universe={len(missing_overall)}\n"
        f"missing_symbols={sorted(missing_overall)}\n"
    )


if __name__ == "__main__":
    main()
