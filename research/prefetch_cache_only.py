"""One-off prefetch: warms the on-disk NSE archive cache for a date range
WITHOUT touching the database. Safe to run alongside a live
backfill-nse-archive process, since DuckDB allows only one writer and
this script never opens the DB at all -- it just calls the same
content-hash-cached fetchers so a later `backfill-nse-archive` for this
range replays from disk instead of hitting the network.

Usage: python research/prefetch_cache_only.py 2010-01-01 2014-12-31
"""

from __future__ import annotations

import sys
from datetime import date, datetime

from stocksense.data.nse_archive import fetch_range


def main() -> None:
    start = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
    end = datetime.strptime(sys.argv[2], "%Y-%m-%d").date()
    n_ok = n_holiday = n_err = 0
    for d, df in fetch_range(start, end, kind="cm"):
        if df is None:
            n_holiday += 1
        else:
            n_ok += 1
        if (n_ok + n_holiday) % 50 == 0:
            print(f"...at {d}: {n_ok} fetched, {n_holiday} holidays/weekends", flush=True)
    print(f"done: {n_ok} fetched, {n_holiday} holidays/weekends skipped, range {start}..{end}")


if __name__ == "__main__":
    main()
