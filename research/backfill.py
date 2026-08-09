"""
Phase 0 backfill: pull daily OHLCV for the Phase 0 universe via yfinance
and load into the DuckDB store. Resumable — re-running skips symbols
already present with a full date range, per the idempotency requirement
in docs/05-nightly-pipeline.md.

Usage:  python research/backfill.py [--start 2010-01-01] [--end today]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date

import structlog

sys.path.insert(0, str(__file__.rsplit("research", 1)[0] + "src"))

from stocksense.core.config import get_settings
from stocksense.data.store import Store
from stocksense.data.universe import PHASE0_UNIVERSE
from stocksense.data.yfinance_source import fetch_history

log = structlog.get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=str(date.today()))
    args = parser.parse_args()

    settings = get_settings()
    store = Store(settings.duckdb_path)

    existing = store.con.execute(
        "SELECT symbol, COUNT(*) AS n, MIN(date) AS min_d, MAX(date) AS max_d FROM candles GROUP BY symbol"
    ).fetchdf()
    existing_map = {row["symbol"]: row for _, row in existing.iterrows()}

    total_rows = 0
    ok, skipped, failed = 0, 0, []

    for i, symbol in enumerate(PHASE0_UNIVERSE, 1):
        prior = existing_map.get(symbol)
        if prior is not None and prior["n"] > 200 and str(prior["min_d"]) <= args.start:
            log.info("skip_existing", symbol=symbol, rows=int(prior["n"]))
            skipped += 1
            continue

        df = fetch_history(symbol, args.start, args.end)
        if df.empty:
            log.warning("no_data", symbol=symbol)
            failed.append(symbol)
            continue

        n = store.upsert_candles(df)
        total_rows += n
        ok += 1
        log.info("ingested", symbol=symbol, rows=n, progress=f"{i}/{len(PHASE0_UNIVERSE)}")
        time.sleep(0.3)  # be polite to the endpoint

    store.close()

    print(f"\n=== BACKFILL COMPLETE ===")
    print(f"ok={ok} skipped={skipped} failed={len(failed)} total_rows_upserted={total_rows}")
    if failed:
        print(f"failed symbols: {failed}")


if __name__ == "__main__":
    main()
