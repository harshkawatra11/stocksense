"""One-off live probe of the dashboard sources (PLAN-M card M2.1). Prints shapes only, saves trimmed fixtures.

Run: .venv\\Scripts\\python.exe scripts\\m_probe_sources.py [--out tests/fixtures/marketlab]
No credentials are used or printed; every call is a public read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stocksense.core.clock import SystemClock
from stocksense.marketlab.sources import Sources, dumps_probe
from stocksense.marketlab.store import LabStore


def shape(x: Any) -> str:
    """Short description of a payload: type, top-level keys or row count and the first row's keys."""
    if isinstance(x, dict):
        return f"dict keys={list(x)[:12]}"
    if isinstance(x, list):
        first = x[0] if x else None
        keys = list(first)[:12] if isinstance(first, dict) else type(first).__name__
        return f"list n={len(x)} first_keys={keys}"
    return type(x).__name__


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/fixtures/marketlab")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    clock = SystemClock()
    src = Sources(None, clock, LabStore())  # type: ignore[arg-type]
    probes = {
        "all_indices": lambda: src.nse_all_indices(),
        "watch_nifty500": lambda: src.nse_watch("NIFTY 500"),
        "watch_fo": lambda: src.nse_watch("SECURITIES IN F&O"),
        "symbol_data": lambda: src.nse_symbol_data("RELIANCE"),
        "filing": lambda: src.nse_integrated_filing("RELIANCE"),
        "results_cmp": lambda: src.nse_results_comparison("RELIANCE"),
        "option_expiries": lambda: src.nse_option_expiries("NIFTY"),
        "option_chain_index": lambda: src.nse_option_chain("NIFTY", True),
        "option_chain_equity": lambda: src.nse_option_chain("RELIANCE", False),
        "oi_spurts": lambda: src.nse_oi_spurts(),
        "preopen_nifty": lambda: src.nse_preopen("NIFTY"),
        "nifty500_list": lambda: src.nifty500_list(),
        "yahoo_spark": lambda: src.yahoo_spark(["RELIANCE.NS", "^NSEI"]),
        "yahoo_chart": lambda: src.yahoo_chart("RELIANCE.NS", "1d", "1mo"),
        "google_news": lambda: src.google_news("nifty", 7, 5),
        "wiki": lambda: src.wiki_summary("Reliance Industries Limited"),
    }
    for name, fn in probes.items():
        try:
            res = fn()
        except Exception as e:  # probe script: report every failure and go on
            print(f"{name:20s} FAILED {type(e).__name__}: {str(e)[:100]}")
            continue
        body = res
        if isinstance(res, dict) and len(res) > 5 and all(isinstance(v, dict) for v in res.values()):
            body = dict(list(res.items())[:5])  # keyed table: keep five entries
        (out / f"probe_{name}.json").write_text(dumps_probe(body), encoding="utf-8")
        print(f"{name:20s} ok {shape(res)}")
    print("calls:", json.dumps(src.calls))


if __name__ == "__main__":
    main()
