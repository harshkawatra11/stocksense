"""
Market-reality layer — Upstox live data as a sanity-check on signals, NOT a
forecaster.

Upstox live data isn't used as a forecasting input — it's a "is this signal
still true right now" check. A signal computed from EOD/intraday data can go stale by the time
it's actually reviewed/approved (dashboard latency, human approval delay,
etc). This module compares a signal's price-at-signal-time against the
CURRENT live LTP (backend/services/quote_cache.py, populated by
data/pipeline/upstox_feed.py's WS feed) and flags meaningful drift.

Everything here follows the Stage 0 "no silent fallbacks" rule: if live data
isn't available (ticker not subscribed, feed not running, tick too stale),
that is reported honestly as "unavailable"/"stale" — never silently treated
as "no drift" or "fine".
"""
from __future__ import annotations

import logging

from backend.services import quote_cache

log = logging.getLogger(__name__)

# Ticks older than this are considered too stale to trust for a live sanity check.
STALE_TICK_SECONDS = 60.0


def check_price_drift(ticker: str, signal_price: float, max_drift_pct: float = 1.0) -> dict:
    """
    Compare a signal's price_at_signal against the current live LTP.

    Returns:
        {
          "drifted": bool | None,        # None when we genuinely can't tell
          "current_ltp": float | None,
          "drift_pct": float | None,
          "stale_price_data": bool,
          "detail": str,
        }

    "drifted" is only ever True/False when we have a fresh (<= STALE_TICK_SECONDS
    old) live tick for the ticker. Missing/stale data is reported explicitly
    rather than defaulting to "no drift" — a missing feed must never look like
    a clean signal.
    """
    if signal_price is None:
        return {
            "drifted": None,
            "current_ltp": None,
            "drift_pct": None,
            "stale_price_data": False,
            "detail": "no signal_price provided — cannot check drift",
        }

    quote = quote_cache.get(ticker)
    if quote is None:
        return {
            "drifted": None,
            "current_ltp": None,
            "drift_pct": None,
            "stale_price_data": False,
            "detail": f"no live quote for {ticker} — not subscribed on the Upstox feed, or feed not running (unavailable, not 'no drift')",
        }

    age = quote_cache.age_seconds(ticker)
    stale = age is None or age > STALE_TICK_SECONDS
    current_ltp = quote.get("ltp")

    if stale:
        return {
            "drifted": None,
            "current_ltp": current_ltp,
            "drift_pct": None,
            "stale_price_data": True,
            "detail": f"live quote for {ticker} is {age:.0f}s old (> {STALE_TICK_SECONDS:.0f}s threshold) — too stale to trust for drift check" if age is not None else f"live quote for {ticker} has no age info — treating as stale",
        }

    if current_ltp is None:
        return {
            "drifted": None,
            "current_ltp": None,
            "drift_pct": None,
            "stale_price_data": False,
            "detail": f"live quote for {ticker} has no ltp field — unavailable",
        }

    drift_pct = ((current_ltp - signal_price) / signal_price) * 100 if signal_price else None
    drifted = abs(drift_pct) > max_drift_pct if drift_pct is not None else None

    detail = (
        f"live LTP ₹{current_ltp:.2f} vs signal price ₹{signal_price:.2f} "
        f"({drift_pct:+.2f}%) — {'exceeds' if drifted else 'within'} {max_drift_pct:.1f}% threshold"
    )
    return {
        "drifted": drifted,
        "current_ltp": current_ltp,
        "drift_pct": round(drift_pct, 4) if drift_pct is not None else None,
        "stale_price_data": False,
        "detail": detail,
    }


def check_circuit_limit(ticker: str) -> dict:
    """
    Would check whether `ticker` is at/near its exchange circuit limit right
    now (a signal to buy a stock already frozen at upper circuit is unsafe to
    act on).

    NOT IMPLEMENTED — honestly unavailable, not faked. Circuit-limit /
    upper-lower-band data is only present in Upstox's WebSocket "full" mode
    payload. Per docs/UPSTOX_API_NOTES.md §4, Stage 1's live feed
    (data/pipeline/upstox_feed.py) subscribes in "ltpc" mode only (LTP +
    close), which does NOT carry circuit-limit fields — so quote_cache has
    nothing to check here.

    TODO (future enhancement, not required for Stage 3): switch the WS
    subscription to "full" mode (or add a second "full" mode subscription
    for a watch-list subset) and thread circuit-limit fields through
    upstox_feed.py's tick shape and quote_cache's entry dict before this
    function can return a real answer.
    """
    return {
        "available": False,
        "detail": "requires full-mode WS subscription (not yet wired — Stage 1 feed subscribes in ltpc mode only, see docs/UPSTOX_API_NOTES.md §4)",
    }


def enrich_signal_with_market_reality(signal: dict) -> dict:
    """
    Compute the market-reality checks for one signal dict and return the
    {"market_reality": {...}, "component_status": {...}} pair to attach.

    Callers should do:
        mr = enrich_signal_with_market_reality(signal)
        signal["market_reality"] = mr["market_reality"]
        components["market_reality"] = mr["component_status"]

    component_status follows the exact Stage 0 {status, detail, source}
    contract used by intelligence.signal_pipeline.get_component_statuses():
      - status: "ok" | "degraded" | "unavailable"
      - source: "live_quote_cache" | "stale" | "unavailable"
    """
    ticker = signal.get("ticker")
    signal_price = signal.get("price")

    drift = check_price_drift(ticker, signal_price)
    circuit = check_circuit_limit(ticker)

    market_reality = {
        "price_drift": drift,
        "circuit_limit": circuit,
    }

    if drift["stale_price_data"]:
        component_status = {
            "status": "degraded",
            "detail": drift["detail"],
            "source": "stale",
        }
    elif drift["drifted"] is None:
        component_status = {
            "status": "unavailable",
            "detail": drift["detail"],
            "source": "unavailable",
        }
    elif drift["drifted"]:
        component_status = {
            "status": "degraded",
            "detail": drift["detail"],
            "source": "live_quote_cache",
        }
    else:
        component_status = {
            "status": "ok",
            "detail": drift["detail"],
            "source": "live_quote_cache",
        }

    return {"market_reality": market_reality, "component_status": component_status}
