"""Q0.2 / Q0.3 -- can we actually read the market and the account from here?

Read-only by construction. Nothing in this module places, modifies or cancels an
order; tests/unit/test_readonly.py asserts that by AST-scanning the source. The
static-IP question that gates live EXECUTION is answered by q0_network, not here.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import requests

from stocksense.core.config import get_settings
from stocksense.probes.base import ProbeResult

_UPSTOX_BASE = "https://api.upstox.com/v2"
_NIFTY = "NSE_INDEX|Nifty 50"
_RELIANCE = "NSE_EQ|INE002A01018"


def probe_upstox(result: ProbeResult) -> None:
    """Does the stored token still authenticate, and how fresh is the data?"""
    s = get_settings()
    result.findings["token_present"] = bool(s.upstox_access_token)
    if not s.upstox_access_token:
        result.verdict = "BLOCKED"
        result.note("STOCKSENSE_UPSTOX_ACCESS_TOKEN is unset in .env")
        return

    headers = {"Authorization": f"Bearer {s.upstox_access_token}", "Accept": "application/json"}

    # 1. Authenticated quote. Upstox tokens expire daily (~03:30 IST), so a 401
    #    here means "re-auth needed today", not "the integration is broken".
    t0 = time.perf_counter()
    r = requests.get(
        f"{_UPSTOX_BASE}/market-quote/ltp", params={"instrument_key": _NIFTY}, headers=headers, timeout=15
    )
    result.findings["ltp_status"] = r.status_code
    result.findings["ltp_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    result.note(f"LTP  HTTP {r.status_code} in {result.findings['ltp_latency_ms']}ms")

    if r.status_code == 401:
        result.verdict = "BLOCKED"
        result.note("token expired -- Upstox tokens die daily around 03:30 IST; re-run the OAuth flow")
        result.findings["body"] = r.text[:300]
        return
    if r.ok:
        result.findings["ltp_body"] = r.json().get("data", {})

    # 2. Historical 1-minute depth. The whole intraday spine depends on how far
    #    back this actually goes -- a previous build measured 2022 as the floor.
    end = date.today()
    start = end - timedelta(days=5)
    t0 = time.perf_counter()
    h = requests.get(
        f"{_UPSTOX_BASE.replace('/v2', '/v3')}/historical-candle/{_RELIANCE}/minutes/1/"
        f"{end.isoformat()}/{start.isoformat()}",
        headers=headers,
        timeout=30,
    )
    result.findings["hist_status"] = h.status_code
    result.findings["hist_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if h.ok:
        candles = h.json().get("data", {}).get("candles", [])
        result.findings["hist_bars"] = len(candles)
        result.findings["hist_first"] = candles[-1][:2] if candles else None
        result.findings["hist_last"] = candles[0][:2] if candles else None
        result.note(f"1-min history: {len(candles)} bars over 5 days")
    else:
        result.note(f"history HTTP {h.status_code}: {h.text[:200]}")

    result.verdict = "PASS" if r.ok else "FAIL"


def probe_angel_readonly(result: ProbeResult) -> None:
    """Fresh TOTP login + read holdings/positions/tradebook.

    Always a fresh login: a previous build found that reusing a SmartAPI session
    across processes silently fails on this SDK rather than erroring.
    """
    s = get_settings()
    have = all([s.angel_api_key, s.angel_client_code, s.angel_password, s.angel_totp_secret])
    result.findings["credentials_present"] = have
    if not have:
        result.verdict = "BLOCKED"
        result.note("one or more STOCKSENSE_ANGEL_* values missing from .env")
        return

    try:
        import pyotp  # noqa: PLC0415
        from SmartApi import SmartConnect  # noqa: PLC0415
    except ImportError as exc:
        result.verdict = "BLOCKED"
        result.note(f"SDK missing: {exc}")
        return

    totp = pyotp.TOTP(s.angel_totp_secret).now()
    result.note("generated TOTP, logging in fresh")

    api = SmartConnect(api_key=s.angel_api_key)
    t0 = time.perf_counter()
    session = api.generateSession(s.angel_client_code, s.angel_password, totp)
    result.findings["login_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    ok = bool(session and session.get("status"))
    result.findings["login_ok"] = ok
    if not ok:
        result.verdict = "FAIL"
        result.findings["login_message"] = (session or {}).get("message")
        result.note(f"login failed: {(session or {}).get('message')}")
        return
    result.note(f"login OK in {result.findings['login_latency_ms']}ms")

    # Read-only calls. Counts only -- never dump holdings into a committed file.
    for label, fn in (
        ("holdings", api.holding),
        ("positions", api.position),
        ("tradebook", api.tradeBook),
        ("rms", api.rmsLimit),
    ):
        try:
            resp = fn()
            data = (resp or {}).get("data")
            n = len(data) if isinstance(data, list) else (1 if data else 0)
            result.findings[f"{label}_count"] = n
            result.findings[f"{label}_status"] = (resp or {}).get("status")
            result.note(f"{label:10s} status={(resp or {}).get('status')} rows={n}")
        except Exception as exc:
            result.findings[f"{label}_error"] = f"{type(exc).__name__}: {exc}"
            result.note(f"{label:10s} ERROR {type(exc).__name__}")

    try:
        api.terminateSession(s.angel_client_code)
    except Exception:
        pass

    result.verdict = "PASS"
