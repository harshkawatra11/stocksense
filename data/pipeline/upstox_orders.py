"""
Upstox SANDBOX order placement — api-sandbox.upstox.com/v2, risk-free rehearsal.

This is the ONLY order-placement code in the app, and it only ever talks to
Upstox's sandbox environment: no real money, no real market impact, a
separate access token from both the live OAuth token and the Analytics
Token used elsewhere. See docs/UPSTOX_API_NOTES.md §2c.

Every order this module places follows the official Upstox Agent Skill's
own guardrails (docs/UPSTOX_API_NOTES.md §8), adopted here as fixed
behavior rather than optional flags:
  - LIMIT orders only, never MARKET — a bad limit price fails safely, a bad
    market order fills at whatever price is available.
  - Called ONLY from backend/routers/confirmations.py's approve() endpoint,
    i.e. only after a human has explicitly approved a specific proposed
    trade shown in the UI. Nothing in this module or its caller is reachable
    from an unattended scheduler path.

This module does NOT place live orders and never will without a deliberate,
separate, later change — see WHAT_TO_DO_NEXT.txt Section 5 and the
confirmation-gated execution design discussed with the user.
"""
from __future__ import annotations

import asyncio
import logging

import httpx

from config import settings
from data.pipeline.upstox_client import resolve_instrument_key

log = logging.getLogger(__name__)

# The correct sandbox host is api-sandbox.upstox.com, NOT sandbox.upstox.com
# (the latter does not resolve at all — confirmed via DNS lookup, "Non-existent
# domain" — an earlier version of this file had it wrong and every sandbox
# call was silently failing with a network error, not an auth error).
# place_sandbox_order now goes through the official SDK's
# Configuration(sandbox=True), which resolves to this same host internally.
# get_order_status and get_available_funds stay on direct REST calls — per
# the SDK's own Configuration.sandbox_urls allowlist, sandbox only serves
# order place/modify/cancel; order/details and funds-and-margin are real,
# confirmed gaps in Upstox's sandbox (not something the SDK can route
# around), so those two endpoints keep the original hand-rolled calls.
SANDBOX_ORDER_DETAILS_URL = "https://api-sandbox.upstox.com/v2/order/details"
SANDBOX_FUNDS_URL = "https://api-sandbox.upstox.com/v2/user/get-funds-and-margin"

# Upstox order statuses that mean "still working" — poll again later.
# Everything else (complete, rejected, cancelled) is terminal.
OPEN_ORDER_STATUSES = {"open", "open pending", "trigger pending", "after market order req received"}


def sandbox_configured() -> bool:
    return bool(settings.UPSTOX_SANDBOX_TOKEN)


async def place_sandbox_order(ticker: str, action: str, quantity: int, price: float) -> dict:
    """
    Place a LIMIT order on Upstox's sandbox environment.

    Returns {"status": "PLACED", "order_id": str} on success, or
    {"status": "FAILED", "detail": str} on any failure — never raises, so
    callers (the approve() endpoint) can always record a clear outcome
    against the confirmation row instead of a 500.
    """
    if not sandbox_configured():
        return {
            "status": "FAILED",
            "detail": "UPSTOX_SANDBOX_TOKEN not set — generate one from the "
                      "Upstox developer portal's Sandbox section and add it to .env",
        }
    if action not in ("BUY", "SELL"):
        return {"status": "FAILED", "detail": f"invalid action {action!r} — must be BUY or SELL"}
    if quantity < 1:
        return {"status": "FAILED", "detail": f"invalid quantity {quantity}"}

    # Position-sizing sanity guard — this is sandbox/fake money, but a
    # runaway loop or a bad shares_affordable calculation placing a
    # 10,000-share order is still a real bug worth catching loudly instead
    # of quietly rehearsing it. See config.MAX_SANDBOX_ORDER_VALUE.
    order_value = quantity * price
    if order_value > settings.MAX_SANDBOX_ORDER_VALUE:
        detail = (
            f"order value ₹{order_value:,.2f} (qty={quantity} x price={price}) exceeds "
            f"MAX_SANDBOX_ORDER_VALUE ₹{settings.MAX_SANDBOX_ORDER_VALUE:,.2f} — refusing "
            f"to place, this smells like a sizing bug even in sandbox"
        )
        log.warning("Sandbox order blocked for %s: %s", ticker, detail)
        return {"status": "FAILED", "detail": detail}

    instrument_key = await resolve_instrument_key(ticker)
    if instrument_key is None:
        return {"status": "FAILED", "detail": f"no Upstox instrument_key for {ticker} (missing ISIN)"}

    import upstox_client

    def _place():
        configuration = upstox_client.Configuration(sandbox=True)
        configuration.access_token = settings.UPSTOX_SANDBOX_TOKEN
        api = upstox_client.OrderApi(upstox_client.ApiClient(configuration))
        body = upstox_client.PlaceOrderRequest(
            quantity=quantity,
            product="D",           # delivery — matches StockSense's swing-trade holding style
            validity="DAY",
            price=round(price, 2),
            instrument_token=instrument_key,
            order_type="LIMIT",    # never MARKET — see module docstring
            transaction_type=action,
            disclosed_quantity=0,
            trigger_price=0,
            is_amo=False,
        )
        return api.place_order(body, api_version="2.0")

    try:
        # SDK's generated client is synchronous — offload to a thread so it
        # doesn't block the event loop, same as the httpx.AsyncClient call
        # this replaces.
        response = await asyncio.to_thread(_place)
    except Exception as e:
        detail = getattr(e, "body", None) or str(e)
        log.warning("Upstox sandbox order failed for %s: %s", ticker, detail)
        return {"status": "FAILED", "detail": str(detail)[:300]}

    order_id = response.data.order_id if response and response.data else None
    if not order_id:
        return {"status": "FAILED", "detail": f"no order_id in response: {response}"}

    log.info("Sandbox order placed: %s %s x%d @ ~%.2f -> order_id=%s", action, ticker, quantity, price, order_id)
    return {"status": "PLACED", "order_id": order_id}


async def get_order_status(order_id: str) -> dict:
    """
    Fetch the current exchange-side status of a sandbox order via
    GET /v2/order/details?order_id=... — same endpoint shape sandbox and
    live share (sandbox is an API-parity rehearsal environment, see
    docs/UPSTOX_API_NOTES.md §2c), so this function works unchanged if a
    live order path is ever built later.

    place_sandbox_order() only tells you Upstox *accepted* the order for
    processing — "PLACED" is not "FILLED". The exchange-side sandbox still
    runs the order through its own matching/validation and can reject it
    afterward. Without this call, a rejected order would sit in
    pending_trade_confirmations forever reporting the misleadingly final-
    sounding "PLACED" status. Callers should map the returned `status`
    (Upstox's own vocabulary: "complete" | "rejected" | "cancelled" |
    "open" | "trigger pending" | ...) onto execution_status.

    Returns {"status": "complete"|"rejected"|..., "detail": str} on success,
    or {"status": "UNKNOWN", "detail": str} on any failure — never raises,
    same never-raise contract as place_sandbox_order() so a polling loop
    can't be taken down by one bad response.
    """
    # NOT migrated to the SDK: Upstox's sandbox only supports order
    # place/modify/cancel (confirmed straight from the installed SDK's own
    # Configuration.sandbox_urls allowlist — order/details isn't in it, and
    # a live SDK call against it returns "This API is not available in
    # sandbox mode"). Same constraint already documented for
    # get_available_funds below. Stays on the direct REST call, which
    # sandbox does genuinely serve.
    if not sandbox_configured():
        return {"status": "UNKNOWN", "detail": "UPSTOX_SANDBOX_TOKEN not set"}

    headers = {
        "Authorization": f"Bearer {settings.UPSTOX_SANDBOX_TOKEN}",
        "Accept": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                SANDBOX_ORDER_DETAILS_URL, params={"order_id": order_id}, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.text[:300] if e.response is not None else str(e)
        log.warning("Upstox order status fetch failed for %s: %s", order_id, detail)
        return {"status": "UNKNOWN", "detail": f"HTTP {e.response.status_code if e.response else '?'}: {detail}"}
    except Exception as e:
        log.warning("Upstox order status request failed for %s: %s", order_id, e)
        return {"status": "UNKNOWN", "detail": str(e)}

    order_data = data.get("data")
    # /v2/order/details returns either a single object or (rarely, for
    # multi-leg orders) a list — normalize to the most recent entry.
    if isinstance(order_data, list):
        order_data = order_data[-1] if order_data else None
    if not order_data:
        return {"status": "UNKNOWN", "detail": f"no order data in response: {data}"}

    status = str(order_data.get("status", "")).lower()
    detail = order_data.get("status_message") or status
    return {"status": status, "detail": detail}


async def get_available_funds() -> dict:
    """
    Fetch real available equity funds from Upstox sandbox via
    GET /v2/user/get-funds-and-margin — used to size trades on the
    human-confirmation path (backend/routers/confirmations.py's approve(),
    intelligence/live_confirmation.py) against what's ACTUALLY sitting in
    the account, rather than a static config number. Sandbox mirrors this
    endpoint the same way it mirrors /v2/order/place and /v2/order/details
    (see docs/UPSTOX_API_NOTES.md §2c — "sandbox emulates the real API").

    Upstox's documented response shape is:
        {"status": "success", "data": {"equity": {"available_margin": ...,
         "used_margin": ..., ...}, "commodity": {...}}}
    but exact field naming has drifted across Upstox API versions/docs, so
    this parses defensively: tries a set of known/likely field names for the
    equity segment's available balance, and logs the raw shape if none match
    so a mismatch is debuggable instead of silently wrong.

    Returns {"status": "ok", "available": <float>} on success, or
    {"status": "unavailable", "detail": <str>, "available": None} on ANY
    failure (no token configured, network error, unparseable response) —
    never raises, same never-raise contract as place_sandbox_order() and
    get_order_status(). Callers must treat available=None as "cannot size a
    trade right now" — never coerce to 0 or infinity.
    """
    if not sandbox_configured():
        return {
            "status": "unavailable",
            "detail": "UPSTOX_SANDBOX_TOKEN not set",
            "available": None,
        }

    headers = {
        "Authorization": f"Bearer {settings.UPSTOX_SANDBOX_TOKEN}",
        "Accept": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(SANDBOX_FUNDS_URL, params={"segment": "SEC"}, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.text[:300] if e.response is not None else str(e)
        log.warning("Upstox funds fetch failed: %s", detail)
        return {
            "status": "unavailable",
            "detail": f"HTTP {e.response.status_code if e.response else '?'}: {detail}",
            "available": None,
        }
    except Exception as e:
        log.warning("Upstox funds request failed: %s", e)
        return {"status": "unavailable", "detail": str(e), "available": None}

    payload = data.get("data")
    equity = None
    if isinstance(payload, dict):
        # Common shapes: {"data": {"equity": {...}, "commodity": {...}}}
        equity = payload.get("equity")
        if equity is None and ("available_margin" in payload or "availablemargin" in payload):
            # Some responses are flat (no per-segment nesting).
            equity = payload

    if not isinstance(equity, dict):
        log.warning("Upstox funds response had no parseable equity segment: %s", data)
        return {
            "status": "unavailable",
            "detail": f"unexpected funds response shape: {data}",
            "available": None,
        }

    available = None
    for key in ("available_margin", "availablemargin", "available_balance", "net"):
        if key in equity and equity[key] is not None:
            try:
                available = float(equity[key])
                break
            except (TypeError, ValueError):
                continue

    if available is None:
        log.warning("Upstox funds equity segment had no recognizable balance field: %s", equity)
        return {
            "status": "unavailable",
            "detail": f"no recognizable balance field in equity segment: {equity}",
            "available": None,
        }

    return {"status": "ok", "available": available}


def is_order_open(status: str) -> bool:
    """True if `status` (as returned by get_order_status) means the order is
    still working and should be polled again; False if it's terminal
    (filled/complete, rejected, or cancelled)."""
    return status.lower() in OPEN_ORDER_STATUSES
