"""
Upstox SANDBOX order placement — sandbox.upstox.com/v2, risk-free rehearsal.

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

import logging

import httpx

from config import settings
from data.pipeline.upstox_client import resolve_instrument_key

log = logging.getLogger(__name__)

SANDBOX_ORDER_URL = "https://sandbox.upstox.com/v2/order/place"


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

    instrument_key = await resolve_instrument_key(ticker)
    if instrument_key is None:
        return {"status": "FAILED", "detail": f"no Upstox instrument_key for {ticker} (missing ISIN)"}

    payload = {
        "quantity": quantity,
        "product": "D",           # delivery — matches StockSense's swing-trade holding style
        "validity": "DAY",
        "price": round(price, 2),
        "instrument_token": instrument_key,
        "order_type": "LIMIT",    # never MARKET — see module docstring
        "transaction_type": action,
        "disclosed_quantity": 0,
        "trigger_price": 0,
        "is_amo": False,
    }
    headers = {
        "Authorization": f"Bearer {settings.UPSTOX_SANDBOX_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(SANDBOX_ORDER_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.text[:300] if e.response is not None else str(e)
        log.warning("Upstox sandbox order failed for %s: %s", ticker, detail)
        return {"status": "FAILED", "detail": f"HTTP {e.response.status_code if e.response else '?'}: {detail}"}
    except Exception as e:
        log.warning("Upstox sandbox order request failed for %s: %s", ticker, e)
        return {"status": "FAILED", "detail": str(e)}

    order_id = (data.get("data") or {}).get("order_id")
    if not order_id:
        return {"status": "FAILED", "detail": f"no order_id in response: {data}"}

    log.info("Sandbox order placed: %s %s x%d @ ~%.2f -> order_id=%s", action, ticker, quantity, price, order_id)
    return {"status": "PLACED", "order_id": order_id}
