"""
WS /api/ws/prices — fan-out of the live quote cache to the frontend.

On connect: sends a full snapshot of backend.services.quote_cache.get_all().
After that: streams throttled deltas (~1s) of whatever changed since the last
send. No subscribe/unsubscribe protocol yet — the client implicitly tracks
whatever symbols were in its snapshot; we just broadcast the current state of
every tracked symbol whenever something changed.

Message shapes sent to the client:
    {"type": "snapshot", "quotes": {symbol: {"ltp": float, "close": float|None, "ts": float}}}
    {"type": "delta", "quotes": {symbol: {"ltp": float, "close": float|None, "ts": float}}}
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services import quote_cache

log = logging.getLogger(__name__)
router = APIRouter()

BROADCAST_INTERVAL_SECONDS = 1.0


@router.websocket("/prices")
async def ws_prices(websocket: WebSocket) -> None:
    await websocket.accept()

    snapshot = quote_cache.get_all()
    try:
        # `history` = recent LTPs per symbol (oldest first) to seed client-side
        # sparklines; after this the client appends deltas itself.
        await websocket.send_json({
            "type": "snapshot",
            "quotes": snapshot,
            "history": quote_cache.get_all_history(),
        })
    except Exception:
        log.debug("ws_prices: client disconnected before snapshot could be sent")
        return

    last_sent = dict(snapshot)

    try:
        while True:
            await asyncio.sleep(BROADCAST_INTERVAL_SECONDS)

            current = quote_cache.get_all()
            changed = {
                sym: quote
                for sym, quote in current.items()
                if last_sent.get(sym) != quote
            }
            if not changed:
                continue

            try:
                await websocket.send_json({"type": "delta", "quotes": changed})
            except Exception:
                # Client gone or socket broken — stop this connection's loop
                # without affecting any other connected client.
                break

            last_sent.update(changed)
    except WebSocketDisconnect:
        log.debug("ws_prices: client disconnected")
    except Exception:
        log.exception("ws_prices: broadcast loop error")
    finally:
        log.debug("ws_prices: connection closed")
