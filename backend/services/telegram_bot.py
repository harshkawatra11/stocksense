"""
Telegram approve/reject control surface for the human trade-confirmation
queue — lets the human act on each proposed SANDBOX trade from their phone.

Design constraints (deliberate, do not "improve" away):
  * Approve/reject ONLY. There is no "approve all", no auto-approve mode,
    no bulk action of any kind — per-trade explicit human action is the
    product's core safety invariant. One button tap = one confirmation row.
  * The bot talks to exactly ONE chat (settings.TELEGRAM_CHAT_ID). Updates
    from any other chat/user/group are logged and silently ignored — never
    answered, never sent trade data.
  * No bot-framework dependency: plain httpx long-polling against the
    Telegram Bot API (getUpdates timeout=50 + offset tracking), sendMessage
    with inline_keyboard, answerCallbackQuery, editMessageText/ReplyMarkup.
  * Decisions go through intelligence/confirmation_actions.py — the exact
    same code path as the web UI's /api/confirmations endpoints, so the
    atomic PENDING flip, stale-quote re-check and sandbox-only order
    placement are shared verbatim (zero duplicated SQL, idempotent races).

Unconfigured (no token/chat id) = clean no-op: start_telegram_bot() logs
once and returns; send_message()/notify_new_confirmation() do nothing.
"""
from __future__ import annotations

import asyncio
import logging

import httpx

from config import settings

log = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org/bot{token}/{method}"
_POLL_TIMEOUT_S = 50          # Telegram long-poll timeout (their max is ~50s)
_HTTP_TIMEOUT_S = 65.0        # must exceed the long-poll timeout
_BACKOFF_START_S = 2.0
_BACKOFF_MAX_S = 60.0


# ------------------------------------------------------------------ #
# Status / config                                                      #
# ------------------------------------------------------------------ #

def telegram_configured() -> bool:
    """Both the bot token AND the single allowed chat id must be set."""
    return bool(settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID)


def get_component_status() -> dict:
    """Stage-0-style {status, detail, source} snapshot for system_health."""
    if not telegram_configured():
        return {
            "status": "unavailable",
            "detail": "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — Telegram surface off",
            "source": "unconfigured",
        }
    return {
        "status": "ok",
        "detail": f"bot configured, locked to chat {settings.TELEGRAM_CHAT_ID}",
        "source": "telegram",
    }


# ------------------------------------------------------------------ #
# Low-level API helper                                                 #
# ------------------------------------------------------------------ #

async def _api(method: str, payload: dict, timeout: float = 30.0) -> dict | None:
    """POST one Bot API method. Returns the `result` field, or None on any
    failure (logged) — callers must tolerate None; the bot never crashes
    the app over a Telegram hiccup."""
    url = _API_BASE.format(token=settings.TELEGRAM_BOT_TOKEN, method=method)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
        data = resp.json()
        if not data.get("ok"):
            log.warning("Telegram %s failed: %s", method, data.get("description"))
            return None
        return data.get("result")
    except Exception as e:
        log.warning("Telegram %s error: %s", method, e)
        return None


# ------------------------------------------------------------------ #
# Outbound messages                                                    #
# ------------------------------------------------------------------ #

async def send_message(text: str, reply_markup: dict | None = None) -> dict | None:
    """Send plain text to THE configured chat. Safe no-op when unconfigured —
    other modules (EOD summary, order reconciliation, alerts) may call this
    unconditionally. Returns the sent message dict or None."""
    if not telegram_configured():
        return None
    payload: dict = {"chat_id": settings.TELEGRAM_CHAT_ID, "text": text}
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    return await _api("sendMessage", payload)


def format_confirmation_message(row: dict) -> str:
    """The message shown for one queued trade. Kept scannable on a phone."""
    price = float(row.get("price") or 0)
    return (
        f"Trade confirmation #{row['id']} — SANDBOX ONLY (no real money)\n"
        f"{row.get('action', 'BUY')} {row['ticker']} x {row.get('quantity')} @ ₹{price:,.2f}\n"
        f"Reasoning: {row.get('reasoning') or '—'}\n"
        f"Expires {settings.CONFIRMATION_EXPIRY_MINUTES} min after queueing. "
        f"Approve applies to THIS trade only."
    )


async def notify_new_confirmation(row: dict) -> None:
    """Push one freshly-queued confirmation with Approve/Reject buttons.
    `row` needs: id, ticker, action, quantity, price, reasoning.
    No-op when unconfigured; never raises (callers in the queueing path
    must not be broken by a Telegram failure)."""
    if not telegram_configured():
        return
    try:
        keyboard = {
            "inline_keyboard": [[
                {"text": "Approve", "callback_data": f"approve:{row['id']}"},
                {"text": "Reject", "callback_data": f"reject:{row['id']}"},
            ]]
        }
        await send_message(format_confirmation_message(row), reply_markup=keyboard)
    except Exception as e:
        log.warning("notify_new_confirmation failed (queueing unaffected): %s", e)


# ------------------------------------------------------------------ #
# Callback handling (button taps)                                      #
# ------------------------------------------------------------------ #

def parse_callback_data(data: str | None) -> tuple[str, int] | None:
    """Parse "approve:<id>" / "reject:<id>" -> (action, id). None for
    anything else — unknown actions, junk, missing/non-integer ids. Pure
    function, unit-tested offline in tests/test_telegram_callbacks.py."""
    if not data or ":" not in data:
        return None
    action, _, raw_id = data.partition(":")
    if action not in ("approve", "reject"):
        return None
    try:
        cid = int(raw_id)
    except ValueError:
        return None
    return action, cid


def _decision_summary(action: str, result: dict) -> str:
    """One-line outcome appended to the edited message / callback toast."""
    if action == "reject":
        return f"REJECTED — no order placed (#{result['id']} {result['ticker']})"
    status = result.get("status")
    if status == "EXPIRED":
        return (f"EXPIRED — {result['ticker']} was too stale to trade: "
                f"{result.get('execution_detail')}")
    exec_status = result.get("execution_status") or "no execution attempted"
    detail = result.get("execution_detail") or ""
    line = f"APPROVED — sandbox order {exec_status} for {result['ticker']}"
    if result.get("order_id"):
        line += f" (order {result['order_id']})"
    if detail:
        line += f": {detail}"
    return line


async def _handle_callback(cb: dict) -> None:
    """One button tap: verify chat, run the SHARED approve/reject action,
    toast the outcome, edit the original message and strip its buttons."""
    cb_id = cb.get("id")
    msg = cb.get("message") or {}
    chat_id = str(((msg.get("chat")) or {}).get("id", ""))
    from_id = str((cb.get("from") or {}).get("id", ""))

    # Hard gate: only the configured chat may drive decisions. Silently
    # ignore everyone else (log only — no answer, no data leak).
    if chat_id != str(settings.TELEGRAM_CHAT_ID):
        log.warning("Ignoring callback from unauthorized chat %s (user %s)", chat_id, from_id)
        return

    parsed = parse_callback_data(cb.get("data"))
    if parsed is None:
        await _api("answerCallbackQuery", {"callback_query_id": cb_id, "text": "Unrecognized action."})
        return
    action, confirmation_id = parsed

    # Import here (not module top) so this service stays importable even if
    # the intelligence package is mid-refactor; failure degrades to a toast.
    try:
        from intelligence.confirmation_actions import approve_confirmation, reject_confirmation
    except ImportError as e:
        log.error("confirmation_actions unavailable: %s", e)
        await _api("answerCallbackQuery", {"callback_query_id": cb_id, "text": "Backend unavailable, try the web UI."})
        return

    try:
        if action == "approve":
            result = await approve_confirmation(confirmation_id)
        else:
            result = await reject_confirmation(confirmation_id)
    except Exception as e:
        log.error("Telegram %s of confirmation %d failed: %s", action, confirmation_id, e, exc_info=True)
        await _api("answerCallbackQuery", {"callback_query_id": cb_id,
                                           "text": f"Error while processing: {e}"[:190]})
        return

    if result is None:
        # Idempotency: already approved/rejected/expired (web UI, a second
        # tap, or the expiry job got there first). Acknowledge gracefully
        # and disable the now-dead buttons.
        await _api("answerCallbackQuery", {"callback_query_id": cb_id,
                                           "text": "Already handled (or expired) — no action taken."})
        if msg.get("message_id"):
            await _api("editMessageReplyMarkup", {
                "chat_id": settings.TELEGRAM_CHAT_ID,
                "message_id": msg["message_id"],
                "reply_markup": {"inline_keyboard": []},
            })
        return

    summary = _decision_summary(action, result)
    await _api("answerCallbackQuery", {"callback_query_id": cb_id, "text": summary[:190]})
    # Rewrite the original message with the decision + execution outcome and
    # remove the buttons (editMessageText with no reply_markup drops them).
    if msg.get("message_id"):
        original = msg.get("text") or format_confirmation_message(result)
        await _api("editMessageText", {
            "chat_id": settings.TELEGRAM_CHAT_ID,
            "message_id": msg["message_id"],
            "text": f"{original}\n\n>> {summary}",
        })


async def _handle_message(m: dict) -> None:
    """Plain text messages: only the configured chat gets any reply, and
    even it gets no command surface — decisions happen via buttons only."""
    chat_id = str(((m.get("chat")) or {}).get("id", ""))
    if chat_id != str(settings.TELEGRAM_CHAT_ID):
        log.warning("Ignoring message from unauthorized chat %s", chat_id)
        return
    text = (m.get("text") or "").strip().lower()
    if text in ("/start", "/help", "help", "status"):
        await send_message(
            "StockSense confirmation bot. I push each proposed SANDBOX trade "
            "here with Approve/Reject buttons — one decision per trade, no "
            "bulk or auto-approve, ever. No real money is involved."
        )
    elif text.startswith("/kill"):
        from intelligence.kill_switch import trip
        reason = m.get("text", "").strip()[5:].strip() or "manual kill switch via Telegram"
        await trip(reason)
        await send_message(f"🛑 Kill switch TRIPPED: {reason}\nNo new orders will be placed until /reset.")
    elif text.startswith("/reset"):
        from intelligence.kill_switch import reset
        await reset()
        await send_message("✅ Kill switch reset — order placement resumed.")


# ------------------------------------------------------------------ #
# Long-poll loop (lifespan task)                                       #
# ------------------------------------------------------------------ #

async def start_telegram_bot() -> None:
    """Lifespan-compatible forever task (asyncio.create_task in
    backend/main.py, same pattern as start_quote_cache_feed). Unconfigured:
    log once, return — clean no-op. Otherwise: getUpdates long-poll with
    offset tracking, reconnect/backoff on any error, never crashes the app."""
    if not telegram_configured():
        log.info("Telegram bot unconfigured (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID unset) — approve/reject via web UI only")
        return

    log.info("Telegram bot starting — long-polling, locked to chat %s", settings.TELEGRAM_CHAT_ID)
    offset: int | None = None
    backoff = _BACKOFF_START_S
    while True:
        try:
            payload: dict = {
                "timeout": _POLL_TIMEOUT_S,
                "allowed_updates": ["message", "callback_query"],
            }
            if offset is not None:
                payload["offset"] = offset
            updates = await _api("getUpdates", payload, timeout=_HTTP_TIMEOUT_S)
            if updates is None:
                # Network/API failure already logged — back off and retry.
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX_S)
                continue
            backoff = _BACKOFF_START_S

            for u in updates:
                offset = u["update_id"] + 1  # ack even ones we ignore/fail on
                try:
                    if "callback_query" in u:
                        await _handle_callback(u["callback_query"])
                    elif "message" in u:
                        await _handle_message(u["message"])
                except Exception as e:
                    log.error("Telegram update %s handling failed: %s", u.get("update_id"), e, exc_info=True)
        except asyncio.CancelledError:
            log.info("Telegram bot stopping (app shutdown)")
            raise
        except Exception as e:
            log.error("Telegram poll loop error: %s — retrying in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _BACKOFF_MAX_S)
