"""
Offline tests for the Telegram bot's callback handling — no network, no DB.
Covers: callback_data parsing, the unauthorized-chat gate, and idempotency
("already handled" path when approve/reject returns None).

Run: python -m tests.test_telegram_callbacks   (or via pytest)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services import telegram_bot as tb


def test_parse_callback_data():
    assert tb.parse_callback_data("approve:42") == ("approve", 42)
    assert tb.parse_callback_data("reject:7") == ("reject", 7)
    # junk / hostile inputs all -> None
    for bad in (None, "", "approve", "approve:", "approve:abc", "nuke:1",
                "approve_all:1", ":5", "approve:1:2x", "APPROVE:1"):
        assert tb.parse_callback_data(bad) is None, bad
    # "approve:1:2" partitions to id "1:2" -> non-int -> None
    assert tb.parse_callback_data("approve:1:2") is None


def _cb(chat_id, data, cb_id="cb1", message_id=99):
    return {
        "id": cb_id,
        "from": {"id": chat_id},
        "data": data,
        "message": {"message_id": message_id, "chat": {"id": chat_id},
                    "text": "Trade confirmation #5"},
    }


def test_unauthorized_chat_is_silently_ignored():
    calls = []

    async def fake_api(method, payload, timeout=30.0):
        calls.append(method)
        return {}

    with patch.object(tb.settings.__class__, "TELEGRAM_CHAT_ID", property(lambda self: "111")), \
         patch.object(tb, "_api", fake_api):
        asyncio.run(tb._handle_callback(_cb(chat_id=999, data="approve:5")))
    # No API call at all — no answer, no edit, no data leak.
    assert calls == [], calls


def test_already_handled_is_idempotent():
    calls = []

    async def fake_api(method, payload, timeout=30.0):
        calls.append((method, payload))
        return {}

    async def fake_approve(cid):
        return None  # row already decided / not pending

    with patch.object(tb.settings.__class__, "TELEGRAM_BOT_TOKEN", property(lambda self: "t")), \
         patch.object(tb.settings.__class__, "TELEGRAM_CHAT_ID", property(lambda self: "111")), \
         patch.object(tb, "_api", fake_api), \
         patch("intelligence.confirmation_actions.approve_confirmation", fake_approve):
        asyncio.run(tb._handle_callback(_cb(chat_id=111, data="approve:5")))

    methods = [m for m, _ in calls]
    assert "answerCallbackQuery" in methods
    toast = next(p for m, p in calls if m == "answerCallbackQuery")
    assert "Already handled" in toast["text"]
    # Buttons stripped from the original message.
    assert "editMessageReplyMarkup" in methods
    edit = next(p for m, p in calls if m == "editMessageReplyMarkup")
    assert edit["reply_markup"] == {"inline_keyboard": []}


def test_decided_row_edits_message_and_removes_buttons():
    calls = []

    async def fake_api(method, payload, timeout=30.0):
        calls.append((method, payload))
        return {}

    async def fake_reject(cid):
        return {"id": cid, "ticker": "RELIANCE", "status": "REJECTED",
                "action": "BUY", "quantity": 5, "price": 2850.0, "reasoning": "x"}

    with patch.object(tb.settings.__class__, "TELEGRAM_BOT_TOKEN", property(lambda self: "t")), \
         patch.object(tb.settings.__class__, "TELEGRAM_CHAT_ID", property(lambda self: "111")), \
         patch.object(tb, "_api", fake_api), \
         patch("intelligence.confirmation_actions.reject_confirmation", fake_reject):
        asyncio.run(tb._handle_callback(_cb(chat_id=111, data="reject:5")))

    methods = [m for m, _ in calls]
    assert methods == ["answerCallbackQuery", "editMessageText"], methods
    edit = calls[1][1]
    assert "REJECTED" in edit["text"] and "reply_markup" not in edit
    assert edit["text"].startswith("Trade confirmation #5")  # original kept


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all telegram callback tests passed")
