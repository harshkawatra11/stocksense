"""Telegram notifications; a no-op unless both Telegram secrets are set. Never raises, never logs a secret."""

from __future__ import annotations

import httpx

from stocksense.core.config import Settings
from stocksense.core.log import get_logger

log = get_logger(__name__)

_MAX_TEXT = 3900


class Notifier:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        s = settings.secrets
        self._token, self._chat = s.telegram_bot_token, s.telegram_chat_id
        self._client = client

    @property
    def enabled(self) -> bool:
        return bool(self._token and self._chat)

    def send(self, text: str) -> bool:
        """Send one message; True if Telegram accepted it. Failures are logged by type only."""
        if not self.enabled:
            return False
        client = self._client or httpx.Client(timeout=httpx.Timeout(10, connect=5))
        try:
            r = client.post(
                f"https://api.telegram.org/bot{self._token}/sendMessage",
                json={"chat_id": self._chat, "text": text[:_MAX_TEXT]},
            )
        except httpx.HTTPError as e:  # the message of an httpx error can carry the URL (= the token)
            log.warning("notify_failed", error=type(e).__name__)
            return False
        finally:
            if self._client is None:
                client.close()
        if r.status_code != 200:
            log.warning("notify_failed", status=r.status_code)
            return False
        return True
