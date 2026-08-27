"""
Phase J1.1: Angel One SmartAPI session lifecycle. The one place
credentials and the raw SmartConnect client are handled directly --
every other module in this package gets a ReadOnlyBrokerClient
(angel_readonly.py), never this session's underlying client.

Token storage deliberately does NOT reuse data/upstox_auth.py's
write-token-to-.env pattern: Upstox's token rotates once daily; Angel's
jwt/refresh/feed triple rotates per-session and there are three of them
-- rewriting .env several times a day is a corruption risk on a file
that also holds long-lived secrets. Cached instead to a gitignored JSON
file under data_store/secrets/, atomic write (tmp + os.replace).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from stocksense.core.config import DATA_STORE

SESSION_PATH = DATA_STORE / "secrets" / "angel_session.json"


class TransientBrokerError(Exception):
    """Network/timeout/rate-limit -- retryable later, no human needed."""


class BrokerAuthError(Exception):
    """Credentials, MPIN, or TOTP rejected -- a human needs to fix
    something in .env. Must never be conflated with TransientBrokerError:
    a caller (e.g. the nightly sync graph) needs to know whether to
    retry tonight or stop and tell the user."""


@dataclass(frozen=True)
class AngelSession:
    jwt_token: str
    refresh_token: str
    feed_token: str
    client_code: str
    issued_at: datetime

    def is_expired(self, skew_s: int = 300) -> bool:
        """Angel sessions are valid for the trading day; treated as
        expired after 8 hours as a conservative default (no documented
        exact TTL), with a safety margin so a sync never starts a call
        against a session about to expire mid-request."""
        return datetime.now(timezone.utc) - self.issued_at > timedelta(hours=8) - timedelta(seconds=skew_s)

    def __repr__(self) -> str:
        return f"AngelSession(client_code={self.client_code!r}, issued_at={self.issued_at.isoformat()!r}, tokens=<redacted>)"


def current_totp(secret: str) -> str:
    import pyotp

    return pyotp.TOTP(secret).now()


def _load_cached_session() -> AngelSession | None:
    if not SESSION_PATH.exists():
        return None
    try:
        data = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
        return AngelSession(
            jwt_token=data["jwt_token"], refresh_token=data["refresh_token"],
            feed_token=data["feed_token"], client_code=data["client_code"],
            issued_at=datetime.fromisoformat(data["issued_at"]),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _save_session(session: AngelSession) -> None:
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "jwt_token": session.jwt_token, "refresh_token": session.refresh_token,
        "feed_token": session.feed_token, "client_code": session.client_code,
        "issued_at": session.issued_at.isoformat(),
    }
    tmp_path = SESSION_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp_path, SESSION_PATH)
    try:
        os.chmod(SESSION_PATH, 0o600)
    except OSError:
        pass  # best-effort on platforms/filesystems that don't support it (e.g. some Windows setups)


def login(settings, *, force: bool = True):
    """Returns (AngelSession, raw SmartConnect client) -- the raw client
    is handed ONLY to angel_readonly.ReadOnlyBrokerClient by callers,
    never used directly elsewhere.

    Always performs a fresh login. FOUND LIVE, verified with a real
    two-process test before trusting either behavior: reconstructing a
    SmartConnect session purely from setAccessToken/setRefreshToken/
    setFeedToken/setUserId (this SDK's only public setters) does NOT
    actually work across process boundaries -- position()/orderBook()/
    tradeBook() all came back "Invalid Token" against a reused session
    while holding() (called moments earlier, same tokens) had succeeded.
    A fresh generateSession() call in the SAME process, immediately
    followed by the same three calls, worked for all of them. Since a
    login is cheap (~2s, confirmed via probe_angel_smartapi.py) and this
    is invoked at most a few times a day, the honest fix is to always
    log in fresh rather than ship a cache-reuse path proven not to work.
    The session is still WRITTEN to disk on every successful login, for
    audit/debugging (last successful login time) -- just never read back
    for functional reuse. `force` is kept as a parameter for API
    stability but has no effect; a future fix that actually reconstructs
    a working session would restore its meaning."""
    missing = [
        name for name, v in [
            ("angel_api_key", settings.angel_api_key), ("angel_client_code", settings.angel_client_code),
            ("angel_password", settings.angel_password), ("angel_totp_secret", settings.angel_totp_secret),
        ] if not v
    ]
    if missing:
        raise BrokerAuthError(f"missing Angel One credentials in settings: {missing}")

    try:
        from SmartApi import SmartConnect

        client = SmartConnect(api_key=settings.angel_api_key)
        totp = current_totp(settings.angel_totp_secret)
        response = client.generateSession(settings.angel_client_code, settings.angel_password, totp)
    except (ConnectionError, TimeoutError, OSError) as e:
        raise TransientBrokerError(f"network error during Angel One login: {e}") from e

    if not response or not response.get("status"):
        message = response.get("message") if response else "no response"
        raise BrokerAuthError(f"Angel One login rejected: {message}")

    data = response["data"]
    session = AngelSession(
        jwt_token=data["jwtToken"], refresh_token=data["refreshToken"],
        feed_token=data.get("feedToken", ""), client_code=settings.angel_client_code,
        issued_at=datetime.now(timezone.utc),
    )
    _save_session(session)
    return session, client


def logout(client) -> None:
    try:
        client.terminateSession(client.userId if hasattr(client, "userId") else None)
    except Exception:
        pass  # best-effort; a failed logout call doesn't need to fail the caller
