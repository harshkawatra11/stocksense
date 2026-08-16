"""
Upstox OAuth token exchange.

Upstox access tokens expire daily (~3:30am IST). This script trades a
short-lived authorization `code` (obtained via the browser login-redirect
flow) for an access token, and writes it directly into `.env` — the
token is never printed to stdout or returned to a caller, so it doesn't
end up sitting in shell history or a chat transcript.

Usage:
    python -m stocksense.data.upstox_auth <code> [--redirect-uri URL]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import requests

from stocksense.core.config import REPO_ROOT, get_settings

TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"
DEFAULT_REDIRECT_URI = "https://127.0.0.1:3000/callback"
ENV_PATH = REPO_ROOT / ".env"


def exchange_code_for_token(code: str, redirect_uri: str = DEFAULT_REDIRECT_URI) -> str:
    settings = get_settings()
    if not settings.upstox_api_key or not settings.upstox_api_secret:
        raise RuntimeError(
            "STOCKSENSE_UPSTOX_API_KEY / STOCKSENSE_UPSTOX_API_SECRET missing from .env"
        )

    resp = requests.post(
        TOKEN_URL,
        headers={"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
        data={
            "code": code,
            "client_id": settings.upstox_api_key,
            "client_secret": settings.upstox_api_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in response: {payload}")
    return token


def write_token_to_env(token: str, env_path: Path = ENV_PATH) -> None:
    """Update STOCKSENSE_UPSTOX_ACCESS_TOKEN in .env in place, without
    printing the token or touching any other line."""
    text = env_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^STOCKSENSE_UPSTOX_ACCESS_TOKEN=.*$", flags=re.MULTILINE)
    if pattern.search(text):
        new_text = pattern.sub(f"STOCKSENSE_UPSTOX_ACCESS_TOKEN={token}", text)
    else:
        new_text = text.rstrip("\n") + f"\nSTOCKSENSE_UPSTOX_ACCESS_TOKEN={token}\n"
    env_path.write_text(new_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("code", help="Authorization code from the redirect URL's ?code= param")
    parser.add_argument("--redirect-uri", default=DEFAULT_REDIRECT_URI)
    args = parser.parse_args()

    try:
        token = exchange_code_for_token(args.code, args.redirect_uri)
        write_token_to_env(token)
    except Exception as e:  # noqa: BLE001
        print(f"FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    print("OK: access token written to .env (STOCKSENSE_UPSTOX_ACCESS_TOKEN). Not printed here.")


if __name__ == "__main__":
    main()
