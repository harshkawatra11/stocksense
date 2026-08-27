"""
Phase J1.0: the live probe, run BEFORE writing any real integration
module. A prior SmartAPI login attempt timed out on this home ISP --
this script exists to find that out in under a minute, not after a day
of building a module that can never actually connect.

Deliberately standalone: no package import beyond stdlib + smartapi-
python + pyotp, no database writes, no .env writes, no order-placement
call anywhere. Never prints jwtToken/refreshToken/feedToken -- only
whether each step succeeded.

Usage: python scripts/probe_angel_smartapi.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> dict:
    env_path = REPO_ROOT / ".env"
    values = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    return values


def main() -> int:
    env = _load_env()
    api_key = env.get("STOCKSENSE_ANGEL_API_KEY") or os.environ.get("STOCKSENSE_ANGEL_API_KEY")
    client_code = env.get("STOCKSENSE_ANGEL_CLIENT_CODE") or os.environ.get("STOCKSENSE_ANGEL_CLIENT_CODE")
    password = env.get("STOCKSENSE_ANGEL_PASSWORD") or os.environ.get("STOCKSENSE_ANGEL_PASSWORD")
    totp_secret = env.get("STOCKSENSE_ANGEL_TOTP_SECRET") or os.environ.get("STOCKSENSE_ANGEL_TOTP_SECRET")

    missing = [name for name, v in [
        ("STOCKSENSE_ANGEL_API_KEY", api_key), ("STOCKSENSE_ANGEL_CLIENT_CODE", client_code),
        ("STOCKSENSE_ANGEL_PASSWORD", password), ("STOCKSENSE_ANGEL_TOTP_SECRET", totp_secret),
    ] if not v]
    if missing:
        print(f"MISSING credentials in .env: {missing}")
        return 4

    print("probe: DNS/TLS reachability to apiconnect.angelone.in")
    t0 = time.time()
    try:
        import socket
        import ssl

        ctx = ssl.create_default_context()
        with socket.create_connection(("apiconnect.angelone.in", 443), timeout=15) as sock:
            with ctx.wrap_socket(sock, server_hostname="apiconnect.angelone.in"):
                pass
        print(f"  OK ({time.time() - t0:.2f}s)")
    except Exception as e:
        print(f"  FAILED after {time.time() - t0:.2f}s: {e!r}")
        print("  Likely a network/DNS/ISP-level block, not a credential problem.")
        return 2

    print("probe: TOTP generation")
    try:
        import pyotp

        totp = pyotp.TOTP(totp_secret).now()
        print(f"  OK (6-digit code generated, not printed)")
    except Exception as e:
        print(f"  FAILED: {e!r} -- STOCKSENSE_ANGEL_TOTP_SECRET may not be a valid base32 secret")
        return 3

    print("probe: SmartAPI login (generateSession)")
    t0 = time.time()
    try:
        from SmartApi import SmartConnect

        sc = SmartConnect(api_key=api_key)
        totp_now = pyotp.TOTP(totp_secret).now()
        session = sc.generateSession(client_code, password, totp_now)
        elapsed = time.time() - t0
        if not session or not session.get("status"):
            print(f"  REJECTED after {elapsed:.2f}s: {session.get('message') if session else 'no response'}")
            print("  This is a credential/TOTP problem (wrong client code, MPIN, or TOTP secret), not a network one.")
            return 5
        print(f"  OK ({elapsed:.2f}s) -- session established, tokens NOT printed")
    except Exception as e:
        print(f"  FAILED after {time.time() - t0:.2f}s: {type(e).__name__}: {e}")
        return 6

    print("probe: one read-only call (getProfile)")
    t0 = time.time()
    try:
        profile = sc.getProfile(session["data"]["refreshToken"])
        elapsed = time.time() - t0
        if not profile or not profile.get("status"):
            print(f"  FAILED after {elapsed:.2f}s: {profile.get('message') if profile else 'no response'}")
            return 7
        print(f"  OK ({elapsed:.2f}s) -- profile fetched, name={profile['data'].get('name')!r}")
    except Exception as e:
        print(f"  FAILED after {time.time() - t0:.2f}s: {type(e).__name__}: {e}")
        return 7

    print("\nALL PROBES PASSED. Safe to build the real read-only sync module.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
