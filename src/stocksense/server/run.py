"""
Server entrypoint. The localhost-only binding lives HERE, at the actual
`uvicorn.run` call site — not in app.py, which is just a FastAPI object
with no opinion about what host it gets served on. Anyone adding a new
way to run this app (a different script, a container entrypoint) must
consciously choose a host; there is no "default" path that could
accidentally end up exposed to the network.
"""

from __future__ import annotations

import socket

LOCALHOST_ONLY = "127.0.0.1"


def find_free_port(start: int = 8420, attempts: int = 20) -> int:
    """Electron's main.js needs a port to launch the API on before it
    can point a BrowserWindow at it -- scan for a free one rather than
    hardcoding one that might already be in use."""
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((LOCALHOST_ONLY, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port found in range {start}-{start + attempts}")


def run(port: int | None = None) -> None:
    import uvicorn

    from stocksense.server.app import app

    resolved_port = port or find_free_port()
    uvicorn.run(app, host=LOCALHOST_ONLY, port=resolved_port, log_level="info")


if __name__ == "__main__":
    run()
