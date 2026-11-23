"""Serve the built Next.js dashboard (web/out) at `/`, or a small fallback page when it is not built."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

FALLBACK = "<h1>StockSense</h1><p>Dashboard not built: run run-lab.bat</p>"


def mount_web(app: FastAPI, web_out: Path) -> None:
    """Mount `web_out` at `/` after all routers; without index.html add a fallback GET /."""
    if (web_out / "index.html").exists():
        app.mount("/", StaticFiles(directory=web_out, html=True), name="web")
        return

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def fallback() -> str:
        return FALLBACK
