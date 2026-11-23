"""Middleware that makes the whole app read-only (paper-only rule: no write routes)."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class ReadOnlyMiddleware(BaseHTTPMiddleware):
    """Reject every method except GET, HEAD and OPTIONS (CORS preflight); mark API replies no-store."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            return JSONResponse({"error": "read-only API"}, status_code=405)
        resp = await call_next(request)
        if request.url.path.startswith("/api"):
            resp.headers["Cache-Control"] = "no-store"
        return resp
