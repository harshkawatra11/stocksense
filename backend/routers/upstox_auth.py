"""
Upstox OAuth routes — login redirect + callback.

NOTE: this router is NOT mounted in backend/main.py yet — whoever wires up
main.py should add:
    from backend.routers import upstox_auth
    app.include_router(upstox_auth.router, prefix="/api/upstox", tags=["upstox"])

GET /api/upstox/login    -> redirects to the Upstox authorization dialog
GET /api/upstox/callback -> receives ?code=&state=, exchanges the code for
                             an access token (data/pipeline/upstox_client.py),
                             stores it, returns a simple success/failure response.
"""
from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import RedirectResponse, HTMLResponse

from data.pipeline.upstox_client import get_authorization_url, exchange_code_for_token

router = APIRouter()


@router.get("/login")
async def login():
    """Redirect the user to the Upstox OAuth authorization dialog."""
    return RedirectResponse(url=get_authorization_url(state="stocksense"))


@router.get("/callback")
async def callback(code: str = Query(...), state: str | None = Query(None)):
    """
    Upstox redirects here with ?code=...&state=... after the user authorizes.
    Exchanges the single-use code for an access token and stores it
    (data/pipeline/upstox_client.store_token via exchange_code_for_token).
    """
    token = await exchange_code_for_token(code)

    if token is None:
        return HTMLResponse(
            "<h2>Upstox authorization failed</h2>"
            "<p>Could not exchange the authorization code for an access token. "
            "Check UPSTOX_CLIENT_ID / UPSTOX_CLIENT_SECRET / UPSTOX_REDIRECT_URI in .env "
            "and server logs for details.</p>",
            status_code=502,
        )

    return HTMLResponse(
        "<h2>Upstox authorization successful</h2>"
        "<p>Access token acquired and stored. You can close this tab.</p>"
    )
