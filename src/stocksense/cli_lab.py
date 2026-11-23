"""Dashboard commands (PLAN-M): `serve` starts the local read-only server. Registered from cli.py."""

from __future__ import annotations

from typing import Annotated

import typer


def serve(
    ctx: typer.Context,
    port: Annotated[int, typer.Option(help="Port on 127.0.0.1 (Market Lab uses 8810).")] = 8765,
    no_feeds: Annotated[bool, typer.Option(help="Do not start the live feed threads.")] = False,
    no_catchup: Annotated[
        bool, typer.Option(help="Do not run the startup catch-up (ingest, panel, scan).")
    ] = False,
) -> None:
    """Start the local dashboard server on 127.0.0.1 (manual start only; Ctrl+C stops it)."""
    import uvicorn

    from stocksense.api.app import create_app
    from stocksense.cli import _settings

    app = create_app(_settings(ctx), start_feeds=not no_feeds)
    app.state.catchup_enabled = not no_catchup
    uvicorn.run(app, host="127.0.0.1", port=port, workers=1, log_level="info")


def register(app: typer.Typer) -> None:
    """Hook for cli.py: add `serve`."""
    app.command("serve")(serve)
