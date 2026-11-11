from __future__ import annotations

import typer

from stocksense import __version__

app = typer.Typer(help="StockSense command line.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """StockSense command line."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
