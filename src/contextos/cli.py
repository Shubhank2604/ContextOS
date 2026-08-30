"""Command-line entry point for ContextOS."""

from __future__ import annotations

import typer

from contextos import __version__

app = typer.Typer(
    name="contextos",
    help="Construct and inspect LLM context under explicit token budgets.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Run ContextOS commands."""


@app.command()
def version() -> None:
    """Print the installed ContextOS version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
