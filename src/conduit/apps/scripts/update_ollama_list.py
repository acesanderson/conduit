"""
Update the cached Ollama model list for a specific inference server.

Usage:
    update_ollama --server deepwater
    update_ollama --server bywater
    update_ollama --server backwater

This fetches the live model list from the given server's Ollama HTTP API
(GET /api/tags) and writes it into the local cache at OLLAMA_MODELS_PATH
under that server's key. Only the specified server's entry is updated.
"""

from __future__ import annotations

import asyncio
import logging

import click
from rich.console import Console

from conduit.config import settings
from conduit.core.clients.ollama.server_registry import (
    OLLAMA_SERVERS,
    fetch_server_models,
    write_server_to_cache,
)
from conduit.core.model.models.modelstore import ModelStore

logger = logging.getLogger(__name__)
console = Console(width=80)


@click.command()
@click.option(
    "--server",
    type=click.Choice(list(OLLAMA_SERVERS)),
    required=True,
    help="Which inference server to update.",
)
def main(server: str) -> None:
    """Fetch the live model list from SERVER and update the local cache."""
    console.print(f"[green]Fetching model list from {server}...[/green]")
    try:
        models = asyncio.run(fetch_server_models(server))
    except Exception as exc:
        console.print(f"[red]Failed to reach {server}: {exc}[/red]")
        raise SystemExit(1)

    cache_path = settings.paths["OLLAMA_MODELS_PATH"]
    write_server_to_cache(cache_path, server, models)

    console.print(
        f"[green]Updated {server}:[/green] [yellow]{len(models)} models cached[/yellow]"
    )
    console.print(
        f"[green]All Ollama models:[/green] [yellow]{ModelStore.models()['ollama']}[/yellow]"
    )


if __name__ == "__main__":
    main()
