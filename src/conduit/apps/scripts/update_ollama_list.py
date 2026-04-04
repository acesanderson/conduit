"""
Update the cached Ollama model list for one or all inference servers.

Usage:
    update_ollama_list.py                    # update all servers
    update_ollama_list.py --server deepwater
    update_ollama_list.py --server bywater
    update_ollama_list.py --server backwater

Fetches the live model list from each server's HeadwaterServer /v1/models
endpoint and writes it into the local cache at OLLAMA_MODELS_PATH.
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


async def _update_server(server: str, cache_path) -> bool:
    console.print(f"Fetching model list from [cyan]{server}[/cyan]...")
    try:
        models = await fetch_server_models(server)
    except Exception as exc:
        console.print(f"  [red]Failed to reach {server}: {exc}[/red]")
        return False
    write_server_to_cache(cache_path, server, models)
    console.print(f"  [green]{server}:[/green] {len(models)} models cached")
    return True


@click.command()
@click.option(
    "--server",
    type=click.Choice(list(OLLAMA_SERVERS)),
    default=None,
    help="Which inference server to update (default: all).",
)
def main(server: str | None) -> None:
    """Fetch live model lists and update the local cache."""
    cache_path = settings.paths["OLLAMA_MODELS_PATH"]
    servers = [server] if server else list(OLLAMA_SERVERS)

    async def run():
        return await asyncio.gather(*[_update_server(s, cache_path) for s in servers])

    results = asyncio.run(run())

    if not any(results):
        raise SystemExit(1)

    all_models = ModelStore.models().get("ollama", [])
    console.print(f"\n[green]Total cached Ollama models:[/green] {len(all_models)}")
    for m in sorted(all_models):
        console.print(f"  {m}")

    # Report ModelSpec coverage — does NOT write to Postgres.
    try:
        from conduit.storage.modelspec_repository import ModelSpecRepository
        from conduit.storage.modelspec_repository import ModelSpecRepositoryError

        repo = ModelSpecRepository()
        spec_names = set(repo.get_all_names())
        ollama_models = set(all_models)
        missing = ollama_models - spec_names
        if missing:
            console.print(
                f"\n[yellow]{len(missing)} ollama model(s) have no ModelSpec[/yellow]"
                " — run [cyan]update[/cyan] to generate"
            )
        else:
            console.print("\n[green]All ollama models have ModelSpecs.[/green]")
    except ModelSpecRepositoryError:
        console.print("[dim]Could not reach Postgres to check ModelSpec coverage.[/dim]")


if __name__ == "__main__":
    main()
