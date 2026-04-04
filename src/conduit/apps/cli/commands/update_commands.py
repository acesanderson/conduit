from __future__ import annotations

import asyncio
import click
from rich.console import Console

console = Console(width=80)


@click.group(invoke_without_command=True)
@click.pass_context
def update(ctx: click.Context) -> None:
    """Update conduit model data."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@update.command()
def modelstore() -> None:
    """Sync ModelSpec database records with models.json."""
    from conduit.storage.modelspec_repository import ModelSpecRepository
    from conduit.core.model.models.modelstore import ModelStore

    ModelSpecRepository().initialize()
    ModelStore.update()


@update.command()
@click.option(
    "--server",
    type=str,
    default=None,
    help="Which inference server to update (default: all).",
)
def ollama(server: str | None) -> None:
    """Fetch live model lists from Ollama servers and update local cache."""
    from conduit.config import settings
    from conduit.core.clients.ollama.server_registry import (
        OLLAMA_SERVERS,
        fetch_server_models,
        write_server_to_cache,
    )
    from conduit.core.model.models.modelstore import ModelStore

    if server and server not in OLLAMA_SERVERS:
        raise click.BadParameter(
            f"Unknown server {server!r}. Choose from: {', '.join(OLLAMA_SERVERS)}.",
            param_hint="--server",
        )

    cache_path = settings.paths["OLLAMA_MODELS_PATH"]
    servers = [server] if server else list(OLLAMA_SERVERS)

    async def _update_server(s: str) -> bool:
        console.print(f"Fetching model list from [cyan]{s}[/cyan]...")
        try:
            models = await fetch_server_models(s)
        except Exception as exc:
            console.print(f"  [red]Failed to reach {s}: {exc}[/red]")
            return False
        write_server_to_cache(cache_path, s, models)
        console.print(f"  [green]{s}:[/green] {len(models)} models cached")
        return True

    async def _run():
        return await asyncio.gather(*[_update_server(s) for s in servers])

    results = asyncio.run(_run())

    if not any(results):
        raise SystemExit(1)

    all_models = ModelStore.models().get("ollama", [])
    console.print(f"\n[green]Total cached Ollama models:[/green] {len(all_models)}")
    for m in sorted(all_models):
        console.print(f"  {m}")


@update.command()
def both() -> None:
    """Refresh Ollama cache then sync modelstore DB."""
    ctx = click.get_current_context()
    ctx.invoke(ollama)
    ctx.invoke(modelstore)
