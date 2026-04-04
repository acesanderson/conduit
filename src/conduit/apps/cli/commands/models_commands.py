# src/conduit/apps/cli/commands/models_commands.py
from __future__ import annotations

import click


@click.command("models")
@click.option("-m", "--model", type=str, help="Name of the model to retrieve details for.")
@click.option("-t", "--type", "model_type", type=str, help="Type of the model to filter by.")
@click.option("-p", "--provider", type=str, help="Provider of the model to filter by.")
@click.option("-a", "--aliases", is_flag=True, help="Display model aliases.")
@click.option("-e", "--embeddings", is_flag=True, help="Display models that support embeddings.")
@click.option("-r", "--rerankers", is_flag=True, help="Display models that support reranking.")
@click.option(
    "-s",
    "--server",
    type=click.Choice(["deepwater", "bywater", "backwater"]),
    default=None,
    help="Show models for a specific Ollama server (live-fetches with cache fallback).",
)
def models_command(
    model: str | None,
    model_type: str | None,
    provider: str | None,
    aliases: bool,
    embeddings: bool,
    rerankers: bool,
    server: str | None,
) -> None:
    """List and inspect available models."""
    if rerankers:
        from headwater_client.client.headwater_client import HeadwaterClient
        from rich.console import Console

        specs = HeadwaterClient().reranker.list_reranker_models()
        console = Console()
        console.print("Reranker models:", style="bold green")
        for spec in specs:
            console.print(f"  - {spec.name}", style="cyan")
        return

    if embeddings:
        from conduit.embeddings.generate_embeddings import list_embedding_models
        from rich.console import Console

        specs = list_embedding_models()
        console = Console()
        console.print("Embedding models:", style="bold green")
        for spec in specs:
            console.print(f"  - {spec.model}", style="cyan")
        return

    if aliases:
        from conduit.core.model.models.modelstore import ModelStore
        from rich.console import Console

        console = Console()
        aliases_data = ModelStore.aliases()
        console.print(aliases_data)
        return

    if server:
        _show_server_models(server)
        return

    if model:
        from conduit.core.model.models.modelstore import ModelStore
        from conduit.storage.modelspec_repository import ModelSpecRepositoryError
        from rich.console import Console

        console = Console()
        try:
            modelspec = ModelStore.get_model(model)
            modelspec.card
        except ModelSpecRepositoryError as exc:
            console.print(f"[red]Database unavailable: {exc}[/red]")
            raise SystemExit(1)
        except ValueError:
            from rapidfuzz import process
            from rapidfuzz import fuzz
            from collections import namedtuple

            Match = namedtuple("Match", ["title", "score", "rank"])
            models_list = ModelStore.list_models()
            results = process.extract(model, models_list, scorer=fuzz.WRatio, limit=3)
            matches = [
                Match(title=title, score=score, rank=rank + 1)
                for rank, (title, score, _) in enumerate(results)
            ]
            console.print(f"[red]Model '{model}' not found. Did you mean:[/red]")
            for match in matches:
                console.print(f"  {match.rank}. {match.title}")
        return

    if model_type:
        from conduit.core.model.models.modelstore import ModelStore

        modeltypes = ModelStore.list_model_types()
        if model_type not in modeltypes:
            raise click.BadParameter(
                f"Must be one of: {' | '.join(modeltypes)}",
                param_hint="'--type'",
            )
        modelspecs = ModelStore.by_type(model_type)
        for ms in modelspecs:
            click.echo(ms.model)
        return

    if provider:
        from conduit.core.model.models.modelstore import ModelStore

        provider = provider.lower()
        providers_list = ModelStore.list_providers()
        if provider not in providers_list:
            raise click.BadParameter(
                f"Must be one of: {' | '.join(providers_list)}",
                param_hint="'--provider'",
            )
        if provider == "ollama":
            _show_ollama_table()
        else:
            modelspecs = ModelStore.by_provider(provider)
            for ms in modelspecs:
                click.echo(ms.model)
        return

    from conduit.core.model.models.modelstore import ModelStore
    ModelStore.display()


def _show_server_models(server: str) -> None:
    """Live-fetch models for one server; fall back to cache with a staleness warning."""
    import asyncio
    from rich.console import Console
    from conduit.config import settings
    from conduit.core.clients.ollama.server_registry import fetch_with_cache_fallback

    console = Console()
    result = asyncio.run(
        fetch_with_cache_fallback(server, settings.paths["OLLAMA_MODELS_PATH"])
    )

    if result.from_cache:
        cached_str = result.cached_at or "unknown date"
        console.print(
            f"[yellow]Warning: {server} unreachable — showing cached list from {cached_str}[/yellow]"
        )
        console.print(f"[bold]{server}[/bold] models [dim](cached)[/dim]")
    else:
        console.print(f"[bold]{server}[/bold] models [dim](live)[/dim]")

    for m in sorted(result.models):
        console.print(f"  [cyan]{m}[/cyan]")


def _show_ollama_table() -> None:
    """Async-gather all three servers and render a cross-server model table."""
    import asyncio
    from rich.console import Console
    from rich.table import Table
    from conduit.config import settings
    from conduit.core.clients.ollama.server_registry import (
        OLLAMA_SERVERS,
        fetch_all_servers,
    )
    from conduit.core.model.models.modelstore import ModelStore

    console = Console()

    results = asyncio.run(fetch_all_servers(settings.paths["OLLAMA_MODELS_PATH"]))

    # Staleness warnings first
    for r in results:
        if r.from_cache:
            cached_str = r.cached_at or "unknown date"
            console.print(
                f"[yellow]Warning: {r.server} unreachable — cached list from {cached_str}[/yellow]"
            )

    # Build heavy-model set from DB (best-effort)
    try:
        all_specs = ModelStore.get_all_models()
        heavy_set = {spec.model for spec in all_specs if getattr(spec, "heavy", False)}
    except Exception:
        heavy_set = set()

    # Collect all known models and per-server sets
    server_names = list(OLLAMA_SERVERS)
    server_sets: dict[str, set[str]] = {r.server: set(r.models) for r in results}
    all_models = sorted({m for models in server_sets.values() for m in models})

    table = Table(title="Ollama Models")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Heavy", justify="center")
    for name in server_names:
        table.add_column(name.capitalize(), justify="center")

    for m in all_models:
        heavy_marker = "*" if m in heavy_set else ""
        server_markers = ["x" if m in server_sets.get(name, set()) else "" for name in server_names]
        table.add_row(m, heavy_marker, *server_markers)

    console.print(table)
