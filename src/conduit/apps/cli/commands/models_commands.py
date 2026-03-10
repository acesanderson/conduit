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
def models_command(
    model: str | None,
    model_type: str | None,
    provider: str | None,
    aliases: bool,
    embeddings: bool,
    rerankers: bool,
) -> None:
    """List and inspect available models."""
    if rerankers:
        pass  # placeholder — implemented in Task 7
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

    if model:
        from conduit.core.model.models.modelstore import ModelStore

        try:
            modelspec = ModelStore.get_model(model)
            modelspec.card
        except ValueError:
            from rapidfuzz import process
            from rapidfuzz import fuzz
            from collections import namedtuple
            from rich.console import Console

            Match = namedtuple("Match", ["title", "score", "rank"])
            models_list = ModelStore.list_models()
            results = process.extract(model, models_list, scorer=fuzz.WRatio, limit=3)
            matches = [
                Match(title=title, score=score, rank=rank + 1)
                for rank, (title, score, _) in enumerate(results)
            ]
            console = Console()
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
        modelspecs = ModelStore.by_provider(provider)
        for ms in modelspecs:
            click.echo(ms.model)
        return

    from conduit.core.model.models.modelstore import ModelStore
    ModelStore.display()
