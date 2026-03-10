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
        pass  # placeholder — implemented in Task 6
        return

    if aliases:
        pass  # placeholder — implemented in Task 5
        return

    if model:
        from conduit.core.model.models.modelstore import ModelStore
        from rich.console import Console

        try:
            modelspec = ModelStore.get_model(model)
            modelspec.card
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
            console = Console()
            console.print(f"[red]Model '{model}' not found. Did you mean:[/red]")
            for match in matches:
                console.print(f"  {match.rank}. {match.title}")
        return

    if model_type:
        pass  # placeholder — implemented in Task 4
        return

    if provider:
        pass  # placeholder — implemented in Task 5
        return

    from conduit.core.model.models.modelstore import ModelStore
    ModelStore.display()
