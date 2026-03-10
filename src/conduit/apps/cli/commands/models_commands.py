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
    pass
