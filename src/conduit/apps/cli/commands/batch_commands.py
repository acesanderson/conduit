from __future__ import annotations

import sys
import logging
from pathlib import Path

import click

from conduit.apps.cli.handlers.batch_handlers import BatchHandlers

logger = logging.getLogger(__name__)


@click.command("batch")
@click.option("-m", "--model", type=str, default=None, help="Model for all prompts.")
@click.option("-t", "--temperature", type=float, default=None, help="Temperature (0.0-1.0).")
@click.option(
    "-f",
    "--file",
    "prompt_file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="File of prompts, one per line.",
)
@click.option("-n", "--max-concurrent", type=int, default=None, help="Max parallel requests.")
@click.option("-a", "--append", type=str, default=None, help="Suffix appended to every prompt.")
@click.option("-r", "--raw", is_flag=True, default=False, help="Plain text output, separated by ---.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON array.")
@click.argument("prompts", nargs=-1)
@click.pass_context
def batch_command(
    ctx: click.Context,
    model: str | None,
    temperature: float | None,
    prompt_file: str | None,
    max_concurrent: int | None,
    append: str | None,
    raw: bool,
    as_json: bool,
    prompts: tuple[str, ...],
) -> None:
    """Run multiple prompts in parallel against an LLM.

    Prompts can be passed as arguments, read from --file (one per line),
    or piped via stdin. All sources are merged.

    Examples:

        conduit batch "What is X?" "What is Y?" -m sonar-pro

        conduit batch -f prompts.txt -m gpt-4o -n 5

        cat prompts.txt | conduit batch -m claude
    """
    if raw and as_json:
        raise click.UsageError("--raw and --json are mutually exclusive.")

    printer = ctx.obj["printer"]
    preferred_model = ctx.obj.get("preferred_model")
    resolved_model = model or preferred_model or "gpt-4o"

    # Collect prompts from all sources
    collected: list[str] = []

    # 1. File
    if prompt_file:
        text = Path(prompt_file).read_text(encoding="utf-8")
        collected.extend(line for line in text.splitlines() if line.strip())

    # 2. Inline args
    collected.extend(prompts)

    # 3. Stdin (only when no args and no file, and stdin is not a TTY)
    if not collected and not sys.stdin.isatty():
        stdin_text = sys.stdin.read()
        collected.extend(line for line in stdin_text.splitlines() if line.strip())

    if not collected:
        raise click.UsageError(
            "No prompts provided. Pass prompts as arguments, use --file, or pipe via stdin."
        )

    # Apply --append suffix to every prompt
    if append:
        collected = [f"{p}\n{append}" for p in collected]

    BatchHandlers.handle_batch(
        prompts=collected,
        model=resolved_model,
        temperature=temperature,
        max_concurrent=max_concurrent,
        raw=raw,
        as_json=as_json,
        printer=printer,
    )
