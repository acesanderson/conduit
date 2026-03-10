from __future__ import annotations
import json
import click
import logging
from typing import TYPE_CHECKING

from conduit.batch import ConduitBatchSync

if TYPE_CHECKING:
    from conduit.apps.cli.utils.printer import Printer

logger = logging.getLogger(__name__)


class BatchHandlers:
    @staticmethod
    def handle_batch(
        prompts: list[str],
        model: str,
        temperature: float | None,
        local: bool,
        citations: bool,
        max_concurrent: int | None,
        raw: bool,
        as_json: bool,
        printer: Printer,
    ) -> None:
        """Run prompts in parallel and display results."""
        from conduit.config import settings

        param_kwargs: dict = {}
        if temperature is not None:
            param_kwargs["temperature"] = temperature

        batch = ConduitBatchSync.create(
            model=model,
            verbosity=settings.default_verbosity,
            **param_kwargs,
        )

        conversations = batch.run(
            prompt_strings_list=prompts,
            max_concurrent=max_concurrent,
        )

        results = [
            {"index": i, "prompt": p, "response": conv.content}
            for i, (p, conv) in enumerate(zip(prompts, conversations))
        ]

        if as_json:
            click.echo(json.dumps(results, ensure_ascii=False, indent=2))
            return

        if raw:
            for i, item in enumerate(results):
                click.echo(item["response"])
                if i < len(results) - 1:
                    click.echo("---")
            return

        # Pretty mode
        total = len(results)
        for item in results:
            idx = item["index"] + 1
            truncated = item["prompt"][:50].replace("\n", " ")
            if len(item["prompt"]) > 50:
                truncated += "..."
            header = f"[{idx}/{total}] {truncated}"
            printer.print_pretty(header, style="bold cyan")
            printer.print_markdown(item["response"])
