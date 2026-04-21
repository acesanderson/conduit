from __future__ import annotations
import json
import click
import logging
from typing import TYPE_CHECKING

from conduit.batch import ConduitBatchSync
from conduit.config import settings

if TYPE_CHECKING:
    from conduit.apps.cli.utils.printer import Printer

logger = logging.getLogger(__name__)


class BatchHandlers:
    @staticmethod
    def handle_batch(
        prompts: list[str],
        model: str,
        temperature: float | None,
        max_concurrent: int | None,
        raw: bool,
        as_json: bool,
        citations: bool,
        printer: Printer,
    ) -> None:
        """Run prompts in parallel and display results."""
        param_kwargs: dict[str, object] = {}
        if temperature is not None:
            param_kwargs["temperature"] = temperature
        if citations:
            param_kwargs["client_params"] = {"return_citations": True}

        from conduit.core.model.models.modelstore import ModelStore
        is_ollama = ModelStore.identify_provider(model) == "ollama"
        batch = ConduitBatchSync.create(
            model=model,
            verbosity=settings.default_verbosity,
            use_remote=is_ollama,
            **param_kwargs,
        )

        conversations = batch.run(
            prompt_strings_list=prompts,
            max_concurrent=max_concurrent,
        )

        def _extract_citations(conv: object) -> list[dict]:
            last = getattr(conv, "last", None)
            meta: dict = getattr(last, "metadata", {}) or {}
            return meta.get("citations", [])

        results = [
            {
                "index": i,
                "prompt": p,
                "response": str(conv.content),
                "citations": _extract_citations(conv) if citations else [],
            }
            for i, (p, conv) in enumerate(zip(prompts, conversations))
        ]

        if as_json:
            click.echo(json.dumps(results, ensure_ascii=False, indent=2))
            return

        if raw:
            for i, item in enumerate(results):
                click.echo(item["response"])
                if citations and item["citations"]:
                    click.echo(json.dumps(item["citations"]))
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
            if citations and item["citations"]:
                printer.print_citations(item["citations"])
