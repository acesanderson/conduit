"""
```python
from conduit.core.workflow.harness import ConduitHarness
from conduit.strategies.summarize.summarizers.atomic_proposition import AtomicPropositionSummarizer

config = {
    "model": "gpt-oss:latest",
    "chunk_size": 8000,
    "overlap": 500,
}

harness = ConduitHarness(config=config)
result = await harness.run(AtomicPropositionSummarizer(), text=document)
```
"""

from __future__ import annotations

import re
import logging
from typing import override, Any
from conduit.core.workflow.step import step, get_param, add_metadata
from conduit.core.workflow.context import context
from conduit.strategies.summarize.strategy import SummarizationStrategy, _TextInput
from conduit.strategies.summarize.summarizers.chunker import Chunker
from conduit.core.model.model_async import ModelAsync
from conduit.core.prompt.prompt import Prompt
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.domain.result.response import GenerationResponse
from conduit.utils.progress.verbosity import Verbosity
import asyncio

logger = logging.getLogger(__name__)


atomic_extraction_prompt = """
Extract all atomic propositions from the following text. An atomic proposition is a
simple, indivisible statement that can be true or false.

Requirements:
1. Each proposition should be a single factual claim
2. Keep propositions concise but self-contained
3. Preserve specific entities, numbers, and relationships
4. One proposition per line, formatted as: "Subject - Predicate - Object"
5. If a sentence contains multiple claims, split them

<chunk>
{{ chunk }}
</chunk>
""".strip()


atomic_combine_prompt = """
Combine the following atomic propositions into a coherent summary. Maintain all
factual information while eliminating redundancies and preserving key relationships.

Propositions:
{{ propositions }}

Provide a concise paragraph summary that incorporates all propositions.
""".strip()


class AtomicPropositionSummarizer(SummarizationStrategy):
    """
    Extract atomic propositions from text, then recombine into a summary.

    Workflow:
    1. Chunk the document.
    2. For each chunk, extract atomic propositions (one per line).
    3. Combine all propositions into a single list.
    4. Deduplicate and recombine propositions into a final summary.

    This approach produces structured summaries that preserve factual granularity.
    """

    @step
    @override
    async def __call__(self, input: Any, config: dict) -> str:
        token_conf = context.config.set(config)
        token_defaults = context.use_defaults.set(True)
        try:
            text = input.data
            model = get_param("model", default="gpt3")
            concurrency_limit = get_param("concurrency_limit", default=5)

            # Chunk the text
            chunker = Chunker()
            chunks = await chunker(text)
            total_chunks = len(chunks)
            logger.info(f"AtomicPropositionSummarizer: {total_chunks} chunks")

            # Extract propositions from each chunk
            generation_params = GenerationParams(
                model=model,
                max_tokens=get_param("max_tokens", default=None),
                temperature=get_param("temperature", default=None),
                top_p=get_param("top_p", default=None),
            )
            options = ConduitOptions(
                project_name=get_param("project_name", default="conduit"),
                verbosity=Verbosity.SILENT,
                debug_payload=True,
            )
            model_instance = ModelAsync(model=model)

            coroutines = []
            for i, chunk in enumerate(chunks):
                logger.debug(f"Extracting propositions from chunk {i + 1}/{total_chunks}")
                rendered = Prompt(atomic_extraction_prompt).render({"chunk": chunk})
                coroutine = model_instance.query(
                    query_input=rendered,
                    params=generation_params,
                    options=options,
                )
                coroutines.append(coroutine)

            semaphore = asyncio.Semaphore(concurrency_limit)
            async with semaphore:
                responses: list[GenerationResponse] = await asyncio.gather(*coroutines)

            # Collect all propositions, filtering empty lines
            all_propositions: list[str] = []
            for response in responses:
                content = str(response.content)
                # Split by newlines and filter
                lines = [line.strip() for line in content.split("\n") if line.strip()]
                # Filter out lines that seem like commentary rather than propositions
                for line in lines:
                    # Skip common non-proposition patterns
                    if not any(line.lower().startswith(prefix) for prefix in ["sure", "here", "as an", "i will"]):
                        all_propositions.append(line)

            total_propositions = len(all_propositions)
            logger.info(f"Extracted {total_propositions} propositions from {total_chunks} chunks")

            # Combine propositions into a summary
            propositions_text = "\n".join(all_propositions)
            combine_rendered = Prompt(atomic_combine_prompt).render({"propositions": propositions_text})

            combine_response = await model_instance.query(
                query_input=combine_rendered,
                params=generation_params,
                options=options,
            )
            assert isinstance(combine_response, GenerationResponse)

            # Collect metadata
            total_input_tokens = sum(r.metadata.input_tokens for r in responses)
            total_input_tokens += combine_response.metadata.input_tokens
            total_output_tokens = sum(r.metadata.output_tokens for r in responses)
            total_output_tokens += combine_response.metadata.output_tokens

            add_metadata("input_tokens", total_input_tokens)
            add_metadata("output_tokens", total_output_tokens)
            add_metadata("num_chunks", total_chunks)
            add_metadata("num_propositions", total_propositions)

            return str(combine_response.content)
        finally:
            context.config.reset(token_conf)
            context.use_defaults.reset(token_defaults)
