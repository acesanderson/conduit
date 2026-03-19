"""
```python
from conduit.core.workflow.harness import ConduitHarness
from conduit.strategies.summarize.summarizers.map_dedupe_reduce import MapDedupeReduceSummarizer

config = {
    "model": "gpt-oss:latest",
    "chunk_size": 8000,
    "overlap": 500,
    "dedupe_similarity_threshold": 0.85,
}

harness = ConduitHarness(config=config)
result = await harness.run(MapDedupeReduceSummarizer(), text=document)
```
"""

from __future__ import annotations

import logging
from typing import override, Any
from conduit.core.workflow.step import step, get_param, add_metadata
from conduit.core.workflow.context import context
from conduit.strategies.summarize.strategy import SummarizationStrategy, _TextInput
from conduit.strategies.summarize.summarizers.chunker import Chunker
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer
from conduit.core.model.model_async import ModelAsync
from conduit.core.prompt.prompt import Prompt
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.domain.result.response import GenerationResponse
from conduit.utils.progress.verbosity import Verbosity
import asyncio

logger = logging.getLogger(__name__)


dedupe_prompt = """
Identify and remove duplicate or near-duplicate statements from the following text.
Keep only unique, non-redundant information while preserving the key facts.

Original text:
<text>
{{ text }}
</text>

Return only the deduplicated content.
""".strip()


class MapDedupeReduceSummarizer(SummarizationStrategy):
    """
    Map-Dedupe-Reduce summarization strategy.

    Workflow:
    1. MAP: Summarize each chunk to reduce its size.
    2. DEDUPE: Remove duplicate/redundant information across summaries.
    3. REDUCE: Combine all deduplicated summaries into a final summary.

    This approach reduces redundancy at multiple levels, potentially improving
    information density and coherence.
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
            logger.info(f"MapDedupeReduceSummarizer: {total_chunks} chunks")

            if total_chunks == 0:
                return ""

            # MAP: Summarize each chunk
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

            # Use one-shot summarizer for the map phase
            map_summarizer = OneShotSummarizer()

            # Create coroutines for parallel chunk summarization
            coroutines = []
            for i, chunk in enumerate(chunks):
                logger.debug(f"Summarizing chunk {i + 1}/{total_chunks}")
                coroutine = map_summarizer(_TextInput(chunk), config)
                coroutines.append(coroutine)

            semaphore = asyncio.Semaphore(concurrency_limit)
            async with semaphore:
                chunk_summaries: list[str] = await asyncio.gather(*coroutines)

            # Join summaries for deduplication
            combined_before_dedupe = "\n\n".join(chunk_summaries)

            # DEDUPE: Remove redundancy across chunks
            dedupe_rendered = Prompt(dedupe_prompt).render({"text": combined_before_dedupe})
            dedupe_response = await model_instance.query(
                query_input=dedupe_rendered,
                params=generation_params,
                options=options,
            )
            assert isinstance(dedupe_response, GenerationResponse)

            deduped_text = str(dedupe_response.content)
            logger.info(f"Reduced from {len(chunk_summaries)} summaries to deduplicated content")

            # REDUCE: Final summary of deduplicated content
            one_shot = OneShotSummarizer()
            final_summary = await one_shot(_TextInput(deduped_text), config)

            # Collect metadata
            total_input_tokens = sum(r.metadata.input_tokens for r in chunk_summaries if hasattr(r, 'metadata'))
            total_input_tokens += dedupe_response.metadata.input_tokens
            total_output_tokens = sum(r.metadata.output_tokens for r in chunk_summaries if hasattr(r, 'metadata'))
            total_output_tokens += dedupe_response.metadata.output_tokens

            add_metadata("input_tokens", total_input_tokens)
            add_metadata("output_tokens", total_output_tokens)
            add_metadata("num_chunks", total_chunks)

            return final_summary
        finally:
            context.config.reset(token_conf)
            context.use_defaults.reset(token_defaults)
