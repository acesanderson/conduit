"""
```python
from conduit.core.workflow.harness import ConduitHarness
from conduit.strategies.summarize.summarizers.rolling_refine import RollingRefineSummarizer

config = {
    "model": "gpt-oss:latest",
    "chunk_size": 8000,
    "overlap": 500,
    "RollingRefineSummarizer.refine_prompt": "...",  # optional override
}

harness = ConduitHarness(config=config)
result = await harness.run(RollingRefineSummarizer(), text=long_document)
```
"""

from __future__ import annotations

import logging
from typing import override
from conduit.core.workflow.step import step, get_param, add_metadata
from conduit.strategies.summarize.strategy import SummarizationStrategy
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer
from conduit.strategies.summarize.summarizers.chunker import Chunker

logger = logging.getLogger(__name__)

refine_prompt_default = """
You are refining an evolving summary with new information.

Current summary:
<current_summary>
{{ current_summary }}
</current_summary>

New content (section {{ chunk_index }} of {{ total_chunks }}):
<new_chunk>
{{ new_chunk }}
</new_chunk>

Update the summary to incorporate relevant new information from the new content.
Preserve all existing important facts. Do not repeat information already captured.
Return only the updated summary.
""".strip()


class RollingRefineSummarizer(SummarizationStrategy):
    """
    Sequential refine strategy for narrative-preserving summarization.

    Workflow:
    1. Chunk the text linearly.
    2. Summarize chunk 1 via OneShotSummarizer.
    3. For each subsequent chunk, call the LLM with (current_summary + new_chunk)
       using a refine prompt, producing an updated summary.
    4. Return the final evolved summary.

    This preserves narrative continuity better than map-reduce because each
    refinement step has the full accumulated context of all prior chunks.
    """

    @step
    @override
    async def __call__(self, text: str, *args, **kwargs) -> str:
        from conduit.core.model.model_async import ModelAsync
        from conduit.core.prompt.prompt import Prompt
        from conduit.domain.request.generation_params import GenerationParams
        from conduit.domain.config.conduit_options import ConduitOptions
        from conduit.domain.result.response import GenerationResponse
        from conduit.utils.progress.verbosity import Verbosity

        # 1. Resolve params
        model = get_param("model", default="gpt3")
        refine_prompt = get_param("refine_prompt", default=refine_prompt_default)

        # 2. Chunk
        chunker = Chunker()
        chunks = await chunker(text)
        total_chunks = len(chunks)
        logger.info(f"RollingRefineSummarizer: {total_chunks} chunks")

        add_metadata("num_chunks", total_chunks)

        # 3. Base case: single chunk — delegate directly to OneShotSummarizer
        if total_chunks == 1:
            logger.info("Single chunk — delegating to OneShotSummarizer")
            return await OneShotSummarizer()(text=chunks[0])

        # 4. Summarize first chunk to seed the rolling summary
        logger.info("Seeding summary from chunk 1")
        current_summary = await OneShotSummarizer()(text=chunks[0])

        # 5. Refine iteratively over remaining chunks
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

        total_input_tokens = 0
        total_output_tokens = 0

        for i, chunk in enumerate(chunks[1:], start=2):
            logger.info(f"Refining with chunk {i}/{total_chunks}")
            rendered = Prompt(refine_prompt).render(
                {
                    "current_summary": current_summary,
                    "new_chunk": chunk,
                    "chunk_index": str(i),
                    "total_chunks": str(total_chunks),
                }
            )
            response = await model_instance.query(
                query_input=rendered,
                params=generation_params,
                options=options,
            )
            assert isinstance(response, GenerationResponse)
            current_summary = str(response.content)
            total_input_tokens += response.metadata.input_tokens
            total_output_tokens += response.metadata.output_tokens

        add_metadata("refine_input_tokens", total_input_tokens)
        add_metadata("refine_output_tokens", total_output_tokens)

        return current_summary


if __name__ == "__main__":
    import asyncio
    from conduit.core.workflow.harness import ConduitHarness

    async def main():
        summarizer = RollingRefineSummarizer()

        config = {
            "model": "gpt-oss:latest",
            "chunk_size": 4000,
            "overlap": 200,
        }

        harness = ConduitHarness(config=config)
        test_text = "This is a test sentence with some content. " * 200
        result = await harness.run(summarizer, text=test_text)

        print(f"Summary: {result}")
        print(f"Trace: {harness.trace}")

    asyncio.run(main())
