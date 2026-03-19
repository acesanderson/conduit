from __future__ import annotations

import logging
from typing import override, Any
from pydantic import BaseModel, ConfigDict
from conduit.strategies.summarize.strategy import SummarizationStrategy, _TextInput
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer
from conduit.strategies.summarize.summarizers.chunker import Chunker
from conduit.domain.result.response import GenerationResponse
from conduit.core.workflow.step import step, add_metadata

logger = logging.getLogger(__name__)

chunk_summarization_prompt = """
Summarize the following content. This is section {{ chunk_index }} of {{ total_chunks}}.
Preserve key facts, entities, and relationships.
Target ~{{ target_tokens }} tokens.

<chunk>
{{ chunk }}
</chunk>
""".strip()


class MapReduceSummarizer(SummarizationStrategy):
    class Config(BaseModel):
        model_config = ConfigDict(extra="ignore")
        model: str = "gpt-oss:latest"
        prompt: str = chunk_summarization_prompt
        chunk_size: int = 12000
        overlap: int = 500
        concurrency_limit: int = 5
        target_tokens: int = 150
        max_tokens: int | None = None
        temperature: float | None = None
        top_p: float | None = None
        project_name: str = "conduit"

    config_model = Config

    @step
    @override
    async def __call__(self, input: Any, config: dict) -> str:
        cfg = self.Config(**config)
        text = input.data
        logger.info("Starting MapReduceSummarizer")

        chunker = Chunker()
        chunks = await chunker(text, config)
        logger.info(f"Text chunked into {len(chunks)} chunks.")

        from conduit.core.model.model_async import ModelAsync
        from conduit.core.prompt.prompt import Prompt
        from conduit.domain.request.generation_params import GenerationParams
        from conduit.domain.config.conduit_options import ConduitOptions
        from conduit.utils.progress.verbosity import Verbosity
        import asyncio

        generation_params = GenerationParams(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        options = ConduitOptions(
            project_name=cfg.project_name,
            verbosity=Verbosity.SILENT,
            debug_payload=True,
        )
        model_instance = ModelAsync(model=cfg.model)

        coroutines = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"Creating coroutine for chunk {i + 1}/{len(chunks)}")
            summarization_prompt = Prompt(cfg.prompt).render(
                input_variables={
                    "chunk": chunk,
                    "chunk_index": str(i + 1),
                    "total_chunks": str(len(chunks)),
                    "target_tokens": str(cfg.target_tokens),
                }
            )
            coroutine = model_instance.query(
                query_input=summarization_prompt,
                params=generation_params,
                options=options,
            )
            coroutines.append(coroutine)

        semaphore = asyncio.Semaphore(cfg.concurrency_limit)
        logger.debug("Awaiting all chunk summarization coroutines")
        async with semaphore:
            responses: list[GenerationResponse] = await asyncio.gather(*coroutines)

        response_strings = [str(r.content) for r in responses]
        combined = "\n\n".join(response_strings)

        logger.debug("Starting final summarization step")
        final_summary = await OneShotSummarizer()(_TextInput(combined), config)

        total_input_tokens = sum(r.metadata.input_tokens for r in responses)
        total_output_tokens = sum(r.metadata.output_tokens for r in responses)
        add_metadata("input_tokens", total_input_tokens)
        add_metadata("output_tokens", total_output_tokens)

        return final_summary
