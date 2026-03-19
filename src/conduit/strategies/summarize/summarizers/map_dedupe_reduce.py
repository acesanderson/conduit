from __future__ import annotations

import asyncio
import logging
from typing import override, Any
from pydantic import BaseModel, ConfigDict
from conduit.core.workflow.step import step, add_metadata
from conduit.strategies.summarize.strategy import SummarizationStrategy, _TextInput
from conduit.strategies.summarize.summarizers.chunker import Chunker
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer

logger = logging.getLogger(__name__)

default_extraction_prompt = """
Extract the key facts, entities, and decisions from the following text.
Return a concise bulleted list. Each bullet should be a single, self-contained item.
Do not include commentary or headers.

<text>
{{ text }}
</text>
""".strip()

deduplicate_prompt = """
Below are extracted facts and entities from multiple sections of a document.
Some items may be duplicated or near-identical across sections.

Your task:
1. Merge exact and near-duplicate entries.
2. Normalize formatting across all items.
3. Return the deduplicated, normalized list — one item per line.
Do not add commentary or headers.

<items>
{{ items }}
</items>
""".strip()


class MapDedupeReduceSummarizer(SummarizationStrategy):
    """
    Map-reduce with an explicit deduplication pass.

    Workflow:
    1. Chunk the text.
    2. Map (parallel): run an extraction prompt on every chunk to extract
       key facts, entities, and decisions.
    3. Collect all extracted lists into one combined string.
    4. Dedupe pass: prompt the model to merge duplicates, normalize format,
       and remove near-identical entries.
    5. Final reduce: pass the deduplicated list to OneShotSummarizer.

    Config params:
        model:              LLM (default: gpt3)
        extraction_prompt:  override the default map-phase extraction prompt
        concurrency_limit:  max parallel map calls (default: 5)
        max_tokens:         max tokens per call
        temperature:        sampling temperature
    """

    class Config(BaseModel):
        model_config = ConfigDict(extra="ignore")
        model: str = "gpt3"
        extraction_prompt: str = default_extraction_prompt
        concurrency_limit: int = 5
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

        from conduit.core.model.model_async import ModelAsync
        from conduit.core.prompt.prompt import Prompt
        from conduit.domain.request.generation_params import GenerationParams
        from conduit.domain.config.conduit_options import ConduitOptions
        from conduit.domain.result.response import GenerationResponse
        from conduit.utils.progress.verbosity import Verbosity

        chunker = Chunker()
        chunks = await chunker(text, config)
        total_chunks = len(chunks)
        logger.info(f"MapDedupeReduceSummarizer: {total_chunks} chunks")

        if total_chunks == 0:
            return ""

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
        semaphore = asyncio.Semaphore(cfg.concurrency_limit)

        async def extract_chunk(chunk: str) -> GenerationResponse:
            rendered = Prompt(cfg.extraction_prompt).render({"text": chunk})
            async with semaphore:
                response = await model_instance.query(
                    query_input=rendered,
                    params=generation_params,
                    options=options,
                )
            assert isinstance(response, GenerationResponse)
            return response

        map_responses: list[GenerationResponse] = await asyncio.gather(
            *[extract_chunk(chunk) for chunk in chunks]
        )

        map_input_tokens = sum(r.metadata.input_tokens for r in map_responses)
        map_output_tokens = sum(r.metadata.output_tokens for r in map_responses)

        combined = "\n\n".join(str(r.content) for r in map_responses)

        # Dedupe pass
        dedupe_rendered = Prompt(deduplicate_prompt).render({"items": combined})
        dedupe_response = await model_instance.query(
            query_input=dedupe_rendered,
            params=generation_params,
            options=options,
        )
        assert isinstance(dedupe_response, GenerationResponse)
        deduped_text = str(dedupe_response.content)

        add_metadata("num_chunks", total_chunks)
        add_metadata("map_input_tokens", map_input_tokens)
        add_metadata("map_output_tokens", map_output_tokens)
        add_metadata("dedupe_input_tokens", dedupe_response.metadata.input_tokens)
        add_metadata("dedupe_output_tokens", dedupe_response.metadata.output_tokens)

        return await OneShotSummarizer()(_TextInput(deduped_text), config)


if __name__ == "__main__":
    import asyncio

    _sample = (
        "The Apollo 11 mission launched on July 16, 1969, carrying astronauts Neil Armstrong, "
        "Buzz Aldrin, and Michael Collins. On July 20, Armstrong and Aldrin landed on the Moon "
        "in the Sea of Tranquility while Collins orbited above. Armstrong became the first human "
        "to walk on the Moon at 02:56 UTC, followed by Aldrin. They collected 21.5 kg of lunar "
        "material and deployed several scientific instruments. The mission returned to Earth on "
        "July 24, splashing down in the Pacific Ocean. It was the fifth crewed mission of NASA's "
        "Apollo program and fulfilled President Kennedy's 1961 goal of landing on the Moon before "
        "the end of the decade."
    )

    async def _main() -> None:
        result = await MapDedupeReduceSummarizer()(_TextInput(_sample), {"model": "gpt3"})
        print(result)

    asyncio.run(_main())
