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
        model:              LLM (default: gpt3) — used for any phase whose
                            phase-specific model override is unset.
        chunk_model:        override model for the map (chunk extraction) phase
        chunk_host_alias:   override host for the map phase
        dedupe_model:       override model for the dedupe phase
        dedupe_host_alias:  override host for the dedupe phase
        reduce_model:       override model for the final OneShot reduce phase
        reduce_host_alias:  override host for the final reduce phase
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
        use_remote: bool = False
        host_alias: str = "headwater"
        use_cache: bool = True
        chunk_model: str | None = None
        chunk_host_alias: str | None = None
        dedupe_model: str | None = None
        dedupe_host_alias: str | None = None
        reduce_model: str | None = None
        reduce_host_alias: str | None = None

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
        logger.info(f"{self.__class__.__name__}: {total_chunks} chunks")

        if total_chunks == 0:
            return ""

        chunk_model_name = cfg.chunk_model or cfg.model
        chunk_host = cfg.chunk_host_alias or cfg.host_alias
        dedupe_model_name = cfg.dedupe_model or cfg.model
        dedupe_host = cfg.dedupe_host_alias or cfg.host_alias
        reduce_model_name = cfg.reduce_model or cfg.model
        reduce_host = cfg.reduce_host_alias or cfg.host_alias

        options = ConduitOptions(
            project_name=cfg.project_name,
            verbosity=Verbosity.SILENT,
            use_remote=cfg.use_remote,
            use_cache=cfg.use_cache,
        )

        def _build_model(model_name: str, host: str) -> Any:
            if cfg.use_remote:
                from conduit.core.model.model_remote import RemoteModelAsync
                return RemoteModelAsync(model=model_name, host_alias=host)
            return ModelAsync(model=model_name)

        def _params(model_name: str) -> GenerationParams:
            return GenerationParams(
                model=model_name,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )

        chunk_model_instance = _build_model(chunk_model_name, chunk_host)
        dedupe_model_instance = _build_model(dedupe_model_name, dedupe_host)
        chunk_params = _params(chunk_model_name)
        dedupe_params = _params(dedupe_model_name)

        semaphore = asyncio.Semaphore(cfg.concurrency_limit)

        async def extract_chunk(chunk: str) -> GenerationResponse:
            rendered = Prompt(cfg.extraction_prompt).render({"text": chunk})
            async with semaphore:
                response = await chunk_model_instance.query(
                    query_input=rendered,
                    params=chunk_params,
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
        dedupe_response = await dedupe_model_instance.query(
            query_input=dedupe_rendered,
            params=dedupe_params,
            options=options,
        )
        assert isinstance(dedupe_response, GenerationResponse)
        deduped_text = str(dedupe_response.content)

        add_metadata("num_chunks", total_chunks)
        add_metadata("map_input_tokens", map_input_tokens)
        add_metadata("map_output_tokens", map_output_tokens)
        add_metadata("dedupe_input_tokens", dedupe_response.metadata.input_tokens)
        add_metadata("dedupe_output_tokens", dedupe_response.metadata.output_tokens)
        add_metadata("chunk_model", chunk_model_name)
        add_metadata("dedupe_model", dedupe_model_name)
        add_metadata("reduce_model", reduce_model_name)

        reduce_config = {**config, "model": reduce_model_name, "host_alias": reduce_host}
        return await OneShotSummarizer()(_TextInput(deduped_text), reduce_config)


class MapDedupeReduceHybridModelSummarizer(MapDedupeReduceSummarizer):
    """
    Identical workflow to MapDedupeReduceSummarizer, but expected to be configured
    with distinct chunk_model / dedupe_model / reduce_model (e.g. cheap fast model
    for chunks, higher-quality model for dedupe and reduce). The separate class
    name lets eval results distinguish hybrid runs from single-model runs.
    """
    pass


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
