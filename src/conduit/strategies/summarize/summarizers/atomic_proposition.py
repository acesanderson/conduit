from __future__ import annotations

import asyncio
import logging
from typing import override, Any
from pydantic import BaseModel, ConfigDict
from conduit.core.workflow.step import step, add_metadata
from conduit.strategies.summarize.strategy import SummarizationStrategy, _TextInput
from conduit.strategies.summarize.summarizers.chunker import Chunker

logger = logging.getLogger(__name__)

extract_propositions_prompt = """
Decompose the following text into a list of atomic, self-contained statements.

Rules:
- Each statement must be independent and self-contained.
- Replace ALL pronouns (he, she, it, they, the company, etc.) with the full named entity.
- Each statement expresses exactly one fact or claim.
- One statement per line.
- Do not include commentary, headers, or numbering — just the statements.

<text>
{{ text }}
</text>
""".strip()

deduplicate_propositions_prompt = """
Below is a list of atomic propositions extracted from a document.
Some propositions may be near-duplicates or express the same fact in different words.

Your task:
1. Remove exact duplicates and near-duplicates.
2. Keep the most informative version of each unique fact.
3. Return the canonical, deduplicated list — one proposition per line.
4. Do not add commentary, headers, or numbering — just the propositions.

<propositions>
{{ propositions }}
</propositions>
""".strip()


class AtomicPropositionSummarizer(SummarizationStrategy):
    """
    Propositional decomposition summarizer.

    Workflow:
    1. Chunk the text.
    2. For each chunk in parallel, decompose into independent atomic statements
       with all pronouns dereferenced to named entities.
    3. Collect all propositions across chunks.
    4. Run a final LLM deduplication pass to remove near-duplicates and return
       the canonical set.
    5. Return the canonical propositions as a numbered list (one per line).

    Config params:
        model:             LLM (default: gpt3)
        concurrency_limit: max parallel chunk calls (default: 5)
        max_tokens:        max tokens per call
        temperature:       sampling temperature
    """

    class Config(BaseModel):
        model_config = ConfigDict(extra="ignore")
        model: str = "gpt3"
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
        logger.info(f"AtomicPropositionSummarizer: {total_chunks} chunks")

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
            rendered = Prompt(extract_propositions_prompt).render({"text": chunk})
            async with semaphore:
                response = await model_instance.query(
                    query_input=rendered,
                    params=generation_params,
                    options=options,
                )
            assert isinstance(response, GenerationResponse)
            return response

        extraction_responses: list[GenerationResponse] = await asyncio.gather(
            *[extract_chunk(chunk) for chunk in chunks]
        )

        all_propositions: list[str] = []
        for response in extraction_responses:
            lines = [ln.strip() for ln in str(response.content).splitlines() if ln.strip()]
            all_propositions.extend(lines)

        total_before_dedupe = len(all_propositions)
        logger.info(f"Extracted {total_before_dedupe} propositions before dedupe")

        # LLM deduplication pass
        dedupe_rendered = Prompt(deduplicate_propositions_prompt).render({
            "propositions": "\n".join(all_propositions),
        })
        dedupe_response = await model_instance.query(
            query_input=dedupe_rendered,
            params=generation_params,
            options=options,
        )
        assert isinstance(dedupe_response, GenerationResponse)

        deduped_lines = [
            ln.strip()
            for ln in str(dedupe_response.content).splitlines()
            if ln.strip()
        ]
        total_after_dedupe = len(deduped_lines)
        logger.info(f"After dedupe: {total_after_dedupe} propositions")

        add_metadata("num_chunks", total_chunks)
        add_metadata("total_propositions_before_dedupe", total_before_dedupe)
        add_metadata("total_propositions_after_dedupe", total_after_dedupe)

        return "\n".join(f"{i + 1}. {prop}" for i, prop in enumerate(deduped_lines))


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
        result = await AtomicPropositionSummarizer()(_TextInput(_sample), {"model": "gpt3"})
        print(result)

    asyncio.run(_main())
