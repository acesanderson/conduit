from __future__ import annotations

import logging
from typing import override, Any
from pydantic import BaseModel, ConfigDict
from conduit.core.workflow.step import step, add_metadata
from conduit.strategies.summarize.strategy import SummarizationStrategy, _TextInput
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer

logger = logging.getLogger(__name__)

identify_entities_prompt = """
You are analyzing a document and its current summary to identify missing information.

Source document:
<document>
{{ document }}
</document>

Current summary:
<summary>
{{ summary }}
</summary>

Identify 5-10 specific entities (people, organizations, dates, numbers, events, locations)
that are present in the source document but missing or underrepresented in the current summary.
Return only a numbered list of the missing entities, one per line. No explanation.
""".strip()

fuse_entities_prompt = """
You are densifying a summary by incorporating missing entities.

Current summary:
<summary>
{{ summary }}
</summary>

Missing entities to incorporate:
<missing_entities>
{{ missing_entities }}
</missing_entities>

Rewrite the summary to fuse ALL of the missing entities into the text.
Rules:
- Do not increase the total word count.
- Preserve all information already in the summary.
- Every missing entity must appear in the new summary.
- Return only the rewritten summary, no explanation.
""".strip()


class ChainOfDensitySummarizer(SummarizationStrategy):
    """
    Chain-of-Density summarization: iterative densification via two-step loops.

    Workflow:
    1. Generate an initial verbose summary via OneShotSummarizer.
    2. For N iterations:
       a. Identify 5-10 entities present in the source but missing from the draft.
       b. Fuse those entities into the draft without increasing word count.
    3. Return the final densified summary.

    Config params:
        iterations:   number of densification loops (default: 3)
        model:        LLM to use (default: gpt3)
        max_tokens:   max tokens for generation
        temperature:  sampling temperature
    """

    class Config(BaseModel):
        model_config = ConfigDict(extra="ignore")
        model: str = "gpt3"
        iterations: int = 3
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

        current_summary = await OneShotSummarizer()(_TextInput(text), config)

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

        total_input_tokens = 0
        total_output_tokens = 0

        for i in range(1, cfg.iterations + 1):
            logger.info(f"ChainOfDensity iteration {i}/{cfg.iterations}")

            # 2a: identify missing entities
            identify_rendered = Prompt(identify_entities_prompt).render({
                "document": text,
                "summary": current_summary,
            })
            identify_response = await model_instance.query(
                query_input=identify_rendered,
                params=generation_params,
                options=options,
            )
            assert isinstance(identify_response, GenerationResponse)
            missing_entities = str(identify_response.content).strip()
            total_input_tokens += identify_response.metadata.input_tokens
            total_output_tokens += identify_response.metadata.output_tokens

            # 2b: fuse missing entities into draft without increasing word count
            fuse_rendered = Prompt(fuse_entities_prompt).render({
                "summary": current_summary,
                "missing_entities": missing_entities,
            })
            fuse_response = await model_instance.query(
                query_input=fuse_rendered,
                params=generation_params,
                options=options,
            )
            assert isinstance(fuse_response, GenerationResponse)
            current_summary = str(fuse_response.content).strip()
            total_input_tokens += fuse_response.metadata.input_tokens
            total_output_tokens += fuse_response.metadata.output_tokens

        add_metadata("iterations_run", cfg.iterations)
        add_metadata("input_tokens_total", total_input_tokens)
        add_metadata("output_tokens_total", total_output_tokens)

        return current_summary


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
        result = await ChainOfDensitySummarizer()(_TextInput(_sample), {"model": "gpt3"})
        print(result)

    asyncio.run(_main())
