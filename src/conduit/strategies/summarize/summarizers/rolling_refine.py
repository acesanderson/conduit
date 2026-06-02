from __future__ import annotations

import logging
from typing import override, Any
from pydantic import BaseModel, ConfigDict
from conduit.core.workflow.step import step, add_metadata
from conduit.strategies.summarize.strategy import SummarizationStrategy, _TextInput
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer
from conduit.strategies.summarize.summarizers.chunker import Chunker

logger = logging.getLogger(__name__)

# Bridge text between an opaque guideline and the rolling summary, used only
# when the caller supplied input.guideline. Refinement does content
# accumulation; this final pass owns the user-facing output shape.
_FORMAT_PASS_INSTRUCTION = """

Using the instructions above, restructure the following summary content
into the specified final form. Preserve all factual content, entities,
dates, and attribution. Restructure only. Do not summarize further or
introduce new information.

<summary>
""".lstrip()


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

    class Config(BaseModel):
        model_config = ConfigDict(extra="ignore")
        model: str = "gpt-oss:latest"
        refine_prompt: str = refine_prompt_default
        chunk_size: int = 12000
        overlap: int = 500
        max_summary_chars: int = 24000  # ~6k tokens; prevents context overflow on small-context models
        max_tokens: int | None = None
        temperature: float | None = None
        top_p: float | None = None
        project_name: str = "conduit"
        use_remote: bool = False
        host_alias: str = "headwater"
        use_cache: bool = True

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
        logger.info(f"RollingRefineSummarizer: {total_chunks} chunks")

        add_metadata("num_chunks", total_chunks)

        guideline = getattr(input, "guideline", None)

        if total_chunks == 1:
            # Single-chunk path: this OneShot call IS the final output. Pass guideline through.
            logger.info("Single chunk, delegating to OneShotSummarizer")
            return await OneShotSummarizer()(_TextInput(chunks[0], guideline=guideline), config)

        # Multi-chunk seed: intermediate accumulation step, no guideline.
        # Guideline fires only on the post-loop format pass (see end of method).
        logger.info("Seeding summary from chunk 1")
        current_summary = await OneShotSummarizer()(_TextInput(chunks[0]), config)

        generation_params = GenerationParams(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        options = ConduitOptions(
            project_name=cfg.project_name,
            verbosity=Verbosity.SILENT,
            use_remote=cfg.use_remote,
            use_cache=cfg.use_cache,
        )
        if cfg.use_remote:
            from conduit.core.model.model_remote import RemoteModelAsync
            model_instance = RemoteModelAsync(model=cfg.model, host_alias=cfg.host_alias)
        else:
            model_instance = ModelAsync(model=cfg.model)

        total_input_tokens = 0
        total_output_tokens = 0

        for i, chunk in enumerate(chunks[1:], start=2):
            logger.info(f"Refining with chunk {i}/{total_chunks}")
            # Truncate from the front to stay within the model's context window.
            # Keeps the most recent accumulated context when the summary grows large.
            if len(current_summary) > cfg.max_summary_chars:
                current_summary = current_summary[-cfg.max_summary_chars:]
            rendered = Prompt(cfg.refine_prompt).render(
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

        if guideline:
            logger.info("Running final format pass")
            format_prompt = (
                guideline
                + "\n\n"
                + _FORMAT_PASS_INSTRUCTION
                + current_summary
                + "\n</summary>"
            )
            response = await model_instance.query(
                query_input=format_prompt,
                params=generation_params,
                options=options,
            )
            assert isinstance(response, GenerationResponse)
            current_summary = str(response.content)
            add_metadata("format_pass_input_tokens", response.metadata.input_tokens)
            add_metadata("format_pass_output_tokens", response.metadata.output_tokens)
            add_metadata("guideline_applied", True)

        return current_summary
