from conduit.extensions.summarize.strategy import SummarizationStrategy
from conduit.extensions.summarize.summarizers.one_shot import OneShotSummarizer
from conduit.extensions.summarize.summarizers.chunker import Chunker
from conduit.domain.result.response import GenerationResponse
from conduit.core.workflow.workflow import step, get_param, add_metadata
from typing import override
import logging

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
    @step
    @override
    async def __call__(self, text: str, **kwargs) -> str:
        logger.info("Starting MapReduceSummarizer")
        # Grab llm params
        model = get_param("model", default="gpt-3.5")
        prompt = get_param("prompt", default=chunk_summarization_prompt)

        # Chunk that shit
        chunker = Chunker()
        chunks = await chunker(text)
        logger.info(f"Text chunked into {len(chunks)} chunks.")

        # MAP: Run the conduit
        from conduit.core.model.model_async import ModelAsync
        from conduit.core.prompt.prompt import Prompt
        from conduit.domain.request.generation_params import GenerationParams
        from conduit.domain.config.conduit_options import ConduitOptions
        import asyncio

        generation_params = GenerationParams(
            model=model,
            max_tokens=get_param("max_tokens", default=None),
            temperature=get_param("temperature", default=None),
            top_p=get_param("top_p", default=None),
        )
        options = ConduitOptions(
            project_name=get_param("project_name", default="conduit")
        )
        model_instance = ModelAsync(model=model)

        coroutines = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"Creating coroutine for chunk {i + 1}/{len(chunks)}")
            summarization_prompt = Prompt(prompt).render(
                input_variables={
                    "chunk": chunk,
                    "chunk_index": str(i + 1),
                    "total_chunks": str(len(chunks)),
                    "target_tokens": get_param("target_tokens", default=150),
                    "text": text,
                }
            )
            coroutine = model_instance.query(
                query_input=summarization_prompt,
                params=generation_params,
                options=options,
            )
            coroutines.append(coroutine)

        logger.debug("Awaiting all chunk summarization coroutines")
        responses: list[GenerationResponse] = await asyncio.gather(*coroutines)
        # REDUCE: Join structured summaries
        response_strings = [str(r.content) for r in responses]
        combined = "\n\n".join(response_strings)
        one_shot_summarizer = OneShotSummarizer()
        logger.debug("Starting final summarization step")
        final_summary = await one_shot_summarizer(combined)

        # Collect trace metadata
        total_input_tokens = sum(r.metadata.input_tokens for r in responses)
        total_output_tokens = sum(r.metadata.output_tokens for r in responses)
        add_metadata("input_tokens", total_input_tokens)
        add_metadata("output_tokens", total_output_tokens)

        return final_summary


async def main():
    from conduit.core.workflow.workflow import ConduitHarness
    from pathlib import Path

    ESSAYS_DIR = Path(__file__).parent.parent / "essays"
    text = (ESSAYS_DIR / "conduit.txt").read_text()
    config = {
        "model": "gpt3",
        "prompt": "Summarize the key speakers in this text:\n\n{{text}}",
    }
    workflow = MapReduceSummarizer()
    harness = ConduitHarness(config=config)
    response = await harness.run(workflow, text=text)
    print(response)
    return response, harness


if __name__ == "__main__":
    import asyncio

    response, harness = asyncio.run(main())
