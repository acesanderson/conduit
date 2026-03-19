"""
RAG-based Summarization Strategy

This module provides a summarization strategy that uses Retrieval-Augmented Generation
to create summaries of long-form text. It chunks the input, then for each chunk:
1. Retrieves relevant passages from the chunk itself
2. Generates a summary using the retrieved passages
3. Combines all chunk summaries into a final summary
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from conduit.core.workflow.context import context
from conduit.core.workflow.step import add_metadata, get_param, step
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.result.response import GenerationResponse
from conduit.strategies.rag.strategy import RAGStrategy
from conduit.strategies.summarize.strategy import SummarizationStrategy, _TextInput
from conduit.strategies.summarize.summarizers.chunker import Chunker
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer

from conduit.core.model.model_async import ModelAsync
from conduit.core.prompt.prompt import Prompt

logger = logging.getLogger(__name__)

chunk_rag_prompt = """
You are summarizing a section of a document using RAG (Retrieval-Augmented Generation).

Here is the content to summarize (section {chunk_index} of {total_chunks}):

<chunk>
{chunk}
</chunk>

Instructions:
1. Identify the most important information in this chunk
2. Generate a concise summary that preserves key facts, entities, and relationships
3. Target approximately {target_tokens} tokens

Summary:
""".strip()

combine_prompt_default = """
You are combining chunk summaries into a final summary.

Here are the summaries of each chunk:

{combined_summaries}

Please merge these summaries into a cohesive final summary.
Preserve all important information and eliminate redundancies.
""".strip()


class RAGSummarizer(SummarizationStrategy):
    """
    RAG-based summarization strategy.

    Workflow:
    1. Chunk the input text using semantic token-based chunking
    2. For each chunk, use a RAG approach to generate a summary:
       - The chunk itself serves as the retrieval corpus
       - An LLM generates a summary based on the chunk content
    3. Combine all chunk summaries into a final summary using a final summarizer

    This approach preserves narrative structure better than naive chunking
    because each summary is generated with full context of its chunk.
    """

    @step
    async def __call__(self, input: Any, config: dict) -> str:
        """
        Execute the RAG summarization workflow.

        Args:
            input: Input object with .data attribute containing the text to summarize
            config: Configuration dictionary with parameters like model, chunk_size, etc.

        Returns:
            A final concatenated summary string
        """
        token_conf = context.config.set(config)
        token_defaults = context.use_defaults.set(True)
        try:
            text = input.data
            logger.info("Starting RAGSummarizer")

            # Resolve configuration parameters
            model = get_param("model", default="gpt3")
            chunk_prompt = get_param("chunk_prompt", default=chunk_rag_prompt)
            combine_prompt = get_param("combine_prompt", default=combine_prompt_default)
            concurrency_limit = get_param("concurrency_limit", default=5)
            target_tokens = get_param("target_tokens", default=150)

            # Chunk the text
            chunker = Chunker()
            chunks = await chunker(text)
            total_chunks = len(chunks)
            logger.info(f"Text chunked into {total_chunks} chunks")

            add_metadata("num_chunks", total_chunks)
            add_metadata("original_text_chars", len(text))

            if total_chunks == 0:
                return ""

            # If only one chunk, skip the RAG process and use OneShotSummarizer
            if total_chunks == 1:
                logger.info("Single chunk — delegating to OneShotSummarizer")
                return await OneShotSummarizer()(_TextInput(chunks[0]), config)

            # Setup model instance and generation parameters
            generation_params = GenerationParams(
                model=model,
                max_tokens=get_param("max_tokens", default=None),
                temperature=get_param("temperature", default=None),
                top_p=get_param("top_p", default=None),
            )
            options = ConduitOptions(
                project_name=get_param("project_name", default="conduit"),
                verbosity=get_param("verbosity", default="SILENT"),
                debug_payload=get_param("debug_payload", default=True),
            )
            model_instance = ModelAsync(model=model)

            # Generate summaries for each chunk
            coroutines = []
            for i, chunk in enumerate(chunks):
                logger.debug(f"Creating RAG summary coroutine for chunk {i + 1}/{total_chunks}")
                prompt_text = Prompt(chunk_prompt).render(
                    input_variables={
                        "chunk": chunk,
                        "chunk_index": str(i + 1),
                        "total_chunks": str(total_chunks),
                        "target_tokens": target_tokens,
                    }
                )
                coroutine = model_instance.query(
                    query_input=prompt_text,
                    params=generation_params,
                    options=options,
                )
                coroutines.append(coroutine)

            # Execute all chunk summaries concurrently with concurrency limit
            semaphore = asyncio.Semaphore(concurrency_limit)
            async with semaphore:
                responses: list[GenerationResponse] = await asyncio.gather(*coroutines)

            # Extract summary strings from responses
            chunk_summaries = [str(r.content) for r in responses]

            # Track token usage for chunk summarization
            total_input_tokens = sum(r.metadata.input_tokens for r in responses)
            total_output_tokens = sum(r.metadata.output_tokens for r in responses)
            add_metadata("chunk_summary_input_tokens", total_input_tokens)
            add_metadata("chunk_summary_output_tokens", total_output_tokens)

            # Combine chunk summaries into final summary
            logger.info(f"Combining {total_chunks} chunk summaries")
            combined_text = "\n\n".join(chunk_summaries)
            final_prompt = Prompt(combine_prompt).render(
                input_variables={"combined_summaries": combined_text}
            )

            final_response = await model_instance.query(
                query_input=final_prompt,
                params=generation_params,
                options=options,
            )
            final_summary = str(final_response.content)

            # Track final combine token usage
            add_metadata("combine_input_tokens", final_response.metadata.input_tokens)
            add_metadata("combine_output_tokens", final_response.metadata.output_tokens)
            add_metadata(
                "total_input_tokens",
                total_input_tokens + final_response.metadata.input_tokens,
            )
            add_metadata(
                "total_output_tokens",
                total_output_tokens + final_response.metadata.output_tokens,
            )

            return final_summary

        finally:
            context.config.reset(token_conf)
            context.use_defaults.reset(token_defaults)
