"""
Gold standard summarizations generation script.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from asyncio import Semaphore
from datasets import Dataset

from conduit.config import settings
from conduit.strategies.summarize.datasets.load_datasets import load_corpus
from conduit.strategies.summarize.datasets.gold_standard import (
    GoldStandardSummary,
    GoldStandardSummaryWithMetadata,
    GoldStandardEntry,
    GoldStandardDatum,
)
from conduit.strategies.summarize.compression import get_target_summary_length
from conduit.core.prompt.prompt_loader import PromptLoader
from headwater_api.classes import EmbeddingsRequest, ChromaBatch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from headwater_client.api.embeddings_async_api import EmbeddingsAsyncAPI

# General configs
MODEL_NAME = "gemini3"
OUTPUT_FILE = settings.paths["DATA_DIR"] / "gold_standard.json"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
PROMPT_LOADER = PromptLoader(PROMPTS_DIR)
PROJECT_NAME = "gold_standard_summarization"
GOLD_STANDARD_PROMPT = PROMPT_LOADER["gold_standard"]
CORPUS_DATASET = load_corpus()
GOLD_STANDARD_DATASET_PATH = (
    settings.paths["DATASETS_DIR"] / "gold_standard_dataset.parquet"
)
EMBEDDINGS_MODEL = "google/embeddinggemma-300m"

# Test configs
DRY_RUN = False
DEBUG_PAYLOAD = False


async def generate_gold_standards(
    docs: list[GoldStandardEntry], dry_run: bool = False
) -> list[GoldStandardDatum]:
    """
    Generate gold standard summaries for a list of documents.
    """
    from headwater_client.client.headwater_client_async import HeadwaterAsyncClient

    from conduit.batch import (
        ConduitBatchAsync,
        GenerationParams,
        ConduitOptions,
        Verbosity,
    )

    params = GenerationParams(
        model=MODEL_NAME,
        output_type="structured_response",
        response_model=GoldStandardSummary,
        temperature=0.0,
    )
    options = ConduitOptions(
        project_name=PROJECT_NAME,
        cache=settings.default_cache(PROJECT_NAME),
        verbosity=Verbosity.PROGRESS,
        debug_payload=DEBUG_PAYLOAD,
    )

    if dry_run:
        for doc in docs:
            prompt_str = GOLD_STANDARD_PROMPT.render(
                input_variables={
                    "text": doc.text,
                    "target_tokens": get_target_summary_length(doc.token_count),
                }
            )
            print(prompt_str)
        return []

    conduit = ConduitBatchAsync(prompt=GOLD_STANDARD_PROMPT)
    input_variables_list = [
        {"text": doc.text, "target_tokens": get_target_summary_length(doc.token_count)}
        for doc in docs
    ]

    summaries_responses = await conduit.run(
        input_variables_list=input_variables_list,
        prompt_strings_list=[],
        params=params,
        options=options,
        max_concurrent=5,
    )
    summaries = [summary.last.parsed for summary in summaries_responses]

    # Generate metadata for summaries using a shared client to optimize network usage
    async with HeadwaterAsyncClient() as client:
        summaries_with_metadata = await generate_summary_metadata(
            summaries, client.embeddings
        )

    gold_standard_data = []
    for entry, summary in zip(docs, summaries_with_metadata):
        gold_standard_data.append(GoldStandardDatum(entry=entry, summary=summary))
    return gold_standard_data


async def generate_summary_metadata(
    summaries: list[GoldStandardSummary],
    embeddings_client: EmbeddingsAsyncAPI,
) -> list[GoldStandardSummaryWithMetadata]:
    """
    Generate metadata for summaries, including token counts.
    """
    summary_lengths = await get_summary_lengths(summaries)
    summary_embeddings = await generate_summary_embeddings(summaries, embeddings_client)
    entity_list_embeddings = await generate_entity_list_embeddings(
        summaries, embeddings_client
    )

    summaries_with_metadata = []
    for summary, length, embedding, entity_embeddings in zip(
        summaries, summary_lengths, summary_embeddings, entity_list_embeddings
    ):
        summaries_with_metadata.append(
            GoldStandardSummaryWithMetadata(
                **summary.model_dump(),
                summary_length=length,
                summary_embeddings=embedding,
                entity_list_embeddings=entity_embeddings,
            )
        )
    return summaries_with_metadata


async def get_summary_lengths(summaries: list[GoldStandardSummary]) -> list[int]:
    """
    Get token counts for each summary using the same client for efficiency.
    """
    from conduit.async_ import ModelAsync

    tokenize = ModelAsync(MODEL_NAME).tokenize
    summary_texts = [summary.summary for summary in summaries]
    token_counts = await asyncio.gather(*(tokenize(text) for text in summary_texts))
    return token_counts


async def generate_summary_embeddings(
    summaries: list[GoldStandardSummary], embeddings_client: EmbeddingsAsyncAPI
) -> list[list[float]]:
    """
    Generate embeddings for the summary using a shared client.
    """
    ids = [str(i) for i in range(len(summaries))]
    documents = [summary.summary for summary in summaries]
    batch = ChromaBatch(ids=ids, documents=documents)
    embedding_request = EmbeddingsRequest(
        model=EMBEDDINGS_MODEL,
        batch=batch,
    )
    embedding_response = await embeddings_client.generate_embeddings(embedding_request)
    return embedding_response.embeddings


async def generate_entity_list_embeddings(
    summaries: list[GoldStandardSummary], embeddings_client: EmbeddingsAsyncAPI
) -> list[list[list[float]]]:
    """
    Generate embeddings for entity lists with a semaphore to prevent network exhaustion.
    """
    # Semaphore limits concurrent HTTP requests to the server
    sem = Semaphore(2)

    async def sem_task(summary):
        async with sem:
            ids = [str(i) for i in range(len(summary.entity_list))]
            batch = ChromaBatch(ids=ids, documents=summary.entity_list)
            req = EmbeddingsRequest(model=EMBEDDINGS_MODEL, batch=batch)
            return await embeddings_client.generate_embeddings(req)

    # Gather all tasks, but the semaphore inside sem_task ensures only 5 run at once
    embedding_responses = await asyncio.gather(*(sem_task(s) for s in summaries))

    return [resp.embeddings for resp in embedding_responses]


async def validate_datum(datum: GoldStandardDatum) -> bool:
    """
    Validate that the summary meets the target token count.
    """
    from conduit.async_ import ModelAsync

    tokenize = ModelAsync(MODEL_NAME).tokenize
    summary_token_count = await tokenize(datum.summary.summary)
    target_token_count = get_target_summary_length(datum.entry.token_count)

    print(
        f"Original text tokens: {datum.entry.token_count}, "
        f"Summary tokens: {summary_token_count}, "
        f"Target tokens: {target_token_count}"
    )
    return summary_token_count <= target_token_count


def save_gold_standard_data(dataset: Dataset, path: Path = GOLD_STANDARD_DATASET_PATH):
    """
    Save the gold standard data to a Parquet file.
    """
    dataset.to_parquet(str(path))
    print(f"Gold standard dataset saved to {path}")


if __name__ == "__main__":

    async def main():
        entries = [GoldStandardEntry(**d) for d in CORPUS_DATASET]
        gold_standard_data = await generate_gold_standards(entries, dry_run=DRY_RUN)

        for datum in gold_standard_data:
            is_valid = await validate_datum(datum)
            print(f"Datum valid: {is_valid}")

        if gold_standard_data:
            gold_standard_dataset = Dataset.from_list(
                [datum.model_dump() for datum in gold_standard_data]
            )
            save_gold_standard_data(gold_standard_dataset)

    asyncio.run(main())
