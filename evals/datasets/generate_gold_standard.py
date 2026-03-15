"""
Gold standard summarizations generation script.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from asyncio import Semaphore
import pandas as pd
from conduit.config import settings
from conduit.strategies.summarize.datasets.load_datasets import load_corpus
from conduit.core.eval.models import GoldSummary, GoldDatum, Document
from conduit.strategies.summarize.compression import get_target_summary_length
from conduit.core.prompt.prompt_loader import PromptLoader
from headwater_api.classes import EmbeddingsRequest, ChromaBatch

from typing import TYPE_CHECKING, Any

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
    docs: list[Document], dry_run: bool = False
) -> list[GoldDatum]:
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
                    "text": doc.content,
                    "target_tokens": get_target_summary_length(
                        doc.metadata["token_count"]
                    ),
                }
            )
            print(prompt_str)
        return []

    conduit = ConduitBatchAsync(prompt=GOLD_STANDARD_PROMPT)
    input_variables_list = [
        {
            "text": doc.content,
            "target_tokens": get_target_summary_length(doc.metadata["token_count"]),
        }
        for doc in docs
    ]

    summaries_responses = await conduit.run(
        input_variables_list=input_variables_list,
        prompt_strings_list=[],
        params=params,
        options=options,
        max_concurrent=5,
    )
    summaries = [str(summary.content) for summary in summaries_responses]

    # Generate metadata for summaries using a shared client to optimize network usage
    async with HeadwaterAsyncClient() as client:
        gold_summaries = await generate_summary_metadata(summaries, client.embeddings)

    gold_standard_data = []
    # Zip docs and gold_summaries together to create a list of GoldDatums
    for doc, gold_summary in zip(docs, gold_summaries):
        gold_standard_data.append(
            GoldDatum(
                document=doc,
                gold_summary=gold_summary,
            )
        )
    return gold_standard_data


async def generate_summary_metadata(
    summaries: list[str],
    embeddings_client: EmbeddingsAsyncAPI,
) -> list[GoldSummary]:
    """
    Generate metadata for summaries, including token counts.
    """
    summary_lengths = await get_summary_lengths(summaries)
    summary_embeddings = await generate_summary_embeddings(summaries, embeddings_client)

    gold_summaries = []
    # Combine the summaries with a metadata dict containing: summary_length, summary_embeddings, in one GoldSummary object
    for summary, length, embedding in zip(
        summaries, summary_lengths, summary_embeddings
    ):
        gold_summaries.append(
            GoldSummary(
                summary=summary,
                metadata={
                    "summary_length": length,
                    "summary_embedding": embedding,
                },
            )
        )
    return gold_summaries


async def get_summary_lengths(summaries: list[str]) -> list[int]:
    """
    Get token counts for each summary using the same client for efficiency.
    """
    from conduit.async_ import ModelAsync

    tokenize = ModelAsync(MODEL_NAME).tokenize
    token_counts = await asyncio.gather(*(tokenize(text) for text in summaries))
    return token_counts


async def generate_summary_embeddings(
    summaries: list[str], embeddings_client: EmbeddingsAsyncAPI
) -> list[list[float]]:
    """
    Generate embeddings for the summary using a shared client.
    """
    ids = [str(i) for i in range(len(summaries))]
    batch = ChromaBatch(ids=ids, documents=summaries)
    embedding_request = EmbeddingsRequest(
        model=EMBEDDINGS_MODEL,
        batch=batch,
    )
    embedding_response = await embeddings_client.generate_embeddings(embedding_request)
    return embedding_response.embeddings


def save_gold_standard_data(
    dataset: list[GoldDatum], path: Path = GOLD_STANDARD_DATASET_PATH
):
    """
    Save the gold standard data to a Parquet file.
    """
    df = pd.DataFrame([datum.model_dump() for datum in dataset])
    df.to_parquet(path, index=False)
    print(f"Gold standard dataset saved to {path}")


if __name__ == "__main__":

    async def main():
        docs = load_corpus()
        gold_standard_data = await generate_gold_standards(docs, dry_run=DRY_RUN)
        return gold_standard_data

    data = asyncio.run(main())
    save_gold_standard_data(data)
