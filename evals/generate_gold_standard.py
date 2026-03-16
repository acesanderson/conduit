"""
Gold standard summarizations generation script.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
import pandas as pd
from conduit.config import settings
from load_datasets import load_corpus

sys.path.insert(0, str(Path(__file__).parent.parent))
from evals import RunInput

from conduit.strategies.summarize.compression import get_target_summary_length
from conduit.core.prompt.prompt_loader import PromptLoader

# General configs
MODEL_NAME = "gemini3"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
PROMPT_LOADER = PromptLoader(PROMPTS_DIR)
PROJECT_NAME = "gold_standard_summarization"
GOLD_STANDARD_PROMPT = PROMPT_LOADER["gold_standard"]
GOLD_STANDARD_DATASET_PATH = (
    settings.paths["DATASETS_DIR"] / "gold_standard_dataset.parquet"
)

# Test configs
DRY_RUN = False
DEBUG_PAYLOAD = False


async def generate_gold_standards(
    docs: list[RunInput], dry_run: bool = False
) -> list[RunInput]:
    """
    Generate gold standard summaries for a list of documents.
    """
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
                    "text": doc.data,
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
            "text": doc.data,
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
    summaries = [str(r.content) for r in summaries_responses]

    return [
        doc.model_copy(update={"reference": summary})
        for doc, summary in zip(docs, summaries)
    ]


def save_gold_standard_data(
    docs: list[RunInput], path: Path = GOLD_STANDARD_DATASET_PATH
):
    """
    Save the corpus with populated reference summaries to a Parquet file.
    """
    records = [
        {
            "source_id": doc.source_id,
            "content": doc.data,
            "category": doc.metadata["category"],
            "token_count": doc.metadata["token_count"],
            "reference": doc.reference,
        }
        for doc in docs
    ]
    pd.DataFrame(records).to_parquet(path, index=False)
    print(f"Gold standard dataset saved to {path}")


if __name__ == "__main__":

    async def main():
        docs = load_corpus()
        gold_standard_data = await generate_gold_standards(docs, dry_run=DRY_RUN)
        return gold_standard_data

    data = asyncio.run(main())
    save_gold_standard_data(data)
