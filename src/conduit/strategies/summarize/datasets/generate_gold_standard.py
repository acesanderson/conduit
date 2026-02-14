"""
Gold standard summarizations, for this exercise, is these params:
- temp: 0.0 (we want the best possible summary, not a variety)
- model: Gemini 3 (the best we have access to at the moment)
- summarization is one-shot, with the gold standard prompt

We are generating the following for each doc:
- summary
- token count of the summary (to validate that it meets the target compression ratio)
- entities and key info in the summary (to validate that it captures the important info from the original text)
"""

from conduit.config import settings
from conduit.strategies.summarize.datasets.load_datasets import load_corpus
from conduit.strategies.summarize.datasets.gold_standard import (
    GoldStandardSummary,
    GoldStandardEntry,
    GoldStandardDatum,
)
from conduit.strategies.summarize.compression import get_target_summary_length
from conduit.core.prompt.prompt_loader import PromptLoader
from conduit.config import settings
from pathlib import Path
from datasets import Dataset

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

# Test configs
DRY_RUN = False  # Set to True to just print prompts without making API calls
DEBUG_PAYLOAD = (
    False  # Set to True to include the full API payload in the logs for debugginTrueg
)


async def generate_single_gold_standard(doc: GoldStandardEntry) -> GoldStandardDatum:
    """
    For testing.
    """
    from conduit.async_ import ConduitAsync, GenerationParams, ConduitOptions, Verbosity

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
    conduit = ConduitAsync(prompt=GOLD_STANDARD_PROMPT)
    conversation = await conduit.run(
        input_variables={
            "text": doc.text,
            "target_tokens": get_target_summary_length(doc.token_count),
        },
        params=params,
        options=options,
    )
    summary = conversation.last.parsed
    return GoldStandardDatum(entry=doc, summary=summary)


async def generate_gold_standards(
    docs: list[GoldStandardEntry], dry_run: bool = False
) -> list[GoldStandardDatum]:
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
        # Just print the rendered prompts
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
    summaries = await conduit.run(
        input_variables_list=input_variables_list,
        prompt_strings_list=[],
        params=params,
        options=options,
        max_concurrent=5,  # Adjust based on your resources and rate limits
    )
    data = [
        GoldStandardDatum(entry=doc, summary=summary.last.parsed)
        for doc, summary in zip(docs, summaries)
    ]
    return data


async def validate_datum(
    datum: GoldStandardDatum,
) -> bool:
    """
    Validate that the summary meets the target token count.
    """
    from conduit.async_ import ModelAsync

    tokenize = ModelAsync(MODEL_NAME).tokenize
    summary_token_count = await tokenize(datum.summary.summary)
    target_token_count = get_target_summary_length(datum.entry.token_count)
    print(
        f"Original text token count: {datum.entry.token_count}, "
        f"Summary token count: {summary_token_count}, "
        f"Target token count: {target_token_count}",
        sep="\n",
    )
    if summary_token_count > target_token_count:
        return False
    return True


def save_gold_standard_data(dataset: Dataset, path: Path = GOLD_STANDARD_DATASET_PATH):
    """
    Save the gold standard data to a Parquet file.
    """
    dataset.to_parquet(str(path))
    print(f"Gold standard dataset saved to {path}")


if __name__ == "__main__":
    import asyncio

    entries = [GoldStandardEntry(**d) for d in CORPUS_DATASET]
    gold_standard_data = asyncio.run(generate_gold_standards(entries, dry_run=DRY_RUN))
    for datum in gold_standard_data:
        is_valid = asyncio.run(validate_datum(datum))
        print(f"Datum valid: {is_valid}")
    gold_standard_dataset = Dataset.from_list(
        [datum.model_dump() for datum in gold_standard_data]
    )
    save_gold_standard_data(gold_standard_dataset)
