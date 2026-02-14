from conduit.config import settings
from conduit.strategies.summarize.datasets.load_corpus import load_corpus
from conduit.strategies.summarize.datasets.gold_standard import (
    GoldStandardSummary,
    GoldStandardEntry,
    GoldStandardDatum,
)
from conduit.strategies.summarize.compression import get_target_summary_length
from conduit.core.prompt.prompt_loader import PromptLoader
from conduit.config import settings
from pathlib import Path
import asyncio

MODEL_NAME = "gemini3"
OUTPUT_FILE = settings.paths["DATA_DIR"] / "gold_standard.json"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
PROMPT_LOADER = PromptLoader(PROMPTS_DIR)
PROJECT_NAME = "gold_standard_summarization"
GOLD_STANDARD_PROMPT = PROMPT_LOADER["gold_standard"]
CORPUS_DATASET = load_corpus()
DRY_RUN = False  # Set to True to just print prompts without making API calls
DEBUG_PAYLOAD = True

"""
Dataset schema:
"category": category,
"source_id": source_id_fn(item, i),
"text": text,
"token_count": count,
"""


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


if __name__ == "__main__":
    # Grab just one GoldStandardEntry from the dataset
    # for d in CORPUS_DATASET:
    #     entry = GoldStandardEntry(**d)
    #     print(entry.token_count, get_target_summary_length(entry.token_count), sep="\t")
    # sample_doc = GoldStandardEntry(**CORPUS_DATASET[0])
    # sample_summary = asyncio.run(generate_single_gold_standard(sample_doc))
    # print(sample_doc.token_count)
    # input_vars = asyncio.run(
    #     generate_gold_standards([GoldStandardEntry(**d) for d in CORPUS_DATASET])
    # )
    entries = [GoldStandardEntry(**d) for d in list(CORPUS_DATASET)[:3]]
    gold_standard_data = asyncio.run(generate_gold_standards(entries, dry_run=DRY_RUN))
    for datum in gold_standard_data:
        is_valid = asyncio.run(validate_datum(datum))
        print(f"Datum valid: {is_valid}")
    # Save the gold standard data to a JSON file
