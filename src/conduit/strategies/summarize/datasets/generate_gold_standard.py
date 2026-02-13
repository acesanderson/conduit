from conduit.config import settings
from conduit.strategies.summarize.datasets.corpus import build_composite_dataset
from conduit.strategies.summarize.datasets.gold_standard import (
    GoldStandardSummary,
    GoldStandardEntry,
    GoldStandardDatum,
)
from conduit.core.prompt.prompt_loader import PromptLoader
from conduit.async_ import ConduitAsync, GenerationParams, ConduitOptions, Verbosity
from conduit.config import settings
from pathlib import Path
import asyncio

MODEL_NAME = "gemini3"
OUTPUT_FILE = settings.paths["DATA_DIR"] / "gold_standard.json"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
PROMPT_LOADER = PromptLoader(PROMPTS_DIR)
PROJECT_NAME = "gold_standard_summarization"
GOLD_STANDARD_PROMPT = PROMPT_LOADER["gold_standard"]
DATASET = asyncio.run(build_composite_dataset())

"""
Dataset schema:
"category": category,
"source_id": source_id_fn(item, i),
"text": text,
"token_count": count,
"""


async def generate_gold_standard(doc: GoldStandardEntry) -> GoldStandardDatum:
    params = GenerationParams(
        model=MODEL_NAME,
        output_type="structured_response",
        response_model=GoldStandardSummary,
    )
    options = ConduitOptions(
        project_name=PROJECT_NAME,
        cache=settings.default_cache(PROJECT_NAME),
        verbosity=Verbosity.PROGRESS,
    )
    conduit = ConduitAsync(prompt=GOLD_STANDARD_PROMPT)
    conversation = await conduit.run(
        input_variables={"text": doc.text}, params=params, options=options
    )
    summary = conversation.last.parsed
    return GoldStandardDatum(entry=doc, summary=summary)


if __name__ == "__main__":
    # Grab just one GoldStandardEntry from the dataset
    sample_doc = GoldStandardEntry(**DATASET[0])
    sample_summary = asyncio.run(generate_gold_standard(sample_doc))
