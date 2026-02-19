from conduit.core.workflow.harness import ConduitHarness
from conduit.strategies.summarize.summarizers.recursive import RecursiveSummarizer
from conduit.strategies.summarize.datasets.load_datasets import load_golden_dataset
from conduit.strategies.summarize.datasets.gold_standard import GeneratedSummary
import json
from pathlib import Path
import asyncio

# Construct configs
CONFIG_PATH = Path("config.json")
CONFIG_DICT = json.loads(CONFIG_PATH.read_text())
ONE_SHOT_PROMPT = Path("one_shot_prompt.jinja2").read_text()
MAP_REDUCE_PROMPT = Path("map_reduce_prompt.jinja2").read_text()
CONFIG_DICT["MapReduceSummarizer.prompt"] = MAP_REDUCE_PROMPT
CONFIG_DICT["OneShotSummarizer.prompt"] = ONE_SHOT_PROMPT

# Grab our dataset
DATASET = load_golden_dataset()
EXAMPLE = DATASET[0]
EXAMPLE_TEXT = EXAMPLE["entry"]["text"]


async def run_config(config: dict[str, str | float], text: str) -> GeneratedSummary:
    # Instantiate the strategy
    summarizer = RecursiveSummarizer()

    # Create harness and run
    harness = ConduitHarness(config=config)

    # Test with a simple text input
    result = await harness.run(summarizer, text=text)

    trace = harness.trace
    generated_summary = GeneratedSummary(
        summary=result,
        token_count=len(result.split()),
        config_dict=config,
        trace=trace,
    )
    return generated_summary


if __name__ == "__main__":
    summary = asyncio.run(run_config(CONFIG_DICT, text=EXAMPLE_TEXT))
    print(summary.model_dump_json(indent=4))
