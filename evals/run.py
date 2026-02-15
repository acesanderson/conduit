from conduit.core.workflow.harness import ConduitHarness
from conduit.strategies.summarize.summarizers.recursive import RecursiveSummarizer
from conduit.strategies.summarize.datasets.load_datasets import load_golden_dataset
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


async def run_config(config: dict[str, str | float], text: str) -> None:
    # Instantiate the strategy
    summarizer = RecursiveSummarizer()

    # Create harness and run
    harness = ConduitHarness(config=config)

    # Test with a simple text input
    test_text = "This is a test document. " * 100
    result = await harness.run(summarizer, text=text)

    print(f"Summary: {result}")

    trace = harness.trace
    trace_str = json.dumps(trace, indent=2)
    print(f"Trace:\n{trace_str}")


if __name__ == "__main__":
    EXAMPLE = DATASET[0]
    EXAMPLE_TEXT = EXAMPLE["entry"]["text"]

    print("RecursiveSummarizer Schema:")
    print(json.dumps(RecursiveSummarizer().schema, indent=2))
    print("\nConfig Dictionary:")
    print(json.dumps(CONFIG_DICT, indent=2))
    asyncio.run(run_config(CONFIG_DICT, text=EXAMPLE_TEXT))
