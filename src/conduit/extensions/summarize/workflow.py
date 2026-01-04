from __future__ import annotations
from conduit.core.workflow.workflow import ConduitHarness
from conduit.extensions.summarize.summarizers.one_shot import OneShotSummarizer
from conduit.extensions.summarize.summarizers.map_reduce import MapReduceSummarizer
from conduit.extensions.summarize.summarizers.hierarchal import HierarchalSummarizer

from typing import TYPE_CHECKING
from pathlib import Path
import asyncio

if TYPE_CHECKING:
    from conduit.extensions.summarize.strategy import SummarizationStrategy
    from conduit.core.workflow.workflow import ConduitHarness

ESSAYS_DIR = Path(__file__).parent / "essays"
text = (ESSAYS_DIR / "essay_5333_words.txt").read_text()
config = {
    "model": "gpt3",
    "prompt": "Summarize the key speakers in this text:\n\n{{text}}",
}


class Summarizer:
    def __init__(self, strategy: SummarizationStrategy):
        self.strategy: SummarizationStrategy = strategy

    async def __call__(self, text: str) -> str:
        return await self.strategy(text)


def test_one_shot_summarizer(config: dict[str, str]) -> tuple[str, ConduitHarness]:
    workflow = OneShotSummarizer()
    harness = ConduitHarness(config=config)

    async def main():
        response = await harness.run(workflow, text=text)
        print(response)
        return response, harness

    response, harness = asyncio.run(main())
    return response, harness


if __name__ == "__main__":
    response, harness = test_one_shot_summarizer(config)
