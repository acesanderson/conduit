from load_datasets import load_golden_dataset
from evals import run_eval
from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer
from conduit.strategies.summarize.summarizers.recursive import RecursiveSummarizer
from conduit.strategies.summarize.summarizers.rolling_refine import (
    RollingRefineSummarizer,
)
from conduit.strategies.summarize.summarizers.extractive_pre_filter import (
    ExtractivePreFilterSummarizer,
)
from conduit.strategies.summarize.summarizers.hierarchical_tree import (
    HierarchicalTreeSummarizer,
)

strategies = [
    OneShotSummarizer(),
    RecursiveSummarizer(),
    RollingRefineSummarizer(),
    ExtractivePreFilterSummarizer(),
    HierarchicalTreeSummarizer(),
]


ds = load_golden_dataset()

# OneShot run
"""
Config schema:
    - model: str (default: "gpt3")
    - prompt: str (default: "Summarize the following text:\n\n{{text}}")
    - max_tokens: int (optional) - max tokens for the summary
    - temperature: float (optional) - temperature for generation
    - top_p: float (optional) - top_p for generation
"""
one_shot_config = {
    "model": "gpt-oss:latest",
}

run_output = run_eval(ds[0], config={}, strategy=strategies[0])
