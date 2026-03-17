import asyncio
from load_datasets import load_golden_dataset
from evals import run_eval, generate_runs
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

# Recursive run
"""
    config = {
        "model": "gpt-oss:latest",
        "effective_context_window_ratio": 0.6,
        "OneShotSummarizer.prompt": "Summarize concisely: {{ text }}",
        "MapReduceSummarizer.prompt": "Summarize chunk {{ chunk_index }}/{{ total_chunks }}: {{ chunk }}",
        "chunk_size": 8000,
        "overlap": 500,
    }
"""
recursive_config = {
    "model": "gpt-oss:latest",
    "effective_context_window_ratio": 0.6,
    "OneShotSummarizer.prompt": "Summarize concisely: {{ text }}",
    "MapReduceSummarizer.prompt": "Summarize chunk {{ chunk_index }}/{{ total_chunks }}: {{ chunk }}",
    "chunk_size": 8000,
    "overlap": 500,
}

# RollingRefine run
"""
Config schema:
    - model: str (default: "gpt3")
    - refine_prompt: str (optional) - prompt template for each refinement step
    - chunk_size: int (optional) - tokens per chunk
    - overlap: int (optional) - overlap tokens between chunks
    - max_tokens: int (optional)
    - temperature: float (optional)
    - top_p: float (optional)
"""
rolling_refine_config = {
    "model": "gpt-oss:latest",
    "chunk_size": 8000,
    "overlap": 500,
}

# ExtractivePreFilter run
"""
Config schema:
    - model: str (default: "gpt3")
    - keep_ratio: float (default: 0.3) - fraction of chunks to retain after embedding filter
    - embedding_model: str (default: "sentence-transformers/all-MiniLM-L6-v2")
    - chunk_size: int (optional)
    - overlap: int (optional)
    - max_tokens: int (optional)
    - temperature: float (optional)
    - top_p: float (optional)
"""
extractive_pre_filter_config = {
    "model": "gpt-oss:latest",
    "keep_ratio": 0.3,
    "chunk_size": 8000,
    "overlap": 500,
}

# HierarchicalTree run
"""
Config schema:
    - model: str (default: "gpt3")
    - group_size: int (default: 4) - number of summaries merged per tree node per level
    - chunk_size: int (optional)
    - overlap: int (optional)
    - max_tokens: int (optional)
    - temperature: float (optional)
    - top_p: float (optional)
"""
hierarchical_tree_config = {
    "model": "gpt-oss:latest",
    "group_size": 4,
    "chunk_size": 8000,
    "overlap": 500,
}


# Now, our run matrix. Start with just one strategy. (generate_runs only uses one strategy anyways.
"""
    - model: str (default: "gpt3")
    - prompt: str (default: "Summarize the following text:\n\n{{text}}")
    - max_tokens: int (optional) - max tokens for the summary
    - temperature: float (optional) - temperature for generation
    - top_p: float (optional) - top_p for generation
"""
# models = [
#     "gpt-oss:latest",
#     "deepseek-v3:latest",
#     "llama3.1:70b",
#     "qwen2.5:32b",
#     "granite4:32b-a9b-h",
#     "gemma3:27b",
# ]
# temperatures = [0.0, 0.4, 0.7, 1.0]
# configs = []
# for model in models:
#     for temp in temperatures:
#         configs.append(
#             {
#                 "model": model,
#                 "prompt": "Summarize the following text:\n\n{{text}}",
#                 "temperature": temp,
#             }
#         )
configs = [
    {"model": "gpt-oss:latest", "prompt": "Summarize the following text:\n\n{{text}}"}
]


async def main():
    runs = await generate_runs(inputs=ds, configs=configs, strategy=OneShotSummarizer())
    return runs


if __name__ == "__main__":
    runs = asyncio.run(main())
