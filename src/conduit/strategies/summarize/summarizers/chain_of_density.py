"""
```python
from conduit.core.workflow.harness import ConduitHarness
from conduit.strategies.summarize.summarizers.chain_of_density import ChainOfDensitySummarizer

config = {
    "model": "gpt-oss:latest",
    "density_iterations": 3,
    "density_target_tokens": 100,
}

harness = ConduitHarness(config=config)
result = await harness.run(ChainOfDensitySummarizer(), text=document)
```
"""

from __future__ import annotations

import logging
from typing import override, Any
from conduit.core.workflow.step import step, get_param, add_metadata
from conduit.core.workflow.context import context
from conduit.strategies.summarize.strategy import SummarizationStrategy, _TextInput
from conduit.core.model.model_async import ModelAsync
from conduit.core.prompt.prompt import Prompt
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.domain.result.response import GenerationResponse
from conduit.utils.progress.verbosity import Verbosity

logger = logging.getLogger(__name__)


density_iteration_prompt = """
You are generating a increasingly dense summary of the same document.

Previous summary (less dense):
<previous_summary>
{{ previous_summary }}
</previous_summary>

Original document:
<document>
{{ document }}
</document>

Generate a new summary that:
1. Retains all key factual information from the previous summary
2. Adds missing entities, relationships, and numerical details
3. Is more information-dense while remaining coherent
4. Target ~{{ target_tokens }} tokens

Return only the new summary.
""".strip()


class ChainOfDensitySummarizer(SummarizationStrategy):
    """
    Chain-of-Density style summarization that iteratively adds missing details.

    Workflow:
    1. Generate an initial abstract summary.
    2. For N iterations, ask the model to add missing entities/facts to the summary.
    3. Return the final dense summary.

    This approach produces summaries that progressively become more informative
    while maintaining coherence, similar to the Chain-of-Density method.
    """

    @step
    @override
    async def __call__(self, input: Any, config: dict) -> str:
        token_conf = context.config.set(config)
        token_defaults = context.use_defaults.set(True)
        try:
            text = input.data
            model = get_param("model", default="gpt3")
            num_iterations = get_param("density_iterations", default=3)
            target_tokens = get_param("density_target_tokens", default=100)
            iteration_prompt = get_param("density_iteration_prompt", default=density_iteration_prompt)

            # 1. Initial abstract summary (same prompt as OneShotSummarizer)
            from conduit.strategies.summarize.summarizers.one_shot import OneShotSummarizer
            current_summary = await OneShotSummarizer()(_TextInput(text), config)

            # 2. Iteratively refine
            generation_params = GenerationParams(
                model=model,
                max_tokens=get_param("max_tokens", default=None),
                temperature=get_param("temperature", default=None),
                top_p=get_param("top_p", default=None),
            )
            options = ConduitOptions(
                project_name=get_param("project_name", default="conduit"),
                verbosity=Verbosity.SILENT,
                debug_payload=True,
            )
            model_instance = ModelAsync(model=model)

            total_input_tokens = 0
            total_output_tokens = 0

            for i in range(1, num_iterations + 1):
                logger.info(f"Chain-of-Density iteration {i}/{num_iterations}")
                rendered = Prompt(iteration_prompt).render({
                    "previous_summary": current_summary,
                    "document": text,
                    "target_tokens": str(target_tokens),
                })
                response = await model_instance.query(
                    query_input=rendered,
                    params=generation_params,
                    options=options,
                )
                assert isinstance(response, GenerationResponse)
                current_summary = str(response.content)
                total_input_tokens += response.metadata.input_tokens
                total_output_tokens += response.metadata.output_tokens

            add_metadata("density_iterations", num_iterations)
            add_metadata("density_input_tokens", total_input_tokens)
            add_metadata("density_output_tokens", total_output_tokens)

            return current_summary
        finally:
            context.config.reset(token_conf)
            context.use_defaults.reset(token_defaults)
