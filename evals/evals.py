"""
Lightweight evaluation framework for comparing strategy outputs against ground truth using async execution and pluggable scoring functions. This module provides the core infrastructure for running LLM strategies with different configurations, collecting their outputs, and scoring them via custom evaluation functions in parallel.

The framework orchestrates a three-stage pipeline: `run()` executes a single strategy with a given input and config, producing a `RunResult` that captures identity, configuration, and output metadata; `generate_runs()` creates the full combinatorial matrix of inputs×configs and executes them concurrently; and `evaluate()` applies a scoring function to each run result, returning normalized `EvalResult` objects. Strategy functions and evaluation functions are protocol-based, allowing any async callable that matches the expected signatures to be plugged in.

Usage:
```python
# Define a strategy and eval function, then benchmark them
results = await generate_runs(inputs=[RunInput(data="text", source_id="doc1")],
                               configs=[{"model": "gpt-4"}],
                               strategy=my_strategy)
scores = await evaluate(results, eval_function=semantic_similarity_scorer)
```
"""

from __future__ import annotations
from typing import Any, Protocol
from collections.abc import Callable
from pydantic import BaseModel, Field, ValidationError
from conduit.core.workflow.context import context
import asyncio
import hashlib
import json

CONCURRENCY_LIMIT = 10


# Protocols
class Strategy(Protocol):
    async def __call__(self, input: RunInput, config: dict) -> str: ...


class EvalFunction(Protocol):
    async def __call__(self, run_result: RunResult) -> float: ...


# Data Models
class RunInput(BaseModel):
    """
    Has the full input data and an optional reference output for evaluation.
    """

    source_id: str
    data: Any
    reference: str | None = None  # optional reference output for evaluation
    metadata: dict | None = None  # optional metadata about the input


class RunOutput(BaseModel):
    output: str
    metadata: dict


class RunResult(BaseModel):
    # Identity: hash of the strategy, config, and input to uniquely identify this run
    strategy: str  # __name__ of the strategy function
    config_id: str  # hash of the config dict or pydantic model
    source_id: str  # source_id from input
    reference_id: str | None = (
        None  # optional reference_id derived from input.reference for evaluation purposes
    )

    # Config
    config: dict | BaseModel  # either dict or a pydantic model

    # Results
    output: RunOutput
    warnings: list[str] = []


class EvalResult(BaseModel):
    run_result: RunResult
    score: float = Field(..., ge=0.0, le=1.0)


def validate_config(strategy: Any, config: dict) -> list[str]:
    model = getattr(strategy, "config_model", None)
    if model is None:
        return []
    try:
        model(**config)
        return []
    except ValidationError as e:
        return [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]


# Generate an evaluation run.
async def run_eval(
    input: RunInput, config: dict | BaseModel, strategy: Callable
) -> RunResult:
    """
    Run a single evaluation with the given input, configuration, and strategy.

    Args:
        input (Any): The input data for the evaluation.
        config (dict): The configuration settings for the evaluation.
        strategy (Callable): The strategy function to execute the evaluation.

    Returns:
        Any: The result of the evaluation.

    NOTE: The config schema is tightly coupled with the strategy, so we assume that the strategy can handle the provided config.
    """
    config = config.model_dump() if isinstance(config, BaseModel) else config
    warnings = validate_config(strategy, config)

    trace: list = []
    token = context.trace.set(trace)
    try:
        output = await strategy(input, config)
    finally:
        context.trace.reset(token)

    config_id = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    return RunResult(
        strategy=strategy.__class__.__name__,
        config_id=config_id,
        source_id=input.source_id,
        config=config,
        output=RunOutput(output=output, metadata={"trace": trace}),
        warnings=warnings,
    )


async def generate_runs(
    inputs: list[RunInput], configs: list[dict], strategy: Callable
) -> list[RunResult]:
    """
    Generate multiple evaluation runs based on the provided inputs, configurations, and strategy.

    Args:
        inputs (list[Any]): A list of input data for the evaluations.
        configs (list[dict]): A list of configuration settings for the evaluations.
        strategy (Callable): The strategy function to execute each evaluation.

    Returns:
        list[Any]: A list of results from each evaluation run.

    NOTE: config schema is tightly coupled with the strategy, so we assume that the strategy can handle the provided configs.
    """
    coroutines = []
    for input in inputs:
        for config in configs:
            coroutines.append(run_eval(input, config, strategy))

    # Run all evaluations concurrently and gather results
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def bounded(coro):
        async with sem:
            return await coro

    run_results = await asyncio.gather(*[bounded(c) for c in coroutines])

    return run_results


async def evaluate(
    run_results: list[RunResult], eval_function: Callable
) -> list[EvalResult]:
    """
    Evaluate the results of multiple runs using a specified evaluation function.

    Args:
        run_results (list[RunResult]): A list of RunResult objects to be evaluated.
        eval (Callable): The evaluation function that takes a RunResult and returns a score.

    Returns:
        list[RunResult]: A list of RunResult objects with updated scores based on the evaluation.
    """
    coroutines = [eval_function(run_result) for run_result in run_results]
    # Run all evaluations concurrently and gather scores
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def bounded(coro):
        async with sem:
            return await coro

    scores = await asyncio.gather(*[bounded(c) for c in coroutines])

    eval_results = [
        EvalResult(run_result=run_result, score=score)
        for run_result, score in zip(run_results, scores)
    ]
    return eval_results


if __name__ == "__main__":
    # Define a simple test strategy
    async def test_strategy(input: RunInput, config: dict) -> str:
        return f"Processed: {input.data} with model {config.get('model')}"

    # Define a simple eval function
    async def test_eval_function(run_result: RunResult) -> float:
        return 0.85

    # Run the evaluation pipeline
    async def main():
        inputs = [RunInput(data="hello world", source_id="doc1")]
        configs = [{"model": "gpt-4"}, {"model": "claude"}]

        # Generate runs
        results = await generate_runs(inputs, configs, test_strategy)
        print(f"Generated {len(results)} run results")
        print(f"First result output: {results[0].output.output}")

        # Evaluate runs
        eval_results = await evaluate(results, test_eval_function)
        print(f"Evaluated {len(eval_results)} results")
        print(f"First eval score: {eval_results[0].score}")

    asyncio.run(main())
