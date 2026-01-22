"""
Convenience module for batch processing workflows in Conduit. Provides unified imports for asynchronous and synchronous batch operations on LLM requests, along with core configuration and result types needed to orchestrate multi-item pipelines.

This module serves as the primary entry point for developers working with batch processing—either through ConduitBatchAsync for concurrent async operations or ConduitBatchSync for simpler synchronous batch flows. It aggregates both execution orchestrators, prompt templating, and all necessary parameter/option/response DTOs into a single import surface to avoid deep module traversal.

Usage:
```python
from conduit.batch import ConduitBatchSync, Prompt, GenerationParams

batch = ConduitBatchSync.create(
    model="gpt-4o",
    prompt=Prompt("Summarize: {{text}}"),
    persist=True
)
results = batch.run(prompt_strings_list=["text1", "text2", "text3"])
```
"""

# Orchestration classes
from conduit.core.conduit.batch.conduit_batch_async import ConduitBatchAsync
from conduit.core.conduit.batch.conduit_batch_sync import ConduitBatchSync
from conduit.core.prompt.prompt import Prompt

# Primitives: dataclasses / enums
from conduit.domain.result.response import GenerationResponse
from conduit.utils.progress.verbosity import Verbosity
from conduit.domain.request.request import GenerationRequest

# Configs
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions

__all__ = [
    "ConduitBatchAsync",
    "ConduitBatchSync",
    "ConduitOptions",
    "GenerationParams",
    "GenerationRequest",
    "GenerationResponse",
    "Prompt",
    "Verbosity",
]
