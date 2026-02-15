"""
Unified re-export module that aggregates remote execution capabilities for the Conduit LLM framework, enabling seamless switching between local and server-based model inference. This module serves as the primary public API for developers working with remote models—it bundles both synchronous (`RemoteModelSync`, `remote_model_sync`) and asynchronous (`RemoteModelAsync`, `remote_model_async`) remote execution interfaces alongside core orchestration primitives (prompts, generation parameters, verbosity controls) to support multi-modal LLM pipelines executed via Headwater/Siphon remote servers. By centralizing these imports, the module simplifies dependency management and ensures consistent configuration across local-vs-remote execution paths without requiring developers to know the internal module structure.

The module complements `sync.py` and `async_.py` by providing a third orchestration tier for distributed inference; developers can instantiate a `RemoteModelSync` or `RemoteModelAsync` directly and execute queries identically to their local counterparts, with the framework transparently routing requests to a remote server for execution.

Usage:
```python
from conduit.remote import RemoteModel

model = RemoteModel(model="gpt-oss:latest")
result = model.query("Name ten mammals")
```
"""

# Orchestration classes
from conduit.core.model.model_remote import (
    RemoteModelSync,
    RemoteModelAsync,
    remote_model_sync,
    remote_model_async,
)
from conduit.core.model.model_remote import RemoteModelSync as RemoteModel
from conduit.core.prompt.prompt import Prompt
from conduit.core.conduit.conduit_sync import ConduitSync
from conduit.core.conduit.conduit_async import ConduitAsync

# Primitives: dataclasses / enums
from conduit.domain.result.response import GenerationResponse
from conduit.domain.request.request import GenerationRequest
from conduit.utils.progress.verbosity import Verbosity
from conduit.domain.message.message import Message

# Configs
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions

__all__ = [
    "ConduitAsync",
    "ConduitOptions",
    "ConduitSync",
    "GenerationParams",
    "GenerationRequest",
    "GenerationResponse",
    "Message",
    "Prompt",
    "RemoteModelAsync",
    "RemoteModelSync",
    "Verbosity",
    "remote_model_async",
    "remote_model_sync",
    "RemoteModel",
]
