"""
Convenience imports for the most common Conduit use case:

prompt = Prompt(prompt_str)
model = Model(preferred_model)
conduit = Conduit(prompt, model)
response = conduit.run(input_data)
"""

# Orchestration classes
from conduit.core.conduit.sync_conduit import SyncConduit
from conduit.core.model.model_sync import ModelSync
from conduit.core.prompt.prompt import Prompt

# Cache
from conduit.storage.cache.cache import ConduitCache

# Primitives: dataclasses / enums
from conduit.domain.result.response import Response
from conduit.utils.progress.verbosity import Verbosity
from conduit.domain.message.message import Message
from conduit.domain.request.request import Request

Conduit = SyncConduit  # Alias for easier imports
Model = ModelSync  # Alias for easier imports


__all__ = [
    "Conduit",
    "ConduitCache",
    "Message",
    "Model",
    "ModelSync",
    "Prompt",
    "Request",
    "Response",
    "SyncConduit",
    "Verbosity",
]
