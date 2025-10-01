"""
Convenience imports for the most common Conduit use case:

prompt = Prompt(prompt_str)
model = Model(preferred_model)
conduit = Conduit(prompt, model)
response = conduit.run(input_data)
"""

# Orchestration classes
from conduit.conduit.sync_conduit import SyncConduit
from conduit.model.model import Model
from conduit.prompt.prompt import Prompt

# Cache
from conduit.cache.cache import ConduitCache

# Primitives: dataclasses / enums
from conduit.result.response import Response
from conduit.progress.verbosity import Verbosity
from conduit.message.message import Message
from conduit.request.request import Request

Conduit = SyncConduit  # Alias for easier imports


__all__ = [
    "Conduit",
    "ConduitCache",
    "Message",
    "Model",
    "Prompt",
    "Request",
    "Response",
    "SyncConduit",
    "Verbosity",
]
