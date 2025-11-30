"""
Convenience imports for Headwater server:

model = RemoteModel(preferred_model)
response = model.query(prompt_str)
"""

# Orchestration classes
from conduit.model.model_remote import RemoteModel
from conduit.prompt.prompt import Prompt
from conduit.conduit.sync_conduit import SyncConduit as Conduit

# Cache
from conduit.cache.cache import ConduitCache

# Primitives: dataclasses / enums
from conduit.result.response import Response
from conduit.progress.verbosity import Verbosity
from conduit.message.message import Message
from conduit.request.request import Request

__all__ = [
    "Conduit",
    "ConduitCache",
    "Message",
    "Prompt",
    "RemoteModel",
    "Request",
    "Response",
    "Verbosity",
]
