"""
Convenience imports for Headwater server:

model = RemoteModel(preferred_model)
response = model.query(prompt_str)
"""

# Orchestration classes
from conduit.core.model.model_remote import RemoteModel
from conduit.core.prompt.prompt import Prompt
from conduit.core.conduit.conduit_sync import ConduitSync as Conduit

# Cache
from conduit.storage.cache.cache import ConduitCache

# Primitives: dataclasses / enums
from conduit.domain.result.response import Response
from conduit.utils.progress.verbosity import Verbosity
from conduit.domain.message.message import Message
from conduit.domain.request.request import Request

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
