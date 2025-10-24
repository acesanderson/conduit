"""
Convenience imports for Headwater server:

model = RemoteModel(preferred_model)
response = model.query(prompt_str)
"""

# Orchestration classes
from conduit.model.remote_model import RemoteModel
from conduit.prompt.prompt import Prompt

# Cache
from conduit.cache.cache import ConduitCache

# Primitives: dataclasses / enums
from conduit.result.response import Response
from conduit.progress.verbosity import Verbosity
from conduit.message.message import Message
from conduit.request.request import Request

__all__ = [
    "ConduitCache",
    "Message",
    "Prompt",
    "RemoteModel",
    "Request",
    "Response",
    "Verbosity",
]
