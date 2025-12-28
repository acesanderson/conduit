"""
Convenience imports for Headwater server:

model = RemoteModel(preferred_model)
response = model.query(prompt_str)
"""

# Orchestration classes
from conduit.core.model.model_remote import RemoteModelSync, RemoteModelAsync
from conduit.core.prompt.prompt import Prompt
from conduit.core.conduit.conduit_sync import ConduitSync
from conduit.core.conduit.conduit_async import ConduitAsync

# Primitives: dataclasses / enums
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
    "Message",
    "Prompt",
    "RemoteModelAsync",
    "RemoteModelSync",
    "Verbosity",
]
