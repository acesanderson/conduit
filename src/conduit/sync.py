"""
Convenience imports for the most common Conduit use case:

prompt = Prompt(prompt_str)
model = Model(preferred_model)
conduit = Conduit(prompt, model)
response = conduit.run(input_data)
"""

# Orchestration classes
from conduit.core.conduit.conduit_sync import ConduitSync
from conduit.core.model.model_sync import ModelSync
from conduit.core.prompt.prompt import Prompt

# Primitives: dataclasses / enums
from conduit.utils.progress.verbosity import Verbosity
from conduit.domain.request.request import GenerationRequest

Conduit = ConduitSync  # Alias for easier imports
Model = ModelSync  # Alias for easier imports


__all__ = [
    "Conduit",
    "GenerationRequest",
    "Model",
    "ModelSync",
    "Prompt",
    "Verbosity",
]
