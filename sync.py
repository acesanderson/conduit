"""
Convenience imports for the most common Chain use case:

prompt = Prompt(prompt_str)
model = Model(preferred_model)
chain = Chain(prompt, model)
response = chain.run(input_data)
"""

# Orchestration classes
from Chain.chain.chain import Chain
from Chain.model.model import Model
from Chain.prompt.prompt import Prompt

# Primitives: dataclasses / enums
from Chain.result.response import Response
from Chain.progress.verbosity import Verbosity
from Chain.message.message import Message
from Chain.request.request import Request

__all__ = [
    "Chain",
    "Model",
    "Prompt",
    "Response",
    "Verbosity",
    "Message",
    "Request",
]
