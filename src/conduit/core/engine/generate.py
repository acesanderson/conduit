"""
We need request params.
How does request params differ from request class?
"""

from conduit.domain.conversation.conversation import Conversation
from conduit.domain.result.response import GenerationResponse
from conduit.core.model.model_async import ModelAsync
from conduit.domain.request.request import GenerationRequest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.domain.message.message import Message
    from conduit.domain.request.generation_params import GenerationParams


async def generate(
    conversation: Conversation, params: GenerationParams
) -> Conversation:
    messages = conversation.messages
    model = ModelAsync("gpt")
    request = GenerationRequest.from_conversation(conversation, params)
    response = model.query(request)
    assert isinstance(response, GenerationResponse), (
        "Expected response to be of type Response"
    )
    new_message: Message = response.message
    messages.append(new_message)
    return Conversation(messages=messages)
