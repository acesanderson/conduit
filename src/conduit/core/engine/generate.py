"""
We need request params.
How does request params differ from request class?
"""

from conduit.domain.conversation.conversation import Conversation
from conduit.domain.result.response import Response
from conduit.sync import Model
from conduit.domain.request.request import Request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.domain.message.message import Message
    from conduit.domain.request.generation_params import GenerationParams


def generate(conversation: Conversation, params: GenerationParams) -> Conversation:
    messages = conversation.messages
    model = Model("gpt")
    request = Request.from_conversation(conversation, params)
    response = model.query(request)
    assert isinstance(response, Response), "Expected response to be of type Response"
    new_message: Message = response.message
    messages.append(new_message)
    return Conversation(messages=messages)
