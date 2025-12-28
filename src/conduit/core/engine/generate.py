"""
We need request params.
How does request params differ from request class?
"""

from conduit.domain.conversation.conversation import Conversation
from conduit.domain.result.response import GenerationResponse
from conduit.domain.request.request import GenerationRequest
from conduit.core.model.model_async import ModelAsync
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.domain.message.message import Message


async def generate(
    conversation: Conversation, params: GenerationParams, options: ConduitOptions
) -> Conversation:
    # 1. Create a mutable copy of the messages to avoid side effects.
    messages = list(conversation.messages)

    # 2. Ensure system prompt is correctly placed.
    has_system_message = any(msg.role == "system" for msg in messages)
    if not has_system_message and params.system:
        # Prepend a new SystemMessage if one doesn't exist and is provided in params.
        from conduit.domain.message.message import SystemMessage
        messages.insert(0, SystemMessage(content=params.system))

    # 3. Prepare and execute the request.
    # We create a temporary request object with the potentially modified message list.
    request = GenerationRequest(messages=messages, params=params, options=options)
    
    model = ModelAsync(params.model)
    response = await model.pipe(request)
    
    assert isinstance(response, GenerationResponse), (
        "Expected response to be of type Response"
    )

    # 4. Append the new assistant message to our copied list.
    new_message: Message = response.message
    messages.append(new_message)
    
    # 5. Return a new Conversation object, preserving the original's topic/id.
    return conversation.model_copy(update={"messages": messages})
