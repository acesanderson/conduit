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
    # 1. Clone the conversation to work on it safely
    working_conversation = conversation.model_copy(deep=True)

    # 2. Ensure system prompt is correctly placed.
    messages = working_conversation.messages
    has_system_message = any(msg.role.value == "system" for msg in messages)
    if not has_system_message and params.system:
        working_conversation.ensure_system_message(params.system)

    # 3. Prepare and execute the request.
    request = GenerationRequest(
        messages=working_conversation.messages, params=params, options=options
    )

    model = ModelAsync(params.model)
    response = await model.pipe(request)

    assert isinstance(response, GenerationResponse), (
        "Expected response to be of type Response"
    )

    # 4. Add the new assistant message
    new_message: Message = response.message
    working_conversation.add(new_message)

    # 5. Return the updated conversation
    return working_conversation
