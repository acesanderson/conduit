from conduit.domain.result.response import Response
from conduit.domain.message.message import AssistantMessage, SystemMessage, UserMessage
from conduit.domain.result.response_metadata import ResponseMetadata, StopReason
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.request.request import Request
from conduit.domain.conversation.conversation import Conversation

mock_params = GenerationParams(model="gpt3")
mock_request = Request(
    params=mock_params,
    messages=[],
)
mock_response_metadata = ResponseMetadata(
    model_slug="gpt3",
    duration=1.23,
    input_tokens=10,
    output_tokens=20,
    stop_reason=StopReason.STOP,
)
mock_assistant_message = AssistantMessage(content="Hello, world!")
mock_response = Response(
    request=mock_request,
    message=mock_assistant_message,
    metadata=mock_response_metadata,
)
mock_list_messages = [
    system_message := SystemMessage(content="You are a helpful assistant."),
    user_message := UserMessage(content="Hello, assistant!"),
    assistant_message := AssistantMessage(content="Hello, human!"),
]
mock_conversation = Conversation(
    messages=mock_list_messages,
)
