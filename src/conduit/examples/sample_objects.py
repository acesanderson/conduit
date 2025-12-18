from conduit.domain.result.response import GenerationResponse
from conduit.domain.request.request import GenerationRequest
from conduit.domain.message.message import (
    UserMessage,
    AssistantMessage,
)
from conduit.domain.conversation.conversation import Conversation
from pathlib import Path

dir_path = Path(__file__).parent

# Messages
sample_message = UserMessage(content="Hello, world!")
sample_messages = [
    UserMessage(content="Hello, world!"),
    AssistantMessage(content="Hello! How can I assist you today?"),
    UserMessage(content="What is the weather like?"),
]
sample_conversation = Conversation(messages=sample_messages)

sample_audio_file = dir_path / "audio.mp3"
sample_image_file = dir_path / "image.png"

# Requests, results, etc. TBD
sample_response: GenerationResponse
sample_request: GenerationRequest

# For Async testing
sample_async_prompt = """Name ten {{things}}."""
sample_input_variables_list = ["mammals", "birds", "villains"]
sample_prompt_strings = ["Name ten mammals.", "Name ten birds.", "Name ten villains."]
