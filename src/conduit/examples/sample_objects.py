from conduit.domain.request.request import GenerationRequest
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions
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

# Requests, results
sample_params = GenerationParams(model="gpt3")
sample_options = ConduitOptions(project_name="test")
sample_request = GenerationRequest(
    messages=sample_messages, params=sample_params, options=sample_options
)

# For Async testing
sample_async_prompt = """Name ten {{things}}."""
sample_input_variables_list = ["mammals", "birds", "villains"]
sample_prompt_strings = ["Name ten mammals.", "Name ten birds.", "Name ten villains."]
