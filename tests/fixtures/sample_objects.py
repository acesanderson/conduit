from Chain.result.error import ChainError, ErrorInfo, ErrorDetail
from Chain.result.response import Response
from Chain.request.request import Request
from Chain.message.textmessage import TextMessage
from Chain.message.imagemessage import ImageMessage
from Chain.message.audiomessage import AudioMessage
from Chain.message.messages import Messages
from datetime import datetime
from pathlib import Path

dir_path = Path(__file__).parent

# Messages
sample_message = TextMessage(role="user", content="Hello, world!")
sample_messages = Messages(
    [
        TextMessage(role="user", content="Hello, world!"),
        TextMessage(role="assistant", content="Hello! How can I assist you today?"),
        TextMessage(role="user", content="What is the weather like?"),
    ]
)

sample_audio_file = dir_path / "audio.mp3"
sample_audio_message = AudioMessage.from_audio_file(
    role="user",
    text_content="This is a sample audio message.",
    audio_file=sample_audio_file,
)
sample_image_file = dir_path / "image.png"
sample_image_message = ImageMessage.from_image_file(
    role="user",
    text_content="This is a sample image message.",
    image_file=sample_image_file,
)
# Requests, results, etc.
sample_error = ChainError(
    info=ErrorInfo(
        code="ERR001",
        message="An unexpected error occurred",
        category="RuntimeError",
        timestamp=datetime.now(),
    ),
    detail=ErrorDetail(
        exception_type="ValueError",
        stack_trace="Traceback (most recent call last): ...",  # Example stack trace
        raw_response=None,  # Could be a response object or None
        request_params=None,  # Could be a dict of request parameters or None
        retry_count=0,  # Number of retries attempted
    ),
)
sample_response = Response(
    message=sample_message,
    request=Request(model="gpt-3.5-turbo-0125", messages=sample_messages),
    duration=1.23,
    input_tokens=60,
    output_tokens=120,
)
sample_request = Request(
    model="gpt-3.5-turbo-0125",
    messages=[
        TextMessage(role="user", content="Hello, world!"),
        TextMessage(role="assistant", content="Hello! How can I assist you today?"),
    ],
    temperature=0.7,
    stream=True,
    parser=None,  # Assuming no parser for this example
)

# For Async testing
sample_async_prompt = """Name ten {{things}}."""
sample_input_variables_list = ["mammals", "birds", "villains"]
sample_prompt_strings = ["Name ten mammals.", "Name ten birds.", "Name ten villains."]
