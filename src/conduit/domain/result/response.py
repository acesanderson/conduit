"""
Model and client interaction:
- Model sends a Request, which is: conversation (list[Message]) + generation_params
- Request sends Response, which is: the request (list[Message]) + generation_params) + the assistant message + response metadata
"""

from __future__ import annotations
from conduit.domain.message.message import AssistantMessage, Message
from conduit.domain.message.role import Role
from conduit.domain.request.request import GenerationRequest
from conduit.domain.result.response_metadata import ResponseMetadata
from pydantic import BaseModel
from typing import TYPE_CHECKING, override
import logging

if TYPE_CHECKING:
    from conduit.domain.conversation.conversation import Conversation

logger = logging.getLogger(__name__)


class GenerationResponse(BaseModel):
    """
    Our class for a successful Result.
    Message + Request + ResponseMetadata
    """

    # Core attributes
    message: AssistantMessage
    request: GenerationRequest
    metadata: ResponseMetadata

    @property
    def prompt(self) -> str | None:
        """
        This is the last user message.
        """
        if not self.request.messages:
            raise ValueError("No messages in the request to extract prompt from.")
        if not self.request.messages[-1]:
            raise ValueError("The last message in the request is None.")
        if not self.request.messages[-1].role:
            raise ValueError("The last message in the request has no role defined.")
        if self.request.messages[-1].role != Role.USER:
            raise ValueError("The last message in the request is not from the user.")
        return str(self.request.messages[-1].content)

    @property
    def messages(self) -> list[Message]:
        return [*self.request.messages, self.message]

    @property
    def conversation(self) -> Conversation:
        from conduit.domain.conversation.conversation import Conversation

        return Conversation(messages=self.messages)

    @property
    def total_tokens(self) -> int:
        return self.metadata.input_tokens + self.metadata.output_tokens

    @property
    def content(self) -> str | object:
        """
        This is the last assistant message content.
        """
        if not self.message:
            raise ValueError("No message in the response to extract content from.")
        if not self.message.content:
            return ""
        return self.message.content

    @property
    def model(self) -> str:
        """
        This is the model used for the response.
        """
        return self.request.model

    @override
    def __str__(self) -> str:
        """
        We want to pass as string when possible.
        Allow json objects (dict) to be pretty printed.
        """
        content = self.content
        if content == None or content == "":
            return ""
        if content.__class__.__name__ == "PerplexityContent":
            output = content.text + "\n\n"
            for index, citation in enumerate(content.citations):
                output += f"{index + 1}. - [{citation.title}]({citation.url})\n"
        else:
            output = content
        return output

    # Interaction methods
    def play(self) -> None:
        """
        Play audio if available.
        """
        if self.request.params.output_type != "audio":
            raise ValueError("Response is not audio type.")
        from pydub import AudioSegment
        from pydub.playback import play
        import io
        import base64

        audio_data_b64 = self.message.content
        audio_data = base64.b64decode(audio_data_b64)
        audio_format = self.request.params.client_params.get("response_format", "mp3")
        audio_segment = AudioSegment.from_file(
            io.BytesIO(audio_data), format=audio_format
        )
        play(audio_segment)

    def display(self) -> None:
        """
        Display image if available, using viu in a subprocess.
        """
        if self.request.params.output_type != "image":
            raise ValueError("Response is not image type.")
        if self.message.images is None or len(self.message.images) == 0:
            raise ValueError("No images in the response to display.")
        import subprocess
        import base64
        import io
        from PIL import Image

        image_data_b64 = self.message.images[0].b64_json
        image_data = base64.b64decode(image_data_b64)
        image = Image.open(io.BytesIO(image_data))
        image.save("/tmp/temp_image.png")
        subprocess.run(["viu", "/tmp/temp_image.png"])
