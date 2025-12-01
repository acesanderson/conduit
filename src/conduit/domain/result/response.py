"""
A successful Result.
"""

from conduit.domain.message.message import Message
from conduit.domain.request.request import Request
from conduit.utils.progress.display_mixins import (
    RichDisplayResponseMixin,
    PlainDisplayResponseMixin,
)
from pydantic import BaseModel, Field, model_validator
import time
import logging

logger = logging.getLogger(__name__)


class Response(BaseModel, RichDisplayResponseMixin, PlainDisplayResponseMixin):
    """
    Our class for a successful Result.
    We mixin display modules so that Responses can to_plain, to_rich as part of our progress tracking / verbosity system.
    """

    # Core attributes
    message: Message
    request: Request
    input_tokens: int
    output_tokens: int
    duration: float

    # Control flag -- if this is rehydrated from cache, we don't want to emit token events again
    emit_tokens: bool = Field(default=True, exclude=True)  # Exclude from serialization

    # Initialization attributes
    timestamp: int = Field(
        default_factory=lambda: int(time.time() * 1000),
        description="Timestamp of the response creation",
    )

    @model_validator(mode="after")
    def emit_token_event_if_enabled(self):
        if self.emit_tokens:
            try:
                self.emit_token_event()
            except Exception as e:
                logger.error(f"Failed to emit token event: {e}")
        return self

    def emit_token_event(self):
        """
        Emit a TokenEvent to the OdometerRegistry if it exists.
        """
        from conduit.storage.odometer.TokenEvent import TokenEvent
        from conduit.core.model.model_sync import ModelSync

        assert self.request.provider, "Provider must be set in the request"

        # Get hostname
        import socket

        try:
            host = socket.gethostname()
        except Exception as e:
            logger.error(f"Failed to get hostname: {e}")
            host = "unknown"

        event = TokenEvent(
            provider=self.request.provider,
            model=self.request.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            timestamp=int(datetime.fromisoformat(self.timestamp).timestamp()),
            host=host,
        )
        ModelSync._odometer_registry.emit_token_event(event)

    @property
    def prompt(self) -> str | None:
        """
        This is the last user message.
        """
        return self.request.messages[-1].content

    @property
    def messages(self) -> list[Message]:
        return [*self.request.messages, self.message]

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def content(self) -> str | BaseModel | list[BaseModel] | list[str]:
        """
        This is the last assistant message content.
        """
        return self.message.content

    @property
    def model(self) -> str:
        """
        This is the model used for the response.
        """
        return self.request.model

    def __repr__(self):
        attr_list = []
        for k, v in self.__dict__.items():
            if k == "message" and hasattr(v, "content"):
                # Special handling for message content
                content = v.content
                if isinstance(content, str):
                    word_count = len(content.split())
                    if len(content) > 20:
                        truncated_content = (
                            content[:20] + f"... [{word_count} words total]"
                        )
                    else:
                        truncated_content = content
                elif isinstance(content, BaseModel):
                    word_count = len(content.model_dump_json().split())
                    truncated_content = f"{content.__class__.__name__}(...)"
                elif isinstance(content, list):
                    word_count = sum(
                        len(item.split()) if isinstance(item, str) else 0
                        for item in content
                    )
                    truncated_content = f"List[{len(content)} items] (...)"

                # Build the message repr manually with truncated content
                message_repr = (
                    f"TextMessage(role='{v.role}', content='{truncated_content}')"
                )
                attr_list.append(f"{k}={message_repr}")
            else:
                attr_list.append(f"{k}={repr(v)[:50]}")

        attributes = ", ".join(attr_list)
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self):
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

    def __len__(self):
        """
        We want to be able to check the length of the content.
        """
        return len(self.__str__())
