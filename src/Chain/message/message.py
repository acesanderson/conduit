"""
Message is the default message type recognized as industry standard (role + content).
Our Message class is inherited from specialized types like TextMessage, AudioMessage, ImageMessage, etc.

All Messages have:
(1) serialization methods to convert to/from dictionaries for caching and API compatibility.

```
to_cache_dict()
from_cache_dict()
```

(2) methods for API compatibility with OpenAI, Anthropic, Google, and Perplexity, which are invoked by the individual provider clients.

```
to_openai()
to_anthropic()
to_google()
...
```
"""

from abc import abstractmethod, ABC
from Chain.logs.logging_config import get_logger
from pydantic import BaseModel
from typing import Literal, Any

logger = get_logger(__name__)

# Useful type aliases
Role = Literal["user", "assistant", "system"]
MessageType = Literal["text", "audio", "image"]


class Message(BaseModel, ABC):
    """Base message class - abstract with required Pydantic functionality"""
    message_type: MessageType
    role: Role
    content: Any

    # Serialization methods to convert the message to a dictionary
    @abstractmethod
    def to_cache_dict(self) -> dict:
        """
        Serializes the message to a dictionary for caching.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
   
    @classmethod
    def from_cache_dict(cls, cache_dict: dict) -> "Message":
        """Parse JSON with lazy imports to avoid circular dependencies"""
        message_type = cache_dict["message_type"]
        
        # Import only when needed
        if message_type == "text":
            from Chain.message.textmessage import TextMessage
            return TextMessage.from_cache_dict(cache_dict)
        elif message_type == "audio":
            from Chain.message.audiomessage import AudioMessage  
            return AudioMessage.from_cache_dict(cache_dict)
        elif message_type == "image":
            from Chain.message.imagemessage import ImageMessage
            return ImageMessage.from_cache_dict(cache_dict)
        else:
            raise ValueError(f"Unknown message type: {message_type}")
    
    # API compatibility methods. These are overridden in AudioMessage, ImageMessage, and other specialized message types (TBD).
    def to_openai(self) -> dict:
        """
        Convert message to OpenAI API format.
        """
        return self.to_cache_dict()

    def to_anthropic(self) -> dict:
        """
        Convert message to Anthropic API format.
        """
        return self.to_cache_dict()

    def to_google(self) -> dict:
        """
        Convert message to Google API format.
        """
        return self.to_cache_dict()

    def to_ollama(self) -> dict:
        """
        Convert message to Ollama API format.
        """
        return self.to_cache_dict()

    def to_perplexity(self) -> dict:
        """
        Convert message to Perplexity API format.
        """
        return self.to_cache_dict()
