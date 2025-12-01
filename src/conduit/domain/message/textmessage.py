"""
Message is the default message type recognized as industry standard (role + content).
Our Message class is inherited from specialized types like AudioMessage, ImageMessage, etc.
"""

from conduit.prompt.prompt import Prompt
from conduit.message.message import Message, MessageType
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class TextMessage(Message):
    """
    Industry standard, more or less, for messaging with LLMs.
    System roles can have some weirdness (like with Anthropic), but role/content is standard.
    """

    message_type: MessageType = Field(default="text", exclude=True, repr=False)
    content: str | BaseModel | list[BaseModel]

    def __str__(self):
        """
        Returns the message in a human-readable format.
        """
        return f"{self.role}: {self.content}"

    def __getitem__(self, key):
        """
        Allows for dictionary-style access to the object.
        """
        return getattr(self, key)

    # Serialization methods
    def to_cache_dict(self) -> dict:
        """
        Serializes the message to a dictionary for caching.
        """
        role = self.role
        if isinstance(self.content, str):
            content = self.content
        elif isinstance(self.content, BaseModel):
            content = self._serialize_pydantic(self.content)
        elif isinstance(self.content, list):
            content = [self._serialize_pydantic(item) for item in self.content]
        else:
            logger.error(f"Unsupported content type: {type(self.content)}")
            raise TypeError(
                "Content must be a string, BaseModel, or list of BaseModels."
            )
        assert isinstance(content, (str, dict, list)), (
            "Content must be a string, dict, or list of dicts, "
        )
        return {
            "message_type": self.message_type,
            "role": role,
            "content": content,
        }

    @classmethod
    def from_cache_dict(cls, cache_dict: dict):
        """
        Deserializes the message from a dictionary.
        """
        role = cache_dict["role"]
        content = cache_dict["content"]
        # If it's a BaseModel, we need to deserialize it
        if isinstance(content, str):
            content = content
        elif isinstance(content, dict) and "__class__" in content:
            if "__class__" in cache_dict["content"]:
                content = cls._deserialize_pydantic(cache_dict["content"])
            else:
                logger.error("Content dictionary does not contain '__class__' key.")
                raise ValueError(
                    "Content dictionary must contain '__class__' key for deserialization."
                )
        # If it's a list of BaseModels, we need to deserialize each item
        elif isinstance(content, list):
            content = [cls._deserialize_pydantic(item) for item in content]
        else:
            logger.error(f"Unsupported content type: {type(content)}")
            raise TypeError("Content must be a string, dict, or list of dicts.")
        return cls(role=role, content=content)

    def _serialize_pydantic(self, obj: BaseModel) -> dict:
        """
        Serializes a Pydantic model to a dictionary.
        """
        obj_dict = obj.model_dump()
        obj_dict["__class__"] = obj.__class__.__name__
        return obj_dict

    @classmethod
    def _deserialize_pydantic(cls, obj_dict: dict) -> BaseModel:
        # First, handle PerplexityContent specifically, since it's not technically a structured response request.
        if obj_dict.get("__class__") == "PerplexityContent":
            from conduit.model.clients.perplexity_content import PerplexityContent

            return PerplexityContent.model_validate(obj_dict)

        from conduit.parser.parser import Parser

        class_name = obj_dict.pop("__class__")

        for response_model in Parser._response_models:
            if response_model.__name__ == class_name:
                try:
                    return response_model.model_validate(obj_dict)
                except Exception as e:
                    logger.error(f"Error deserializing {class_name}: {e}")
                    raise ValueError(
                        f"Error deserializing {class_name}: {e}. There maybe version conflicts with the response model."
                    )
        logger.error(f"Unknown class name: {class_name}")
        raise ValueError(
            f"Unknown class name: {class_name}. Please check the response models in Parser._response_model."
        )

    # API compatibility methods -- these are inherited from Message so no need to override here.


# Some helpful functions
def create_system_message(
    system_prompt: str | Prompt, input_variables=None
) -> list[Message]:
    """
    Takes a system prompt object (Prompt()) or a string, an optional input object, and returns a Message object.
    """
    if isinstance(system_prompt, str):
        system_prompt = Prompt(system_prompt)
    if input_variables:
        system_message = [
            TextMessage(
                role="system",
                content=system_prompt.render(input_variables=input_variables),
            )
        ]
    else:
        system_message = [
            TextMessage(role="system", content=system_prompt.prompt_string)
        ]
    return system_message
