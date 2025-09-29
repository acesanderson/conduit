"""
Base class for clients; openai, anthropic, etc. inherit from this class.
ABC abstract methods are like pydantic validators for classes. They ensure that the child classes implement the methods defined in the parent class.
If a client subclass doesn't implement _get_api_key, for example, the code will raise an error when trying to instantiate the subclass.
This guarantees all client subclasses have the methods below.
TODO: implement a class SDK as a protocol for all the library clients (openai, ollama, groq, etc.). This would define an object that has a chat function, fex.
"""

from abc import ABC, abstractmethod
from Chain.request.request import Request
from pydantic import BaseModel


class Usage(BaseModel):
    """
    Simple data class for usage statistics, standardized across providers.
    """

    input_tokens: int
    output_tokens: int


class Client(ABC):
    @abstractmethod
    def __init__(self):
        """
        Typically we should see this here:
        self._client = self._initialize_client()
        """
        pass

    @abstractmethod
    def _initialize_client(self) -> object:
        """
        This method should initialize the client object, this is idiosyncratic to each SDK.
        """
        pass

    @abstractmethod
    def _get_api_key(self) -> str:
        """
        API keys are accessed via dotenv.
        This should be in an .env file in the root directory.
        """
        pass

    @abstractmethod
    def query(self, request: Request) -> tuple:
        """
        All client subclasses must have a query function that can take:
        - a Request object, which contains all the parameters needed for the query

        And returns
        - A tuple of
            - either a string (i.e. text generation) or a Pydantic model (function calling)
            - a Usage object containing input and output token counts
        """
        pass

    @abstractmethod
    def tokenize(self, model: str, text: str) -> int:
        """
        Get the token count for a text, per a given model's tokenization function.
        """
        pass

    def __repr__(self):
        """
        Standard repr.
        """
        attributes = ", ".join(
            [f"{k}={repr(v)[:50]}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({attributes})"
