from conduit.config import settings
from conduit.core.model.clients.client_base import Client
from conduit.domain.result.result import ConduitResult
from headwater_api.classes import StatusResponse
from headwater_client.client.headwater_client import HeadwaterClient
from typing import override, TYPE_CHECKING
import json
import logging

if TYPE_CHECKING:
    from conduit.domain.request.request import Request
    from conduit.domain.message.message import Message

logger = logging.getLogger(__name__)


class RemoteClient(Client):
    def __init__(self):
        self._client = self._initialize_client()

    @override
    def _initialize_client(self) -> HeadwaterClient:
        """Initialize SiphonClient connection"""
        client = HeadwaterClient()

        # Test connection
        try:
            status = client.ping()
            if status == False:
                raise ConnectionError("Server is not healthy")
            return client
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Headwater server: {e}")

    def _validate_server_model(self, model_name: str) -> bool:
        """Validate that the model is available on the server"""
        try:
            status: StatusResponse = self._client.get_status()
            available_models = getattr(status, "models_available", [])
            logger.info(f"Available models on server: {available_models}")
            # Update server models file
            if available_models:
                with open(settings.paths["SERVER_MODELS_PATH"], "w") as f:
                    json_dict = {"ollama": available_models}
                    _ = f.write(json.dumps(json_dict, indent=4))
                logger.debug(
                    f"Updated server models file at {settings.paths['SERVER_MODELS_PATH']}"
                )
            else:
                raise ValueError("No models available on server.")

            if model_name not in available_models:
                raise ValueError(
                    f"Model '{model_name}' not available on server. "
                    f"Available models: {available_models}"
                )
            else:
                logger.info(f"Model '{model_name}' is available on server.")
                return True
        except Exception as e:
            raise ValueError(f"Failed to validate model on server: {e}")

    @override
    def _get_api_key(self) -> str: ...

    @override
    def query(self, request: Request) -> ConduitResult:
        """
        Query the remote model via HeadwaterClient.
        Unlike other clients, we get a full Response object (including token usage) back.
        """
        _ = self._validate_server_model(model_name=request.model)
        response = self._client.conduit.query_sync(request)
        # Dummy usage object since we get full response from server
        usage = Usage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        return response, usage  # usage: response, _

    @override
    def tokenize(self, model: str, payload: str | list["Message"]) -> int:
        """
        Get the token count for a text, per a given model's tokenization function.
        If payload is a list of Messages, we serialize to JSON to approximate the weight
        for the server-side tokenizer which expects a string.
        """
        from headwater_api.classes import TokenizationRequest

        _ = self._validate_server_model(model_name=model)

        if isinstance(payload, str):
            text = payload
        elif isinstance(payload, list):
            # Serialize list[Message] to a JSON string to pass to the text-only endpoint.
            # We use to_openai() (which returns dicts) then dump to json.
            # This captures the overhead of keys/structure for the tokenizer.
            text = json.dumps([m.to_openai() for m in payload])
        else:
            raise ValueError("Payload must be string or list[Message]")

        request = TokenizationRequest(model=model, text=text)
        response = self._client.conduit.tokenize(request)
        token_count = response.token_count
        return token_count

    # Client/server specific methods
    def get_status(self) -> StatusResponse:
        """Get server status"""
        return self._client.get_status()

    def ping(self) -> bool:
        """Ping server to check health"""
        return self._client.ping()
