from __future__ import annotations
from conduit.config import settings
from conduit.core.clients.client_base import Client
from conduit.core.clients.payload_base import Payload
from conduit.domain.result.response import GenerationResponse
from conduit.domain.result.response_metadata import ResponseMetadata, StopReason
from conduit.domain.message.message import AssistantMessage
from headwater_api.classes import StatusResponse
from headwater_client.client.headwater_client import HeadwaterClient
from typing import override, TYPE_CHECKING, Any
import json
import logging
import time

if TYPE_CHECKING:
    from conduit.domain.request.request import GenerationRequest
    from conduit.domain.result.result import GenerationResult
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
    def _get_api_key(self) -> str:
        """Remote client doesn't use API keys - authentication handled by HeadwaterClient"""
        return ""

    @override
    def _convert_message(self, message: Message) -> dict[str, Any]:
        """
        Convert internal Message to format expected by remote server.
        TBD: Determine if server expects OpenAI format or custom format.
        """
        # TBD: Implementation needed - may delegate to message.to_openai() or custom format
        raise NotImplementedError("Remote message conversion logic needed")

    @override
    def _convert_request(self, request: GenerationRequest) -> Payload:
        """
        Convert GenerationRequest to format expected by remote server.
        TBD: May not need Payload if server expects raw GenerationRequest.
        """
        # TBD: Implementation needed - determine server's expected request format
        raise NotImplementedError("Remote request conversion logic needed")

    @override
    async def query(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        """
        Query the remote model via HeadwaterClient.
        Routes to appropriate generation method based on output_type.
        """
        match request.params.output_type:
            case "text":
                return await self._generate_text(request)
            case "image":
                return await self._generate_image(request)
            case "audio":
                return await self._generate_audio(request)
            case "structured_response":
                return await self._generate_structured_response(request)
            case _:
                raise ValueError(
                    f"Unsupported output type: {request.params.output_type}"
                )

    async def _generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text via remote server.
        TBD: Update HeadwaterClient API call to match new async pattern.
        """
        start_time = time.time()

        # Validate model availability on server
        _ = self._validate_server_model(model_name=request.params.model)

        # TBD: Update this to use async HeadwaterClient method
        # response = await self._client.conduit.query_async(request)
        # For now, placeholder:
        raise NotImplementedError("Remote text generation logic needed - update HeadwaterClient call to async")

        # TBD: Parse server response and construct GenerationResponse
        # duration = (time.time() - start_time) * 1000
        # assistant_message = AssistantMessage(content=response.content)
        # metadata = ResponseMetadata(
        #     duration=duration,
        #     model_slug=request.params.model,
        #     input_tokens=response.input_tokens,
        #     output_tokens=response.output_tokens,
        #     stop_reason=StopReason.STOP,
        # )
        # return GenerationResponse(
        #     message=assistant_message,
        #     request=request,
        #     metadata=metadata,
        # )

    async def _generate_image(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate image via remote server.
        TBD: Implement if remote server supports image generation.
        """
        raise NotImplementedError("Remote image generation not yet supported")

    async def _generate_audio(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate audio via remote server.
        TBD: Implement if remote server supports audio generation.
        """
        raise NotImplementedError("Remote audio generation not yet supported")

    async def _generate_structured_response(
        self, request: GenerationRequest
    ) -> GenerationResponse:
        """
        Generate structured response via remote server.
        TBD: Update HeadwaterClient API call for structured responses.
        """
        start_time = time.time()

        # Validate model availability on server
        _ = self._validate_server_model(model_name=request.params.model)

        # TBD: Update this to use async HeadwaterClient method with structured output
        raise NotImplementedError("Remote structured response logic needed - update HeadwaterClient call")

    @override
    async def tokenize(self, model: str, payload: str | list["Message"]) -> int:
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
            # TBD: Update to use new message conversion method once implemented
            # text = json.dumps([self._convert_message(m) for m in payload])
            text = json.dumps([m.to_openai() for m in payload])
        else:
            raise ValueError("Payload must be string or list[Message]")

        request = TokenizationRequest(model=model, text=text)
        # TBD: Update to async HeadwaterClient method
        # response = await self._client.conduit.tokenize_async(request)
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
