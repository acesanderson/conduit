from __future__ import annotations
from conduit.config import settings
from conduit.core.clients.client_base import Client
from conduit.core.clients.payload_base import Payload
from conduit.domain.result.response import GenerationResponse
from conduit.domain.result.response_metadata import ResponseMetadata, StopReason
from conduit.domain.message.message import AssistantMessage
from headwater_api.classes import StatusResponse
from headwater_client.client.headwater_client_async import HeadwaterAsyncClient
from typing import override, TYPE_CHECKING, Any
import json
import logging
import time
import asyncio

if TYPE_CHECKING:
    from collections.abc import Sequence
    from conduit.domain.request.request import GenerationRequest
    from conduit.domain.result.result import GenerationResult
    from conduit.domain.message.message import Message

logger = logging.getLogger(__name__)


class RemoteClient(Client):
    def __init__(self):
        self.is_healthy: None | bool = None
        self.status: None | StatusResponse = None
        # Internal cache storage
        self._cached_client: HeadwaterAsyncClient | None = None
        self._cached_loop: asyncio.AbstractEventLoop | None = None

    @property
    def _client(self) -> HeadwaterAsyncClient:
        """
        Lazily initialized HeadwaterAsyncClient.
        Smart caching: Re-initializes if the asyncio Event Loop has changed
        (which happens between calls in RemoteModelSync).
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # Should not happen inside async methods, but safe fallback
            current_loop = None

        # Check if we have a valid cached client for THIS loop
        if (
            self._cached_client
            and self._cached_loop is current_loop
            and current_loop is not None
            and not current_loop.is_closed()
        ):
            return self._cached_client

        # Initialize new client
        logger.debug("Initializing new HeadwaterAsyncClient (New Event Loop detected)")
        self._cached_client = self._initialize_client()
        self._cached_loop = current_loop
        return self._cached_client

    def _initialize_client(self) -> HeadwaterAsyncClient:
        """Initialize SiphonClient connection"""
        client = HeadwaterAsyncClient()
        return client

    async def _ping_server(self) -> bool:
        """Ping remote server to check health"""
        logger.debug("Pinging remote server...")
        is_healthy = await self._client.ping()
        return is_healthy

    async def _validate_server_model(self, model_name: str) -> bool:
        """
        Validate that the model is available on the server, IF not already in our canonical list of models.
        """
        from conduit.core.model.models.modelstore import ModelStore

        if model_name in ModelStore.cloud_models():
            logger.debug(f"Model '{model_name}' is a registered cloud model.")
            return True
        else:
            logger.debug(f"Validating model '{model_name}' on remote server.")
            available_models = getattr(self.status, "models_available", [])
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
                    f"Model '{model_name}' not available on server. Available models: {available_models}"
                )
            else:
                logger.info(f"Model '{model_name}' is available on server.")
                return True

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
        # Initial handshake / health check
        if self.is_healthy is None:  # First time check
            self.is_healthy = await self._ping_server()
            if self.is_healthy:
                self.status = await self._client.get_status()
        if not self.is_healthy:  # Subsequent checks
            raise ConnectionError("Cannot connect to remote server.")

        # Validate model availability on server
        _ = await self._validate_server_model(model_name=request.params.model)

        # Route to appropriate generation method
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

        response = await self._client.conduit.query_generate(request)

        duration = (time.time() - start_time) * 1000
        assistant_message = AssistantMessage(content=str(response.content))
        metadata = ResponseMetadata(
            duration=duration,
            model_slug=request.params.model,
            input_tokens=response.metadata.input_tokens,
            output_tokens=response.metadata.output_tokens,
            stop_reason=StopReason.STOP,
        )
        return GenerationResponse(
            message=assistant_message,
            request=request,
            metadata=metadata,
        )

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
        raise NotImplementedError(
            "Remote structured response logic needed - update HeadwaterClient call"
        )

    @override
    async def tokenize(self, model: str, payload: str | Sequence[Message]) -> int:
        """
        Get the token count for a text, per a given model's tokenization function.
        If payload is a Sequence of Messages, we serialize to JSON to approximate the weight
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
            raise ValueError("Payload must be string or Sequence[Message]")

        request = TokenizationRequest(model=model, text=text)
        # TBD: Update to async HeadwaterClient method
        # response = await self._client.conduit.tokenize_async(request)
        response = self._client.conduit.tokenize(request)
        token_count = response.token_count
        return token_count

    # Convenience methods (ping, status)
    async def ping(self) -> bool:
        """Ping the remote server to check connectivity."""
        return await self._ping_server()

    async def get_status(self) -> StatusResponse:
        """Get the status of the remote server."""
        if self.status is None:
            self.status = await self._client.get_status()
        return self.status
