from __future__ import annotations
from conduit.core.clients.client_base import Client
from conduit.core.clients.payload_base import Payload
from conduit.core.clients.openai.payload import OpenAIPayload
from conduit.core.clients.openai.adapter import convert_message_to_openai
from conduit.core.clients.openai.audio_params import OpenAIAudioParams
from conduit.core.clients.openai.image_params import OpenAIImageParams
from conduit.core.clients.openai.transcription_params import OpenAITranscriptionParams
from conduit.domain.result.response import GenerationResponse
from conduit.domain.result.response_metadata import ResponseMetadata, StopReason
from conduit.domain.message.message import AssistantMessage, ImageOutput
from openai import AsyncOpenAI
from openai import AsyncStream
from abc import ABC
import instructor
from instructor import Instructor
import logging
import os
import time
import base64
from typing import TYPE_CHECKING, override, Any

if TYPE_CHECKING:
    from conduit.domain.result.result import GenerationResult
    from conduit.core.parser.stream.protocol import AsyncStream
    from conduit.domain.request.request import GenerationRequest
    from conduit.domain.message.message import Message

logger = logging.getLogger(__name__)


class OpenAIClient(Client, ABC):
    """
    Client implementation for OpenAI's API using the official OpenAI Python SDK and Instructor library.
    Async by default.
    """

    # Initialize the OpenAI client
    def __init__(self):
        instructor_client, raw_client = self._initialize_client()
        self._client: Instructor = instructor_client
        self._raw_client: AsyncOpenAI = raw_client

    @override
    def _initialize_client(self) -> tuple[Instructor, AsyncOpenAI]:
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        raw_client = AsyncOpenAI(api_key=self._get_api_key())
        instructor_client = instructor.from_openai(raw_client)
        return instructor_client, raw_client

    @override
    def _get_api_key(self) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        return api_key

    # Convert internal Message DTOs to OpenAI format
    @override
    def _convert_message(self, message: Message) -> dict[str, Any]:
        """
        Converts a single internal Message DTO into OpenAI's specific dictionary format.
        """
        return convert_message_to_openai(message)

    @override
    def _convert_request(self, request: GenerationRequest) -> Payload:
        """
        Translates the internal generic Request DTO into the specific
        dictionary parameters required by OpenAI's SDK.
        """
        converted_messages = self._convert_messages(request.messages)
        openai_payload = OpenAIPayload(
            model=request.params.model,
            messages=converted_messages,
            temperature=request.params.temperature,
            top_p=request.params.top_p,
            max_tokens=request.params.max_tokens,
            stream=request.params.stream,
        )
        return openai_payload

    # Tokenization
    @override
    async def tokenize(self, model: str, payload: str | list[Message]) -> int:
        """
        Return the token count for a string or a message list.
        For list[Message], it calculates the overhead per OpenAI ChatML format.
        """
        encoding = self._get_encoding(model)

        if isinstance(payload, str):
            return len(encoding.encode(payload))

        elif isinstance(payload, list):
            # Lazy import to avoid circular dependency
            raise NotImplementedError(
                "Tokenization for list[Message] is not implemented yet."
            )
        raise ValueError("Payload must be string or list[Message]")

    def _get_encoding(self, model: str):
        import tiktoken

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def discover_models(self) -> list[str]:
        import openai

        API_KEY = self._get_api_key()
        openai.api_key = API_KEY
        models = openai.models.list()

        return [model.id for model in models.data]

    # Query methods
    @override
    async def query(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        match request.params.output_type:
            case "text":
                return await self._generate_text(request)
            case "image":
                return await self._generate_image(request)
            case "audio":
                return await self._generate_audio(request)
            case "transcription":
                return await self._transcribe_audio(request)
            case "structured":
                return await self._generate_structured_response(request)
            case _:
                raise ValueError(f"Unsupported output type: {request.output_type}")

    async def _generate_text(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate text using OpenAI's API and return a ConduitResult.

        Returns:
            - Response object for successful non-streaming requests
            - AsyncStream object for streaming requests
        """
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)

        # Track timing
        start_time = time.time()

        # Use the standard completion method
        result = await self._raw_client.chat.completions.create(**payload_dict)

        # Handle streaming response
        if isinstance(result, AsyncStream):
            # For streaming, return the AsyncStream object directly (it's part of ConduitResult)
            return result

        # Assemble response metadata
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        model_stem = result.model
        input_tokens = result.usage.prompt_tokens
        output_tokens = result.usage.completion_tokens

        # Determine stop reason
        stop_reason = StopReason.STOP
        if hasattr(result.choices[0], "finish_reason"):
            finish_reason = result.choices[0].finish_reason
            if finish_reason == "length":
                stop_reason = StopReason.LENGTH
            elif finish_reason == "tool_calls":
                stop_reason = StopReason.TOOL_CALLS
            elif finish_reason == "content_filter":
                stop_reason = StopReason.CONTENT_FILTER

        # Extract the text content
        content = result.choices[0].message.content
        assistant_message = AssistantMessage(content=content)

        # Create ResponseMetadata
        metadata = ResponseMetadata(
            duration=duration,
            model_slug=model_stem,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
        )

        # Create and return Response
        return GenerationResponse(
            message=assistant_message,
            request=request,
            metadata=metadata,
        )

    async def _generate_image(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate an image using OpenAI's DALL-E and return a ConduitResult.

        Returns:
            - Response object with ImageOutput in AssistantMessage.images
        """
        start_time = time.time()

        # Extract text prompt from the last message
        last_message = request.messages[-1]
        if isinstance(last_message.content, str):
            prompt = last_message.content
        else:
            # Handle multimodal content - extract text
            prompt = " ".join(
                [block.text for block in last_message.content if hasattr(block, "text")]
            )

        # Get image parameters (use defaults if not provided)
        image_params = OpenAIImageParams()
        if (
            request.params.client_params
            and "image_params" in request.params.client_params
        ):
            image_params = request.params.client_params["image_params"]

        # Call the images.generate endpoint
        response = await self._raw_client.images.generate(
            model=image_params.model.value,
            prompt=prompt,
            size=image_params.size.value,
            quality=image_params.quality.value,
            style=image_params.style.value,
            response_format=image_params.response_format.value,
            n=image_params.n,
        )

        duration = (time.time() - start_time) * 1000

        # Convert response to ImageOutput objects
        image_outputs = []
        for image_data in response.data:
            image_output = ImageOutput(
                url=image_data.url if hasattr(image_data, "url") else None,
                b64_json=image_data.b64_json
                if hasattr(image_data, "b64_json")
                else None,
                revised_prompt=image_data.revised_prompt
                if hasattr(image_data, "revised_prompt")
                else None,
            )
            image_outputs.append(image_output)

        # Create AssistantMessage with images
        assistant_message = AssistantMessage(images=image_outputs)

        # Create ResponseMetadata (DALL-E doesn't provide token counts)
        metadata = ResponseMetadata(
            duration=duration,
            model_slug=image_params.model.value,
            input_tokens=0,  # DALL-E doesn't provide token counts
            output_tokens=0,
            stop_reason=StopReason.STOP,
        )

        # Create and return Response
        return GenerationResponse(
            message=assistant_message,
            request=request,
            metadata=metadata,
        )

    async def _generate_audio(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate audio using OpenAI's TTS API and return a ConduitResult.

        Returns:
            - Response object with base64-encoded audio data
        """
        start_time = time.time()

        # Extract text from the last message
        last_message = request.messages[-1]
        if isinstance(last_message.content, str):
            text_input = last_message.content
        else:
            # Handle multimodal content - extract text
            text_input = " ".join(
                [block.text for block in last_message.content if hasattr(block, "text")]
            )

        # Get audio parameters (use defaults if not provided)
        audio_params = OpenAIAudioParams()
        if (
            request.params.client_params
            and "audio_params" in request.params.client_params
        ):
            audio_params = request.params.client_params["audio_params"]

        # Call the audio.speech.create endpoint
        response = await self._raw_client.audio.speech.create(
            model=audio_params.model.value,
            voice=audio_params.voice.value,
            input=text_input,
            response_format=audio_params.response_format.value,
            speed=audio_params.speed,
        )

        duration = (time.time() - start_time) * 1000

        # Convert audio bytes to base64
        audio_bytes = response.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Create AssistantMessage with audio data
        assistant_message = AssistantMessage(content=audio_base64)

        # Create ResponseMetadata (TTS doesn't provide token counts)
        metadata = ResponseMetadata(
            duration=duration,
            model_slug=audio_params.model.value,
            input_tokens=0,  # TTS doesn't provide token counts
            output_tokens=0,
            stop_reason=StopReason.STOP,
        )

        # Create and return Response
        return GenerationResponse(
            message=assistant_message,
            request=request,
            metadata=metadata,
        )

    async def _transcribe_audio(self, request: GenerationRequest) -> GenerationResponse:
        """
        Transcribe audio using OpenAI's Whisper API and return a ConduitResult.

        Returns:
            - Response object with transcription text in AssistantMessage.content
        """
        start_time = time.time()

        # Extract audio data from the message
        last_message = request.messages[-1]
        audio_data = None
        audio_format = "mp3"

        if isinstance(last_message.content, list):
            # Find AudioContent in the message
            for block in last_message.content:
                if hasattr(block, "data") and hasattr(block, "format"):
                    audio_data = block.data
                    audio_format = block.format
                    break

        if not audio_data:
            raise ValueError("No audio content found in message")

        # Get transcription parameters (use defaults if not provided)
        transcription_params = OpenAITranscriptionParams()
        if (
            request.params.client_params
            and "transcription_params" in request.params.client_params
        ):
            transcription_params = request.params.client_params["transcription_params"]

        # Convert base64 audio to bytes
        import io

        audio_bytes = base64.b64decode(audio_data)
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = f"audio.{audio_format}"

        # Call the audio.transcriptions.create endpoint
        response = await self._raw_client.audio.transcriptions.create(
            model=transcription_params.model.value,
            file=audio_file,
            language=transcription_params.language,
            prompt=transcription_params.prompt,
            response_format=transcription_params.response_format.value,
            temperature=transcription_params.temperature,
        )

        duration = (time.time() - start_time) * 1000

        # Extract transcription text
        if isinstance(response, str):
            transcription_text = response
        else:
            # For JSON format, extract text field
            transcription_text = response.text

        # Create AssistantMessage with transcription
        assistant_message = AssistantMessage(content=transcription_text)

        # Create ResponseMetadata (Whisper doesn't provide token counts)
        metadata = ResponseMetadata(
            duration=duration,
            model_slug=transcription_params.model.value,
            input_tokens=0,  # Whisper doesn't provide token counts
            output_tokens=0,
            stop_reason=StopReason.STOP,
        )

        # Create and return Response
        return GenerationResponse(
            message=assistant_message,
            request=request,
            metadata=metadata,
        )

    async def _generate_structured_response(
        self, request: GenerationRequest
    ) -> GenerationResponse:
        """
        Generate a structured response using OpenAI's function calling and return a ConduitResult.

        Returns:
            - Response object with parsed structured data in AssistantMessage.parsed
        """
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)

        # Track timing
        start_time = time.time()

        # Make the API call with function calling
        result = await self._client.chat.completions.create(
            response_model=request.params.response_model, **payload_dict
        )

        # Handle streaming response
        if isinstance(result, AsyncStream):
            # For streaming, return the AsyncStream object directly (it's part of ConduitResult)
            return result

        # Assemble response metadata
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        model_stem = result.model
        input_tokens = result.usage.prompt_tokens
        output_tokens = result.usage.completion_tokens

        # Determine stop reason
        stop_reason = StopReason.STOP
        if hasattr(result.choices[0], "finish_reason"):
            finish_reason = result.choices[0].finish_reason
            if finish_reason == "length":
                stop_reason = StopReason.LENGTH
            elif finish_reason == "tool_calls":
                stop_reason = StopReason.TOOL_CALLS
            elif finish_reason == "content_filter":
                stop_reason = StopReason.CONTENT_FILTER

        # Extract structured response from the function call
        function_call = result.choices[0].message.function_call
        structured_response = function_call.arguments if function_call else None

        # Create AssistantMessage with parsed structured data
        assistant_message = AssistantMessage(
            content=None,  # No text content for structured responses
            parsed=structured_response,
        )

        # Create ResponseMetadata
        metadata = ResponseMetadata(
            duration=duration,
            model_slug=model_stem,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
        )

        # Create and return Response
        return GenerationResponse(
            message=assistant_message,
            request=request,
            metadata=metadata,
        )


if __name__ == "__main__":
    client = OpenAIClient()
    # Generate text
    import asyncio
    from conduit.domain.request.request import GenerationRequest
    from conduit.domain.request.generation_params import GenerationParams
    from conduit.domain.message.message import UserMessage

    async def test_text_generation():
        request = GenerationRequest(
            messages=[UserMessage(content="Hello, how are you?")],
            params=GenerationParams(model="gpt-4o", temperature=0.7, max_tokens=100),
        )
        response = await client.query(request)
        print("Text Generation Response:", response)

    asyncio.run(test_text_generation())
