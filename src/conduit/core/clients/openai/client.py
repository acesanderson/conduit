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
from conduit.domain.message.message import (
    AssistantMessage,
    ImageOutput,
    ToolCall,
    Message,
)
from abc import ABC
from functools import cached_property
import logging
import os
import json
import time
import base64
from typing import TYPE_CHECKING, override, Any

if TYPE_CHECKING:
    from instructor import Instructor
    from openai import AsyncOpenAI, AsyncStream
    from conduit.domain.result.result import GenerationResult
    from conduit.domain.request.request import GenerationRequest

logger = logging.getLogger(__name__)


class OpenAIClient(Client, ABC):
    """
    Client implementation for OpenAI's API using the official OpenAI Python SDK and Instructor library.
    Async by default.
    """

    @cached_property
    def async_client(self) -> AsyncOpenAI:
        """
        Provides access to the raw AsyncOpenAI client for advanced use cases.
        """
        from openai import AsyncOpenAI

        async_client = AsyncOpenAI(api_key=self._get_api_key())
        return async_client

    @cached_property
    def instructor_client(self) -> Instructor:
        """
        Provides access to the Instructor client for advanced use cases.
        """
        import instructor

        instructor_client = instructor.from_openai(self.async_client)
        return instructor_client

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

        Estimations:
        - Images: 85 tokens (low detail) or 765 tokens (high/auto detail estimate).
        - Tool Calls: Counts function name + serialized JSON arguments.
        """
        encoding = self._get_encoding(model)

        # CASE 1: Raw String
        if isinstance(payload, str):
            return len(encoding.encode(payload))

        # CASE 2: Message History (ChatML)
        elif isinstance(payload, list):
            num_tokens = 0

            # Constants for ChatML (gpt-3.5-turbo-0613 / gpt-4)
            # <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_message = 3
            tokens_per_name = 1
            tokens_reply_primer = 3

            for message in payload:
                num_tokens += tokens_per_message

                # 1. Role
                # Handle Enum or string role
                role_str = (
                    message.role.value
                    if hasattr(message.role, "value")
                    else str(message.role)
                )
                num_tokens += len(encoding.encode(role_str))

                # 2. Content (Text or Multimodal)
                if message.content:
                    if isinstance(message.content, str):
                        num_tokens += len(encoding.encode(message.content))
                    elif isinstance(message.content, list):
                        for block in message.content:
                            # We use hasattr/getattr to support both Pydantic objects and dicts
                            block_type = getattr(block, "type", None)

                            # Text Block
                            if block_type == "text":
                                text = getattr(block, "text", "")
                                num_tokens += len(encoding.encode(text))

                            # Image Block (Estimation)
                            elif block_type == "image_url":
                                # Standard OpenAI Pricing:
                                # Low detail: 85 tokens
                                # High detail: 85 (base) + 170 per 512x512 tile.
                                # Without dimensions, we estimate a standard 1024x1024 (4 tiles) for High/Auto.
                                detail = getattr(block, "detail", "auto")
                                if detail == "low":
                                    num_tokens += 85
                                else:
                                    num_tokens += 765  # 85 + (4 * 170)

                # 3. Tool Calls (Assistant)
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for call in message.tool_calls:
                        # Function name
                        num_tokens += len(encoding.encode(call.function_name))
                        # Arguments (JSON string)
                        # We dump the dict to a string to estimate tokens
                        args_str = json.dumps(call.arguments)
                        num_tokens += len(encoding.encode(args_str))

                # 4. Name (Optional override)
                if hasattr(message, "name") and message.name:
                    num_tokens += tokens_per_name
                    num_tokens += len(encoding.encode(message.name))

            # Add reply primer: <|start|>assistant<|message|>
            num_tokens += tokens_reply_primer
            return num_tokens

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
            case "structured_response":
                return await self._generate_structured_response(request)
            case _:
                raise ValueError(
                    f"Unsupported output type: {request.params.output_type}"
                )

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
        result = await self.async_client.chat.completions.create(**payload_dict)

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
        response = await self.async_client.images.generate(
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
        response = await self.async_client.audio.speech.create(
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
        response = await self.async_client.audio.transcriptions.create(
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
        (
            user_obj,
            completion,
        ) = await self.instructor_client.chat.completions.create_with_completion(
            response_model=request.params.response_model, **payload_dict
        )

        # Assemble response metadata
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        model_stem = completion.model
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens

        # Determine stop reason
        stop_reason = StopReason.STOP
        if hasattr(completion.choices[0], "finish_reason"):
            finish_reason = completion.choices[0].finish_reason
            if finish_reason == "length":
                stop_reason = StopReason.LENGTH
            elif finish_reason == "tool_calls":
                stop_reason = StopReason.TOOL_CALLS
            elif finish_reason == "content_filter":
                stop_reason = StopReason.CONTENT_FILTER

        # Construct the ToolCall object
        type = "function"  # Structured response must use function calling
        function_name = completion.choices[0].message.tool_calls[0].function.name
        arguments = completion.choices[0].message.tool_calls[0].function.arguments
        arguments_dict = json.loads(arguments)

        tool_call = ToolCall(
            type=type, function_name=function_name, arguments=arguments_dict
        )

        # Create AssistantMessage with parsed structured data
        assistant_message = AssistantMessage(
            content=completion.choices[0].message.content,
            tool_calls=[tool_call],
            parsed=user_obj,
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
