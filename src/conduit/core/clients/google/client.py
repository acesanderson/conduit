"""
For Google Gemini models.
"""

from __future__ import annotations
from functools import cached_property
from conduit.core.clients.client_base import Client
from conduit.core.clients.payload_base import Payload
from conduit.core.clients.google.payload import GooglePayload
from conduit.core.clients.google.message_adapter import convert_message_to_google
from conduit.core.clients.google.tool_adapter import convert_tool_to_google
from conduit.core.clients.google.image_params import GoogleImageParams
from conduit.core.clients.google.audio_params import GoogleAudioParams
from conduit.domain.result.response import GenerationResponse
from conduit.domain.result.response_metadata import ResponseMetadata, StopReason
from conduit.domain.message.message import AssistantMessage, ImageOutput, ToolCall
from typing import TYPE_CHECKING, override, Any
import json
import os
import time
import base64

if TYPE_CHECKING:
    from collections.abc import Sequence
    from openai import AsyncOpenAI
    from instructor import Instructor
    from conduit.domain.result.result import GenerationResult
    from conduit.domain.request.request import GenerationRequest
    from conduit.domain.message.message import Message


class GoogleClient(Client):
    """
    Client implementation for Google's Gemini API using the OpenAI-compatible endpoint.
    Async by default.
    """

    @cached_property
    def async_client(self) -> AsyncOpenAI:
        """
        Exposes the raw AsyncOpenAI client for direct use if needed.
        """
        from openai import AsyncOpenAI

        async_client = AsyncOpenAI(
            api_key=self._get_api_key(),
            base_url="https://generativelanguage.googleapis.com/v1beta/",
        )
        return async_client

    @cached_property
    def instructor_client(self) -> Instructor:
        """
        Exposes the Instructor-wrapped client for structured responses.
        """
        import instructor

        instructor_client = instructor.from_openai(
            self.async_client,
            mode=instructor.Mode.JSON,
        )
        return instructor_client

    def _get_api_key(self) -> str:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        return api_key

    @override
    def _convert_message(self, message: Message) -> dict[str, Any]:
        """
        Converts a single internal Message DTO into Google's specific dictionary format.
        Since Google uses OpenAI spec, we delegate to the OpenAI adapter.
        """
        return convert_message_to_google(message)

    @override
    def _convert_request(self, request: GenerationRequest) -> Payload:
        """
        Translates the internal generic Request DTO into the specific
        dictionary parameters required by Google's SDK (via OpenAI spec).
        """
        # Load client params
        client_params = request.params.client_params or {}
        allowed_params = {"frequency_penalty", "presence_penalty"}
        for param in client_params.keys():
            if param not in allowed_params:
                raise ValueError(f"Unsupported Google client parameter: {param}")
        # Convert messages
        converted_messages = self._convert_messages(request.messages)

        # Convert tools and enable parallel tool calls if tools are present
        tools = None
        parallel_tool_calls = None
        if request.options.tool_registry:
            tools = [
                convert_tool_to_google(tool)
                for tool in request.options.tool_registry.tools
            ]
            parallel_tool_calls = request.options.parallel_tool_calls

        # Build payload
        google_payload = GooglePayload(
            model=request.params.model,
            messages=converted_messages,
            temperature=request.params.temperature,
            top_p=request.params.top_p,
            max_tokens=request.params.max_tokens,
            stream=request.params.stream,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
            # Google-specific params
            **client_params,
        )
        return google_payload

    @override
    async def tokenize(self, model: str, payload: str | Sequence[Message]) -> int:
        """
        Get the token count per official tokenizer (through Google Native API).
        We use the google.genai SDK for this because Gemini tokens != tiktoken.
        """
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._get_api_key())

        # CASE 1: Raw String
        if isinstance(payload, str):
            response = await client.aio.models.count_tokens(
                model=model, contents=payload
            )
            return response.total_tokens

        # CASE 2: Message History
        if isinstance(payload, list):
            native_contents = []

            for msg in payload:
                role = "model" if msg.role == "assistant" else "user"

                text = None
                if hasattr(msg, "text_content") and msg.text_content:
                    text = msg.text_content
                elif isinstance(msg.content, str):
                    text = msg.content

                if text:
                    native_contents.append(
                        types.Content(role=role, parts=[types.Part(text=text)])
                    )

            if not native_contents:
                return 0

            response = await client.aio.models.count_tokens(
                model=model, contents=native_contents
            )
            return response.total_tokens

        raise ValueError("Payload must be string or Sequence[Message]")

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
            case "structured_response":
                return await self._generate_structured_response(request)
            case _:
                raise ValueError(
                    f"Unsupported output type: {request.params.output_type}"
                )

    async def _generate_text(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate text using Google's Gemini API and return a GenerationResponse.

        Returns:
            - GenerationResponse object for successful non-streaming requests
            - AsyncStream object for streaming requests
        """
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)

        # Track timing
        start_time = time.time()

        # Use the raw client for standard completions
        result = await self.async_client.chat.completions.create(**payload_dict)

        from openai import AsyncStream

        # Handle streaming response
        if isinstance(result, AsyncStream):
            # For streaming, return the AsyncStream object directly
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
            if finish_reason == "tool_calls":
                stop_reason = StopReason.TOOL_CALLS
            elif finish_reason == "content_filter":
                stop_reason = StopReason.CONTENT_FILTER

        # Process tool calls if present
        if stop_reason == StopReason.TOOL_CALLS:
            # Handle tool calls - iterate through all parallel calls
            tool_calls = []
            for tool_call_data in result.choices[0].message.tool_calls:
                arguments_dict = json.loads(tool_call_data.function.arguments)

                tool_call = ToolCall(
                    id=tool_call_data.id,  # Use provider-supplied ID
                    type="function",
                    function_name=tool_call_data.function.name,
                    arguments=arguments_dict,
                    provider="google",
                    raw=tool_call_data.dict(),
                )
                tool_calls.append(tool_call)

            # Create AssistantMessage with all tool calls
            assistant_message = AssistantMessage(
                content=result.choices[0].message.content
                if hasattr(result.choices[0].message, "content")
                else "",
                tool_calls=tool_calls,
            )
        else:
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

    async def _generate_structured_response(
        self, request: GenerationRequest
    ) -> GenerationResponse:
        """
        Generate a structured response using Google's function calling and return a GenerationResponse.

        Returns:
            - GenerationResponse object with parsed structured data in AssistantMessage.parsed
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

        # Create AssistantMessage with parsed structured data
        assistant_message = AssistantMessage(
            content=completion.choices[0].message.content,
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

    async def _generate_image(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate an image via the Gemini API using generate_content.
        """
        start_time = time.time()

        last_message = request.messages[-1]
        if isinstance(last_message.content, str):
            prompt = last_message.content
        else:
            prompt = " ".join(
                [block.text for block in last_message.content if hasattr(block, "text")]
            )

        from google import genai
        from google.genai import types

        native_client = genai.Client(api_key=self._get_api_key())
        model_name = request.params.model

        # Define the most permissive safety settings available
        safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_NONE"
            ),
        ]

        response = await native_client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                safety_settings=safety_settings,
            ),
        )

        duration = (time.time() - start_time) * 1000

        # --- FIX: Guard against Safety Refusals or Empty Content ---
        if not response.candidates or not response.candidates[0].content:
            finish_reason = "UNKNOWN"
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason

            raise ValueError(
                f"Google Gemini refused to generate content. Finish Reason: {finish_reason}. "
                "This usually happens due to safety redlines that cannot be disabled."
            )
        # -----------------------------------------------------------

        image_outputs = []
        for part in response.candidates[0].content.parts:
            if getattr(part, "inline_data", None):
                b64_data = base64.b64encode(part.inline_data.data).decode("utf-8")
                image_outputs.append(ImageOutput(b64_json=b64_data))

        assistant_message = AssistantMessage(images=image_outputs)
        metadata = ResponseMetadata(
            duration=duration,
            model_slug=model_name,
            input_tokens=0,
            output_tokens=0,
            stop_reason=StopReason.STOP,
        )
        return GenerationResponse(
            message=assistant_message,
            request=request,
            metadata=metadata,
        )

    async def _generate_audio(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate audio using Google's TTS API and return a GenerationResponse.

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
        audio_params = GoogleAudioParams()
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
