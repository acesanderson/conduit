from __future__ import annotations
from conduit.core.clients.client_base import Client
from conduit.core.clients.payload_base import Payload
from conduit.core.clients.mistral.payload import MistralPayload
from conduit.core.clients.mistral.message_adapter import convert_message_to_mistral
from conduit.core.clients.mistral.tool_adapter import convert_tool_to_mistral
from conduit.domain.result.response import GenerationResponse
from conduit.domain.result.response_metadata import ResponseMetadata, StopReason
from conduit.domain.message.message import AssistantMessage, ToolCall
from typing import TYPE_CHECKING, override, Any
import os
import time
import json

if TYPE_CHECKING:
    from collections.abc import Sequence
    from openai import AsyncOpenAI, AsyncStream
    from instructor import Instructor
    from conduit.domain.result.result import GenerationResult
    from conduit.domain.request.request import GenerationRequest
    from conduit.domain.message.message import Message


class MistralClient(Client):
    """
    Client for Mistral AI API using the OpenAI-compatible endpoint.
    Async only.
    """

    def __init__(self):
        instructor_client, raw_client = self._initialize_clients()
        self._client: Instructor = instructor_client
        self._raw_client: AsyncOpenAI = raw_client

    def _initialize_clients(self):
        from openai import AsyncOpenAI
        import instructor

        raw_client = AsyncOpenAI(
            api_key=self._get_api_key(),
            base_url="https://api.mistral.ai/v1",
        )
        instructor_client = instructor.from_openai(raw_client)
        return instructor_client, raw_client

    def _get_api_key(self) -> str:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set.")
        return api_key

    @override
    def _convert_message(self, message: Message) -> dict[str, Any]:
        return convert_message_to_mistral(message)

    @override
    def _convert_request(self, request: GenerationRequest) -> Payload:
        converted_messages = self._convert_messages(request.messages)

        tools = []
        if request.options.tool_registry:
            tools.extend(
                [
                    convert_tool_to_mistral(tool)
                    for tool in request.options.tool_registry.tools
                ]
            )

        parallel_tool_calls = request.options.parallel_tool_calls if tools else None
        final_tools = tools if tools else None

        client_params = request.params.client_params or {}
        allowed_params = {"safe_prompt", "random_seed", "prompt_mode", "prediction"}
        for param in client_params:
            if param not in allowed_params:
                raise ValueError(
                    f"Invalid client_param '{param}' for MistralClient. Allowed: {allowed_params}"
                )

        return MistralPayload(
            model=request.params.model,
            messages=converted_messages,
            temperature=request.params.temperature,
            top_p=request.params.top_p,
            max_tokens=request.params.max_tokens,
            stream=request.params.stream,
            tools=final_tools,
            parallel_tool_calls=parallel_tool_calls,
            **client_params,
        )

    @override
    async def tokenize(self, model: str, payload: str | Sequence[Message]) -> int:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")

        if isinstance(payload, str):
            return len(encoding.encode(payload))

        if isinstance(payload, list):
            tokens_per_message = 3
            num_tokens = 0
            for message in payload:
                num_tokens += tokens_per_message
                num_tokens += len(encoding.encode(message.role.value))
                if isinstance(message.content, str):
                    num_tokens += len(encoding.encode(message.content))
                elif isinstance(message.content, list):
                    for block in message.content:
                        if hasattr(block, "text"):
                            num_tokens += len(encoding.encode(block.text))
            num_tokens += 3  # reply primer
            return num_tokens

        raise ValueError("Payload must be string or Sequence[Message]")

    @override
    async def query(self, request: GenerationRequest) -> GenerationResult:
        match request.params.output_type:
            case "text":
                return await self._generate_text(request)
            case "structured_response":
                return await self._generate_structured_response(request)
            case _:
                raise ValueError(
                    f"Unsupported output type: {request.params.output_type}"
                )

    async def _generate_text(self, request: GenerationRequest) -> GenerationResult:
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)

        start_time = time.time()
        result = await self._raw_client.chat.completions.create(**payload_dict)

        from openai import AsyncStream

        if isinstance(result, AsyncStream):
            return result

        duration = (time.time() - start_time) * 1000
        model_stem = result.model
        input_tokens = result.usage.prompt_tokens
        output_tokens = result.usage.completion_tokens

        stop_reason = StopReason.STOP
        if hasattr(result.choices[0], "finish_reason"):
            finish_reason = result.choices[0].finish_reason
            if finish_reason == "length":
                stop_reason = StopReason.LENGTH
            elif finish_reason == "tool_calls":
                stop_reason = StopReason.TOOL_CALLS
            elif finish_reason == "content_filter":
                stop_reason = StopReason.CONTENT_FILTER

        if stop_reason == StopReason.TOOL_CALLS:
            tool_calls = []
            for tc in result.choices[0].message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        type="function",
                        function_name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                        provider="mistral",
                        raw=tc.dict(),
                    )
                )
            assistant_message = AssistantMessage(
                content=result.choices[0].message.content or "",
                tool_calls=tool_calls,
            )
        else:
            assistant_message = AssistantMessage(
                content=result.choices[0].message.content
            )

        metadata = ResponseMetadata(
            duration=duration,
            model_slug=model_stem,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
        )

        return GenerationResponse(
            message=assistant_message,
            request=request,
            metadata=metadata,
        )

    async def _generate_structured_response(
        self, request: GenerationRequest
    ) -> GenerationResponse:
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)

        start_time = time.time()
        (
            user_obj,
            completion,
        ) = await self._client.chat.completions.create_with_completion(
            response_model=request.params.response_model, **payload_dict
        )

        duration = (time.time() - start_time) * 1000
        model_stem = completion.model
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens

        stop_reason = StopReason.STOP
        if hasattr(completion.choices[0], "finish_reason"):
            finish_reason = completion.choices[0].finish_reason
            if finish_reason == "length":
                stop_reason = StopReason.LENGTH
            elif finish_reason == "tool_calls":
                stop_reason = StopReason.TOOL_CALLS

        function_name = completion.choices[0].message.tool_calls[0].function.name
        arguments = json.loads(
            completion.choices[0].message.tool_calls[0].function.arguments
        )
        tool_call = ToolCall(
            type="function", function_name=function_name, arguments=arguments
        )

        assistant_message = AssistantMessage(
            content=completion.choices[0].message.content,
            tool_calls=[tool_call],
            parsed=user_obj,
        )

        metadata = ResponseMetadata(
            duration=duration,
            model_slug=model_stem,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
        )

        return GenerationResponse(
            message=assistant_message,
            request=request,
            metadata=metadata,
        )
