"""
Client for llama.cpp's llama-server (OpenAI-compatible endpoint).

Key differences from OllamaClient:
- base_url is configurable via LLAMACPP_BASE_URL env var (default: localhost:8080)
- Supports GBNF grammar constraints via client_params["grammar"]
- Supports per-token logprobs with alternatives via client_params["top_logprobs"]
- Advanced sampling params (mirostat, min_p, tfs_z, etc.) go in extra_body
- No num_ctx injection: context size is fixed at server launch time
- No tokenize(): llama-server has no equivalent endpoint
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, override

import instructor
from instructor import Instructor
from openai import AsyncOpenAI, AsyncStream

from conduit.core.clients.client_base import Client
from conduit.core.clients.llamacpp.message_adapter import convert_message_to_llamacpp
from conduit.core.clients.llamacpp.payload import LlamaCppPayload
from conduit.core.clients.llamacpp.tool_adapter import convert_tool_to_llamacpp
from conduit.core.clients.payload_base import Payload
from conduit.domain.message.message import AssistantMessage, ToolCall
from conduit.domain.result.response import GenerationResponse
from conduit.domain.result.response_metadata import ResponseMetadata, StopReason

if TYPE_CHECKING:
    from collections.abc import Sequence
    from conduit.domain.message.message import Message
    from conduit.domain.request.request import GenerationRequest
    from conduit.domain.result.result import GenerationResult

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8080/v1"

# Params that belong in extra_body (llama.cpp-specific, not standard OpenAI)
_EXTRA_PARAMS: frozenset[str] = frozenset({
    "grammar",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "min_p",
    "tfs_z",
    "repeat_last_n",
    "repeat_penalty",
    "top_k",
})


class LlamaCppClient(Client):
    """
    Client for llama-server's OpenAI-compatible API.

    Configure the endpoint via the LLAMACPP_BASE_URL environment variable.
    Advanced sampling and GBNF grammar are passed through client_params.
    """

    def __init__(self, base_url: str | None = None):
        self._base_url = base_url or os.environ.get("LLAMACPP_BASE_URL", DEFAULT_BASE_URL)
        instructor_client, raw_client = self._initialize_client()
        self._client: Instructor = instructor_client
        self._raw_client: AsyncOpenAI = raw_client

    @override
    def _initialize_client(self) -> tuple[Instructor, AsyncOpenAI]:
        raw_client = AsyncOpenAI(
            base_url=self._base_url,
            api_key="llama.cpp",
        )
        instructor_client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
        return instructor_client, raw_client

    @override
    def _get_api_key(self) -> str:
        return ""

    @override
    def _convert_message(self, message: Message) -> dict[str, Any]:
        return convert_message_to_llamacpp(message)

    @override
    def _convert_request(self, request: GenerationRequest) -> Payload:
        client_params = request.params.client_params or {}

        # top_logprobs goes on the payload directly; logprobs must be True to enable it
        top_logprobs = client_params.get("top_logprobs")
        logprobs = True if top_logprobs is not None else None

        # llama.cpp-specific params go in extra_body (merged into request root by SDK)
        extra = {k: v for k, v in client_params.items() if k in _EXTRA_PARAMS}

        response_format = None
        if (
            request.params.output_type == "structured_response"
            and request.params.response_model is None
            and request.params.response_model_schema is not None
        ):
            schema = request.params.response_model_schema
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("title", "Response"),
                    "schema": schema,
                    "strict": True,
                },
            }

        return LlamaCppPayload(
            model=request.params.model,
            messages=self._convert_messages(request.messages),
            temperature=request.params.temperature,
            top_p=request.params.top_p,
            max_tokens=request.params.max_tokens,
            stream=request.params.stream,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            extra_body=extra if extra else None,
            response_format=response_format,
        )

    @override
    async def query(self, request: GenerationRequest) -> GenerationResult:
        match request.params.output_type:
            case "text":
                return await self._generate_text(request)
            case "structured_response":
                return await self._generate_structured_response(request)
            case _:
                raise ValueError(f"Unsupported output type: {request.params.output_type}")

    async def _generate_text(self, request: GenerationRequest) -> GenerationResult:
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)

        start_time = time.time()
        result = await self._raw_client.chat.completions.create(**payload_dict)

        if isinstance(result, AsyncStream):
            return result

        content = result.choices[0].message.content
        has_tool_calls = bool(getattr(result.choices[0].message, "tool_calls", None))

        if not content and not has_tool_calls:
            max_tokens = payload_dict.get("max_tokens")
            if max_tokens is not None:
                duration = (time.time() - start_time) * 1000
                metadata = ResponseMetadata(
                    duration=duration,
                    model_slug=result.model,
                    input_tokens=result.usage.prompt_tokens,
                    output_tokens=result.usage.completion_tokens,
                    stop_reason=StopReason.LENGTH,
                    cache_hit=False,
                )
                return GenerationResponse(
                    message=AssistantMessage(content=" "),
                    request=request,
                    metadata=metadata,
                )
            else:
                grammar = (payload_dict.get("extra_body") or {}).get("grammar")
                raise ValueError(
                    f"llama-server returned an empty response for model {result.model}. "
                    f"Likely cause: input exceeded context window, or a grammar constraint "
                    f"({grammar!r}) blocked all valid continuations."
                )

        duration = (time.time() - start_time) * 1000
        stop_reason = StopReason.STOP
        if hasattr(result.choices[0], "finish_reason"):
            finish_reason = result.choices[0].finish_reason
            if finish_reason == "length":
                stop_reason = StopReason.LENGTH
            elif finish_reason == "tool_calls":
                stop_reason = StopReason.TOOL_CALLS

        tool_calls = []
        if stop_reason == StopReason.TOOL_CALLS:
            for tc in result.choices[0].message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        type="function",
                        function_name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                        provider="llamacpp",
                        raw=tc.dict(),
                    )
                )

        # Extract logprobs: each position includes the sampled token and top alternatives.
        # Format: [{"token": str, "logprob": float, "top": [{"token": str, "logprob": float}, ...]}, ...]
        raw_logprobs: list[dict] | None = None
        choice_logprobs = getattr(result.choices[0], "logprobs", None)
        if choice_logprobs and getattr(choice_logprobs, "content", None):
            raw_logprobs = [
                {
                    "token": t.token,
                    "logprob": t.logprob,
                    "top": [
                        {"token": alt.token, "logprob": alt.logprob}
                        for alt in (t.top_logprobs or [])
                    ],
                }
                for t in choice_logprobs.content
            ]

        metadata = ResponseMetadata(
            duration=duration,
            model_slug=result.model,
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
            stop_reason=stop_reason,
            logprobs=raw_logprobs,
        )

        return GenerationResponse(
            message=AssistantMessage(
                content=content or "",
                tool_calls=tool_calls if tool_calls else None,
            ),
            request=request,
            metadata=metadata,
        )

    async def _generate_structured_response(
        self, request: GenerationRequest
    ) -> GenerationResponse:
        if request.params.response_model is not None:
            payload = self._convert_request(request)
            payload_dict = payload.model_dump(exclude_none=True)

            start_time = time.time()
            user_obj, completion = (
                await self._client.chat.completions.create_with_completion(
                    response_model=request.params.response_model, **payload_dict
                )
            )

            if not user_obj and not completion.choices[0].message.content:
                raise ValueError(
                    f"llama-server structured response failed for model {request.params.model}."
                )

            return GenerationResponse(
                message=AssistantMessage(
                    content=completion.choices[0].message.content,
                    parsed=user_obj,
                ),
                request=request,
                metadata=ResponseMetadata(
                    duration=(time.time() - start_time) * 1000,
                    model_slug=completion.model,
                    input_tokens=completion.usage.prompt_tokens,
                    output_tokens=completion.usage.completion_tokens,
                    stop_reason=StopReason.STOP,
                ),
            )

        if request.params.response_model_schema is None:
            raise ValueError(
                "structured_response requires response_model or response_model_schema"
            )

        # json_schema path: llama-server converts to GBNF internally
        payload = self._convert_request(request)
        payload_dict = payload.model_dump(exclude_none=True)

        start_time = time.time()
        result = await self._raw_client.chat.completions.create(**payload_dict)

        content = result.choices[0].message.content
        if not content:
            raise ValueError(
                f"llama-server schema path: empty content for model {result.model}"
            )

        return GenerationResponse(
            message=AssistantMessage(content=content, parsed=None),
            request=request,
            metadata=ResponseMetadata(
                duration=(time.time() - start_time) * 1000,
                model_slug=result.model,
                input_tokens=result.usage.prompt_tokens,
                output_tokens=result.usage.completion_tokens,
                stop_reason=StopReason.STOP,
            ),
        )
