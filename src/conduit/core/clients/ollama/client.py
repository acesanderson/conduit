"""
Client subclass for Ollama models.
This doesn't require an API key since these are locally hosted models.
We can use openai api calls to the ollama server, but we use the instructor library to handle the API calls.
This has special logic for updating the models.json file, since the available Ollama models will depend on what we have pulled.
We define preferred defaults for context sizes in a separate json file.
"""

from __future__ import annotations
from conduit.config import settings
from conduit.core.clients.client_base import Client
from conduit.core.clients.payload_base import Payload
from conduit.core.clients.ollama.payload import OllamaPayload
from conduit.core.clients.ollama.message_adapter import convert_message_to_ollama
from conduit.core.clients.ollama.tool_adapter import convert_tool_to_ollama
from conduit.domain.result.response import GenerationResponse
from conduit.domain.result.response_metadata import ResponseMetadata, StopReason
from conduit.domain.message.message import AssistantMessage, ToolCall
from openai import AsyncOpenAI, AsyncStream
from collections import defaultdict
from typing import TYPE_CHECKING, override, Any
import instructor
from instructor import Instructor
import json
import logging
import time

if TYPE_CHECKING:
    from collections.abc import Sequence
    from conduit.domain.result.result import GenerationResult
    from conduit.domain.request.request import GenerationRequest
    from conduit.domain.message.message import Message

logger = logging.getLogger(__name__)


class OllamaClient(Client):
    """
    Client implementation for Ollama models using OpenAI-compatible endpoint.
    Async only.
    """

    def __init__(self):
        # 1. Load Ollama context sizes from the JSON file
        self._ollama_context_sizes = self._load_context_sizes()

        # 2. Initialize connection clients
        instructor_client, raw_client = self._initialize_client()
        self._client: Instructor = instructor_client
        self._raw_client: AsyncOpenAI = raw_client

        # 3. Refresh local model list
        self.update_ollama_models()

    def _load_context_sizes(self) -> defaultdict[str, int]:
        """
        Loads the context size mapping from disk.
        Defaults to 32,768 if the model or file is missing.
        """
        default_val = 32768
        path = settings.paths.get("OLLAMA_CONTEXT_SIZES_PATH")

        if path and path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                    logger.debug(f"Loaded Ollama context sizes from {path}")
                    return defaultdict(lambda: default_val, data)
            except Exception as e:
                logger.warning(f"Failed to parse context sizes file: {e}")

        return defaultdict(lambda: default_val)

    @override
    def _initialize_client(self) -> tuple[Instructor, AsyncOpenAI]:
        """
        Creates both raw and instructor-wrapped clients.
        Raw client for standard completions, Instructor for structured responses.
        """
        raw_client = AsyncOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required by SDK but unused by Ollama
        )
        instructor_client = instructor.from_openai(
            raw_client,
            mode=instructor.Mode.JSON,
        )
        return instructor_client, raw_client

    @override
    def _get_api_key(self) -> str:
        return ""

    @override
    def _convert_message(self, message: Message) -> dict[str, Any]:
        return convert_message_to_ollama(message)

    @override
    def _convert_request(self, request: GenerationRequest) -> Payload:
        """
        Translates Generic Request to Ollama-specific Payload.
        Injects num_ctx from local config to override Ollama/Library defaults.
        """
        model_name = request.params.model
        client_params = request.params.client_params or {}

        # --- CONTEXT WINDOW CUSTOMIZATION ---
        # Ensure we use our high-end hardware settings if not specified in the call
        if "num_ctx" not in client_params:
            client_params["num_ctx"] = self._ollama_context_sizes[model_name]
            logger.debug(
                f"Injected num_ctx={client_params['num_ctx']} for model {model_name}"
            )

        # Filter for supported Ollama options in extra_body
        allowed_params = {"num_ctx", "repeat_penalty", "seed", "top_k", "num_predict"}
        filtered_extra = {k: v for k, v in client_params.items() if k in allowed_params}

        return OllamaPayload(
            model=model_name,
            messages=self._convert_messages(request.messages),
            temperature=request.params.temperature,
            top_p=request.params.top_p,
            max_tokens=request.params.max_tokens,
            stream=request.params.stream,
            extra_body=filtered_extra if filtered_extra else None,
        )

    def update_ollama_models(self):
        """
        Syncs the local models.json with currently pulled Ollama models.
        """
        import ollama

        try:
            settings.paths["OLLAMA_MODELS_PATH"].parent.mkdir(
                parents=True, exist_ok=True
            )
            ollama_models = [m["model"] for m in ollama.list()["models"]]
            ollama_model_dict = {"ollama": ollama_models}
            with open(settings.paths["OLLAMA_MODELS_PATH"], "w") as f:
                json.dump(ollama_model_dict, f)
        except Exception as e:
            logger.error(f"Failed to update Ollama model list: {e}")

    @override
    async def tokenize(self, model: str, payload: str | Sequence[Message]) -> int:
        import ollama

        if isinstance(payload, str):
            response = ollama.generate(
                model=model, prompt=payload, options={"num_predict": 0}
            )
            return int(response.get("prompt_eval_count", 0))
        if isinstance(payload, list):
            messages_payload = [self._convert_message(m) for m in payload]
            response = ollama.chat(
                model=model, messages=messages_payload, options={"num_predict": 0}
            )
            return int(response.get("prompt_eval_count", 0))
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

        if isinstance(result, AsyncStream):
            return result

        # --- RESTORED EMPTY RESPONSE CHECK ---
        content = result.choices[0].message.content
        has_tool_calls = bool(getattr(result.choices[0].message, "tool_calls", None))

        if not content and not has_tool_calls:
            allocated_ctx = payload_dict.get("extra_body", {}).get("num_ctx", "unknown")
            error_msg = (
                f"Ollama model {result.model} returned an EMPTY response. "
                f"Likely cause: Input exceeded the allocated num_ctx ({allocated_ctx}) "
                f"causing silent truncation, or a local inference glitch."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

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
                        provider="ollama",
                        raw=tc.dict(),
                    )
                )

        assistant_message = AssistantMessage(
            content=content or "",
            tool_calls=tool_calls if tool_calls else None,
        )

        metadata = ResponseMetadata(
            duration=duration,
            model_slug=result.model,
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
            stop_reason=stop_reason,
        )

        return GenerationResponse(
            message=assistant_message, request=request, metadata=metadata
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

        # Fail-safe check for structured response
        if not user_obj and not completion.choices[0].message.content:
            allocated_ctx = payload_dict.get("extra_body", {}).get("num_ctx", "unknown")
            raise ValueError(
                f"Ollama structured response failed. num_ctx: {allocated_ctx}"
            )

        metadata = ResponseMetadata(
            duration=(time.time() - start_time) * 1000,
            model_slug=completion.model,
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
            stop_reason=StopReason.STOP,
        )

        assistant_message = AssistantMessage(
            content=completion.choices[0].message.content,
            parsed=user_obj,
        )

        return GenerationResponse(
            message=assistant_message, request=request, metadata=metadata
        )
