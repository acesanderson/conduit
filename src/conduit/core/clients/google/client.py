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
from conduit.domain.message.role import Role
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
        client_params = request.params.client_params or {}
        match request.params.output_type:
            case "text":
                if client_params.get("deep_research"):
                    return await self._generate_deep_research(request)
                if client_params.get("return_citations"):
                    return await self._generate_grounded_text(request)
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

    async def _generate_grounded_text(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text with Google Search grounding via the native google.genai SDK.
        Citations are stored in AssistantMessage.metadata["citations"] (same shape as Perplexity).
        Triggered when client_params contains return_citations=True.
        """
        from google import genai
        from google.genai import types

        start_time = time.time()

        native_client = genai.Client(api_key=self._get_api_key())
        model_name = request.params.model

        contents = []
        system_instruction = None
        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                system_instruction = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                continue
            role = "model" if msg.role == Role.ASSISTANT else "user"
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))

        config_kwargs: dict[str, Any] = {
            "tools": [types.Tool(google_search=types.GoogleSearch())],
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if request.params.temperature is not None:
            config_kwargs["temperature"] = request.params.temperature
        if request.params.max_tokens is not None:
            config_kwargs["max_output_tokens"] = request.params.max_tokens

        response = await native_client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        duration = (time.time() - start_time) * 1000

        text_content = ""
        if response.candidates and response.candidates[0].content:
            parts = response.candidates[0].content.parts or []
            text_content = "".join(
                part.text
                for part in parts
                if hasattr(part, "text") and part.text
            )

        citations: list[dict[str, str]] = []
        if response.candidates:
            grounding = getattr(response.candidates[0], "grounding_metadata", None)
            if grounding:
                for chunk in getattr(grounding, "grounding_chunks", []):
                    web = getattr(chunk, "web", None)
                    if web:
                        citations.append({
                            "title": getattr(web, "title", ""),
                            "url": getattr(web, "uri", ""),
                            "source": "",
                            "date": "",
                        })

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        assistant_message = AssistantMessage(
            content=text_content or None,
            metadata={"citations": citations, "provider": "google"},
        )
        metadata = ResponseMetadata(
            duration=duration,
            model_slug=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=StopReason.STOP,
        )
        return GenerationResponse(
            message=assistant_message,
            request=request,
            metadata=metadata,
        )

    # Agent tier → SDK agent literal. See _gaos/types/interactions/agentoption.py
    _DEEP_RESEARCH_AGENTS: dict[str, str] = {
        "pro": "deep-research-pro-preview-12-2025",
        "standard": "deep-research-preview-04-2026",
        "max": "deep-research-max-preview-04-2026",
    }
    _DEFAULT_DEEP_RESEARCH_AGENT = "deep-research-preview-04-2026"

    @staticmethod
    def _record_deep_research_job(interaction_id: str, agent: str, query_preview: str) -> None:
        """
        Append a job record to ~/.conduit/deep_research_jobs.json so a crashed
        CLI can be recovered with `conduit deep-research-resume <id>`.
        """
        import json as _json
        from pathlib import Path
        from datetime import datetime, timezone

        state_dir = Path.home() / ".conduit"
        state_dir.mkdir(exist_ok=True)
        state_file = state_dir / "deep_research_jobs.json"

        jobs: list[dict] = []
        if state_file.exists():
            try:
                jobs = _json.loads(state_file.read_text())
            except (_json.JSONDecodeError, OSError):
                jobs = []

        jobs.append({
            "interaction_id": interaction_id,
            "agent": agent,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "query_preview": query_preview[:200],
        })
        state_file.write_text(_json.dumps(jobs, indent=2))

    async def _generate_deep_research(self, request: GenerationRequest) -> GenerationResponse:
        """
        Run Gemini Deep Research via the Interactions API (async polling).
        Takes 5-20 minutes. Citations in AssistantMessage.metadata["citations"].
        Triggered when client_params contains deep_research=True.

        Supported client_params keys:
          deep_research: bool — required to trigger
          deep_research_tier: "pro" | "standard" | "max" (default: standard)
          deep_research_agent: explicit agent literal (overrides tier)
          thinking_summaries: bool
          visualization: "off" | "auto"
          collaborative_planning: bool
        """
        import asyncio
        import warnings
        from google import genai
        from google.genai._gaos.types.interactions.modeloutputstep import ModelOutputStep
        from google.genai._gaos.types.interactions.textcontent import TextContent
        from google.genai._gaos.types.interactions.urlcitation import URLCitation

        client_params = request.params.client_params or {}
        agent_literal = client_params.get("deep_research_agent") or self._DEEP_RESEARCH_AGENTS.get(
            client_params.get("deep_research_tier", "standard"),
            self._DEFAULT_DEEP_RESEARCH_AGENT,
        )

        start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            native_client = genai.Client(api_key=self._get_api_key())

        # Extract system/user messages. The SDK *type signature* exposes a
        # system_instruction kwarg on the agent-create overload, but the server
        # rejects it for the deep-research agents ("not supported … include in
        # the 'input' prompt instead"). So we concatenate, same as the original
        # code — but with the correct reason recorded here.
        query_text = ""
        system_instruction: str | None = None
        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content if isinstance(msg.content, str) else str(msg.content)
            elif msg.role == Role.USER:
                query_text = msg.content if isinstance(msg.content, str) else str(msg.content)
        if system_instruction:
            query_text = f"{system_instruction}\n\n{query_text}"

        create_kwargs: dict[str, Any] = {
            "input": query_text,
            "agent": agent_literal,
            "background": True,
            "store": True,
        }

        # Deep Research agent config knobs
        agent_config: dict[str, Any] = {"type": "deep-research"}
        for key in ("thinking_summaries", "visualization", "collaborative_planning"):
            if key in client_params:
                agent_config[key] = client_params[key]
        if len(agent_config) > 1:
            create_kwargs["agent_config"] = agent_config

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interaction = await native_client.aio.interactions.create(**create_kwargs)

        interaction_id = interaction.id
        # Persist so a crashed CLI can recover via deep-research-resume.
        try:
            self._record_deep_research_job(interaction_id, agent_literal, query_text)
        except Exception:
            pass  # state-file failures must not break the actual call

        max_polls = 180  # 30 minutes at 10s intervals
        for _ in range(max_polls):
            await asyncio.sleep(10)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                interaction = await native_client.aio.interactions.get(interaction_id)
            if interaction.status in ("completed", "failed", "cancelled", "incomplete"):
                break

        if interaction.status != "completed":
            raise RuntimeError(
                f"Deep research {interaction_id} ended with status {interaction.status!r}. "
                f"Recover via: conduit deep-research-resume {interaction_id}"
            )

        return self._build_deep_research_response(
            interaction, request, start_time,
            ModelOutputStep, TextContent, URLCitation,
        )

    @staticmethod
    def _build_deep_research_response(
        interaction,
        request: GenerationRequest,
        start_time: float,
        ModelOutputStep: type,
        TextContent: type,
        URLCitation: type,
    ) -> GenerationResponse:
        """Extract text, citations, and token counts from a completed interaction."""
        duration = (time.time() - start_time) * 1000

        # Text: prefer the SDK-provided concatenation; fall back to walking steps.
        text_content = getattr(interaction, "output_text", None) or ""
        if not text_content:
            text_parts: list[str] = []
            for step in getattr(interaction, "steps", None) or []:
                if isinstance(step, ModelOutputStep):
                    for content in step.content or []:
                        if isinstance(content, TextContent) and content.text:
                            text_parts.append(content.text)
            text_content = "\n\n".join(text_parts)

        # Citations: URLCitation annotations on model output text content.
        citations: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for step in getattr(interaction, "steps", None) or []:
            if not isinstance(step, ModelOutputStep):
                continue
            for content in step.content or []:
                if not isinstance(content, TextContent):
                    continue
                for ann in content.annotations or []:
                    if not isinstance(ann, URLCitation):
                        continue
                    url = ann.url
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        citations.append({
                            "title": ann.title or "",
                            "url": url,
                            "source": "",
                            "date": "",
                        })

        usage = getattr(interaction, "usage", None)
        input_tokens = getattr(usage, "total_input_tokens", 0) or 0
        output_tokens = getattr(usage, "total_output_tokens", 0) or 0

        assistant_message = AssistantMessage(
            content=text_content,
            metadata={
                "citations": citations,
                "provider": "google",
                "interaction_id": getattr(interaction, "id", None),
                "agent": getattr(interaction, "agent", None),
            },
        )
        metadata = ResponseMetadata(
            duration=duration,
            model_slug=request.params.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=StopReason.STOP,
        )
        return GenerationResponse(
            message=assistant_message,
            request=request,
            metadata=metadata,
        )

    async def resume_deep_research(self, interaction_id: str) -> dict[str, Any]:
        """
        Poll an existing deep-research interaction by ID and return its raw result
        as a dict — used by `conduit deep-research-resume <id>` after a CLI crash.

        Returns: {"text": str, "citations": list, "interaction_id": str,
                  "agent": str, "input_tokens": int, "output_tokens": int}
        """
        import asyncio
        import warnings
        from google import genai
        from google.genai._gaos.types.interactions.modeloutputstep import ModelOutputStep
        from google.genai._gaos.types.interactions.textcontent import TextContent
        from google.genai._gaos.types.interactions.urlcitation import URLCitation

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            native_client = genai.Client(api_key=self._get_api_key())

        max_polls = 180
        interaction = None
        for _ in range(max_polls):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                interaction = await native_client.aio.interactions.get(interaction_id)
            if interaction.status in ("completed", "failed", "cancelled", "incomplete"):
                break
            await asyncio.sleep(10)

        if interaction is None or interaction.status != "completed":
            status = getattr(interaction, "status", "unknown")
            raise RuntimeError(f"Deep research {interaction_id} ended with status {status!r}")

        text_content = getattr(interaction, "output_text", None) or ""
        if not text_content:
            parts: list[str] = []
            for step in getattr(interaction, "steps", None) or []:
                if isinstance(step, ModelOutputStep):
                    for c in step.content or []:
                        if isinstance(c, TextContent) and c.text:
                            parts.append(c.text)
            text_content = "\n\n".join(parts)

        citations: list[dict[str, str]] = []
        seen: set[str] = set()
        for step in getattr(interaction, "steps", None) or []:
            if not isinstance(step, ModelOutputStep):
                continue
            for c in step.content or []:
                if not isinstance(c, TextContent):
                    continue
                for ann in c.annotations or []:
                    if isinstance(ann, URLCitation) and ann.url and ann.url not in seen:
                        seen.add(ann.url)
                        citations.append({"title": ann.title or "", "url": ann.url})

        usage = getattr(interaction, "usage", None)
        return {
            "text": text_content,
            "citations": citations,
            "interaction_id": interaction_id,
            "agent": getattr(interaction, "agent", None),
            "input_tokens": getattr(usage, "total_input_tokens", 0) or 0,
            "output_tokens": getattr(usage, "total_output_tokens", 0) or 0,
        }

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
