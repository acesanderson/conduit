from __future__ import annotations
from typing import TYPE_CHECKING
from datetime import datetime
import json
from conduit.domain.result.response import Response
from conduit.utils.progress.verbosity import Verbosity
from conduit.domain.message.message import UserMessage

if TYPE_CHECKING:
    from conduit.domain.request.request import Request

# --- Helpers ---


def _extract_user_prompt(request: Request) -> str:
    """Extract a preview of the user prompt from the request."""
    user_message = request.messages[-1].content
    assert isinstance(user_message, UserMessage)
    return str(user_message)


# --- Plain Text Formatters: Response ---


def format_response_plain(response: Response, verbosity: Verbosity) -> str:
    """Entry point for formatting a Response object into plain text."""
    if verbosity == Verbosity.SUMMARY:
        return _response_summary_plain(response)
    elif verbosity == Verbosity.DETAILED:
        return _response_detailed_plain(response)
    elif verbosity == Verbosity.COMPLETE:
        return _response_complete_plain(response)
    elif verbosity == Verbosity.DEBUG:
        return _response_debug_plain(response)
    return ""


def _response_summary_plain(response: Response) -> str:
    lines = []
    timestamp = datetime.now().strftime("%H:%M:%S")
    duration = getattr(response.metadata, "duration", 0)

    # Header
    lines.append(f"[{timestamp}] RESPONSE {duration:.1f}s")

    # Content (Truncated)
    content = str(response.content or "No content")
    if len(content) > 100:
        content = content[:100] + "..."
    lines.append(f"  {content}")

    return "\n".join(lines)


def _response_detailed_plain(response: Response) -> str:
    lines = []
    timestamp = datetime.now().strftime("%H:%M:%S")
    duration = getattr(response.metadata, "duration", 0)

    # Header
    lines.append(f"[{timestamp}] CONVERSATION {duration:.1f}s (Detailed)")

    # User Prompt
    if response.request:
        user_prompt = _extract_user_prompt(response.request)
        if user_prompt:
            if len(user_prompt) > 200:
                user_prompt = user_prompt[:200] + "..."
            lines.append(f"User: {user_prompt}")

    # Assistant Content (Truncated)
    content = str(response.content or "No content")
    if len(content) > 200:
        content = content[:200] + "..."
    lines.append(f"Assistant: {content}")

    # Metadata
    meta = []
    if response.request:
        meta.append(f"Model: {response.request.params.model}")
        if response.request.params.temperature is not None:
            meta.append(f"Temp: {response.request.params.temperature}")

    if meta:
        lines.append("Metadata: " + " • ".join(meta))

    return "\n".join(lines)


def _response_complete_plain(response: Response) -> str:
    lines = []
    timestamp = datetime.now().strftime("%H:%M:%S")
    duration = getattr(response.metadata, "duration", 0)

    # Header
    lines.append(f"[{timestamp}] FULL CONVERSATION {duration:.1f}s")

    # Full User Prompt
    if response.request:
        user_prompt = _extract_user_prompt(response.request)
        if user_prompt:
            lines.append(f"User: {user_prompt}")

    # Full Assistant Content
    content = str(response.content or "No content")
    lines.append(f"Assistant: {content}")

    # Detailed Metadata
    meta = []
    if response.request:
        meta.append(f"Model: {response.request.params.model}")
        if response.request.params.temperature is not None:
            meta.append(f"Temp: {response.request.params.temperature}")

        # Check for parser
        # Note: Response model logic usually lives in params, tricky to extract name dynamically
        # if the class is hidden, but we try:
        if response.request.params.response_model:
            parser_name = str(response.request.params.response_model)
            meta.append(f"Parser: {parser_name}")

    if response.metadata:
        meta.append(f"Input Tokens: {response.metadata.input_tokens}")
        meta.append(f"Output Tokens: {response.metadata.output_tokens}")
        meta.append(f"Stop Reason: {response.metadata.stop_reason}")

    if meta:
        lines.append("Metadata: " + " • ".join(meta))

    return "\n".join(lines)


def _response_debug_plain(response: Response) -> str:
    timestamp = datetime.now().strftime("%H:%M:%S")
    duration = getattr(response.metadata, "duration", 0)

    lines = [f"[{timestamp}] CONVERSATION DEBUG {duration:.1f}s"]

    # Construct clean debug object
    debug_data = {
        "model": response.request.params.model if response.request else "unknown",
        "duration": duration,
        "metadata": response.metadata,
    }

    if response.request:
        debug_data["user_prompt"] = _extract_user_prompt(response.request)
        debug_data["full_request"] = response.request

    # Handle content serialization
    content = response.content
    try:
        if hasattr(content, "model_dump"):
            debug_data["assistant_response"] = content.model_dump()
        else:
            debug_data["assistant_response"] = str(content)
    except Exception:
        debug_data["assistant_response"] = str(content)

    json_str = json.dumps(debug_data, indent=2, default=str)
    lines.append(json_str)

    return "\n".join(lines)
