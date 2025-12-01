"""
Protocol definition for streaming response objects from LLM providers.

This protocol defines the minimal interface that streaming objects from different
providers (OpenAI, Anthropic, Google, Ollama) must satisfy to work with our parsers.
"""

from typing import Protocol, Any, runtime_checkable
from collections.abc import Iterator, AsyncIterator


@runtime_checkable
class StreamChunk(Protocol):
    """Protocol for individual chunks in a stream."""

    @property
    def content(self) -> str | None:
        """Text content delta in this chunk."""
        ...

    @property
    def usage(self) -> Any | None:
        """Usage information (only present in final chunk for some providers)."""
        ...


@runtime_checkable
class SyncStream(Protocol):
    """Protocol for synchronous streaming response objects."""

    def __iter__(self) -> Iterator[StreamChunk]:
        """Iterate over chunks as they arrive."""
        ...

    def close(self) -> None:
        """Close the stream and release resources."""
        ...


@runtime_checkable
class AsyncStream(Protocol):
    """Protocol for asynchronous streaming response objects."""

    def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Async iterate over chunks as they arrive."""
        ...

    async def close(self) -> None:
        """Close the stream and release resources."""
        ...
