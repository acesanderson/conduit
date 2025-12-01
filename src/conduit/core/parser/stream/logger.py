"""
Logging utilities for debugging stream parsers.

Provides a wrapper that logs chunk arrivals in real-time, making it easy
to see exactly what content arrives in each chunk and diagnose parsing issues.
"""

import logging
from collections.abc import Iterator, AsyncIterator
from conduit.core.parser.stream.protocol import SyncStream, AsyncStream, StreamChunk


logger = logging.getLogger(__name__)


class StreamLogger:
    """
    Wrapper that logs stream chunks as they arrive.

    Three display modes:
    - "inline": Print content on same line as it arrives (for user display)
    - "per-chunk": Log each chunk separately with metadata (for debugging)
    - "silent": Collect chunks without output (for production)

    Usage:
        # For debugging - see each chunk
        logged_stream = StreamLogger(stream, name="debug", mode="per-chunk")

        # For user display - see flowing text
        logged_stream = StreamLogger(stream, name="response", mode="inline")

        # For production - just collect, no output
        logged_stream = StreamLogger(stream, name="prod", mode="silent")
    """

    def __init__(
        self,
        stream: SyncStream | AsyncStream,
        name: str = "stream",
        mode: str = "inline",  # "inline", "per-chunk", "silent"
        max_content_display: int = 100,
    ):
        """
        Initialize stream logger.

        Args:
            stream: The underlying stream to wrap
            name: Identifier for this stream in logs
            mode: Display mode - "inline", "per-chunk", or "silent"
            max_content_display: Max chars to show per chunk in per-chunk mode
        """
        self.stream = stream
        self.name = name
        self.mode = mode
        self.max_content_display = max_content_display
        self.chunk_count = 0
        self.total_chars = 0
        self._buffer = ""

        if mode not in ("inline", "per-chunk", "silent"):
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'inline', 'per-chunk', or 'silent'"
            )

    def __iter__(self) -> Iterator[StreamChunk]:
        """Iterate over chunks with appropriate logging based on mode."""
        if self.mode == "inline":
            print(f"[{self.name}] ", end="", flush=True)
        elif self.mode == "per-chunk":
            logger.debug(f"[{self.name}] Stream starting...")

        for chunk in self.stream:
            self.chunk_count += 1

            # Extract content from chunk
            content = self._get_chunk_content(chunk)

            if content:
                self.total_chars += len(content)
                self._buffer += content

                if self.mode == "inline":
                    # Print on same line
                    print(content, end="", flush=True)
                elif self.mode == "per-chunk":
                    # Log each chunk with metadata
                    display_content = content
                    if len(content) > self.max_content_display:
                        display_content = content[: self.max_content_display] + "..."

                    logger.debug(
                        f"[{self.name}] Chunk {self.chunk_count}: {repr(display_content)} "
                        f"({len(content)} chars, {self.total_chars} total)"
                    )
            else:
                # Empty chunk (likely final usage chunk)
                if self.mode == "per-chunk":
                    logger.debug(
                        f"[{self.name}] Chunk {self.chunk_count}: <empty> "
                        f"(usage: {chunk.usage if hasattr(chunk, 'usage') else None})"
                    )

            yield chunk

        # Final summary
        if self.mode == "inline":
            print()  # Newline after inline content

        if self.mode != "silent":
            logger.info(
                f"[{self.name}] Complete: {self.chunk_count} chunks, {self.total_chars} chars"
            )

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Async iterate over chunks with appropriate logging based on mode."""
        if self.mode == "inline":
            print(f"[{self.name}] ", end="", flush=True)
        elif self.mode == "per-chunk":
            logger.debug(f"[{self.name}] Async stream starting...")

        async for chunk in self.stream:
            self.chunk_count += 1

            content = self._get_chunk_content(chunk)

            if content:
                self.total_chars += len(content)
                self._buffer += content

                if self.mode == "inline":
                    print(content, end="", flush=True)
                elif self.mode == "per-chunk":
                    display_content = content
                    if len(content) > self.max_content_display:
                        display_content = content[: self.max_content_display] + "..."

                    logger.debug(
                        f"[{self.name}] Chunk {self.chunk_count}: {repr(display_content)} "
                        f"({len(content)} chars, {self.total_chars} total)"
                    )
            else:
                if self.mode == "per-chunk":
                    logger.debug(
                        f"[{self.name}] Chunk {self.chunk_count}: <empty> "
                        f"(usage: {chunk.usage if hasattr(chunk, 'usage') else None})"
                    )

            yield chunk

        if self.mode == "inline":
            print()

        if self.mode != "silent":
            logger.info(
                f"[{self.name}] Async complete: {self.chunk_count} chunks, {self.total_chars} chars"
            )

    def close(self) -> None:
        """Close the underlying stream."""
        if self.mode == "per-chunk":
            logger.debug(f"[{self.name}] Closing stream...")
        self.stream.close()

    async def aclose(self) -> None:
        """Close the underlying async stream."""
        if self.mode == "per-chunk":
            logger.debug(f"[{self.name}] Closing async stream...")
        await self.stream.close()

    def _get_chunk_content(self, chunk: StreamChunk) -> str | None:
        """
        Extract content from chunk, handling different provider formats.

        Different providers structure chunks differently:
        - OpenAI: chunk.choices[0].delta.content
        - Anthropic: chunk.delta.text
        - Google: chunk.text
        - Our test chunks: chunk.content
        """
        # Try test format first
        if hasattr(chunk, "content"):
            return chunk.content

        # Try OpenAI format
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content"):
                return delta.content

        # Try Anthropic format
        if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
            return chunk.delta.text

        # Try Google format
        if hasattr(chunk, "text"):
            return chunk.text

        return None

    def get_buffer(self) -> str:
        """Get accumulated content so far."""
        return self._buffer

    def get_stats(self) -> dict:
        """Get statistics about the stream."""
        return {
            "chunk_count": self.chunk_count,
            "total_chars": self.total_chars,
            "buffer_length": len(self._buffer),
        }

    def reset_stats(self) -> None:
        """Reset chunk count and character count."""
        self.chunk_count = 0
        self.total_chars = 0
        self._buffer = ""
