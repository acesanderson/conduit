import json
from abc import ABC, abstractmethod
from typing import Any
from conduit.core.parser.stream.protocol import SyncStream, AsyncStream


class StreamParser(ABC):
    """
    Abstract base class for stream parsers.

    Parsers consume streaming responses and extract structured content
    (XML or JSON) while tracking the full buffer for tokenization.

    Supports early termination: can close stream as soon as first complete
    object is found (useful for token efficiency with parallel async requests).
    """

    def __init__(self, stream: SyncStream | AsyncStream):
        self.stream = stream
        self.buffer = ""

    def parse(
        self, close_on_match: bool = True, check_interval: int = 1
    ) -> tuple[str, Any, str]:
        """
        Consume sync stream and parse.
        """
        self.buffer = ""
        chunk_count = 0

        try:
            for chunk in self.stream:
                content = self._get_chunk_content(chunk)
                if content:
                    self.buffer += content
                    chunk_count += 1

                # Check for complete object periodically
                if close_on_match and chunk_count % check_interval == 0:
                    text, obj = self._parse_buffer(self.buffer)
                    if obj:
                        # Found complete object - close stream to save tokens
                        if hasattr(self.stream, "close"):
                            self.stream.close()
                        break
        finally:
            # Ensure stream is always closed
            if hasattr(self.stream, "close"):
                self.stream.close()

        # Final parse on complete buffer
        text, obj = self._parse_buffer(self.buffer)
        return text, obj, self.buffer

    async def parse_async(
        self, close_on_match: bool = True, check_interval: int = 1
    ) -> tuple[str, Any, str]:
        """
        Consume async stream and parse.
        """
        self.buffer = ""
        chunk_count = 0

        try:
            # Iterate asynchronously over the stream
            async for chunk in self.stream:
                content = self._get_chunk_content(chunk)
                if content:
                    self.buffer += content
                    chunk_count += 1

                if close_on_match and chunk_count % check_interval == 0:
                    text, obj = self._parse_buffer(self.buffer)
                    if obj:
                        # Found object, close stream
                        if hasattr(self.stream, "close"):
                            # check if close is async (it usually is for async streams)
                            close_method = self.stream.close
                            if hasattr(close_method, "__call__") and hasattr(
                                close_method, "__await__"
                            ):
                                await close_method()
                            else:
                                close_method()
                        break
        finally:
            if hasattr(self.stream, "close"):
                close_method = self.stream.close
                if hasattr(close_method, "__call__") and hasattr(
                    close_method, "__await__"
                ):
                    await close_method()
                else:
                    close_method()

        text, obj = self._parse_buffer(self.buffer)
        return text, obj, self.buffer

    @abstractmethod
    def _parse_buffer(self, buffer: str) -> tuple[str, Any]:
        """
        Parse buffer and extract first structured object.
        Returns: (text_before_object, extracted_object)
        """
        ...

    def _get_chunk_content(self, chunk) -> str | None:
        """
        Extract content from chunk, handling different provider formats.
        """
        # Try test fixture format first
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


class XMLStreamParser(StreamParser):
    """
    Parser for extracting XML objects from streams using state machine.
    """

    def __init__(
        self, stream: SyncStream | AsyncStream, tag_name: str = "function_calls"
    ):
        super().__init__(stream)
        self.tag_name = tag_name
        self.start_tag = f"<{tag_name}"
        self.end_tag = f"</{tag_name}>"

    def _parse_buffer(self, buffer: str) -> tuple[str, str | None]:
        # Find first occurrence of opening tag
        start_index = buffer.find(self.start_tag)
        if start_index == -1:
            return buffer, None

        # Initialize state machine
        depth = 1
        scan_head = start_index + len(self.start_tag)

        # Scan through buffer tracking depth
        while depth > 0:
            next_start = buffer.find(self.start_tag, scan_head)
            next_end = buffer.find(self.end_tag, scan_head)

            if next_end == -1:
                return buffer, None

            if next_start != -1 and next_start < next_end:
                depth += 1
                scan_head = next_start + len(self.start_tag)
            else:
                depth -= 1
                scan_head = next_end + len(self.end_tag)

        object_end_index = scan_head

        # Extract the complete XML object
        xml_obj = buffer[start_index:object_end_index]

        # Extract text BEFORE the object.
        # Everything after object_end_index is discarded (hallucination)
        text_before = buffer[:start_index]

        return text_before, xml_obj


class JSONStreamParser(StreamParser):
    """
    Parser for extracting JSON objects from streams using state machine.
    """

    def __init__(self, stream: SyncStream | AsyncStream):
        super().__init__(stream)

    def _parse_buffer(self, buffer: str) -> tuple[str, dict | list | None]:
        scan_head = 0
        while scan_head < len(buffer):
            # 1. Find first opening brace or bracket
            start_index = -1
            for i in range(scan_head, len(buffer)):
                if buffer[i] == "{" or buffer[i] == "[":
                    start_index = i
                    break

            if start_index == -1:
                return buffer, None

            start_char = buffer[start_index]
            end_char = "}" if start_char == "{" else "]"

            # 2. Initialize state machine
            depth = 1
            in_string = False
            escaped = False

            # 3. Scan char by char
            for i in range(start_index + 1, len(buffer)):
                char = buffer[i]

                if escaped:
                    escaped = False
                    continue

                if char == "\\":
                    escaped = True
                    continue

                if char == '"':
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == start_char:
                        depth += 1
                    elif char == end_char:
                        depth -= 1

                    if depth == 0:
                        # 4. Found balanced structure
                        end_index = i
                        candidate_str = buffer[start_index : end_index + 1]

                        try:
                            # 5. Validate with json.loads()
                            parsed_obj = json.loads(candidate_str)

                            # Success!
                            # Return text BEFORE the object. Discard tail.
                            text_before = buffer[:start_index]
                            return text_before, parsed_obj

                        except json.JSONDecodeError:
                            scan_head = start_index + 1
                            break

            if depth != 0:
                return buffer, None

        return buffer, None
