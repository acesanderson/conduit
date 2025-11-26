import json
from abc import ABC, abstractmethod
from typing import Any
from conduit.parser.stream.protocol import SyncStream, AsyncStream


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

        Args:
            close_on_match: If True, close stream immediately after finding
                            first complete object (saves tokens in parallel scenarios)
            check_interval: Check for complete object every N chunks
                            (1 = every chunk, higher = less CPU overhead)

        Returns:
            tuple containing:
                - text: Content with structured object removed (before)
                - obj: Extracted object (XML string or dict), or None if not found
                - buffer: All content received before stream was closed
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

        Args:
            close_on_match: If True, close stream immediately after finding
                            first complete object (saves tokens)
            check_interval: Check for complete object every N chunks

        Returns:
            tuple containing:
                - text: Content with structured object removed
                - obj: Extracted object, or None
                - buffer: All content received before stream was closed
        """
        self.buffer = ""
        chunk_count = 0

        try:
            async for chunk in self.stream:
                content = self._get_chunk_content(chunk)
                if content:
                    self.buffer += content
                    chunk_count += 1

                if close_on_match and chunk_count % check_interval == 0:
                    text, obj = self._parse_buffer(self.buffer)
                    if obj:
                        if hasattr(self.stream, "close"):
                            await self.stream.close()
                        break
        finally:
            if hasattr(self.stream, "close"):
                await self.stream.close()

        text, obj = self._parse_buffer(self.buffer)
        return text, obj, self.buffer

    @abstractmethod
    def _parse_buffer(self, buffer: str) -> tuple[str, Any]:
        """
        Parse buffer and extract first structured object.

        Subclasses must implement this with their specific parsing logic
        (XML state machine, JSON state machine, etc.)

        Args:
            buffer: Complete text content accumulated so far

        Returns:
            tuple containing:
                - text: Buffer with first structured object removed
                - obj: Extracted object (type depends on parser), or None
        """
        ...

    def _get_chunk_content(self, chunk) -> str | None:
        """
        Extract content from chunk, handling different provider formats.

        Different LLM providers structure chunks differently:
        - OpenAI: chunk.choices[0].delta.content
        - Anthropic: chunk.delta.text
        - Google: chunk.text
        - Test fixtures: chunk.content

        Args:
            chunk: Stream chunk from any provider

        Returns:
            Text content from chunk, or None if no content
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

    Robustly detects complete XML elements even when split across chunks,
    handles nested tags with same name, and avoids false positives from
    XML-like text in strings.

    Uses depth tracking to find balanced XML tags:
    - Increments depth when opening tag found
    - Decrements depth when closing tag found
    - Complete object when depth returns to 0
    """

    def __init__(
        self, stream: SyncStream | AsyncStream, tag_name: str = "function_calls"
    ):
        """
        Initialize XML parser.

        Args:
            stream: Sync or async stream to parse
            tag_name: XML tag to search for (without < >)
        """
        super().__init__(stream)
        self.tag_name = tag_name
        self.start_tag = f"<{tag_name}"
        self.end_tag = f"</{tag_name}>"

    def _parse_buffer(self, buffer: str) -> tuple[str, str | None]:
        """
        Extract first complete XML object from buffer using state machine.

        Algorithm:
        1. Find first opening tag
        2. Track depth as we scan (increment on open, decrement on close)
        3. When depth returns to 0, we have a complete balanced object
        4. Handle nested tags with same name correctly

        Args:
            buffer: Text buffer to parse

        Returns:
            tuple containing:
                - text: Buffer with XML object removed (the text before)
                - xml_obj: Complete XML object as string, or None if not found
        """
        # Find first occurrence of opening tag
        start_index = buffer.find(self.start_tag)
        if start_index == -1:
            # No opening tag found
            return buffer, None

        # Initialize state machine
        depth = 1  # We found one opening tag
        scan_head = start_index + len(self.start_tag)

        # Scan through buffer tracking depth
        while depth > 0:
            # Look for next opening and closing tags
            next_start = buffer.find(self.start_tag, scan_head)
            next_end = buffer.find(self.end_tag, scan_head)

            # Check if we have a closing tag
            if next_end == -1:
                # No closing tag found - XML is incomplete
                return buffer, None

            # Determine which comes first: nested opening or closing tag
            if next_start != -1 and next_start < next_end:
                # Found nested opening tag (same tag name)
                depth += 1
                scan_head = next_start + len(self.start_tag)
            else:
                # Found closing tag
                depth -= 1
                scan_head = next_end + len(self.end_tag)

        # When we exit loop, depth is 0 and scan_head points to end of object
        object_end_index = scan_head

        # Extract the complete XML object
        xml_obj = buffer[start_index:object_end_index]

        # Extract text before the object -- text after is hallucinated
        text_before = buffer[:start_index]

        # Return text with object removed, and the object itself
        return text_before, xml_obj


class JSONStreamParser(StreamParser):
    """
    Parser for extracting JSON objects from streams using state machine.

    Tracks brace depth and string context to find complete JSON objects,
    handling nested objects and tricky edge cases like:
    - Braces inside strings: {"key": "has } brace"}
    - Escaped quotes: {"key": "has \" quote"}
    - Nested objects: {"outer": {"inner": "value"}}

    Uses state machine that tracks:
    - Brace depth (increments on {, decrements on })
    - Whether we're inside a string
    - Whether next character is escaped
    """

    def __init__(self, stream: SyncStream | AsyncStream):
        """Initialize JSON parser."""
        super().__init__(stream)

    def _parse_buffer(self, buffer: str) -> tuple[str, dict | list | None]:
        """
        Extract first complete JSON object from buffer using state machine.

        Algorithm:
        1. Find first opening brace { or bracket [
        2. Track depth, string context, and escape state as we scan
        3. When depth returns to 0 (outside any string), we have complete object
        4. Validate with json.loads() to catch false positives

        Args:
            buffer: Text buffer to parse

        Returns:
            tuple containing:
                - text: Buffer with JSON object removed
                - json_obj: Parsed JSON as dict/list, or None if not found
        """

        scan_head = 0
        while scan_head < len(buffer):
            # 1. Find first opening brace or bracket
            start_index = -1
            for i in range(scan_head, len(buffer)):
                if buffer[i] == "{" or buffer[i] == "[":
                    start_index = i
                    break

            if start_index == -1:
                # No (more) starting chars found
                return buffer, None

            start_char = buffer[start_index]
            end_char = "}" if start_char == "{" else "]"

            # 2. Initialize state machine
            depth = 1
            in_string = False
            escaped = False

            # 3. Scan char by char from after the starting brace
            for i in range(start_index + 1, len(buffer)):
                char = buffer[i]

                if escaped:
                    # Previous char was '\', so this char is a literal
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

                            # Success! Extract text before object, text after is hallucinated
                            text_before = buffer[:start_index]
                            return text_before, parsed_obj

                        except json.JSONDecodeError:
                            # False positive (e.g., "{key: 'val'}")
                            # Continue scanning from just after this
                            # invalid starting brace.
                            scan_head = start_index + 1
                            break  # Break inner loop, restart outer

            # If inner loop finished without finding end_char, JSON is incomplete
            if depth != 0:
                return buffer, None

            # If we're here, it was a false positive, outer loop continues

        # Scanned whole buffer, no complete valid JSON found
        return buffer, None


if __name__ == "__main__":
    from conduit.sync import Model
