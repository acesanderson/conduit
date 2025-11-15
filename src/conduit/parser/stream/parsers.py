# src/conduit/parser/stream/parsers.py
"""
Stream parsers for extracting structured content from LLM response streams.

Provides robust parsers that use state machines (not regex) to incrementally
detect and extract XML and JSON objects as stream chunks arrive.
"""

from abc import ABC, abstractmethod
from typing import Any
from conduit.parser.stream.protocol import SyncStream, AsyncStream


class StreamParser(ABC):
    """
    Abstract base class for stream parsers.

    Parsers consume streaming responses and extract structured content
    (XML or JSON) while tracking the full buffer for tokenization.
    """

    def __init__(self, stream: SyncStream | AsyncStream):
        self.stream = stream
        self.buffer = ""  # Full accumulated content

    @abstractmethod
    def parse(self) -> tuple[str, Any, str]:
        """
        Parse the stream and extract structured content.

        Returns:
            tuple containing:
                - extracted_text: Text before/around structured content
                - structured_obj: Parsed object (XML string or dict), or None
                - full_content: Complete stream content for tokenization
        """
        ...

    @abstractmethod
    async def parse_async(self) -> tuple[str, Any, str]:
        """Async version of parse for async streams."""
        ...


class XMLStreamParser(StreamParser):
    """
    Parser for extracting XML objects from streams using state machine.

    Robustly detects complete XML elements even when split across chunks,
    handles nesting, and avoids false positives from XML-like text.
    """

    def __init__(
        self, stream: SyncStream | AsyncStream, tag_name: str = "function_calls"
    ):
        super().__init__(stream)
        self.tag_name = tag_name

    def parse(self) -> tuple[str, str | None, str]:
        """
        Extract XML object from stream.

        Returns:
            - text: Content before the XML object
            - xml: The complete XML object as string, or None if not found
            - full_content: All stream content
        """
        ...

    async def parse_async(self) -> tuple[str, str | None, str]:
        """Async version."""
        ...

    def _extract_xml(self, buffer: str) -> tuple[str, str | None]:
        """
        Extract XML from buffer using state machine.

        Tracks tag depth to find complete, balanced XML object.
        """
        ...


class JSONStreamParser(StreamParser):
    """
    Parser for extracting JSON objects from streams using state machine.

    Tracks brace depth and string context to find complete JSON objects,
    handling nested objects and tricky edge cases like braces in strings.
    """

    def __init__(self, stream: SyncStream | AsyncStream):
        super().__init__(stream)

    def parse(self) -> tuple[str, dict | None, str]:
        """
        Extract JSON object from stream.

        Returns:
            - text: Content before the JSON object
            - json_obj: Parsed JSON as dict, or None if not found
            - full_content: All stream content
        """
        ...

    async def parse_async(self) -> tuple[str, dict | None, str]:
        """Async version."""
        ...

    def _extract_json(self, buffer: str) -> tuple[str, dict | None]:
        """
        Extract JSON from buffer using state machine.

        Tracks brace depth and whether we're inside a string to
        find complete, valid JSON object.
        """
        ...


if __name__ == "__main__":
    # Demonstrate usage with test streams
    from conduit.parser.stream.fixtures import (
        create_test_stream,
        SIMPLE_XML_CHUNKS,
        SIMPLE_JSON_CHUNKS,
        NESTED_XML_CHUNKS,
    )

    print("=== XML Parser Demo ===\n")

    xml_stream = create_test_stream(SIMPLE_XML_CHUNKS)
    xml_parser = XMLStreamParser(xml_stream)
    text, xml_obj, full_content = xml_parser.parse()

    print(f"Text before XML: {repr(text)}")
    print(f"XML object: {repr(xml_obj)}")
    print(f"Full content length: {len(full_content)}")
    print()

    print("=== JSON Parser Demo ===\n")

    json_stream = create_test_stream(SIMPLE_JSON_CHUNKS)
    json_parser = JSONStreamParser(json_stream)
    text, json_obj, full_content = json_parser.parse()

    print(f"Text before JSON: {repr(text)}")
    print(f"JSON object: {json_obj}")
    print(f"Full content length: {len(full_content)}")
    print()

    print("=== Nested XML Demo ===\n")

    nested_stream = create_test_stream(NESTED_XML_CHUNKS)
    nested_parser = XMLStreamParser(nested_stream)
    text, xml_obj, full_content = nested_parser.parse()

    print(f"Text: {repr(text)}")
    print(f"Nested XML: {repr(xml_obj)}")
    print()
